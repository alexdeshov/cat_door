"""Замер самой модели: сколько стоит кадр на каждом устройстве.

Парный скрипт к `scripts/benchmark_detector.py`. Тот меряет **конвейер** —
живые камеры, очереди, ожидание кадра; этот меряет **модель** и ничего кроме.
Числа из них не взаимозаменяемы, и расхождение между ними само по себе полезно:
если модель ускорилась вдвое, а проба конвейера не сдвинулась, значит время
уходит не туда, куда думали.

Замер идёт через `OpenVinoCatDetector`, а не через голый `ov.CompiledModel`.
Сравнивать fp16 с int8 надо по тому пути, которым мы реально ходим: с проверкой
контрольной суммы, манифестом, препроцессингом и разбором выходов. Заодно
скрипт физически не может померить модель, чей манифест ей не соответствует.

Это замена скрипта, который снимал таблицу 16.5 / 10.4 мс 2026-08-08: тот жил в
`/tmp` на узле и потерялся вместе с ним. Замеры, которые нельзя повторить, —
это не замеры.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import median, quantiles
from typing import Any

import numpy as np

from src.inference.cat_detector import OpenVinoCatDetector


def load_frames(path: Path | None, *, input_size: int, count: int, seed: int) -> np.ndarray:
    """Кадры для замера: свои, если дали, иначе шум.

    Шум годится для времени и **не годится** для сравнения уверенностей —
    поэтому колонки расхождения без `--frames` не печатаются вовсе. На шуме они
    выглядели бы как измерение, не будучи им.
    """
    if path is None:
        generator = np.random.default_rng(seed)
        return generator.integers(
            0,
            256,
            size=(count, input_size, input_size, 3),
            dtype=np.uint8,
        )
    with np.load(path, allow_pickle=False) as archive:
        frames = np.asarray(archive["frames"])
    if frames.ndim != 4 or frames.shape[1:] != (input_size, input_size, 3):
        raise ValueError(
            f"frames from {path.name} have shape {frames.shape}, "
            f"expected (N, {input_size}, {input_size}, 3)"
        )
    return frames.astype(np.uint8, copy=False)


def _percentile(values: list[float], percent: int) -> float | None:
    if not values:
        return None
    if len(values) < 2:
        return values[0]
    return quantiles(values, n=100, method="inclusive")[percent - 1]


def measure(
    model_path: Path,
    device: str,
    frames: np.ndarray,
    *,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    detector = OpenVinoCatDetector(model_path, device)
    total = len(frames)
    # Прогрев отдельно и вне статистики: первые вызовы на видеоядре включают
    # компиляцию ядер и стоят в разы больше установившегося времени.
    for index in range(warmup):
        detector.detect(frames[index % total])
    timings: list[float] = []
    confidences: list[float] = []
    for index in range(iterations):
        frame = frames[index % total]
        started_at = time.perf_counter()
        detection = detector.detect(frame)
        timings.append((time.perf_counter() - started_at) * 1000)
        confidences.append(detection.confidence)
    return {
        "model": model_path.name,
        "model_sha256": detector.model_sha256[:12],
        "architecture": detector.architecture,
        "input_size": detector.input_size,
        "device_requested": device,
        "device_actual": detector.device,
        "iterations": iterations,
        "warmup": warmup,
        "median_ms": round(median(timings), 3),
        "p95_ms": round(_percentile(timings, 95) or 0.0, 3),
        "fps": round(1000.0 / median(timings), 1),
        # Уверенности первого круга по кадрам — материал для сравнения с базовой
        # моделью. Хранятся только в памяти: в отчёт уходят уже расхождения.
        "_confidences": confidences[: min(iterations, total)],
    }


def compare(baseline: dict[str, Any], candidate: dict[str, Any], *, verdict: float) -> dict[str, Any]:
    """Насколько кандидат разошёлся с базовой моделью на тех же кадрах.

    Ускорение без этой проверки ничего не значит: модель, которая втрое быстрее
    и слепая, по таблице времени выглядит победителем.
    """
    pairs = list(zip(baseline["_confidences"], candidate["_confidences"], strict=False))
    if not pairs:
        return {}
    deltas = [abs(first - second) for first, second in pairs]
    return {
        "compared_frames": len(pairs),
        "mean_abs_confidence_delta": round(sum(deltas) / len(deltas), 4),
        "max_abs_confidence_delta": round(max(deltas), 4),
        "verdict_threshold": verdict,
        "detections_changed": sum(
            1
            for first, second in pairs
            if (first >= verdict) != (second >= verdict)
        ),
    }


def run_benchmark(
    model_paths: list[Path],
    *,
    devices: list[str],
    warmup: int,
    iterations: int,
    frames_path: Path | None,
    verdict_threshold: float,
    seed: int,
    output: Path | None = None,
) -> dict[str, Any]:
    if not model_paths:
        raise ValueError("at least one model is required")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if output is not None and output.exists():
        raise FileExistsError(f"refusing to overwrite benchmark report {output}")
    # Размер входа берём из манифеста, а не из скомпилированной модели: иначе
    # ради одного числа пришлось бы поднимать её на процессоре — в том числе
    # когда меряют только видеоядро.
    manifest = json.loads(model_paths[0].with_suffix(".json").read_text(encoding="utf-8"))
    frames = load_frames(
        frames_path,
        input_size=int(manifest["input"]["shape"][2]),
        count=min(iterations, 64),
        seed=seed,
    )
    cells: list[dict[str, Any]] = []
    for device in devices:
        baseline: dict[str, Any] | None = None
        for model_path in model_paths:
            cell = measure(
                model_path,
                device,
                frames,
                warmup=warmup,
                iterations=iterations,
            )
            if baseline is None:
                baseline = cell
            else:
                cell["versus_baseline"] = {
                    "model": baseline["model"],
                    "speedup": round(baseline["median_ms"] / cell["median_ms"], 2),
                    # Расхождение уверенностей — только по настоящим кадрам.
                    **(
                        compare(baseline, cell, verdict=verdict_threshold)
                        if frames_path is not None
                        else {}
                    ),
                }
            cells.append(cell)
    for cell in cells:
        cell.pop("_confidences", None)
    report = {
        "frames": frames_path.name if frames_path is not None else "pseudorandom",
        "frame_count": len(frames),
        "cells": cells,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark detector models per device through the real inference path. "
            "For the whole pipeline, see scripts/benchmark_detector.py."
        )
    )
    parser.add_argument("models", type=Path, nargs="+")
    parser.add_argument(
        "--device",
        action="append",
        dest="devices",
        help="OpenVINO device; repeat to compare several (default: CPU)",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument(
        "--frames",
        type=Path,
        help=(
            "npz with own letterboxed frames. Without it timings are still valid "
            "but confidence deltas are not reported at all."
        ),
    )
    parser.add_argument("--verdict-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = run_benchmark(
        arguments.models,
        devices=arguments.devices or ["CPU"],
        warmup=arguments.warmup,
        iterations=arguments.iterations,
        frames_path=arguments.frames,
        verdict_threshold=arguments.verdict_threshold,
        seed=arguments.seed,
        output=arguments.output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
