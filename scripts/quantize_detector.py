"""Квантование детектора в int8 через NNCF, с калибровкой на своих кадрах.

Запускается **на маке**: сбор статистики идёт через onnxruntime, и ни ему, ни
NNCF нечего делать на боевом узле. Кадры приезжают файлом от
`scripts/collect_calibration_frames.py`.

Результат — QDQ ONNX, а не OpenVINO IR. Причина не в удобстве: `AGENTS.md`
объявляет ONNX форматом передачи моделей на сервер, а весь измерительный
аппарат проекта однофайловый — детектор считает хеш одного файла, реестр
моделей идемпотентен по одному sha256, выбор модели для оценки отвергает всё,
кроме `.onnx`. QDQ ONNX проходит существующую офлайн-оценку без единой правки, а
IR потребовал бы сначала переписать пять модулей — и тогда сравнение fp16 с int8
перепуталось бы с рефакторингом. OpenVINO читает QDQ и опускает
`QuantizeLinear`/`DequantizeLinear` в низкоточный путь и на процессоре, и на
видеоядре, так что вызов `compile_model` не меняется вовсе.
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from scripts.prepare_detector_candidate import assert_checksum, sha256
from src.inference.cat_detector import build_detector_tensor
from src.inference.detector_adapters import create_output_adapter

# Постобработка квантованию не подлежит. Игнорируем по типам операций, а не по
# именам узлов: имена в сквозном экспорте ultralytics зависят от версии, а
# `NonMaxSuppression` — имя операции ONNX и стабильно.
IGNORED_TYPES = ("NonMaxSuppression", "TopK", "RoiAlign")

# Проверенные параметры препроцессинга. Набор, собранный для 640-й модели, на
# 416-й «сработает» и даст мусорные диапазоны, поэтому расхождение — отказ.
CHECKED_KEYS = (
    "input_size",
    "letterbox_value",
    "resize_mode",
    "letterbox_alignment",
    "channel_order",
)


def load_calibration(path: Path, manifest: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        frames = np.asarray(archive["frames"])
        provenance = json.loads(bytes(archive["provenance"]).decode("utf-8"))
    source = manifest["input"]
    expected = {
        "input_size": int(source["shape"][2]),
        "letterbox_value": int(source.get("letterbox_value", 0)),
        "resize_mode": str(source.get("resize_mode", "letterbox")),
        "letterbox_alignment": str(source.get("letterbox_alignment", "center")),
        "channel_order": str(source.get("channel_order", "rgb")),
    }
    mismatched = [key for key in CHECKED_KEYS if provenance.get(key) != expected[key]]
    if mismatched:
        raise ValueError(
            "calibration set was built for a different preprocessing: "
            + ", ".join(f"{key}={provenance.get(key)!r} vs {expected[key]!r}" for key in mismatched)
        )
    if provenance.get("model_sha256") != manifest["sha256"]:
        raise ValueError("calibration set was built for a different model")
    return frames, provenance


def tensor_builder(manifest: dict[str, Any]):
    source = manifest["input"]
    name = str(source.get("name", "images"))
    channel_order = str(source.get("channel_order", "rgb"))
    scale = float(source.get("scale", 1.0 / 255.0))
    mean = np.asarray(source.get("mean", [0.0, 0.0, 0.0]), dtype=np.float32)
    std = np.asarray(source.get("std", [1.0, 1.0, 1.0]), dtype=np.float32)

    def build(frame: np.ndarray) -> dict[str, np.ndarray]:
        return {
            name: build_detector_tensor(
                frame,
                channel_order=channel_order,
                scale=scale,
                mean=mean,
                std=std,
            )
        }

    return build


def confidences_of(
    model_path: Path,
    manifest: dict[str, Any],
    frames: np.ndarray,
) -> list[float]:
    """Максимальная уверенность класса «кот» по кадрам — через настоящий адаптер.

    Именно через адаптер, а не по сырому тензору: расхождение, которое не видно
    после разбора выходов, нас не касается, а видное — касается целиком.
    """
    import onnxruntime

    session = onnxruntime.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    adapter = create_output_adapter(manifest, int(manifest["input"]["shape"][2]))
    build = tensor_builder(manifest)
    output_names = [str(name) for name in manifest["outputs"]]
    results: list[float] = []
    for frame in frames:
        raw = session.run(output_names, build(frame))
        parsed = adapter.parse(dict(zip(output_names, map(np.asarray, raw), strict=True)))
        results.append(max((item.confidence for item in parsed), default=0.0))
    return results


def quantize(
    model_path: Path,
    calibration_path: Path,
    output: Path,
    *,
    holdout: int = 32,
    max_confidence_delta: float = 0.05,
    verdict_threshold: float = 0.5,
    ignore_nodes: list[str] | None = None,
) -> Path:
    import nncf
    import onnx

    manifest_path = model_path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert_checksum(model_path, manifest["sha256"])
    output_manifest = output.with_suffix(".json")
    if output.exists() or output_manifest.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    source_model = onnx.load(str(model_path))
    graph_types = {node.op_type for node in source_model.graph.node}
    present = [op_type for op_type in IGNORED_TYPES if op_type in graph_types]
    # `validate=True` отвергает тип, которого в графе нет, поэтому список
    # сужается до фактически присутствующих. Строгость при этом не теряется:
    # опечатка в `--ignore-node` по-прежнему ошибка, а несоответствие манифеста
    # графу ловится проверкой ниже.
    if manifest.get("postprocessing", {}).get("nms_in_graph") and "NonMaxSuppression" not in (
        graph_types
    ):
        raise ValueError(
            "manifest says NMS is in the graph, but the graph has no NonMaxSuppression"
        )

    frames, provenance = load_calibration(calibration_path, manifest)
    if len(frames) <= holdout:
        raise ValueError(
            f"calibration set has {len(frames)} frames, "
            f"not enough to hold {holdout} of them out"
        )
    calibration_frames = frames[:-holdout]
    holdout_frames = frames[-holdout:]

    ignored = nncf.IgnoredScope(
        types=present,
        names=list(ignore_nodes or []),
        # Опечатка в имени узла обязана быть ошибкой, а не пустым множеством:
        # молча пустой ignored scope даёт модель, которая прекрасно квантуется
        # и ничего не находит.
        validate=True,
    )
    # Печатается при каждом запуске: что именно исключили из квантования —
    # ровно тот вопрос, который задают, когда модель повела себя странно.
    print(f"ignored op types: {', '.join(present) or '(none present in the graph)'}")
    if ignore_nodes:
        print(f"ignored node names: {', '.join(ignore_nodes)}")

    quantized = nncf.quantize(
        source_model,
        nncf.Dataset(calibration_frames, transform_func=tensor_builder(manifest)),
        # MIXED — симметричные веса и асимметричные активации, стандартный рецепт
        # для детекторов семейства YOLO. Умолчание PERFORMANCE известно потерей
        # точности на головах с широким диапазоном активаций.
        preset=nncf.QuantizationPreset.MIXED,
        subset_size=len(calibration_frames),
        ignored_scope=ignored,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        onnx.save(quantized, str(output))
        before = confidences_of(model_path, manifest, holdout_frames)
        after = confidences_of(output, manifest, holdout_frames)
        deltas = [abs(first - second) for first, second in zip(before, after, strict=True)]
        # Самопроверка смотрит уверенность класса «кот». Если ни на одном
        # отложенном кадре исходная модель кота не нашла, все расхождения
        # окажутся нулями — и отчёт будет выглядеть безупречно, не проверив
        # ничего. Это тот самый молчаливый ноль, который дороже явного отказа.
        informative = sum(1 for value in before if value > 0.0)
        if not informative:
            raise ValueError(
                "the source model found no cat on any holdout frame, so the "
                "self-check proves nothing: include frames with cats in the "
                "calibration set"
            )
        self_check = {
            "holdout_frames": len(deltas),
            "frames_with_a_cat": informative,
            "mean_abs_delta": round(sum(deltas) / len(deltas), 4),
            "max_abs_delta": round(max(deltas), 4),
            "verdict_threshold": verdict_threshold,
            "verdict_flips": sum(
                1
                for first, second in zip(before, after, strict=True)
                if (first >= verdict_threshold) != (second >= verdict_threshold)
            ),
        }
        print(json.dumps(self_check, ensure_ascii=False))
        if self_check["mean_abs_delta"] > max_confidence_delta:
            # Модель, которая втрое быстрее и слепая, до узла доехать не должна.
            raise ValueError(
                f"quantized model drifted by {self_check['mean_abs_delta']}, "
                f"limit is {max_confidence_delta}"
            )
        quantized_manifest = copy.deepcopy(manifest)
        quantized_manifest["sha256"] = sha256(output)
        quantized_manifest["precision"] = "int8"
        quantized_manifest["quantization"] = {
            "method": "nncf_ptq",
            "backend": "onnx",
            "nncf_version": nncf.__version__,
            "preset": "mixed",
            "subset_size": len(calibration_frames),
            "ignored_types": present,
            "ignored_names": list(ignore_nodes or []),
            "source_model": {"file": model_path.name, "sha256": manifest["sha256"]},
            "calibration": {
                "file": calibration_path.name,
                "sha256": sha256(calibration_path),
                "frames": len(calibration_frames),
                "seed": provenance.get("seed"),
                "negative_fraction": provenance.get("negative_fraction"),
                "events": len(provenance.get("events", [])),
                # Одни и те же проходы идут и в калибровку, и в оценку. При таком
                # объёме данных иначе никак, и это не то же, что загрязнение
                # обучающей выборки — PTQ подбирает масштабы, а не веса. Но
                # читать разницу int8/fp16 как несмещённую оценку обобщения
                # нельзя, и запись об этом обязана ехать вместе с моделью.
                "overlaps_evaluation_set": True,
            },
            "self_check": self_check,
            "created_at": datetime.now(UTC).isoformat(),
        }
        output_manifest.write_text(
            json.dumps(quantized_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        output.unlink(missing_ok=True)
        output_manifest.unlink(missing_ok=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantize a detector to int8 with NNCF, calibrated on own frames"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout", type=int, default=32)
    parser.add_argument("--max-confidence-delta", type=float, default=0.05)
    parser.add_argument("--verdict-threshold", type=float, default=0.5)
    parser.add_argument("--ignore-node", action="append", dest="ignore_nodes")
    arguments = parser.parse_args()
    print(
        quantize(
            arguments.model,
            arguments.calibration,
            arguments.output,
            holdout=arguments.holdout,
            max_confidence_delta=arguments.max_confidence_delta,
            verdict_threshold=arguments.verdict_threshold,
            ignore_nodes=arguments.ignore_nodes,
        )
    )


if __name__ == "__main__":
    main()
