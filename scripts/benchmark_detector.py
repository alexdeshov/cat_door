from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from src.common.detection_stats import read_detector_probe_stats
from src.common.settings import Settings
from src.recorder.status import read_recorder_status, recorder_status_path


def run_probe(
    model_path: Path,
    *,
    seconds: float,
    device: str = "CPU",
    output: Path | None = None,
) -> Path:
    """Проба всего конвейера: модель, препроцессинг, очереди, живые камеры.

    Модель саму по себе меряет `scripts/benchmark_detector_model.py`; здесь
    важно именно то, что теряется между ними.

    Устройство остаётся `CPU` умолчанием, и это не лень. Проба наследует
    окружение, а на узле в нём вполне может стоять `CATDOOR_DETECTOR_DEVICE=GPU`
    — тогда отчёт зависел бы от того, из какой оболочки его запустили. Все
    прежние отчёты в `detector-probes` сняты на процессоре и сравнимы между
    собой ровно потому, что устройство закреплено здесь, а не унаследовано.
    """
    if seconds <= 0:
        raise ValueError("probe duration must be positive")
    model_path = model_path.resolve()
    manifest_path = model_path.with_suffix(".json")
    if not model_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("detector ONNX and adjacent manifest are required")
    settings = Settings()
    environment = os.environ.copy()
    environment["CATDOOR_DETECTOR_MODEL_PATH"] = str(model_path)
    environment["CATDOOR_DETECTOR_DEVICE"] = device
    started_at = datetime.now(UTC)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.recorder.app",
            "--probe-seconds",
            str(seconds),
        ],
        env=environment,
        check=False,
        timeout=seconds + 60,
    )
    ended_at = datetime.now(UTC)
    if result.returncode != 0:
        raise RuntimeError(f"detector probe failed with exit code {result.returncode}")
    stats = read_detector_probe_stats(
        settings.data_dir / "runtime" / "detection.jsonl",
        started_at=started_at,
        ended_at=ended_at,
    )
    if stats["combined"]["observations"] == 0:
        raise RuntimeError("detector probe produced no observations")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Запрошенное устройство — намерение, а OpenVINO вправе исполнить иначе:
    # `GPU` без прав на устройство просто не скомпилируется, а `AUTO` вообще
    # выбирает сам. Отчёт, в котором стоит только намерение, доказывает не то,
    # что от него ждут, — поэтому рядом фактическое, как у `hwaccel_actual`.
    status = read_recorder_status(recorder_status_path(settings)) or {}
    report = {
        "model": model_path.name,
        "model_sha256": manifest.get("sha256"),
        "architecture": manifest.get("architecture"),
        "device": device,
        "device_actual": status.get("detector_device_actual"),
        "duration_seconds": seconds,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "stats": stats,
    }
    if output is None:
        timestamp = started_at.strftime("%Y%m%dT%H%M%SZ")
        output = (
            settings.data_dir
            / "runtime"
            / "detector-probes"
            / f"{timestamp}-{model_path.stem}.json"
        )
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite probe report {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded OpenVINO detector latency probe over the live pipeline. "
            "For the model alone, see scripts/benchmark_detector_model.py."
        )
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument(
        "--device",
        default="CPU",
        help=(
            "OpenVINO device for the probe. Stays CPU by default so that reports "
            "remain comparable with every probe taken before this flag existed."
        ),
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    print(
        run_probe(
            arguments.model,
            seconds=arguments.seconds,
            device=arguments.device,
            output=arguments.output,
        )
    )


if __name__ == "__main__":
    main()
