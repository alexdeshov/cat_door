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


def run_probe(
    model_path: Path,
    *,
    seconds: float,
    output: Path | None = None,
) -> Path:
    if seconds <= 0:
        raise ValueError("probe duration must be positive")
    model_path = model_path.resolve()
    manifest_path = model_path.with_suffix(".json")
    if not model_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("detector ONNX and adjacent manifest are required")
    settings = Settings()
    environment = os.environ.copy()
    environment["CATDOOR_DETECTOR_MODEL_PATH"] = str(model_path)
    environment["CATDOOR_DETECTOR_DEVICE"] = "CPU"
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
    report = {
        "model": model_path.name,
        "model_sha256": manifest.get("sha256"),
        "architecture": manifest.get("architecture"),
        "device": "CPU",
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
        description="Run a bounded OpenVINO CPU detector latency probe"
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    print(
        run_probe(
            arguments.model,
            seconds=arguments.seconds,
            output=arguments.output,
        )
    )


if __name__ == "__main__":
    main()
