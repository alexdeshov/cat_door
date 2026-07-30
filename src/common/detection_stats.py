from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from statistics import median
from typing import Any


def _journal_files(path: Path) -> list[Path]:
    backups: list[tuple[int, Path]] = []
    for candidate in path.parent.glob(f"{path.name}.*"):
        try:
            number = int(candidate.suffix.removeprefix("."))
        except ValueError:
            continue
        backups.append((number, candidate))
    ordered = [candidate for _, candidate in sorted(backups, reverse=True)]
    if path.is_file():
        ordered.append(path)
    return ordered


def read_detection_stats(
    path: Path,
    *,
    max_lines: int = 50_000,
) -> dict[str, dict[str, float | int | None]]:
    observations: deque[dict[str, Any]] = deque(maxlen=max_lines)
    for journal in _journal_files(path):
        try:
            with journal.open("r", encoding="utf-8") as source:
                for line in source:
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(value, dict) and value.get("event") == "observation":
                        observations.append(value)
        except OSError:
            continue
    by_camera: dict[str, list[dict[str, Any]]] = {"face": [], "side": []}
    for observation in observations:
        camera = str(observation.get("camera", ""))
        if camera in by_camera:
            by_camera[camera].append(observation)
    result: dict[str, dict[str, float | int | None]] = {}
    for camera, items in by_camera.items():
        inference = _numbers(items, "inference_ms")
        frame_age = _numbers(items, "frame_age_ms")
        result[camera] = {
            "observations": len(items),
            "above_threshold": sum(bool(item.get("detected")) for item in items),
            "median_inference_ms": median(inference) if inference else None,
            "median_frame_age_ms": median(frame_age) if frame_age else None,
        }
    return result


def _numbers(items: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for item in items:
        value = item.get(key)
        if isinstance(value, (float, int)) and not isinstance(value, bool):
            values.append(float(value))
    return values
