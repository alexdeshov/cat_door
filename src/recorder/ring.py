from __future__ import annotations

import subprocess
import time
from pathlib import Path

from src.recorder.ffmpeg import RecorderError


def prune_segments(directory: Path, retention_seconds: float, now: float | None = None) -> int:
    cutoff = (time.time() if now is None else now) - retention_seconds
    removed = 0
    for segment in directory.glob("*.mp4"):
        try:
            if segment.stat().st_mtime < cutoff:
                segment.unlink()
                removed += 1
        except FileNotFoundError:
            continue
    return removed


def select_segments(
    directory: Path,
    started_at: float,
    ended_at: float,
    segment_seconds: float,
) -> list[Path]:
    if ended_at < started_at:
        raise ValueError("event end cannot be before start")
    selected = []
    for segment in directory.glob("*.mp4"):
        try:
            finalized_at = segment.stat().st_mtime
        except FileNotFoundError:
            continue
        segment_started_at = finalized_at - segment_seconds
        if finalized_at >= started_at and segment_started_at <= ended_at:
            selected.append(segment)
    return sorted(selected, key=lambda path: (path.stat().st_mtime, path.name))


def concatenate_segments(
    ffmpeg_path: str,
    segments: list[Path],
    destination: Path,
) -> None:
    if not segments:
        raise RecorderError("ring_segments_missing")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".part.mp4")
    concat_file = destination.with_suffix(".concat.txt")
    concat_file.write_text(
        "".join(f"file '{segment.resolve().as_posix()}'\n" for segment in segments),
        encoding="utf-8",
    )
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-an",
        "-c:v",
        "copy",
        "-movflags",
        "+faststart",
        "-y",
        str(temporary),
    ]
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RecorderError("ring_concat_failed") from exc
    finally:
        concat_file.unlink(missing_ok=True)
    if result.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise RecorderError("ring_concat_failed")
    temporary.replace(destination)
