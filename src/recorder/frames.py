from __future__ import annotations

import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast

import numpy as np

from src.inference.preprocessing import NormalizedRoi
from src.recorder.ffmpeg import _base_input, ffmpeg_creation_flags
from src.recorder.ring import prune_segments

FRAME_WIDTH = 640
FRAME_HEIGHT = 360
FRAME_BYTES = FRAME_WIDTH * FRAME_HEIGHT * 3


@dataclass(frozen=True)
class FrameSample:
    preview: np.ndarray
    detector: np.ndarray


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            break
        chunks.extend(chunk)
    return bytes(chunks)


def _scale_filter() -> str:
    return (
        f"scale={FRAME_WIDTH}:{FRAME_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={FRAME_WIDTH}:{FRAME_HEIGHT}:(ow-iw)/2:(oh-ih)/2"
    )


def _roi_filter(fps: float, roi: NormalizedRoi, rotate_180: bool) -> str:
    crop = (
        f"crop=iw*{roi.width:g}:ih*{roi.height:g}:"
        f"iw*{roi.x:g}:ih*{roi.y:g}"
    )
    scale = _scale_filter()
    orientation = ",hflip,vflip" if rotate_180 else ""
    return (
        f"[0:v:0]fps={fps}{orientation},split=2[preview_source][detector_source];"
        f"[preview_source]{scale}[preview];"
        f"[detector_source]{crop},{scale}[detector];"
        "[preview][detector]hstack=inputs=2[combined]"
    )


def rgb_frame_stream(
    ffmpeg_path: str,
    rtsp_url: str,
    fps: float,
    *,
    segment_directory: Path | None = None,
    segment_seconds: int = 2,
    retention_seconds: int = 180,
    roi: NormalizedRoi | None = None,
    rotate_180: bool = True,
) -> Iterator[FrameSample]:
    command = [
        *_base_input(ffmpeg_path, rtsp_url),
    ]
    if segment_directory is not None:
        segment_directory.mkdir(parents=True, exist_ok=True)
        command.extend(
            [
                "-map",
                "0:v:0",
                "-an",
                "-c:v",
                "copy",
                "-f",
                "segment",
                "-segment_time",
                str(segment_seconds),
                "-reset_timestamps",
                "1",
                "-strftime",
                "1",
                str(segment_directory / "%Y%m%dT%H%M%S.mp4"),
            ]
        )
    if roi is None:
        command.extend(
            [
                "-map",
                "0:v:0",
                "-vf",
                (
                    f"fps={fps},{_scale_filter()}"
                    f"{',hflip,vflip' if rotate_180 else ''}"
                ),
            ]
        )
        frame_bytes = FRAME_BYTES
        output_width = FRAME_WIDTH
    else:
        command.extend(
            [
                "-filter_complex",
                _roi_filter(fps, roi, rotate_180),
                "-map",
                "[combined]",
            ]
        )
        frame_bytes = FRAME_BYTES * 2
        output_width = FRAME_WIDTH * 2
    command.extend(
        [
            "-an",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
    )
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        creationflags=ffmpeg_creation_flags(),
    )
    try:
        assert process.stdout is not None
        stream = cast(BinaryIO, process.stdout)
        last_prune_at = 0.0
        while True:
            frame = _read_exact(stream, frame_bytes)
            if len(frame) != frame_bytes:
                return
            now = time.monotonic()
            if segment_directory is not None and now - last_prune_at >= 5:
                prune_segments(segment_directory, retention_seconds)
                last_prune_at = now
            decoded = np.frombuffer(frame, dtype=np.uint8).reshape(
                (FRAME_HEIGHT, output_width, 3)
            )
            if roi is None:
                yield FrameSample(preview=decoded, detector=decoded)
            else:
                yield FrameSample(
                    preview=np.ascontiguousarray(decoded[:, :FRAME_WIDTH, :]),
                    detector=np.ascontiguousarray(decoded[:, FRAME_WIDTH:, :]),
                )
    finally:
        if process.stdout:
            process.stdout.close()
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
