from __future__ import annotations

import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO, cast

import numpy as np

from src.recorder.ffmpeg import _base_input
from src.recorder.ring import prune_segments

FRAME_WIDTH = 320
FRAME_HEIGHT = 320
FRAME_BYTES = FRAME_WIDTH * FRAME_HEIGHT * 3


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            break
        chunks.extend(chunk)
    return bytes(chunks)


def rgb_frame_stream(
    ffmpeg_path: str,
    rtsp_url: str,
    fps: float,
    *,
    segment_directory: Path | None = None,
    segment_seconds: int = 2,
    retention_seconds: int = 180,
) -> Iterator[np.ndarray]:
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
    command.extend(
        [
            "-map",
            "0:v:0",
            "-vf",
            f"fps={fps},scale={FRAME_WIDTH}:{FRAME_HEIGHT},hflip,vflip",
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
    )
    try:
        assert process.stdout is not None
        stream = cast(BinaryIO, process.stdout)
        last_prune_at = 0.0
        while True:
            frame = _read_exact(stream, FRAME_BYTES)
            if len(frame) != FRAME_BYTES:
                return
            now = time.monotonic()
            if segment_directory is not None and now - last_prune_at >= 5:
                prune_segments(segment_directory, retention_seconds)
                last_prune_at = now
            yield np.frombuffer(frame, dtype=np.uint8).reshape(
                (FRAME_HEIGHT, FRAME_WIDTH, 3)
            )
    finally:
        if process.stdout:
            process.stdout.close()
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
