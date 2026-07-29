from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image


def preview_path(data_dir: Path, role: str) -> Path:
    return data_dir / "runtime" / "previews" / f"{role}.jpg"


def crop_padding(frame: np.ndarray) -> np.ndarray:
    active_rows = np.flatnonzero(np.any(frame != 0, axis=(1, 2)))
    active_columns = np.flatnonzero(np.any(frame != 0, axis=(0, 2)))
    if active_rows.size == 0 or active_columns.size == 0:
        return frame
    return frame[
        active_rows[0] : active_rows[-1] + 1,
        active_columns[0] : active_columns[-1] + 1,
    ]


def publish_preview(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        Image.fromarray(crop_padding(frame), mode="RGB").save(
            temporary,
            format="JPEG",
            quality=75,
        )
        for attempt in range(10):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.01)
    finally:
        temporary.unlink(missing_ok=True)


def local_mjpeg_stream(path: Path, fps: int) -> Iterator[bytes]:
    last_frame: bytes | None = None
    delay = 1 / fps
    while True:
        try:
            frame = path.read_bytes()
        except OSError:
            time.sleep(delay)
            continue
        if frame != last_frame and frame.startswith(b"\xff\xd8") and frame.endswith(b"\xff\xd9"):
            last_frame = frame
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n"
                + f"Content-Length: {len(frame)}\r\n\r\n".encode()
                + frame
                + b"\r\n"
            )
        time.sleep(delay)
