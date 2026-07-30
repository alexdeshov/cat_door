from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw

from src.inference.cat_detector import CatDetection
from src.inference.preprocessing import (
    crop_frame_padding,
    map_detector_box_to_frame,
)


def preview_path(data_dir: Path, role: str) -> Path:
    return data_dir / "runtime" / "previews" / f"{role}.jpg"


crop_padding = crop_frame_padding


def annotate_detections(
    frame: np.ndarray,
    detections: tuple[CatDetection, ...],
    *,
    event_threshold: float,
    box_min_confidence: float,
) -> np.ndarray:
    image = Image.fromarray(crop_frame_padding(frame), mode="RGB")
    draw = ImageDraw.Draw(image)
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True)[:5]:
        if detection.confidence < box_min_confidence:
            continue
        coordinates = map_detector_box_to_frame(
            detection.box,
            image.width,
            image.height,
        )
        color = "#22c55e" if detection.confidence >= event_threshold else "#f59e0b"
        draw.rectangle(coordinates, outline=color, width=3)
        draw.text(
            (coordinates[0] + 4, max(0, coordinates[1] - 14)),
            f"cat {detection.confidence:.2f}",
            fill=color,
            stroke_width=2,
            stroke_fill="black",
        )
    return np.asarray(image, dtype=np.uint8)


def publish_preview(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        Image.fromarray(crop_frame_padding(frame), mode="RGB").save(
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
