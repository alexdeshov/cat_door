from __future__ import annotations

import os
import time

import numpy as np
from PIL import Image

from src.common.detection_config import DetectorRoi
from src.inference.cat_detector import CatDetection
from src.recorder.preview import (
    annotate_detections,
    crop_padding,
    local_mjpeg_stream,
    preview_is_fresh,
    preview_path,
    publish_preview,
)


def test_preview_is_published_as_atomic_jpeg(runtime_path) -> None:
    path = preview_path(runtime_path, "face")
    frame = np.zeros((320, 320, 3), dtype=np.uint8)

    publish_preview(path, frame)

    payload = path.read_bytes()
    assert payload.startswith(b"\xff\xd8")
    assert payload.endswith(b"\xff\xd9")
    assert not list(path.parent.glob("*.tmp"))


def test_local_mjpeg_stream_reuses_published_frame(runtime_path, monkeypatch) -> None:
    path = preview_path(runtime_path, "face")
    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    publish_preview(path, frame)
    monkeypatch.setattr("src.recorder.preview.time.sleep", lambda _: None)

    payload = next(local_mjpeg_stream(path, 5))

    assert payload.startswith(b"--frame\r\nContent-Type: image/jpeg")
    assert b"\xff\xd8" in payload


def test_preview_crops_exact_letterbox_padding(runtime_path) -> None:
    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    frame[70:250, :, :] = 100
    path = preview_path(runtime_path, "face")

    assert crop_padding(frame).shape == (180, 320, 3)
    publish_preview(path, frame)

    with Image.open(path) as image:
        assert image.size == (320, 180)


def test_preview_draws_detection_box_in_source_coordinates() -> None:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[:, :, :] = 30
    detections = (CatDetection(box=(80, 115, 240, 205), confidence=0.8),)

    annotated = annotate_detections(
        frame,
        detections,
        event_threshold=0.55,
        box_min_confidence=0.2,
    )

    assert annotated.shape == (360, 640, 3)
    assert np.any(annotated[:, :, 1] > 150)


def test_preview_maps_roi_detection_to_full_frame() -> None:
    frame = np.full((360, 640, 3), 30, dtype=np.uint8)
    detector_frame = np.full((360, 640, 3), 30, dtype=np.uint8)
    detections = (CatDetection(box=(80, 115, 240, 205), confidence=0.8),)

    annotated = annotate_detections(
        frame,
        detections,
        event_threshold=0.55,
        box_min_confidence=0.2,
        detector_frame=detector_frame,
        roi=DetectorRoi(x=0.25, y=0.25, width=0.5, height=0.5),
    )

    assert annotated[135, 240, 1] > 150
    assert np.all(annotated[90, 160] == 30)


def test_preview_maps_416_detection_to_full_frame() -> None:
    frame = np.full((360, 640, 3), 30, dtype=np.uint8)
    detections = (
        CatDetection(box=(104.0, 149.5, 312.0, 266.5), confidence=0.8),
    )

    annotated = annotate_detections(
        frame,
        detections,
        event_threshold=0.55,
        box_min_confidence=0.2,
        detector_input_size=416,
    )

    assert annotated[90, 160, 1] > 150


def test_stale_preview_is_not_considered_fresh(runtime_path) -> None:
    """An old file means the recorder stopped, not that the camera died."""
    path = preview_path(runtime_path, "face")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xd8\xff\xd9")
    stale = time.time() - 3600
    os.utime(path, (stale, stale))

    assert not preview_is_fresh(path)


def test_missing_preview_is_not_considered_fresh(runtime_path) -> None:
    assert not preview_is_fresh(preview_path(runtime_path, "face"))


def test_just_written_preview_is_fresh(runtime_path) -> None:
    path = preview_path(runtime_path, "face")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xd8\xff\xd9")

    assert preview_is_fresh(path)
