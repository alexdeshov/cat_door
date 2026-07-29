from __future__ import annotations

import numpy as np
from PIL import Image

from src.recorder.preview import crop_padding, local_mjpeg_stream, preview_path, publish_preview


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
