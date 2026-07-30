import json
import queue
from pathlib import Path

import numpy as np
import pytest

from src.common.detection_config import CameraDetectionConfig
from src.common.settings import Settings
from src.inference.cat_detector import CatDetection, CatDetectionResult
from src.inference.cat_identity import IdentityPrediction
from src.recorder.app import (
    CameraConfirmationTracker,
    _get_fair_frame,
    _predict_identity_if_detected,
    _put_latest,
    build_automatic_config,
    publish_heartbeat,
    recorder_status_path,
)


def test_disabled_recorder_writes_safe_heartbeat(runtime_path: Path) -> None:
    settings = Settings(
        data_dir=runtime_path,
        automatic_capture_enabled=False,
        automatic_capture_dry_run=True,
    )

    publish_heartbeat(settings)

    payload = json.loads(recorder_status_path(settings).read_text(encoding="utf-8"))
    assert payload["process"] == "CatDoorRecorder"
    assert payload["state"] == "disabled"
    assert payload["dry_run"] is True
    assert payload["detector_connected"] is False
    assert "rtsp" not in json.dumps(payload).lower()


def test_enabled_recorder_waits_for_local_detector(runtime_path: Path) -> None:
    settings = Settings(data_dir=runtime_path, automatic_capture_enabled=True)

    publish_heartbeat(settings)

    payload = json.loads(recorder_status_path(settings).read_text(encoding="utf-8"))
    assert payload["state"] == "waiting_for_detector"


def test_invalid_detection_window_is_rejected() -> None:
    settings = Settings(
        detection_confirmation_frames=4,
        detection_window_frames=3,
    )

    with pytest.raises(ValueError):
        build_automatic_config(settings)


def test_latest_frame_replaces_oldest_when_queue_is_full() -> None:
    frames: queue.Queue[int] = queue.Queue(maxsize=2)
    _put_latest(frames, 1)
    _put_latest(frames, 2)
    _put_latest(frames, 3)

    assert [frames.get_nowait(), frames.get_nowait()] == [2, 3]


def test_per_camera_queues_do_not_let_face_frames_starve_side() -> None:
    frames = {
        "face": queue.Queue(maxsize=1),
        "side": queue.Queue(maxsize=1),
    }
    _put_latest(frames["face"], "face-1")
    _put_latest(frames["side"], "side-1")

    first, next_role = _get_fair_frame(frames, "face", 0.1)
    _put_latest(frames["face"], "face-2")
    second, next_role = _get_fair_frame(frames, next_role, 0.1)

    assert first == "face-1"
    assert second == "side-1"
    assert next_role == "face"


def test_camera_confirmation_tracker_uses_own_threshold_and_window() -> None:
    tracker = CameraConfirmationTracker(
        CameraDetectionConfig(
            confidence_threshold=0.30,
            box_min_confidence=0.15,
            confirmation_frames=2,
            window_frames=3,
        )
    )

    assert tracker.observe(0.31) is False
    assert tracker.observe(0.10) is False
    assert tracker.observe(0.40) is True


class RecordingIdentityModel:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, float]] = []

    def predict(self, frame: np.ndarray, threshold: float) -> IdentityPrediction:
        self.calls.append((frame, threshold))
        return IdentityPrediction(name="Барсик", confidence=0.91)


def test_identity_uses_per_camera_threshold_below_legacy_global_threshold() -> None:
    config = CameraDetectionConfig(
        confidence_threshold=0.30,
        box_min_confidence=0.15,
        confirmation_frames=2,
        window_frames=5,
    )
    model = RecordingIdentityModel()
    frame = np.full((360, 640, 3), 50, dtype=np.uint8)
    confidence = 0.40
    assert confidence < Settings().detection_confidence_threshold

    attempt = _predict_identity_if_detected(
        frame,
        frame,
        CatDetectionResult(
            confidence=confidence,
            detections=(
                CatDetection(
                    box=(80.0, 115.0, 240.0, 205.0),
                    confidence=confidence,
                ),
            ),
        ),
        config,
        model,
        0.80,
    )

    assert attempt.prediction == IdentityPrediction(name="Барсик", confidence=0.91)
    assert attempt.skipped_reason is None
    assert len(model.calls) == 1
    assert model.calls[0][0].shape == (252, 448, 3)
    assert model.calls[0][1] == 0.80


def test_identity_is_not_called_below_per_camera_threshold() -> None:
    config = CameraDetectionConfig(
        confidence_threshold=0.30,
        box_min_confidence=0.15,
        confirmation_frames=2,
        window_frames=5,
    )
    model = RecordingIdentityModel()

    attempt = _predict_identity_if_detected(
        np.zeros((360, 640, 3), dtype=np.uint8),
        np.zeros((360, 640, 3), dtype=np.uint8),
        CatDetectionResult(
            confidence=0.29,
            detections=(
                CatDetection(
                    box=(80.0, 115.0, 240.0, 205.0),
                    confidence=0.29,
                ),
            ),
        ),
        config,
        model,
        0.80,
    )

    assert attempt.prediction is None
    assert attempt.skipped_reason == "below_camera_threshold"
    assert model.calls == []


def test_identity_skips_missing_box_without_exception() -> None:
    config = CameraDetectionConfig(
        confidence_threshold=0.30,
        box_min_confidence=0.15,
        confirmation_frames=2,
        window_frames=5,
    )
    model = RecordingIdentityModel()

    attempt = _predict_identity_if_detected(
        np.zeros((360, 640, 3), dtype=np.uint8),
        np.zeros((360, 640, 3), dtype=np.uint8),
        CatDetectionResult(confidence=0.40, detections=()),
        config,
        model,
        0.80,
    )

    assert attempt.prediction is None
    assert attempt.skipped_reason == "no_confident_box"
    assert model.calls == []
