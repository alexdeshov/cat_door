import json
import queue
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy.exc import SQLAlchemyError

from src.common.detection_config import CameraDetectionConfig
from src.common.settings import Settings
from src.inference.cat_detector import CatDetection, CatDetectionResult
from src.inference.cat_identity import IdentityPrediction
from src.recorder.app import (
    CameraConfirmationTracker,
    _detect_with_motion_gate,
    _get_fair_frame,
    _predict_identity_if_detected,
    _put_latest,
    _register_loaded_models,
    build_automatic_config,
    publish_heartbeat,
    recorder_status_path,
    run,
)
from src.recorder.automatic import AutomaticEventState
from src.recorder.motion import MotionGate


def test_disabled_recorder_writes_safe_heartbeat(runtime_path: Path) -> None:
    settings = Settings(
        data_dir=runtime_path,
        automatic_capture_enabled=False,
        automatic_capture_dry_run=True,
        detector_device="GPU",
    )

    publish_heartbeat(settings)

    payload = json.loads(recorder_status_path(settings).read_text(encoding="utf-8"))
    assert payload["process"] == "CatDoorRecorder"
    assert payload["state"] == "disabled"
    assert payload["dry_run"] is True
    assert payload["detector_connected"] is False
    assert payload["detector_device_requested"] == "GPU"
    assert payload["detector_device_actual"] is None
    assert "rtsp" not in json.dumps(payload).lower()


def test_enabled_recorder_waits_for_local_detector(runtime_path: Path) -> None:
    settings = Settings(data_dir=runtime_path, automatic_capture_enabled=True)

    publish_heartbeat(settings)

    payload = json.loads(recorder_status_path(settings).read_text(encoding="utf-8"))
    assert payload["state"] == "waiting_for_detector"
    assert payload["detector_device_requested"] == "CPU"
    assert payload["detector_device_actual"] is None


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


class RecordingDetector:
    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []
        self.result = CatDetectionResult(confidence=0.0, detections=())

    def detect(self, frame: np.ndarray) -> CatDetectionResult:
        self.calls.append(frame)
        return self.result


def motion_config(*, enabled: bool) -> CameraDetectionConfig:
    return CameraDetectionConfig(
        confidence_threshold=0.30,
        box_min_confidence=0.15,
        confirmation_frames=2,
        window_frames=5,
        motion_gating_enabled=enabled,
    )


def test_non_idle_state_always_runs_detector_without_updating_static_gate() -> None:
    detector = RecordingDetector()
    gate = MotionGate()
    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    for _ in range(11):
        gate.has_motion(frame)

    result = _detect_with_motion_gate(
        detector,
        frame,
        motion_config(enabled=True),
        AutomaticEventState.ACTIVE,
        gate,
    )

    assert result is detector.result
    assert detector.calls == [frame]
    assert gate.has_motion(frame) is False


def test_disabled_motion_gating_preserves_detector_path_for_every_sample() -> None:
    detector = RecordingDetector()
    gate = MotionGate()
    frame = np.zeros((320, 320, 3), dtype=np.uint8)

    results = [
        _detect_with_motion_gate(
            detector,
            frame,
            motion_config(enabled=False),
            AutomaticEventState.IDLE,
            gate,
        )
        for _ in range(12)
    ]

    assert results == [detector.result] * 12
    assert detector.calls == [frame] * 12
    assert gate.has_motion(frame) is True


def test_idle_static_scene_skips_detector_after_motion_warmup() -> None:
    detector = RecordingDetector()
    gate = MotionGate()
    frame = np.zeros((320, 320, 3), dtype=np.uint8)

    results = [
        _detect_with_motion_gate(
            detector,
            frame,
            motion_config(enabled=True),
            AutomaticEventState.IDLE,
            gate,
        )
        for _ in range(11)
    ]

    assert results[-1] is None
    assert len(detector.calls) == 10


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


@pytest.mark.parametrize("failure", [OSError("camera"), SQLAlchemyError("database")])
def test_run_retries_unexpected_monitor_failures(
    runtime_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    settings = Settings(data_dir=runtime_path, automatic_capture_enabled=True)
    calls = 0

    def monitor(_settings: Settings, _duration: float | None) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise failure
        raise KeyboardInterrupt

    monkeypatch.setattr("src.recorder.app.Settings", lambda: settings)
    monkeypatch.setattr("src.recorder.app.monitor_automatic", monitor)
    monkeypatch.setattr("src.recorder.app.time.sleep", lambda _seconds: None)

    with pytest.raises(KeyboardInterrupt):
        run()

    payload = json.loads(recorder_status_path(settings).read_text(encoding="utf-8"))
    journal = [
        json.loads(line)
        for line in (runtime_path / "runtime" / "detection.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert calls == 2
    assert payload["state"] == "retrying"
    assert payload["error_type"] == type(failure).__name__
    assert payload["retry_count"] == 1
    assert journal[-1]["event"] == "recorder_retry"
    assert journal[-1]["error_type"] == type(failure).__name__


def test_register_loaded_models_retries_database_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    def register(_settings: Settings, _models: object) -> dict[str, int]:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise sqlite3.OperationalError("database is locked")
        return {"detector": 42}

    monkeypatch.setattr("src.recorder.app._register_loaded_models_once", register)
    monkeypatch.setattr("src.recorder.app.time.sleep", lambda _seconds: None)

    assert _register_loaded_models(Settings(), {}) == {"detector": 42}
    assert attempts == 3
