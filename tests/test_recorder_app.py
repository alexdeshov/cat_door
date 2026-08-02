import json
import queue
import sqlite3
import threading
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy.exc import SQLAlchemyError

from src.common.camera_registry import CameraRecord, CameraRegistry
from src.common.detection_config import CameraDetectionConfig
from src.common.sd_notify import Notifier
from src.common.settings import Settings
from src.inference.cat_detector import CatDetection, CatDetectionResult
from src.inference.cat_identity import IdentityPrediction
from src.recorder.app import (
    CameraConfirmationTracker,
    StreamTarget,
    _frame_worker,
    _get_fair_frame,
    _predict_identity_if_detected,
    _put_latest,
    _register_loaded_models,
    build_automatic_config,
    publish_heartbeat,
    recorder_status_path,
    run,
    stream_targets_from_registry,
)
from src.recorder.frames import FrameSample


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

    def monitor(
        _settings: Settings,
        _duration: float | None,
        _notifier: Notifier | None = None,
    ) -> None:
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


def test_stream_targets_follow_the_role_not_the_camera_id() -> None:
    registry = CameraRegistry(
        cameras=[
            CameraRecord(id="ursa", host="10.0.0.5", stream_path="/one", role="side"),
            CameraRecord(id="hiwatch", host="10.0.0.6", stream_path="/two", role="face"),
            CameraRecord(id="spare", host="10.0.0.7", enabled=False),
        ]
    )

    targets = stream_targets_from_registry(registry)

    assert targets["face"].camera_id == "hiwatch"
    assert targets["face"].rtsp_url == "rtsp://10.0.0.6:554/two"
    assert targets["side"].camera_id == "ursa"


def test_disabled_camera_leaves_its_role_without_a_stream() -> None:
    registry = CameraRegistry(
        cameras=[CameraRecord(id="ursa", host="10.0.0.5", role="face", enabled=False)]
    )

    targets = stream_targets_from_registry(registry)

    assert targets["face"].camera_id == "ursa"
    assert targets["face"].rtsp_url == ""
    assert targets["side"] == StreamTarget()


def test_frame_worker_waits_for_a_camera_instead_of_exiting(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """Роль без камеры — ожидание, а не конец работы.

    Прежний воркер завершался на пустом адресе, поэтому камера, подключённая
    после старта, начинала отдавать кадры только после перезапуска службы.
    """
    connected: list[str] = []
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    def fake_stream(ffmpeg_path: str, rtsp_url: str, fps: float, **options):
        connected.append(rtsp_url)
        yield FrameSample(preview=frame, detector=frame)

    monkeypatch.setattr("src.recorder.app.rgb_frame_stream", fake_stream)
    monkeypatch.setattr("src.recorder.app.prepare_detector_frame", lambda sample, *rest: sample)

    settings = Settings(data_dir=runtime_path)
    output: queue.Queue = queue.Queue(maxsize=1)
    camera_configs = {
        "face": CameraDetectionConfig(
            confidence_threshold=0.5,
            box_min_confidence=0.2,
            confirmation_frames=1,
            window_frames=1,
        )
    }
    targets = {"face": StreamTarget()}
    lock = threading.Lock()
    reload_stream = threading.Event()
    stop = threading.Event()
    worker = threading.Thread(
        target=_frame_worker,
        args=(
            "face",
            settings,
            output,
            camera_configs,
            targets,
            lock,
            reload_stream,
            stop,
            320,
            114,
            "letterbox",
            "center",
        ),
        daemon=True,
    )
    worker.start()
    try:
        assert output.empty()
        with lock:
            targets["face"] = StreamTarget(
                rtsp_url="rtsp://10.0.0.9:554/new",
                camera_id="hiwatch",
            )
        role = output.get(timeout=10)[0]

        assert role == "face"
        assert connected == ["rtsp://10.0.0.9:554/new"]
        assert worker.is_alive()
    finally:
        stop.set()
        worker.join(timeout=10)
        assert not worker.is_alive()
