from __future__ import annotations

import argparse
import queue
import sqlite3
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from sqlalchemy.exc import SQLAlchemyError

from src.common.database import create_session_factory
from src.common.detection_config import (
    CameraDetectionConfig,
    DetectionConfig,
    detection_config_path,
    load_detection_config,
)
from src.common.metrics import ScoredPrediction, aggregate_frame_predictions, mean_or_none
from src.common.model_registry import register_model_version
from src.common.settings import Settings
from src.inference.cat_detector import (
    CatDetectionResult,
    OpenVinoCatDetector,
)
from src.inference.cat_identity import IdentityPrediction, OpenVinoCatIdentity
from src.inference.preprocessing import (
    DETECTOR_INPUT_SIZE,
    crop_strongest_cat,
    prepare_detector_frame,
)
from src.recorder.automatic import AutomaticEventConfig, AutomaticEventMachine
from src.recorder.detection_journal import DetectionJournal
from src.recorder.events import (
    AutomaticEventPrediction,
    AutomaticEventRequest,
    persist_automatic_event,
)
from src.recorder.frames import rgb_frame_stream
from src.recorder.preview import annotate_detections, preview_path, publish_preview
from src.recorder.status import write_recorder_status

DATABASE_STARTUP_ATTEMPTS = 3
DATABASE_STARTUP_RETRY_SECONDS = 1.0
RECORDER_RETRY_SECONDS = 5.0


def recorder_status_path(settings: Settings) -> Path:
    return settings.data_dir / "runtime" / "recorder-status.json"


def build_automatic_config(settings: Settings) -> AutomaticEventConfig:
    return DetectionConfig.from_settings(settings).event_config()


def _detector_device_status(
    settings: Settings,
    detector: OpenVinoCatDetector | None = None,
) -> dict[str, str | None]:
    return {
        "detector_device_requested": settings.detector_device,
        "detector_device_actual": detector.device if detector is not None else None,
    }


def _put_latest(output: queue.Queue[Any], item: Any) -> None:
    try:
        output.put_nowait(item)
    except queue.Full:
        try:
            output.get_nowait()
        except queue.Empty:
            pass
        try:
            output.put_nowait(item)
        except queue.Full:
            pass


def _clear_queue(output: queue.Queue[Any]) -> None:
    while True:
        try:
            output.get_nowait()
        except queue.Empty:
            return


def _get_fair_frame(
    inputs: dict[str, queue.Queue[Any]],
    next_role: str,
    timeout: float,
) -> tuple[Any, str]:
    deadline = time.monotonic() + timeout
    roles = (next_role, "side" if next_role == "face" else "face")
    while True:
        for role in roles:
            try:
                item = inputs[role].get_nowait()
            except queue.Empty:
                continue
            following_role = "side" if role == "face" else "face"
            return item, following_role
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise queue.Empty
        time.sleep(min(0.01, remaining))


class CameraConfirmationTracker:
    def __init__(self, config: CameraDetectionConfig) -> None:
        self.config = config
        self._window: deque[bool] = deque(maxlen=config.window_frames)

    def observe(self, confidence: float) -> bool:
        self._window.append(confidence >= self.config.confidence_threshold)
        return sum(self._window) >= self.config.confirmation_frames


class IdentityModel(Protocol):
    def predict(self, frame: Any, threshold: float) -> IdentityPrediction: ...


@dataclass(frozen=True)
class IdentityAttempt:
    prediction: IdentityPrediction | None
    skipped_reason: str | None
    inference_ms: float | None = None


def _predict_identity_if_detected(
    frame: Any,
    detector_frame: Any,
    detection: CatDetectionResult,
    camera_config: CameraDetectionConfig,
    model: IdentityModel | None,
    identity_threshold: float,
    detector_input_size: int = DETECTOR_INPUT_SIZE,
    detector_resize_mode: str = "letterbox",
    detector_letterbox_alignment: str = "center",
) -> IdentityAttempt:
    if model is None:
        return IdentityAttempt(None, "model_unavailable")
    if detection.confidence < camera_config.confidence_threshold:
        return IdentityAttempt(None, "below_camera_threshold")
    cat_crop = crop_strongest_cat(
        frame,
        detection.detections,
        min_confidence=camera_config.confidence_threshold,
        detector_frame=detector_frame,
        roi=camera_config.roi,
        input_size=detector_input_size,
        resize_mode=detector_resize_mode,
        letterbox_alignment=detector_letterbox_alignment,
    )
    if cat_crop is None:
        return IdentityAttempt(None, "no_confident_box")
    started_at = time.monotonic()
    prediction = model.predict(cat_crop, identity_threshold)
    return IdentityAttempt(
        prediction,
        None,
        (time.monotonic() - started_at) * 1000,
    )


def _register_loaded_models_once(
    settings: Settings,
    identity_models: dict[str, OpenVinoCatIdentity],
) -> dict[str, int]:
    factory = create_session_factory(settings.resolved_database_path)
    with factory() as session:
        detector = register_model_version(
            session,
            kind="detector",
            model_path=settings.detector_model_path,
        )
        versions = {"detector": detector.id}
        for role, model_path in (
            ("face", settings.identity_face_model_path),
            ("side", settings.identity_side_model_path),
        ):
            if role not in identity_models or model_path is None:
                continue
            version = register_model_version(
                session,
                kind=f"identity_{role}",
                model_path=model_path,
            )
            versions[role] = version.id
        session.commit()
    return versions


def _register_loaded_models(
    settings: Settings,
    identity_models: dict[str, OpenVinoCatIdentity],
) -> dict[str, int]:
    for attempt in range(1, DATABASE_STARTUP_ATTEMPTS + 1):
        try:
            return _register_loaded_models_once(settings, identity_models)
        except (sqlite3.OperationalError, SQLAlchemyError):
            if attempt == DATABASE_STARTUP_ATTEMPTS:
                raise
            time.sleep(DATABASE_STARTUP_RETRY_SECONDS)
    raise AssertionError("database startup retry loop ended unexpectedly")


def _build_online_predictions(
    model_versions: dict[str, int],
    identity_scores: dict[str, list[ScoredPrediction]],
    identity_timings: dict[str, list[float]],
    detector_confidences: dict[str, float],
    threshold: float,
) -> tuple[AutomaticEventPrediction, ...]:
    predictions: list[AutomaticEventPrediction] = []
    for role in ("face", "side"):
        model_version_id = model_versions.get(role)
        if model_version_id is None:
            continue
        scores = identity_scores[role]
        aggregate = aggregate_frame_predictions(scores, threshold)
        predictions.append(
            AutomaticEventPrediction(
                model_version_id=model_version_id,
                view=role,
                decision=aggregate.decision,
                predicted_name=aggregate.predicted_name,
                confidence=aggregate.confidence,
                frames_used=aggregate.frames_used,
                max_detector_confidence=detector_confidences[role],
                mean_inference_ms=mean_or_none(identity_timings[role]),
                raw_scores=tuple(
                    (prediction.name, prediction.confidence)
                    for prediction in scores
                ),
            )
        )
    return tuple(predictions)


def publish_heartbeat(settings: Settings) -> None:
    build_automatic_config(settings)
    if not settings.automatic_capture_enabled:
        state = "disabled"
        message = "Автоматический сбор выключен"
    else:
        state = "waiting_for_detector"
        message = "Автомат событий готов, локальный детектор ещё не подключён"
    write_recorder_status(
        recorder_status_path(settings),
        process="CatDoorRecorder",
        state=state,
        message=message,
        automatic_capture_enabled=settings.automatic_capture_enabled,
        dry_run=settings.automatic_capture_dry_run,
        detector_connected=False,
        **_detector_device_status(settings),
    )


def _frame_worker(
    role: str,
    settings: Settings,
    output: queue.Queue[tuple[str, Any, Any, Any, float, CameraDetectionConfig]],
    camera_configs: dict[str, CameraDetectionConfig],
    config_lock: threading.Lock,
    reload_stream: threading.Event,
    stop: threading.Event,
    detector_input_size: int,
    detector_letterbox_value: int,
    detector_resize_mode: str,
    detector_letterbox_alignment: str,
) -> None:
    url = settings.camera_face_rtsp if role == "face" else settings.camera_side_rtsp
    if not url:
        return
    while not stop.is_set():
        with config_lock:
            camera_config = camera_configs[role]
            reload_stream.clear()
        try:
            for sample in rgb_frame_stream(
                settings.ffmpeg_path,
                url,
                camera_config.sample_fps,
                segment_directory=settings.data_dir / "runtime" / "ring" / role,
                segment_seconds=settings.ring_segment_seconds,
                retention_seconds=settings.ring_retention_seconds,
                roi=camera_config.roi,
            ):
                if stop.is_set():
                    return
                if reload_stream.is_set():
                    break
                item = (
                    role,
                    prepare_detector_frame(
                        sample.detector,
                        detector_input_size,
                        detector_letterbox_value,
                        detector_resize_mode,
                        detector_letterbox_alignment,
                    ),
                    sample.preview,
                    sample.detector,
                    time.monotonic(),
                    camera_config,
                )
                _put_latest(output, item)
        except OSError:
            pass
        if not reload_stream.is_set():
            stop.wait(1)


def monitor_automatic(settings: Settings, duration_seconds: float | None) -> None:
    config_path = detection_config_path(settings.data_dir)
    runtime_config = load_detection_config(config_path, DetectionConfig.from_settings(settings))
    journal = DetectionJournal(
        settings.data_dir / "runtime" / "detection.jsonl",
        max_bytes=settings.detection_log_max_bytes,
        backup_count=settings.detection_log_backup_count,
    )
    detector = OpenVinoCatDetector(settings.detector_model_path, settings.detector_device)
    identity_models = {
        role: OpenVinoCatIdentity(path, settings.detector_device)
        for role, path in (
            ("face", settings.identity_face_model_path),
            ("side", settings.identity_side_model_path),
        )
        if path is not None
    }
    model_versions = _register_loaded_models(settings, identity_models)
    machine = AutomaticEventMachine(runtime_config.event_config())
    trackers = {
        role: CameraConfirmationTracker(getattr(runtime_config, role))
        for role in ("face", "side")
    }
    frame_queues: dict[
        str,
        queue.Queue[tuple[str, Any, Any, Any, float, CameraDetectionConfig]],
    ] = {
        role: queue.Queue(maxsize=1)
        for role in ("face", "side")
    }
    worker_configs = {
        role: getattr(runtime_config, role)
        for role in ("face", "side")
    }
    config_lock = threading.Lock()
    reload_events = {
        role: threading.Event()
        for role in ("face", "side")
    }
    stop = threading.Event()
    workers = [
        threading.Thread(
            target=_frame_worker,
            args=(
                role,
                settings,
                frame_queues[role],
                worker_configs,
                config_lock,
                reload_events[role],
                stop,
                detector.input_size,
                detector.letterbox_value,
                detector.resize_mode,
                detector.letterbox_alignment,
            ),
            daemon=True,
            name=f"catdoor-{role}-frames",
        )
        for role in ("face", "side")
    ]
    for worker in workers:
        worker.start()

    started_at = time.monotonic()
    confidences = {"face": 0.0, "side": 0.0}
    identity_predictions: dict[str, dict[str, object]] = {}
    frames_processed = {"face": 0, "side": 0}
    last_action: str | None = None
    last_status_at = 0.0
    active_started_at: float | None = None
    active_initiating_camera: str | None = None
    active_confidences = {"face": 0.0, "side": 0.0}
    active_identity_scores: dict[str, list[ScoredPrediction]] = {
        "face": [],
        "side": [],
    }
    active_identity_timings: dict[str, list[float]] = {
        "face": [],
        "side": [],
    }
    journal.write(
        "session_started",
        camera_settings={
            role: getattr(runtime_config, role).model_dump()
            for role in ("face", "side")
        },
        sample_fps={
            role: getattr(runtime_config, role).sample_fps
            for role in ("face", "side")
        },
        recording=not settings.automatic_capture_dry_run,
        cameras={
            "face": bool(settings.camera_face_rtsp),
            "side": bool(settings.camera_side_rtsp),
        },
    )
    last_config_check_at = 0.0
    next_camera = "face"
    try:
        while duration_seconds is None or time.monotonic() - started_at < duration_seconds:
            now = time.monotonic()
            if now - last_config_check_at >= 1 and machine.state.value == "idle":
                updated_config = load_detection_config(config_path, runtime_config)
                if updated_config != runtime_config:
                    with config_lock:
                        for role in ("face", "side"):
                            previous_camera = getattr(runtime_config, role)
                            updated_camera = getattr(updated_config, role)
                            worker_configs[role] = updated_camera
                            if (
                                previous_camera.sample_fps != updated_camera.sample_fps
                                or previous_camera.roi != updated_camera.roi
                            ):
                                _clear_queue(frame_queues[role])
                                reload_events[role].set()
                    runtime_config = updated_config
                    machine = AutomaticEventMachine(runtime_config.event_config())
                    trackers = {
                        role: CameraConfirmationTracker(getattr(runtime_config, role))
                        for role in ("face", "side")
                    }
                    journal.write("config_updated", **runtime_config.model_dump())
                last_config_check_at = now
            try:
                item, next_camera = _get_fair_frame(
                    frame_queues,
                    next_camera,
                    timeout=2,
                )
                (
                    role,
                    frame,
                    preview_frame,
                    detector_frame,
                    captured_at,
                    camera_config,
                ) = item
            except queue.Empty:
                journal.write(
                    "frames_waiting",
                    state=machine.state.value,
                    frames_processed=frames_processed,
                )
                write_recorder_status(
                    recorder_status_path(settings),
                    process="CatDoorRecorder",
                    state="waiting_for_frames",
                    message="Ожидание кадров камер",
                    automatic_capture_enabled=True,
                    dry_run=True,
                    detector_connected=True,
                    frames_processed=frames_processed,
                    confidences=confidences,
                    **_detector_device_status(settings, detector),
                )
                continue
            inference_started_at = time.monotonic()
            detection = detector.detect(frame)
            inference_ms = (time.monotonic() - inference_started_at) * 1000
            frame_age_ms = (time.monotonic() - captured_at) * 1000
            confidence = detection.confidence
            camera_confirmed = trackers[role].observe(confidence)
            try:
                annotated_frame = annotate_detections(
                    preview_frame,
                    detection.detections,
                    event_threshold=camera_config.confidence_threshold,
                    box_min_confidence=camera_config.box_min_confidence,
                    detector_frame=detector_frame,
                    roi=camera_config.roi,
                    detector_input_size=detector.input_size,
                    detector_resize_mode=detector.resize_mode,
                    detector_letterbox_alignment=detector.letterbox_alignment,
                )
                publish_preview(preview_path(settings.data_dir, role), annotated_frame)
            except OSError:
                pass
            previous_state = machine.state.value
            confidences[role] = round(confidence, 4)
            prediction = _predict_identity_if_detected(
                preview_frame,
                detector_frame,
                detection,
                camera_config,
                identity_models.get(role),
                settings.identity_confidence_threshold,
                detector.input_size,
                detector.resize_mode,
                detector.letterbox_alignment,
            )
            if prediction.skipped_reason == "no_confident_box":
                journal.write(
                    "identity_skipped",
                    camera=role,
                    reason=prediction.skipped_reason,
                    confidence=round(confidence, 4),
                    threshold=camera_config.confidence_threshold,
                )
            if prediction.prediction is not None:
                identity_predictions[role] = {
                    "name": prediction.prediction.name,
                    "confidence": round(prediction.prediction.confidence, 4),
                }
            frames_processed[role] += 1
            update = machine.observe(time.monotonic(), {role: 1.0 if camera_confirmed else 0.0})
            journal.write(
                "observation",
                camera=role,
                confidence=round(confidence, 4),
                threshold=camera_config.confidence_threshold,
                detected=confidence >= camera_config.confidence_threshold,
                camera_confirmed=camera_confirmed,
                confirmation_frames=camera_config.confirmation_frames,
                window_frames=camera_config.window_frames,
                sample_fps=camera_config.sample_fps,
                roi=(
                    camera_config.roi.model_dump()
                    if camera_config.roi is not None
                    else None
                ),
                previous_state=previous_state,
                state=update.state.value,
                action=update.action.value if update.action is not None else None,
                initiating_camera=update.initiating_camera,
                frames_processed=dict(frames_processed),
                inference_ms=round(inference_ms, 1),
                frame_age_ms=round(frame_age_ms, 1),
                boxes_visible=sum(
                    item.confidence >= camera_config.box_min_confidence
                    for item in detection.detections
                ),
            )
            if active_started_at is not None:
                active_confidences[role] = max(active_confidences[role], confidence)
            if update.action is not None:
                last_action = update.action.value
            if update.action is not None and update.action.value == "start":
                active_started_at = time.time() - settings.automatic_prebuffer_seconds
                active_initiating_camera = update.initiating_camera
                active_confidences = dict(confidences)
                active_identity_scores = {"face": [], "side": []}
                active_identity_timings = {"face": [], "side": []}
            if active_started_at is not None and prediction.prediction is not None:
                candidate_name = (
                    prediction.prediction.candidate_name
                    or prediction.prediction.name
                )
                if candidate_name is not None:
                    active_identity_scores[role].append(
                        ScoredPrediction(
                            name=candidate_name,
                            confidence=prediction.prediction.confidence,
                        )
                    )
                    if prediction.inference_ms is not None:
                        active_identity_timings[role].append(prediction.inference_ms)
            if (
                update.action is not None
                and update.action.value == "stop"
                and active_started_at is not None
            ):
                online_predictions = _build_online_predictions(
                    model_versions,
                    active_identity_scores,
                    active_identity_timings,
                    active_confidences,
                    settings.identity_confidence_threshold,
                )
                request = AutomaticEventRequest(
                    started_at=active_started_at,
                    ended_at=time.time(),
                    initiating_camera=active_initiating_camera,
                    confidence_face=active_confidences["face"],
                    confidence_side=active_confidences["side"],
                    predictions=online_predictions,
                )
                if not settings.automatic_capture_dry_run:
                    journal.write(
                        "event_publish_scheduled",
                        started_at=request.started_at,
                        ended_at=request.ended_at,
                        initiating_camera=request.initiating_camera,
                        confidence_face=round(request.confidence_face, 4),
                        confidence_side=round(request.confidence_side, 4),
                    )
                    threading.Thread(
                        target=_persist_after_segment_close,
                        args=(settings, request, journal),
                        daemon=True,
                        name="catdoor-event-writer",
                    ).start()
                active_started_at = None
                active_initiating_camera = None
                active_confidences = {"face": 0.0, "side": 0.0}
                active_identity_scores = {"face": [], "side": []}
                active_identity_timings = {"face": [], "side": []}
            now = time.monotonic()
            if update.action is None and now - last_status_at < 0.5:
                continue
            write_recorder_status(
                recorder_status_path(settings),
                process="CatDoorRecorder",
                state=update.state.value,
                message=(
                    "Dry-run: локальный детектор анализирует камеры"
                    if settings.automatic_capture_dry_run
                    else "Автоматический сбор и запись событий включены"
                ),
                automatic_capture_enabled=True,
                dry_run=settings.automatic_capture_dry_run,
                detector_connected=True,
                frames_processed=frames_processed,
                confidences=confidences,
                identity=identity_predictions,
                last_action=last_action,
                initiating_camera=update.initiating_camera,
                **_detector_device_status(settings, detector),
            )
            last_status_at = now
    except (OSError, RuntimeError, ValueError, SQLAlchemyError) as exc:
        journal.write(
            "session_error",
            error_type=type(exc).__name__,
        )
        raise
    finally:
        stop.set()
        write_recorder_status(
            recorder_status_path(settings),
            process="CatDoorRecorder",
            state="probe_completed",
            message="Dry-run проверка завершена",
            automatic_capture_enabled=True,
            dry_run=True,
            detector_connected=True,
            frames_processed=frames_processed,
            confidences=confidences,
            identity=identity_predictions,
            last_action=last_action,
            **_detector_device_status(settings, detector),
        )
        journal.write(
            "session_stopped",
            frames_processed=frames_processed,
            last_action=last_action,
        )
        journal.close()


def _persist_after_segment_close(
    settings: Settings,
    request: AutomaticEventRequest,
    journal: DetectionJournal,
) -> None:
    time.sleep(settings.ring_segment_seconds + 1)
    try:
        event = persist_automatic_event(
            settings.resolved_database_path,
            settings.data_dir,
            settings.ffmpeg_path,
            settings.ring_segment_seconds,
            request,
        )
    except (OSError, RuntimeError, ValueError, SQLAlchemyError) as exc:
        journal.write(
            "event_publish_failed",
            error_type=type(exc).__name__,
            message=str(exc)[:500],
        )
        return
    journal.write("event_published", event_id=event.id, status=event.status)


def run(*, once: bool = False, probe_seconds: float | None = None) -> None:
    settings = Settings()
    if probe_seconds is not None:
        settings.automatic_capture_enabled = True
        settings.automatic_capture_dry_run = True
        monitor_automatic(settings, probe_seconds)
        return
    consecutive_failures = 0
    while True:
        try:
            if settings.automatic_capture_enabled:
                monitor_automatic(settings, None)
            else:
                publish_heartbeat(settings)
            consecutive_failures = 0
        except Exception as exc:  # noqa: BLE001 - recorder must survive component failures
            consecutive_failures += 1
            error_type = type(exc).__name__
            try:
                write_recorder_status(
                    recorder_status_path(settings),
                    process="CatDoorRecorder",
                    state="retrying",
                    message=f"Ошибка recorder ({error_type}), повторный запуск",
                    error_type=error_type,
                    retry_count=consecutive_failures,
                    retry_in_seconds=RECORDER_RETRY_SECONDS,
                    automatic_capture_enabled=settings.automatic_capture_enabled,
                    dry_run=settings.automatic_capture_dry_run,
                    detector_connected=False,
                    **_detector_device_status(settings),
                )
            except Exception:  # noqa: BLE001,S110 - status failure must not stop recorder
                pass
            try:
                journal = DetectionJournal(
                    settings.data_dir / "runtime" / "detection.jsonl",
                    max_bytes=settings.detection_log_max_bytes,
                    backup_count=settings.detection_log_backup_count,
                )
                journal.write(
                    "recorder_retry",
                    error_type=error_type,
                    retry_count=consecutive_failures,
                    retry_in_seconds=RECORDER_RETRY_SECONDS,
                )
                journal.close()
            except Exception:  # noqa: BLE001,S110 - journal failure must not stop recorder
                pass
        if once:
            return
        time.sleep(RECORDER_RETRY_SECONDS)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CatDoor independent recorder service")
    parser.add_argument("--once", action="store_true", help="write one heartbeat and exit")
    parser.add_argument(
        "--probe-seconds",
        type=float,
        help="run bounded dry-run inference without recording video",
    )
    arguments = parser.parse_args(argv)
    run(once=arguments.once, probe_seconds=arguments.probe_seconds)


if __name__ == "__main__":
    main()
