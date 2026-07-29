from __future__ import annotations

import argparse
import queue
import threading
import time
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from src.common.detection_config import (
    CameraDetectionConfig,
    DetectionConfig,
    detection_config_path,
    load_detection_config,
)
from src.common.settings import Settings
from src.inference.cat_detector import DetectorError, OpenVinoCatDetector
from src.inference.cat_identity import OpenVinoCatIdentity
from src.recorder.automatic import AutomaticEventConfig, AutomaticEventMachine
from src.recorder.detection_journal import DetectionJournal
from src.recorder.events import AutomaticEventRequest, persist_automatic_event
from src.recorder.frames import prepare_detector_frame, rgb_frame_stream
from src.recorder.preview import annotate_detections, preview_path, publish_preview
from src.recorder.status import write_recorder_status


def recorder_status_path(settings: Settings) -> Path:
    return settings.data_dir / "runtime" / "recorder-status.json"


def build_automatic_config(settings: Settings) -> AutomaticEventConfig:
    return DetectionConfig.from_settings(settings).event_config()


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


class CameraConfirmationTracker:
    def __init__(self, config: CameraDetectionConfig) -> None:
        self.config = config
        self._window: deque[bool] = deque(maxlen=config.window_frames)

    def observe(self, confidence: float) -> bool:
        self._window.append(confidence >= self.config.confidence_threshold)
        return sum(self._window) >= self.config.confirmation_frames


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
    )


def _frame_worker(
    role: str,
    settings: Settings,
    output: queue.Queue[tuple[str, Any, Any, float]],
    stop: threading.Event,
) -> None:
    url = settings.camera_face_rtsp if role == "face" else settings.camera_side_rtsp
    if not url:
        return
    try:
        for frame in rgb_frame_stream(
            settings.ffmpeg_path,
            url,
            settings.detector_sample_fps,
            segment_directory=settings.data_dir / "runtime" / "ring" / role,
            segment_seconds=settings.ring_segment_seconds,
            retention_seconds=settings.ring_retention_seconds,
        ):
            if stop.is_set():
                return
            item = (role, prepare_detector_frame(frame), frame, time.monotonic())
            _put_latest(output, item)
    except OSError:
        return


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
    machine = AutomaticEventMachine(runtime_config.event_config())
    trackers = {
        role: CameraConfirmationTracker(getattr(runtime_config, role))
        for role in ("face", "side")
    }
    frames: queue.Queue[tuple[str, Any, Any, float]] = queue.Queue(maxsize=2)
    stop = threading.Event()
    workers = [
        threading.Thread(
            target=_frame_worker,
            args=(role, settings, frames, stop),
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
    journal.write(
        "session_started",
        camera_settings={
            role: getattr(runtime_config, role).model_dump()
            for role in ("face", "side")
        },
        sample_fps=settings.detector_sample_fps,
        recording=not settings.automatic_capture_dry_run,
        cameras={
            "face": bool(settings.camera_face_rtsp),
            "side": bool(settings.camera_side_rtsp),
        },
    )
    last_config_check_at = 0.0
    try:
        while duration_seconds is None or time.monotonic() - started_at < duration_seconds:
            now = time.monotonic()
            if now - last_config_check_at >= 1 and machine.state.value == "idle":
                updated_config = load_detection_config(config_path, runtime_config)
                if updated_config != runtime_config:
                    runtime_config = updated_config
                    machine = AutomaticEventMachine(runtime_config.event_config())
                    trackers = {
                        role: CameraConfirmationTracker(getattr(runtime_config, role))
                        for role in ("face", "side")
                    }
                    journal.write("config_updated", **runtime_config.model_dump())
                last_config_check_at = now
            try:
                role, frame, preview_frame, captured_at = frames.get(timeout=2)
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
                )
                continue
            inference_started_at = time.monotonic()
            detection = detector.detect(frame)
            inference_ms = (time.monotonic() - inference_started_at) * 1000
            frame_age_ms = (time.monotonic() - captured_at) * 1000
            confidence = detection.confidence
            camera_config = getattr(runtime_config, role)
            camera_confirmed = trackers[role].observe(confidence)
            try:
                annotated_frame = annotate_detections(
                    preview_frame,
                    detection.detections,
                    event_threshold=camera_config.confidence_threshold,
                    box_min_confidence=camera_config.box_min_confidence,
                )
                publish_preview(preview_path(settings.data_dir, role), annotated_frame)
            except OSError:
                pass
            previous_state = machine.state.value
            confidences[role] = round(confidence, 4)
            if confidence >= settings.detection_confidence_threshold and role in identity_models:
                prediction = identity_models[role].predict(
                    frame,
                    settings.identity_confidence_threshold,
                )
                identity_predictions[role] = {
                    "name": prediction.name,
                    "confidence": round(prediction.confidence, 4),
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
            elif (
                update.action is not None
                and update.action.value == "stop"
                and active_started_at is not None
            ):
                request = AutomaticEventRequest(
                    started_at=active_started_at,
                    ended_at=time.time(),
                    initiating_camera=active_initiating_camera,
                    confidence_face=active_confidences["face"],
                    confidence_side=active_confidences["side"],
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
            )
            last_status_at = now
    except (OSError, RuntimeError, ValueError, SQLAlchemyError) as exc:
        journal.write(
            "session_error",
            error_type=type(exc).__name__,
            message=str(exc)[:500],
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
    while True:
        try:
            if settings.automatic_capture_enabled:
                monitor_automatic(settings, None)
            else:
                publish_heartbeat(settings)
        except (DetectorError, RuntimeError, ValueError) as exc:
            write_recorder_status(
                recorder_status_path(settings),
                process="CatDoorRecorder",
                state="error",
                message=str(exc),
                automatic_capture_enabled=settings.automatic_capture_enabled,
                dry_run=settings.automatic_capture_dry_run,
                detector_connected=False,
            )
        if once:
            return
        time.sleep(5)


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
