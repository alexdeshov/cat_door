from __future__ import annotations

import argparse
import queue
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.common.settings import Settings
from src.inference.cat_detector import DetectorError, OpenVinoCatDetector
from src.recorder.automatic import AutomaticEventConfig, AutomaticEventMachine
from src.recorder.events import AutomaticEventRequest, persist_automatic_event
from src.recorder.frames import rgb_frame_stream
from src.recorder.status import write_recorder_status


def recorder_status_path(settings: Settings) -> Path:
    return settings.data_dir / "runtime" / "recorder-status.json"


def build_automatic_config(settings: Settings) -> AutomaticEventConfig:
    return AutomaticEventConfig(
        confidence_threshold=settings.detection_confidence_threshold,
        confirmation_frames=settings.detection_confirmation_frames,
        window_frames=settings.detection_window_frames,
        candidate_timeout_seconds=settings.detection_candidate_timeout_seconds,
        absence_seconds=settings.detection_absence_seconds,
        tail_seconds=settings.detection_tail_seconds,
        cooldown_seconds=settings.detection_cooldown_seconds,
    )


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
    output: queue.Queue[tuple[str, Any]],
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
            try:
                output.put((role, frame), timeout=0.1)
            except queue.Full:
                continue
    except OSError:
        return


def monitor_automatic(settings: Settings, duration_seconds: float | None) -> None:
    detector = OpenVinoCatDetector(settings.detector_model_path, settings.detector_device)
    machine = AutomaticEventMachine(build_automatic_config(settings))
    frames: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=4)
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
    frames_processed = {"face": 0, "side": 0}
    last_action: str | None = None
    last_status_at = 0.0
    active_started_at: float | None = None
    active_initiating_camera: str | None = None
    active_confidences = {"face": 0.0, "side": 0.0}
    try:
        while duration_seconds is None or time.monotonic() - started_at < duration_seconds:
            try:
                role, frame = frames.get(timeout=2)
            except queue.Empty:
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
            confidence = detector.confidence(frame)
            confidences[role] = round(confidence, 4)
            frames_processed[role] += 1
            update = machine.observe(time.monotonic(), {role: confidence})
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
                    threading.Thread(
                        target=_persist_after_segment_close,
                        args=(settings, request),
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
                last_action=last_action,
                initiating_camera=update.initiating_camera,
            )
            last_status_at = now
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
            last_action=last_action,
        )


def _persist_after_segment_close(
    settings: Settings,
    request: AutomaticEventRequest,
) -> None:
    time.sleep(settings.ring_segment_seconds + 1)
    persist_automatic_event(
        settings.resolved_database_path,
        settings.data_dir,
        settings.ffmpeg_path,
        settings.ring_segment_seconds,
        request,
    )


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
