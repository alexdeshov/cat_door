from __future__ import annotations

import argparse
import queue
import sqlite3
import subprocess
import threading
import time
from collections import deque
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from sqlalchemy.exc import SQLAlchemyError

from src.common.camera_discovery import addresses_from_leases, read_dhcp_leases
from src.common.camera_registry import (
    ROLES,
    CameraRegistry,
    camera_registry_path,
    ensure_registry,
    load_registry,
    save_registry,
)
from src.common.capture_mode import CaptureMode, capture_mode_path, load_capture_mode
from src.common.database import create_session_factory
from src.common.detection_config import (
    CameraDetectionConfig,
    DetectionConfig,
    detection_config_path,
    load_detection_config,
)
from src.common.metrics import ScoredPrediction, aggregate_frame_predictions, mean_or_none
from src.common.model_registry import register_model_version
from src.common.sd_notify import Notifier
from src.common.settings import ConfigurationError, Settings, validate_runtime_paths
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
from src.recorder.cleanup import sweep_incomplete_files
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
# Пауза воркера роли, у которой сейчас нет камеры. Назначение роли приходит
# файлом инвентаря, а не сигналом, поэтому проверять чаще смысла нет.
CAMERA_WAIT_SECONDS = 2.0
# Как часто перечитываются аренды DHCP в поисках уехавшего адреса. Камера
# поднимается после подачи питания не мгновенно, гнаться незачем.
LEASE_CHECK_SECONDS = 15.0
# Верхняя граница времени загрузки моделей, в течение которой watchdog кормится
# отдельным потоком. Дальше пинги прекращаются, и systemd имеет полное право
# счесть процесс зависшим: загрузка, которая идёт пять минут, уже не загрузка.
MODEL_LOAD_GRACE_SECONDS = 300.0


@dataclass(frozen=True)
class StreamTarget:
    """Куда сейчас подключается воркер роли.

    Отдельно от `CameraDetectionConfig`, потому что меняется по другой причине:
    настройки детекции правит человек, а адрес — перестановка камер.
    """

    rtsp_url: str = ""
    rotate_180: bool = False
    camera_id: str | None = None


@dataclass
class StreamState:
    """Признаки жизни потока роли и способ его разбудить.

    Чтение кадра блокирующее: камера, исчезнувшая без закрытия TCP-сессии,
    оставляет поток роли в `read` навсегда. Главный цикл при этом продолжает
    работать и исправно кормит watchdog, поэтому systemd такой отказ не увидит
    — заметить и прервать его может только тот, кто помнит время последнего
    кадра и держит сам процесс ffmpeg.
    """

    started_at: float = 0.0
    last_frame_at: float = 0.0
    process: subprocess.Popen[bytes] | None = None

    def silent_for(self, now: float) -> float:
        reference = max(self.started_at, self.last_frame_at)
        return 0.0 if reference <= 0 else now - reference


def restart_stalled_streams(
    stream_states: dict[str, StreamState],
    stream_targets: dict[str, StreamTarget],
    frame_queues: dict[str, queue.Queue[Any]],
    reload_events: dict[str, threading.Event],
    config_lock: threading.Lock,
    stall_seconds: float,
    now: float,
) -> list[str]:
    """Прервать потоки, из которых давно не приходило кадров. Вернуть их роли.

    Процесс именно убивается, а не завершается вежливо: он застрял в системном
    вызове, и смысл сигнала в том, чтобы чтение в воркере вернуло конец файла и
    цикл переподключения пошёл заново. Отметки времени сдвигаются здесь же,
    иначе роль убивали бы на каждой итерации, пока новый поток поднимается.
    """
    stalled: list[str] = []
    with config_lock:
        for role, state in stream_states.items():
            if not stream_targets[role].rtsp_url:
                continue
            if state.silent_for(now) < stall_seconds:
                continue
            stalled.append(role)
            state.started_at = now
            state.last_frame_at = now
            if state.process is not None:
                with suppress(OSError):
                    state.process.kill()
                state.process = None
    for role in stalled:
        _clear_queue(frame_queues[role])
        reload_events[role].set()
    return stalled


def recover_stalled_addresses(
    registry: CameraRegistry,
    registry_path: Path,
    leases_path: Path,
    stream_targets: dict[str, StreamTarget],
    roles: Sequence[str],
) -> tuple[CameraRegistry, dict[str, str]]:
    """Подставить молчащим камерам адрес, под которым их видит DHCP.

    Единственный случай, когда система правит инвентарь сама. Основание для
    этого сильное: MAC совпал, значит это та же камера, а старый адрес молчит
    уже дольше допустимого. Ждать здесь решения владельца означало бы, что
    ночной перебой питания стоит записей до утра.
    """
    camera_ids = [
        stream_targets[role].camera_id
        for role in roles
        if stream_targets[role].camera_id is not None
    ]
    if not camera_ids:
        return registry, {}
    updates = addresses_from_leases(
        registry,
        read_dhcp_leases(leases_path),
        [camera_id for camera_id in camera_ids if camera_id is not None],
    )
    if not updates:
        return registry, {}
    updated = registry
    for camera_id, host in updates.items():
        camera = updated.get(camera_id)
        if camera is None:
            continue
        updated = updated.with_camera(camera.model_copy(update={"host": host}))
    try:
        save_registry(registry_path, updated)
    except OSError:
        # Записать не удалось — инвентарь на диске остаётся прежним, и попытка
        # повторится на следующем застое. Отдавать наружу изменение, которого
        # нет в файле, нельзя: админка показала бы несуществующий адрес.
        return registry, {}
    return updated, updates


@contextmanager
def _watchdog_during_startup(notifier: Notifier | None) -> Iterator[None]:
    """Кормить watchdog, пока грузятся модели, но не бесконечно.

    Между `READY=1` и первой публикацией статуса стоит загрузка OpenVINO-модели
    и регистрация её в базе. На холодном старте после отключения питания, когда
    диск ещё занят проверкой файловой системы, это может занять больше отпущенных
    systemd шестидесяти секунд — и служба попадёт в петлю: убита как зависшая,
    перезапущена, снова убита на том же месте.

    Пинги отсюда идут по таймеру, а не по факту работы, поэтому они ограничены
    сверху: загрузка, которая идёт пять минут, зависла по-настоящему, и мешать
    systemd её прервать больше не нужно.
    """
    interval = notifier.watchdog_interval_seconds if notifier is not None else None
    if notifier is None or interval is None:
        yield
        return
    finished = threading.Event()

    def keep_alive() -> None:
        deadline = time.monotonic() + MODEL_LOAD_GRACE_SECONDS
        while not finished.wait(interval) and time.monotonic() < deadline:
            notifier.watchdog()

    pinger = threading.Thread(target=keep_alive, daemon=True, name="catdoor-startup-watchdog")
    pinger.start()
    try:
        yield
    finally:
        finished.set()
        pinger.join(timeout=1)


def recorder_status_path(settings: Settings) -> Path:
    return settings.data_dir / "runtime" / "recorder-status.json"


def publish_status(
    settings: Settings,
    notifier: Notifier | None,
    **values: Any,
) -> None:
    """Publish the recorder status and tell systemd the process is still alive.

    The two go together on purpose. The recorder reports to the administrator
    and to its supervisor at exactly the same moments, so a loop that stops
    making progress goes silent on both channels at once instead of keeping the
    watchdog happy while the status file goes stale.
    """
    write_recorder_status(recorder_status_path(settings), **values)
    if notifier is not None:
        notifier.watchdog()


def build_automatic_config(settings: Settings) -> AutomaticEventConfig:
    return DetectionConfig.from_settings(settings).event_config()


def _camera_status(stream_targets: dict[str, StreamTarget]) -> dict[str, object]:
    """Что админка показывает про каждую роль, не зная адресов.

    Роль без камеры и роль с камерой, которая молчит, — разные неисправности,
    и различить их по одному счётчику кадров нельзя.
    """
    return {
        role: {"camera_id": target.camera_id, "configured": bool(target.rtsp_url)}
        for role, target in stream_targets.items()
    }


def _waiting_message(stream_targets: dict[str, StreamTarget]) -> str:
    unassigned = [role for role, target in stream_targets.items() if not target.rtsp_url]
    if not unassigned:
        return "Ожидание кадров камер"
    return f"Ролям не назначена камера: {', '.join(sorted(unassigned))}"


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


def stream_targets_from_registry(registry: CameraRegistry) -> dict[str, StreamTarget]:
    targets: dict[str, StreamTarget] = {}
    for role in ROLES:
        camera = registry.by_role(role)
        targets[role] = (
            StreamTarget(
                rtsp_url=camera.rtsp_url() if camera.enabled else "",
                rotate_180=camera.rotate_180,
                camera_id=camera.id,
            )
            if camera is not None
            else StreamTarget()
        )
    return targets


def current_capture_mode(settings: Settings) -> CaptureMode:
    """Режим с диска, а при его отсутствии — заданный окружением."""
    return load_capture_mode(
        capture_mode_path(settings.data_dir),
        CaptureMode.from_settings(settings),
    )


def publish_heartbeat(
    settings: Settings,
    notifier: Notifier | None = None,
    mode: CaptureMode | None = None,
) -> None:
    build_automatic_config(settings)
    # Инвентарь заводится и здесь: recorder может подняться раньше веб-службы,
    # и тогда файл камер должен появиться с его стороны.
    registry = ensure_registry(camera_registry_path(settings.data_dir), settings)
    mode = mode or current_capture_mode(settings)
    if not mode.enabled:
        state = "disabled"
        message = "Автоматический сбор выключен"
    else:
        state = "waiting_for_detector"
        message = "Автомат событий готов, локальный детектор ещё не подключён"
    publish_status(
        settings,
        notifier,
        process="CatDoorRecorder",
        state=state,
        message=message,
        automatic_capture_enabled=mode.enabled,
        dry_run=mode.dry_run,
        detector_connected=False,
        cameras=_camera_status(stream_targets_from_registry(registry)),
        **_detector_device_status(settings),
    )


def _frame_worker(
    role: str,
    settings: Settings,
    output: queue.Queue[tuple[str, Any, Any, Any, float, CameraDetectionConfig]],
    camera_configs: dict[str, CameraDetectionConfig],
    stream_targets: dict[str, StreamTarget],
    stream_states: dict[str, StreamState],
    config_lock: threading.Lock,
    reload_stream: threading.Event,
    stop: threading.Event,
    detector_input_size: int,
    detector_letterbox_value: int,
    detector_resize_mode: str,
    detector_letterbox_alignment: str,
) -> None:
    state = stream_states[role]

    def remember_process(process: subprocess.Popen[bytes]) -> None:
        with config_lock:
            state.process = process

    while not stop.is_set():
        with config_lock:
            camera_config = camera_configs[role]
            target = stream_targets[role]
            state.process = None
            # Отсчёт молчания начинается с попытки подключиться, а не с первого
            # кадра: камера, которая приняла соединение и не отдала ни одного
            # кадра, — такой же застой, как и оборвавшаяся посреди потока.
            state.started_at = time.monotonic()
            state.last_frame_at = 0.0
            reload_stream.clear()
        if not target.rtsp_url:
            # Роль без камеры — это ожидание назначения, а не конец работы.
            # Раньше воркер здесь завершался, и камера, подключённая позже,
            # начинала отдавать кадры только после перезапуска службы.
            with config_lock:
                state.started_at = 0.0
            stop.wait(CAMERA_WAIT_SECONDS)
            continue
        try:
            for sample in rgb_frame_stream(
                settings.ffmpeg_path,
                target.rtsp_url,
                camera_config.sample_fps,
                segment_directory=settings.data_dir / "runtime" / "ring" / role,
                segment_seconds=settings.ring_segment_seconds,
                retention_seconds=settings.ring_retention_seconds,
                roi=camera_config.roi,
                rotate_180=target.rotate_180,
                on_process=remember_process,
            ):
                if stop.is_set():
                    return
                if reload_stream.is_set():
                    break
                with config_lock:
                    state.last_frame_at = time.monotonic()
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


def monitor_automatic(
    settings: Settings,
    duration_seconds: float | None,
    notifier: Notifier | None = None,
) -> None:
    config_path = detection_config_path(settings.data_dir)
    mode_path = capture_mode_path(settings.data_dir)
    runtime_mode = current_capture_mode(settings)
    stopped_by_operator = False
    registry_path = camera_registry_path(settings.data_dir)
    registry = ensure_registry(registry_path, settings)
    runtime_config = load_detection_config(config_path, DetectionConfig.from_settings(settings))
    journal = DetectionJournal(
        settings.data_dir / "runtime" / "detection.jsonl",
        max_bytes=settings.detection_log_max_bytes,
        backup_count=settings.detection_log_backup_count,
    )
    with _watchdog_during_startup(notifier):
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
    stream_targets = stream_targets_from_registry(registry)
    stream_states = {role: StreamState() for role in ("face", "side")}
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
                stream_targets,
                stream_states,
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
        recording=runtime_mode.record_events,
        cameras={
            role: target.camera_id if target.rtsp_url else None
            for role, target in stream_targets.items()
        },
    )
    last_config_check_at = 0.0
    last_lease_check_at = 0.0
    next_camera = "face"
    try:
        while duration_seconds is None or time.monotonic() - started_at < duration_seconds:
            now = time.monotonic()
            # Проверка застоя идёт вне зависимости от состояния автомата:
            # камера, замолчавшая посреди прохода, — это как раз тот момент,
            # когда ждать возврата в ожидание бессмысленно.
            stalled = restart_stalled_streams(
                stream_states,
                stream_targets,
                frame_queues,
                reload_events,
                config_lock,
                settings.camera_stall_seconds,
                now,
            )
            if stalled:
                journal.write(
                    "stream_stalled",
                    roles=stalled,
                    silent_seconds=round(settings.camera_stall_seconds, 1),
                    cameras={
                        role: stream_targets[role].camera_id for role in stalled
                    },
                )
                if now - last_lease_check_at >= LEASE_CHECK_SECONDS:
                    last_lease_check_at = now
                    registry, moved = recover_stalled_addresses(
                        registry,
                        registry_path,
                        settings.camera_dhcp_leases_path,
                        stream_targets,
                        stalled,
                    )
                    if moved:
                        # Новый адрес подставляется сразу, а не ждёт обычной
                        # сверки инвентаря: та сравнивает файл с уже обновлённой
                        # копией в памяти и разницы не увидит.
                        recovered_targets = stream_targets_from_registry(registry)
                        with config_lock:
                            for role in stalled:
                                stream_targets[role] = recovered_targets[role]
                        # Адреса в журнал не пишутся: достаточно того, какая
                        # камера переехала. Точное значение видно в админке.
                        journal.write("camera_address_recovered", cameras=sorted(moved))
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
                updated_mode = load_capture_mode(mode_path, runtime_mode)
                if updated_mode != runtime_mode:
                    runtime_mode = updated_mode
                    journal.write("capture_mode_changed", state=runtime_mode.state)
                if not runtime_mode.enabled:
                    # Выключено из админки. Выходим здесь, а не по исключению:
                    # внешний цикл увидит тот же файл и перейдёт на heartbeat.
                    stopped_by_operator = True
                    return
                updated_registry = load_registry(registry_path)
                if updated_registry is not None and updated_registry != registry:
                    registry = updated_registry
                    updated_targets = stream_targets_from_registry(registry)
                    changed_roles = [
                        role
                        for role in ("face", "side")
                        if updated_targets[role] != stream_targets[role]
                    ]
                    with config_lock:
                        for role in changed_roles:
                            stream_targets[role] = updated_targets[role]
                            _clear_queue(frame_queues[role])
                            reload_events[role].set()
                    if changed_roles:
                        # Адрес не пишется даже в замаскированном виде: в
                        # журнале хватает того, какая камера теперь на роли.
                        journal.write(
                            "cameras_updated",
                            roles=changed_roles,
                            cameras={
                                role: updated_targets[role].camera_id
                                for role in ("face", "side")
                            },
                            configured={
                                role: bool(updated_targets[role].rtsp_url)
                                for role in ("face", "side")
                            },
                        )
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
                publish_status(
                    settings,
                    notifier,
                    process="CatDoorRecorder",
                    state="waiting_for_frames",
                    message=_waiting_message(stream_targets),
                    automatic_capture_enabled=True,
                    dry_run=True,
                    detector_connected=True,
                    frames_processed=frames_processed,
                    confidences=confidences,
                    cameras=_camera_status(stream_targets),
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
                if runtime_mode.record_events:
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
            publish_status(
                settings,
                notifier,
                process="CatDoorRecorder",
                state=update.state.value,
                message=(
                    "Наблюдение: детектор анализирует камеры, записи нет"
                    if runtime_mode.dry_run
                    else "Автоматический сбор и запись событий включены"
                ),
                automatic_capture_enabled=True,
                dry_run=runtime_mode.dry_run,
                detector_connected=True,
                frames_processed=frames_processed,
                confidences=confidences,
                identity=identity_predictions,
                last_action=last_action,
                initiating_camera=update.initiating_camera,
                cameras=_camera_status(stream_targets),
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
        publish_status(
            settings,
            notifier,
            process="CatDoorRecorder",
            state="disabled" if stopped_by_operator else "probe_completed",
            message=(
                "Автоматический сбор выключен"
                if stopped_by_operator
                else "Проверка детектора завершена"
            ),
            automatic_capture_enabled=not stopped_by_operator,
            dry_run=runtime_mode.dry_run,
            detector_connected=True,
            frames_processed=frames_processed,
            confidences=confidences,
            identity=identity_predictions,
            last_action=last_action,
            cameras=_camera_status(stream_targets),
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
    validate_runtime_paths(settings)
    # Единственный момент, когда точно известно, что прерванная запись уже не
    # продолжится: процесс только что стартовал. Пропажа питания не исполняет
    # ни одного блока `finally`, поэтому её следы убираются здесь.
    sweep_incomplete_files(settings.data_dir)
    notifier = Notifier.from_environment()
    if probe_seconds is not None:
        settings.automatic_capture_enabled = True
        settings.automatic_capture_dry_run = True
        monitor_automatic(settings, probe_seconds, notifier)
        return
    # Readiness is announced only once the paths are known to work: under
    # Type=notify systemd holds the unit in `activating` until READY=1 arrives,
    # so an unusable data directory shows up as a failed start instead of a
    # service that claims to be running and does nothing.
    notifier.ready("Recorder запущен")
    consecutive_failures = 0
    while True:
        try:
            mode = current_capture_mode(settings)
            if mode.enabled:
                monitor_automatic(settings, None, notifier)
            else:
                publish_heartbeat(settings, notifier, mode)
            consecutive_failures = 0
        except Exception as exc:  # noqa: BLE001 - recorder must survive component failures
            consecutive_failures += 1
            error_type = type(exc).__name__
            try:
                publish_status(
                    settings,
                    notifier,
                    process="CatDoorRecorder",
                    state="retrying",
                    message=f"Ошибка recorder ({error_type}), повторный запуск",
                    error_type=error_type,
                    retry_count=consecutive_failures,
                    retry_in_seconds=RECORDER_RETRY_SECONDS,
                    automatic_capture_enabled=mode.enabled,
                    dry_run=mode.dry_run,
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
    try:
        run(once=arguments.once, probe_seconds=arguments.probe_seconds)
    except ConfigurationError as exc:
        raise SystemExit(f"Конфигурация непригодна для запуска: {exc}") from exc


if __name__ == "__main__":
    main()
