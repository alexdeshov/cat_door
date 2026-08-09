import json
import queue
import sqlite3
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy.exc import SQLAlchemyError

from src.common.camera_registry import (
    CameraRecord,
    CameraRegistry,
    camera_registry_path,
    load_registry,
    save_registry,
)
from src.common.capture_mode import CaptureMode, capture_mode_path, save_capture_mode
from src.common.detection_config import CameraDetectionConfig, DetectionConfig
from src.common.sd_notify import Notifier
from src.common.settings import Settings
from src.inference.cat_detector import CatDetection, CatDetectionResult
from src.inference.cat_identity import IdentityPrediction
from src.recorder.app import (
    CameraConfirmationTracker,
    StreamState,
    StreamTarget,
    _detection_profile_status,
    _frame_worker,
    _get_fair_frame,
    _predict_identity_if_detected,
    _put_latest,
    _register_loaded_models,
    build_automatic_config,
    current_capture_mode,
    monitor_automatic,
    publish_heartbeat,
    recorder_status_path,
    recover_stalled_addresses,
    restart_stalled_streams,
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


def test_status_reports_the_thresholds_the_pipeline_actually_uses() -> None:
    """«Записано» и «применено» обязаны различаться снаружи.

    Журнал детекции, где живёт `config_updated`, лежит под правами службы: без
    пароля его не прочитать даже с узла. Статус читает кто угодно, поэтому
    действующий профиль отвечает именно он.
    """
    config = DetectionConfig.from_settings(Settings())

    profile = _detection_profile_status(config)["detection_profile"]

    assert profile["face"]["confidence_threshold"] == config.face.confidence_threshold
    assert profile["side"]["sample_fps"] == config.side.sample_fps


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
    states = {"face": StreamState()}
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
            states,
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


def _stall_fixtures(role: str = "face") -> tuple[dict, dict, dict, dict, threading.Lock]:
    states = {
        role: StreamState(
            target_key=("rtsp://10.0.0.5:554/one", False),
            target_since=100.0,
            last_frame_at=100.0,
            process_started_at=100.0,
        )
    }
    targets = {role: StreamTarget(rtsp_url="rtsp://10.0.0.5:554/one", camera_id="hiwatch")}
    queues: dict[str, queue.Queue] = {role: queue.Queue(maxsize=1)}
    events = {role: threading.Event()}
    return states, targets, queues, events, threading.Lock()


def test_a_wedged_stream_is_killed_so_the_worker_can_reconnect() -> None:
    """Главная дыра до этой правки: зависший ffmpeg не виден никому.

    Чтение кадра блокирующее, главный цикл при этом жив и исправно кормит
    watchdog, поэтому systemd такой отказ не замечает вовсе.
    """
    states, targets, queues, events, lock = _stall_fixtures()
    killed: list[str] = []
    states["face"].process = SimpleNamespace(kill=lambda: killed.append("face"))
    queues["face"].put_nowait("устаревший кадр")

    starving, dead = restart_stalled_streams(states, targets, queues, events, lock, 30.0, 200.0)

    assert starving == ["face"]
    assert dead == ["face"]
    assert killed == ["face"]
    assert events["face"].is_set()
    assert queues["face"].empty()
    assert states["face"].process is None


def test_a_live_stream_is_left_alone() -> None:
    states, targets, queues, events, lock = _stall_fixtures()
    states["face"].last_frame_at = 199.0

    starving, dead = restart_stalled_streams(states, targets, queues, events, lock, 30.0, 200.0)

    assert (starving, dead) == ([], [])
    assert not events["face"].is_set()


def test_a_role_without_a_camera_is_not_called_stalled() -> None:
    """Пустая роль молчит по совершенно другой причине — её нечего перезапускать."""
    states, targets, queues, events, lock = _stall_fixtures()
    targets["face"] = StreamTarget()

    starving, dead = restart_stalled_streams(states, targets, queues, events, lock, 30.0, 900.0)

    assert (starving, dead) == ([], [])


def test_starvation_persists_across_reconnect_attempts() -> None:
    """Голодание не сбрасывается переподключениями — в этом была главная дыра.

    Мёртвая камера даёт быстрый цикл подключение-отказ по пятисекундному
    таймауту ffmpeg. Прежний код сбрасывал отсчёт на каждой попытке, порог
    застоя не достигался никогда, и поиск камеры в арендах не запускался.
    """
    states, targets, queues, events, lock = _stall_fixtures()
    # Кадров не было вовсе, попытки подключения идут одна за другой.
    states["face"].last_frame_at = 0.0
    states["face"].process = None

    starving, dead = restart_stalled_streams(states, targets, queues, events, lock, 30.0, 200.0)

    assert starving == ["face"]
    assert dead == []


def test_a_fresh_process_is_given_time_before_the_kill() -> None:
    """Свежее подключение не режется, даже когда роль давно голодает."""
    states, targets, queues, events, lock = _stall_fixtures()
    states["face"].last_frame_at = 0.0
    states["face"].process = SimpleNamespace(kill=lambda: None)
    states["face"].process_started_at = 195.0  # процессу пять секунд

    starving, dead = restart_stalled_streams(states, targets, queues, events, lock, 30.0, 200.0)

    assert starving == ["face"]
    assert dead == []


def test_each_fruitless_kill_doubles_the_patience() -> None:
    """Камера с кадром раз в сорок секунд не должна жить в вечном kill-цикле."""
    states, targets, queues, events, lock = _stall_fixtures()
    state = states["face"]
    state.last_frame_at = 0.0
    state.kills_without_frames = 2
    state.process = SimpleNamespace(kill=lambda: None)
    state.process_started_at = 100.0

    # 30 * 2**2 = 120 секунд выдержки: на 219-й секунде процессу 119 — рано.
    starving, dead = restart_stalled_streams(states, targets, queues, events, lock, 30.0, 219.0)
    assert (starving, dead) == (["face"], [])

    starving, dead = restart_stalled_streams(states, targets, queues, events, lock, 30.0, 221.0)
    assert dead == ["face"]
    assert state.kills_without_frames == 3


def _recovery_registry(runtime_path: Path) -> Path:
    registry_path = camera_registry_path(runtime_path)
    save_registry(
        registry_path,
        CameraRegistry(
            cameras=[
                CameraRecord(
                    id="hiwatch",
                    host="192.168.1.101",
                    stream_path="/one",
                    username="admin",
                    password="secret",
                    mac="44:47:cc:11:22:33",
                    role="face",
                )
            ]
        ),
    )
    return registry_path


def _fresh_lease(runtime_path: Path, line: str) -> None:
    (runtime_path / "dnsmasq.leases").write_text(line, encoding="utf-8")


def _recovery_settings(runtime_path: Path) -> Settings:
    return Settings(
        data_dir=runtime_path,
        camera_dhcp_leases_path=runtime_path / "dnsmasq.leases",
    )


def test_silent_camera_gets_its_new_address_from_the_leases(runtime_path: Path) -> None:
    """Обрыв питания у камеры: вернулась с новой арендой, старый адрес мёртв."""
    registry_path = _recovery_registry(runtime_path)
    _fresh_lease(runtime_path, "9999999999 44:47:cc:11:22:33 192.168.1.137 HIWATCH *\n")
    targets = stream_targets_from_registry(load_registry(registry_path))

    updated, moved = recover_stalled_addresses(
        _recovery_settings(runtime_path),
        registry_path,
        targets,
        ["face"],
        probe=lambda url: None,
    )

    assert moved == {"hiwatch": "192.168.1.137"}
    camera = load_registry(registry_path).by_role("face")
    assert camera.host == "192.168.1.137"
    # Меняется только адрес: смена аренды не повод трогать всё остальное.
    assert camera.password == "secret"
    assert camera.stream_path == "/one"
    assert camera.role == "face"
    assert updated is not None and updated.by_role("face").host == "192.168.1.137"


def test_unanswered_candidate_address_is_not_applied(runtime_path: Path) -> None:
    """Проба обязательна: аренда — самое слабое свидетельство из возможных.

    Лежалая запись со старым адресом иначе перезаписала бы рабочий адрес
    мёртвым, и «автопочинка» делала бы хуже, чем бездействие.
    """
    registry_path = _recovery_registry(runtime_path)
    _fresh_lease(runtime_path, "9999999999 44:47:cc:11:22:33 192.168.1.137 HIWATCH *\n")
    targets = stream_targets_from_registry(load_registry(registry_path))

    _updated, moved = recover_stalled_addresses(
        _recovery_settings(runtime_path),
        registry_path,
        targets,
        ["face"],
        probe=lambda url: "камера не ответила",
    )

    assert moved == {}
    assert load_registry(registry_path).by_role("face").host == "192.168.1.101"


def test_expired_lease_is_ignored(runtime_path: Path) -> None:
    registry_path = _recovery_registry(runtime_path)
    _fresh_lease(runtime_path, "1000000000 44:47:cc:11:22:33 192.168.1.137 HIWATCH *\n")
    targets = stream_targets_from_registry(load_registry(registry_path))

    _updated, moved = recover_stalled_addresses(
        _recovery_settings(runtime_path),
        registry_path,
        targets,
        ["face"],
        probe=lambda url: None,
    )

    assert moved == {}


def test_an_address_already_owned_by_another_camera_is_not_stolen(
    runtime_path: Path,
) -> None:
    """Чужой адрес хуже неработающего: роль молча писала бы не тот ракурс."""
    registry_path = camera_registry_path(runtime_path)
    save_registry(
        registry_path,
        CameraRegistry(
            cameras=[
                CameraRecord(
                    id="hiwatch", host="192.168.1.101", mac="44:47:cc:11:22:33", role="face"
                ),
                CameraRecord(id="nobelic", host="192.168.1.137", role="side"),
            ]
        ),
    )
    _fresh_lease(runtime_path, "9999999999 44:47:cc:11:22:33 192.168.1.137 HIWATCH *\n")
    targets = stream_targets_from_registry(load_registry(registry_path))

    _updated, moved = recover_stalled_addresses(
        _recovery_settings(runtime_path),
        registry_path,
        targets,
        ["face"],
        probe=lambda url: None,
    )

    assert moved == {}
    assert load_registry(registry_path).by_role("face").host == "192.168.1.101"


def test_recovery_reads_the_registry_from_disk_not_from_memory(
    runtime_path: Path,
) -> None:
    """Правки админа, сделанные пока recorder был занят, не затираются.

    Recorder держит копию реестра в памяти и перечитывает её только в
    ожидании; recovery же пишет файл целиком. Возьми она копию из памяти —
    пароль, сохранённый админом во время события, молча откатился бы.
    """
    registry_path = _recovery_registry(runtime_path)
    targets = stream_targets_from_registry(load_registry(registry_path))
    # Пока recorder «был занят», админ сменил пароль камеры на диске.
    fresh = load_registry(registry_path)
    save_registry(
        registry_path,
        fresh.with_camera(fresh.get("hiwatch").model_copy(update={"password": "новый"})),
    )
    _fresh_lease(runtime_path, "9999999999 44:47:cc:11:22:33 192.168.1.137 HIWATCH *\n")

    recover_stalled_addresses(
        _recovery_settings(runtime_path),
        registry_path,
        targets,
        ["face"],
        probe=lambda url: None,
    )

    camera = load_registry(registry_path).get("hiwatch")
    assert camera.host == "192.168.1.137"
    assert camera.password == "новый"


def test_camera_without_a_mac_is_left_as_is(runtime_path: Path) -> None:
    """Без MAC совпадение недоказуемо, а чужой адрес хуже неработающего."""
    registry_path = camera_registry_path(runtime_path)
    save_registry(
        registry_path,
        CameraRegistry(cameras=[CameraRecord(id="ursa", host="192.168.1.101", role="face")]),
    )
    _fresh_lease(runtime_path, "9999999999 44:47:cc:11:22:33 192.168.1.137 X *\n")
    registry = load_registry(registry_path)

    _updated, moved = recover_stalled_addresses(
        _recovery_settings(runtime_path),
        registry_path,
        stream_targets_from_registry(registry),
        ["face"],
        probe=lambda url: None,
    )

    assert moved == {}
    assert load_registry(registry_path).by_role("face").host == "192.168.1.101"


def test_missing_leases_file_changes_nothing(runtime_path: Path) -> None:
    registry_path = _recovery_registry(runtime_path)
    registry = load_registry(registry_path)

    _updated, moved = recover_stalled_addresses(
        _recovery_settings(runtime_path),
        registry_path,
        stream_targets_from_registry(registry),
        ["face"],
        probe=lambda url: None,
    )

    assert moved == {}


def test_worker_reconnects_after_a_wedged_process_is_killed(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """Сквозная проверка на настоящих процессах, без подмены ffmpeg заглушкой.

    Имитируется ровно тот отказ, который раньше был невидим: процесс запущен,
    соединение принято, кадров нет и не будет. Воркер обязан выйти из чтения
    после убийства и подключиться заново.
    """
    launches = runtime_path / "launches.txt"
    fake_ffmpeg = runtime_path / "hanging-ffmpeg.sh"
    # `exec` обязателен: без него канал держал бы дочерний `sleep`, а убийство
    # оболочки чтение бы не разбудило. Настоящий ffmpeg — один процесс и держит
    # канал сам, так что здесь воспроизводится именно его случай.
    fake_ffmpeg.write_text(
        f'#!/bin/sh\necho started >> "{launches}"\nexec sleep 30\n',
        encoding="utf-8",
    )
    fake_ffmpeg.chmod(0o755)

    def launch_count() -> int:
        return len(launches.read_text(encoding="utf-8").splitlines()) if launches.exists() else 0

    def wait_for(count: int, timeout: float = 15.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if launch_count() >= count:
                return
            time.sleep(0.05)
        raise AssertionError(f"ожидался {count}-й запуск, было {launch_count()}")

    settings = Settings(data_dir=runtime_path, ffmpeg_path=str(fake_ffmpeg))
    camera_configs = {
        "face": CameraDetectionConfig(
            confidence_threshold=0.5,
            box_min_confidence=0.2,
            confirmation_frames=1,
            window_frames=1,
        )
    }
    targets = {"face": StreamTarget(rtsp_url="rtsp://10.0.0.5:554/one", camera_id="hiwatch")}
    states = {"face": StreamState()}
    queues: dict[str, queue.Queue] = {"face": queue.Queue(maxsize=1)}
    events = {"face": threading.Event()}
    lock = threading.Lock()
    stop = threading.Event()
    worker = threading.Thread(
        target=_frame_worker,
        args=(
            "face",
            settings,
            queues["face"],
            camera_configs,
            targets,
            states,
            lock,
            events["face"],
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
        wait_for(1)
        with lock:
            assert states["face"].process is not None

        # Момент проверки задаётся явно вперёд, а не берётся текущим: иначе
        # тест держится на том, что между подключением воркера и этой строкой
        # пройдёт больше порога, а на быстрой машине не проходит.
        starving, dead = restart_stalled_streams(
            states,
            targets,
            queues,
            events,
            lock,
            0.1,
            time.monotonic() + 60,
        )

        assert starving == ["face"]
        assert dead == ["face"]
        wait_for(2)
        assert worker.is_alive()
    finally:
        stop.set()
        with lock:
            if states["face"].process is not None:
                states["face"].process.kill()
        worker.join(timeout=15)
        assert not worker.is_alive()


def test_recorder_picks_the_mode_from_disk_not_from_the_environment(
    runtime_path: Path,
) -> None:
    """Переключение из админки должно доходить до recorder без перезапуска."""
    settings = Settings(data_dir=runtime_path, automatic_capture_enabled=False)

    assert current_capture_mode(settings).state == "off"

    save_capture_mode(capture_mode_path(runtime_path), CaptureMode.from_state("record"))

    assert current_capture_mode(settings).state == "record"


def test_heartbeat_reports_the_switched_mode(runtime_path: Path) -> None:
    settings = Settings(data_dir=runtime_path, automatic_capture_enabled=False)
    save_capture_mode(capture_mode_path(runtime_path), CaptureMode.from_state("record"))

    publish_heartbeat(settings)

    payload = json.loads(recorder_status_path(settings).read_text(encoding="utf-8"))
    assert payload["state"] == "waiting_for_detector"
    assert payload["automatic_capture_enabled"] is True
    assert payload["dry_run"] is False


def test_disabling_from_the_admin_stops_the_running_monitor(
    runtime_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Выключение посреди работы: монитор обязан выйти сам, а не повиснуть.

    Без этого переключатель работал бы только в одну сторону — включить можно,
    а выключить лишь перезапуском службы.
    """
    save_capture_mode(capture_mode_path(runtime_path), CaptureMode.from_state("observe"))
    # Роли нужен адрес: без него воркер ждёт назначения и до подключения, на
    # котором построена проверка, не доходит.
    settings = Settings(
        data_dir=runtime_path,
        automatic_capture_enabled=True,
        camera_face_rtsp="rtsp://10.0.0.5:554/one",
    )

    class SilentDetector:
        device = "CPU"
        input_size = 320
        letterbox_value = 114
        resize_mode = "letterbox"
        letterbox_alignment = "center"
        model_name = "cat-detector.onnx"
        model_sha256 = "0" * 64

    monkeypatch.setattr("src.recorder.app.OpenVinoCatDetector", lambda *args: SilentDetector())
    monkeypatch.setattr("src.recorder.app._register_loaded_models", lambda *args: {"detector": 1})

    def switch_off_once_running(*args, **kwargs):
        # Кадров не будет: монитор крутится вхолостую, и это ровно то
        # состояние ожидания, в котором переключение должно применяться.
        save_capture_mode(capture_mode_path(runtime_path), CaptureMode.from_state("off"))
        raise OSError("камер нет")

    monkeypatch.setattr("src.recorder.app.rgb_frame_stream", switch_off_once_running)

    monitor_automatic(settings, 30.0, None)

    payload = json.loads(recorder_status_path(settings).read_text(encoding="utf-8"))
    assert payload["state"] == "disabled"
    assert payload["automatic_capture_enabled"] is False


def test_frame_worker_survives_an_unexpected_exception(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """Воркер обязан быть бессмертным.

    Раньше ловился только OSError, и любое другое исключение убивало поток;
    главный цикл продолжал кормить watchdog при полностью мёртвом захвате.
    """
    attempts: list[int] = []
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    def flaky_stream(ffmpeg_path: str, rtsp_url: str, fps: float, **options):
        attempts.append(1)
        if len(attempts) == 1:
            raise ValueError("рассинхронизация пайпа")
        yield FrameSample(preview=frame, detector=frame)

    monkeypatch.setattr("src.recorder.app.rgb_frame_stream", flaky_stream)
    monkeypatch.setattr("src.recorder.app.prepare_detector_frame", lambda sample, *rest: sample)
    monkeypatch.setattr("src.recorder.app.RECORDER_RETRY_SECONDS", 0.05)

    settings = Settings(data_dir=runtime_path, camera_face_rtsp="rtsp://10.0.0.5:554/one")
    camera_configs = {
        "face": CameraDetectionConfig(
            confidence_threshold=0.5,
            box_min_confidence=0.2,
            confirmation_frames=1,
            window_frames=1,
        )
    }
    targets = {"face": StreamTarget(rtsp_url="rtsp://10.0.0.5:554/one", camera_id="x")}
    states = {"face": StreamState()}
    output: queue.Queue = queue.Queue(maxsize=1)
    stop = threading.Event()
    worker = threading.Thread(
        target=_frame_worker,
        args=(
            "face", settings, output, camera_configs, targets, states,
            threading.Lock(), threading.Event(), stop,
            320, 114, "letterbox", "center",
        ),
        daemon=True,
    )
    worker.start()
    try:
        role = output.get(timeout=10)[0]
        assert role == "face"
        assert len(attempts) >= 2
        assert worker.is_alive()
    finally:
        stop.set()
        worker.join(timeout=10)
        assert not worker.is_alive()


def test_status_names_the_model_that_actually_runs(runtime_path: Path) -> None:
    """Путь в конфиге — намерение; служба могла подняться до правки.

    Вопрос «какая модель работает» до этого требовал пароля от файла настроек:
    в статусе не было ничего, кроме устройства OpenVINO.
    """
    from types import SimpleNamespace

    from src.recorder.app import _detector_device_status

    detector = SimpleNamespace(
        device="CPU",
        model_name="yolo11n-416.onnx",
        input_size=416,
        model_sha256="0123456789abcdef" * 4,
    )

    status = _detector_device_status(Settings(data_dir=runtime_path), detector)

    assert status["detector_model"] == "yolo11n-416.onnx"
    assert status["detector_input_size"] == "416"
    assert status["detector_sha256"] == "0123456789ab"


def test_status_without_a_detector_says_nothing_about_the_model(runtime_path: Path) -> None:
    from src.recorder.app import _detector_device_status

    status = _detector_device_status(Settings(data_dir=runtime_path))

    assert status["detector_model"] is None
    assert status["detector_sha256"] is None
