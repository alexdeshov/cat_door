import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient
from sqlalchemy import select

from src.common.camera_discovery import DiscoveredDevice
from src.common.camera_registry import camera_registry_path, load_registry, save_registry
from src.common.capture_mode import CaptureMode, capture_mode_path, load_capture_mode
from src.common.circuits import CIRCUIT_TITLES, circuits_path, load_circuits
from src.common.database import EvaluationRun, Event, create_session_factory
from src.common.gate_service import GateStatus
from src.common.settings import Settings
from src.common.telegram import TelegramError
from src.recorder.status import write_recorder_status
from src.web.app import create_app


def make_client(
    runtime_path: Path,
    monkeypatch,
    *,
    side: bool = False,
    timezone: str = "Europe/Moscow",
) -> TestClient:
    def fake_record(
        ffmpeg_path: str,
        rtsp_url: str,
        destination: Path,
        duration_seconds: int,
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"synthetic-video")

    opened = {"count": 0}

    def fake_camera_stream(ffmpeg_path: str, rtsp_url: str, fps: int, **options):
        # Кадр отдаётся только первой попытке: ответ живого превью не кончается
        # сам, пока камера отвечает, а TestClient не умеет обрывать бесконечный
        # ответ — суита ждала бы его вечно. Дальше поток замолкает и
        # закрывается по пределу тишины.
        opened["count"] += 1
        if opened["count"] == 1:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\ncamera\r\n"

    monkeypatch.setattr("src.web.app.record_clip", fake_record)
    # Живое превью больше не заканчивается вместе с камерой, поэтому суита не
    # должна ни поднимать настоящий ffmpeg (он ушёл бы к несуществующему
    # `camera.local` на все свои таймауты), ни спать по-настоящему в ожидании
    # её возвращения.
    monkeypatch.setattr("src.web.app.mjpeg_stream", fake_camera_stream)
    clock = {"now": 0.0}
    monkeypatch.setattr("src.web.live_stream._now", lambda: clock["now"])
    monkeypatch.setattr("src.web.live_stream._wait", lambda seconds: clock.update(
        now=clock["now"] + seconds,
    ))
    settings = Settings(
        camera_face_rtsp="rtsp://user:password@camera.local/media/video1",
        camera_side_rtsp=(
            "rtsp://user:password@side-camera.local/cam/realmonitor?subtype=0" if side else ""
        ),
        data_dir=runtime_path,
        database_path=runtime_path / "test.sqlite3",
        preview_silence_limit_seconds=10.0,
        display_timezone=timezone,
    )
    return TestClient(create_app(settings))


def test_capture_and_label_known_cat(runtime_path: Path, monkeypatch) -> None:
    client = make_client(runtime_path, monkeypatch)

    capture = client.post("/api/events", json={"duration_seconds": 5})
    assert capture.status_code == 201
    assert capture.json() == {"id": 1, "status": "ready"}
    assert (runtime_path / "events" / "00000001" / "face.mp4").is_file()

    label = client.post(
        "/api/events/1/label",
        json={
            "label": "known_cat",
            "cat_name": "Барсик",
            "training_target": "local",
        },
    )
    assert label.status_code == 200
    assert label.json()["cat_name"] == "Барсик"
    assert label.json()["training_target"] == "local"

    page = client.get("/")
    assert page.status_code == 200
    assert "Барсик" in page.text
    assert 'class="viewport"' in page.text
    assert 'class="pass"' in page.text
    # Оформление переехало в одну таблицу стилей: в разметке его больше нет.
    assert "<style>" not in page.text
    assert '<link rel="stylesheet" href="/static/app.css?v=' in page.text
    assert 'id="swap-cameras"' in page.text
    assert 'localStorage.setItem(cameraLayoutKey, swapped ? "swapped" : "default")' in page.text
    assert page.text.count('class="rotate-camera"') == 1
    assert 'camera.dataset.streamSrc = source;' in page.text
    assert "function streamUrl(role)" in page.text
    # Ответ живого превью всё-таки может кончиться — на перезапуске службы или
    # исчерпанном ожидании камеры. Замерший `<img>` об этом молчит, поэтому
    # страница переспрашивает поток сама.
    assert page.text.count("data-live-stream") == 2
    assert "img[data-live-stream]" in page.text
    assert "/stream/face.mjpeg?v=" in page.text
    assert 'cameraRotations[role] = (cameraRotations[role] + 90) % 360' in page.text
    assert 'localStorage.setItem(cameraRotationKey, JSON.stringify(cameraRotations))' in page.text
    assert 'const cameraRotationKey = "catdoor-camera-rotations-v3"' in page.text
    assert 'id="detector-status"' in page.text
    # Разметка сохраняется нажатием по переключателю, отдельной кнопки нет.
    assert 'class="label-choice" data-label="known_cat"' in page.text
    assert "save-label" not in page.text
    # Настройки детекции переехали на вкладку камер: там же список камер и их
    # роли, поэтому редактировать их в двух местах больше не нужно.
    assert 'id="detection-face-confirmation-frames"' not in page.text
    assert '<video preload="none"' in page.text
    assert "<video controls" not in page.text
    assert 'class="player-bar"' in page.text
    assert 'class="video-seek"' in page.text
    assert 'document.querySelectorAll(".event-player")' in page.text
    assert 'data-camera-role="face"' in page.text
    assert '<div class="event-recording" data-camera-role="side">' not in page.text
    assert "function applyEventMediaLayout(swapped)" in page.text
    assert "applyEventMediaLayout(swapped);" in page.text
    assert "function applyEventVideoRotations()" in page.text
    assert "(base + cameraRotations[role]) % 360" in page.text


def test_summary_is_above_cameras_and_counts_all_events(
    runtime_path: Path,
    monkeypatch,
) -> None:
    client = make_client(runtime_path, monkeypatch)
    client.post("/api/events", json={"duration_seconds": 5})
    client.post("/api/events", json={"duration_seconds": 5})
    client.post(
        "/api/events/1/label",
        json={
            "label": "known_cat",
            "cat_name": "Барсик",
            "training_target": "excluded",
        },
    )
    factory = create_session_factory(runtime_path / "test.sqlite3")
    with factory() as session:
        second = session.get(Event, 2)
        assert second is not None
        second.trigger_source = "automatic"
        session.commit()

    page = client.get("/")

    assert page.status_code == 200
    # Счётчики стоят выше камер: первый вопрос к странице — сколько накопилось
    # неразмеченного, а не как выглядит крыльцо прямо сейчас.
    assert page.text.index("ждут разметки") < page.text.index("<h2>Камеры</h2>")
    assert '<span class="tally-value">2</span>' in page.text
    assert '<span class="tally-key">проходов</span>' in page.text
    # Имена известных котов подсказываются в поле ввода, а не перечисляются
    # отдельным абзацем.
    assert '<option value="Барсик">' in page.text


def test_door_status_reports_unconfigured_without_controller(
    runtime_path: Path, monkeypatch
) -> None:
    """Без контроллера панель дверцы не рисуется, а API честно говорит почему."""
    client = make_client(runtime_path, monkeypatch)

    assert client.get("/api/door/status").json() == {"configured": False}
    assert 'id="door"' not in client.get("/").text


def test_door_status_exposes_monitor_snapshot(runtime_path: Path, monkeypatch) -> None:
    from src.web.door_monitor import DoorSnapshot

    client = make_client(runtime_path, monkeypatch)

    class FakeMonitor:
        def snapshot(self) -> DoorSnapshot:
            return DoorSnapshot(
                available=True,
                position="closed",
                position_stable_seconds=7.34,
                reed_closed=True,
                lock_relay_on=False,
            )

    client.app.state.door_monitor = FakeMonitor()

    payload = client.get("/api/door/status").json()
    assert payload == {
        "configured": True,
        "available": True,
        "position": "closed",
        "position_stable_seconds": 7.3,
        "reed_closed": True,
        "lock_relay_on": False,
        "error": None,
    }
    # Панель появляется вместе с настроенным монитором.
    assert 'id="door"' in client.get("/").text


def test_pending_and_labeled_passes_live_in_separate_sections(
    runtime_path: Path, monkeypatch
) -> None:
    """Реформа вкладки: разметка отдельно, опознанные отдельно."""
    client = make_client(runtime_path, monkeypatch)
    for _ in range(3):
        client.post("/api/events", json={"duration_seconds": 5})
    client.post(
        "/api/events/1/label",
        json={"label": "known_cat", "cat_name": "Барсик", "training_target": "macbook"},
    )
    client.post(
        "/api/events/2/label",
        json={"label": "foreign_cat", "training_target": "excluded"},
    )

    page = client.get("/")
    assert page.status_code == 200
    labeling = page.text[page.text.index('id="labeling"'):page.text.index('id="recognized"')]
    recognized = page.text[page.text.index('id="recognized"'):]
    # Неразмеченный — только в разметке; именованный — только в опознанных.
    assert 'data-event="3"' in labeling
    assert 'data-event="1"' not in labeling
    assert 'data-event="2"' not in labeling
    assert 'data-event="1"' in recognized
    # Чужой кот по умолчанию скрыт (раздел показывает своих с именами)…
    assert 'data-event="2"' not in recognized

    foreign = client.get("/?who=foreign")
    recognized = foreign.text[foreign.text.index('id="recognized"'):]
    # …и появляется под своей фишкой.
    assert 'data-event="2"' in recognized
    assert 'data-event="1"' not in recognized

    by_name = client.get("/?cat=Барсик")
    recognized = by_name.text[by_name.text.index('id="recognized"'):]
    assert 'data-event="1"' in recognized


def test_unknown_labeled_filter_falls_back_to_named(
    runtime_path: Path, monkeypatch
) -> None:
    client = make_client(runtime_path, monkeypatch)
    client.post("/api/events", json={"duration_seconds": 5})
    client.post(
        "/api/events/1/label",
        json={"label": "known_cat", "cat_name": "Барсик", "training_target": "macbook"},
    )

    page = client.get("/?who=nonsense&cat=Чужак")

    assert page.status_code == 200
    recognized = page.text[page.text.index('id="recognized"'):]
    assert 'data-event="1"' in recognized


def test_labeled_section_pagination_is_independent(
    runtime_path: Path, monkeypatch
) -> None:
    client = make_client(runtime_path, monkeypatch)
    for index in range(1, 13):
        client.post("/api/events", json={"duration_seconds": 5})
        client.post(
            f"/api/events/{index}/label",
            json={"label": "known_cat", "cat_name": "Барсик", "training_target": "macbook"},
        )

    first = client.get("/")
    assert "12 шт." in first.text
    assert 'href="/?lpage=2&amp;who=named#recognized"' in first.text

    second = client.get("/?lpage=2")
    recognized = second.text[second.text.index('id="recognized"'):]
    # Самые свежие первыми: на второй странице — самый старый проход.
    assert 'data-event="1"' in recognized


def test_known_cat_requires_name(runtime_path: Path, monkeypatch) -> None:
    client = make_client(runtime_path, monkeypatch)
    client.post("/api/events", json={"duration_seconds": 5})

    response = client.post(
        "/api/events/1/label",
        json={"label": "known_cat", "cat_name": None},
    )
    assert response.status_code == 422


def test_bulk_labels_same_cat_for_macbook(runtime_path: Path, monkeypatch) -> None:
    client = make_client(runtime_path, monkeypatch)
    client.post("/api/events", json={"duration_seconds": 5})
    client.post("/api/events", json={"duration_seconds": 5})

    response = client.post(
        "/api/events/bulk-label",
        json={
            "event_ids": [1, 2],
            "label": "known_cat",
            "cat_name": "Барсик",
            "training_target": "macbook",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"updated": 2, "cat_name": "Барсик"}
    factory = create_session_factory(runtime_path / "test.sqlite3")
    with factory() as session:
        events = session.scalars(select(Event).order_by(Event.id)).all()
        assert [(event.cat_name, event.training_target) for event in events] == [
            ("Барсик", "macbook"),
            ("Барсик", "macbook"),
        ]


def test_bulk_label_rejects_missing_event(runtime_path: Path, monkeypatch) -> None:
    client = make_client(runtime_path, monkeypatch)
    response = client.post(
        "/api/events/bulk-label",
        json={
            "event_ids": [999],
            "label": "known_cat",
            "cat_name": "Барсик",
            "training_target": "macbook",
        },
    )
    assert response.status_code == 404


def test_page_shows_detector_probe_metrics(runtime_path: Path, monkeypatch) -> None:
    write_recorder_status(
        runtime_path / "runtime" / "recorder-status.json",
        state="probe_completed",
        message="Dry-run проверка завершена",
        detector_connected=True,
        frames_processed={"face": 10, "side": 11},
        confidences={"face": 0.25, "side": 0.5},
        detector_device_requested="GPU",
        detector_device_actual="GPU.0",
    )
    client = make_client(runtime_path, monkeypatch, side=True)

    page = client.get("/")

    assert page.status_code == 200
    assert page.text.count('class="rotate-camera"') == 2
    # Показания детектора рисует строка состояния по данным API, а не сервер:
    # раньше та же строка печаталась дважды — в разметке и поверх неё скриптом.
    assert "Кадры face/side:" not in page.text
    assert 'id="detector-readouts"' in page.text

    status = client.get("/api/recorder/status")
    assert status.status_code == 200
    assert status.json()["state"] == "probe_completed"
    assert status.json()["stale"] is False


def test_recorder_status_reports_missing_heartbeat(runtime_path: Path, monkeypatch) -> None:
    client = make_client(runtime_path, monkeypatch)

    status = client.get("/api/recorder/status")

    assert status.status_code == 200
    assert status.json() == {
        "state": "offline",
        "message": "CatDoorRecorder не запущен",
        "stale": True,
    }


def test_starts_detached_offline_evaluation_on_cpu(
    runtime_path: Path,
    monkeypatch,
) -> None:
    models = runtime_path / "models"
    models.mkdir()
    detector = models / "detector.onnx"
    detector.write_bytes(b"detector")
    detector.with_suffix(".json").write_text(
        '{"architecture":"test","cat_class_id":17}',
        encoding="utf-8",
    )
    identity = models / "face.onnx"
    identity.write_bytes(b"identity")
    identity.with_suffix(".json").write_text(
        '{"labels":["Барсик"],"image_size":160}',
        encoding="utf-8",
    )
    launched: dict[str, object] = {}

    class FakeProcess:
        pid = 4321

    def fake_popen(command: list[str], **kwargs: object) -> FakeProcess:
        launched["command"] = command
        launched["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr("src.web.app.subprocess.Popen", fake_popen)
    settings = Settings(
        data_dir=runtime_path,
        database_path=runtime_path / "test.sqlite3",
        detector_model_path=detector,
        identity_face_model_path=identity,
    )
    client = TestClient(create_app(settings))

    response = client.post(
        "/api/evaluations",
        json={"labels": ["known_cat", "foreign_cat"]},
    )

    assert response.status_code == 202
    assert response.json() == {"run_id": 1, "pid": 4321, "state": "starting"}
    command = launched["command"]
    assert isinstance(command, list)
    assert "training.evaluation_job" in command
    assert "--face-model" in command
    # Оценка идёт на процессоре, даже когда recorder работает на видеоядре: у
    # видеоядра одна очередь, и общий с ним замер портит оба числа. Поэтому
    # здесь стоит собственная настройка, а не `detector_device`.
    assert command[command.index("--detector-device") + 1] == "CPU"
    assert "GPU" not in command
    factory = create_session_factory(runtime_path / "test.sqlite3")
    with factory() as session:
        run = session.get(EvaluationRun, 1)
        assert run is not None
        assert run.status == "starting"
        assert run.detector_model_version_id is not None
        assert run.face_model_version_id is not None
        assert run.side_model_version_id is None


def test_evaluation_does_not_inherit_the_recorder_device(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """Рабочее устройство и устройство оценки — разные настройки нарочно.

    Подстановка `detector_device` тихо переключила бы все будущие оценки в тот
    день, когда recorder переехал на видеоядро, и прежние отчёты стали бы
    несравнимы задним числом. Плюс оценка на том же видеоядре, где считает
    живой детектор, портит оба замера сразу.
    """
    models = runtime_path / "models"
    models.mkdir()
    detector = models / "detector.onnx"
    detector.write_bytes(b"detector")
    detector.with_suffix(".json").write_text(
        '{"architecture":"test","cat_class_id":17}',
        encoding="utf-8",
    )
    launched: dict[str, object] = {}

    class FakeProcess:
        pid = 4321

    monkeypatch.setattr(
        "src.web.app.subprocess.Popen",
        lambda command, **kwargs: (launched.update(command=command), FakeProcess())[1],
    )
    settings = Settings(
        data_dir=runtime_path,
        database_path=runtime_path / "test.sqlite3",
        detector_model_path=detector,
        detector_device="GPU",
    )
    client = TestClient(create_app(settings))

    response = client.post(
        "/api/evaluations",
        json={"labels": ["known_cat"], "run_kind": "detector"},
    )

    assert response.status_code == 202
    command = launched["command"]
    assert isinstance(command, list)
    assert command[command.index("--detector-device") + 1] == "CPU"
    assert "GPU" not in command


def test_starts_detector_only_evaluation_without_identity_models(
    runtime_path: Path,
    monkeypatch,
) -> None:
    models = runtime_path / "models"
    models.mkdir()
    detector = models / "candidate.onnx"
    detector.write_bytes(b"candidate")
    detector.with_suffix(".json").write_text(
        '{"architecture":"yolox_nano","cat_class_id":15}',
        encoding="utf-8",
    )
    launched: dict[str, object] = {}

    class FakeProcess:
        pid = 5432

    def fake_popen(command: list[str], **kwargs: object) -> FakeProcess:
        launched["command"] = command
        return FakeProcess()

    monkeypatch.setattr("src.web.app.subprocess.Popen", fake_popen)
    client = TestClient(
        create_app(
            Settings(
                data_dir=runtime_path,
                database_path=runtime_path / "test.sqlite3",
                detector_model_path=detector,
            )
        )
    )

    response = client.post(
        "/api/evaluations",
        json={
            "run_kind": "detector",
            "labels": ["known_cat", "foreign_cat", "not_visible"],
        },
    )

    assert response.status_code == 202
    command = launched["command"]
    assert isinstance(command, list)
    assert command[command.index("--run-kind") + 1] == "detector"
    assert "--face-model" not in command
    assert "--side-model" not in command
    factory = create_session_factory(runtime_path / "test.sqlite3")
    with factory() as session:
        run = session.get(EvaluationRun, 1)
        assert run is not None
        assert run.run_kind == "detector"
        assert run.detector_model_version_id is not None
        assert run.face_model_version_id is None


def test_stale_evaluation_is_failed_and_does_not_block_new_run(
    runtime_path: Path,
    monkeypatch,
) -> None:
    models = runtime_path / "models"
    models.mkdir()
    detector = models / "candidate.onnx"
    detector.write_bytes(b"candidate")
    detector.with_suffix(".json").write_text(
        '{"architecture":"test","cat_class_id":15}',
        encoding="utf-8",
    )

    class FakeProcess:
        pid = 5432

    monkeypatch.setattr(
        "src.web.app.subprocess.Popen",
        lambda *args, **kwargs: FakeProcess(),
    )
    settings = Settings(
        data_dir=runtime_path,
        database_path=runtime_path / "test.sqlite3",
        detector_model_path=detector,
    )
    client = TestClient(create_app(settings))
    old_heartbeat = datetime.now(UTC) - timedelta(minutes=11)
    factory = client.app.state.session_factory
    with factory() as session:
        session.add(
            EvaluationRun(
                created_at=old_heartbeat,
                run_kind="detector",
                filter_json="{}",
                status="running",
            )
        )
        session.commit()
    old_status = runtime_path / "evaluations" / "00000001" / "status.json"
    old_status.parent.mkdir(parents=True)
    old_status.write_text(
        json.dumps(
            {
                "updated_at": old_heartbeat.isoformat(),
                "state": "running",
            }
        ),
        encoding="utf-8",
    )

    response = client.post(
        "/api/evaluations",
        json={"run_kind": "detector", "labels": ["known_cat"]},
    )

    assert response.status_code == 202
    assert response.json()["run_id"] == 2
    with factory() as session:
        stale_run = session.get(EvaluationRun, 1)
        assert stale_run is not None
        assert stale_run.status == "failed"
        assert stale_run.completed_at is not None
        assert "Прерван без завершения" in str(stale_run.error_message)
    assert json.loads(old_status.read_text(encoding="utf-8"))["state"] == "failed"


def test_evaluation_rejects_path_outside_models(
    runtime_path: Path,
    monkeypatch,
) -> None:
    client = make_client(runtime_path, monkeypatch)

    response = client.post(
        "/api/evaluations",
        json={"detector_model": "../secret.onnx"},
    )

    assert response.status_code == 422


def test_metrics_page_shows_online_detection_and_evaluation_sections(
    runtime_path: Path,
    monkeypatch,
) -> None:
    client = make_client(runtime_path, monkeypatch)

    page = client.get("/metrics")

    assert page.status_code == 200
    assert "Последние 7 дней" in page.text
    assert "Нет размеченных событий с online-предсказаниями" in page.text
    assert "Детекция по камерам" in page.text
    assert "Новая офлайн-оценка" in page.text
    assert 'option value="detector"' in page.text
    assert "Сравнить два завершённых" in page.text
    assert client.get("/").text.count('href="/metrics"') == 1


def test_completed_evaluation_report_and_comparison_are_rendered(
    runtime_path: Path,
    monkeypatch,
) -> None:
    client = make_client(runtime_path, monkeypatch)
    factory = create_session_factory(runtime_path / "test.sqlite3")
    reports = [
        {
            "events_processed": 10,
            "identity_events": 8,
            "clean_events": 8,
            "training_seen_events": 2,
            "metrics": {"combined": None},
            "night_metrics": {"combined": None},
            "detection": {},
            "clean_metrics": {
                "combined": {
                    "false_accept_rate": far,
                    "false_reject_rate": 0.2,
                    "misidentified_known": 1,
                    "uncertain_rate": 0.1,
                    "accuracy": 0.8,
                    "mean_inference_ms": 12.0,
                    "threshold_sweep": [],
                    "confusion": {},
                }
            },
        }
        for far in (0.2, 0.1)
    ]
    with factory() as session:
        for report in reports:
            session.add(
                EvaluationRun(
                    run_kind="identity",
                    filter_json="{}",
                    status="completed",
                    metrics_json=json.dumps(report),
                )
            )
        session.commit()

    report_page = client.get("/metrics/runs/1")
    comparison = client.get("/metrics/compare?left=1&right=2")

    assert report_page.status_code == 200
    assert "Чистые события" in report_page.text
    assert "Охват" in report_page.text
    assert comparison.status_code == 200
    assert "Сравнивается combined-метрика" in comparison.text
    assert "-0.1000" in comparison.text


def test_detector_evaluation_report_and_comparison_are_rendered(
    runtime_path: Path,
    monkeypatch,
) -> None:
    client = make_client(runtime_path, monkeypatch)
    factory = create_session_factory(runtime_path / "test.sqlite3")

    def report(miss_rate: float) -> dict[str, object]:
        values = {
            "events": 4,
            "missing_media": 0,
            "cat_events": 2,
            "misses": int(miss_rate * 2),
            "miss_rate": miss_rate,
            "not_visible_events": 2,
            "false_triggers": 0,
            "false_trigger_rate": 0.0,
            "threshold": 0.3,
            "median_inference_ms": 50.0,
            "p95_inference_ms": 60.0,
            "confidence_distribution": {
                group: {
                    "samples": 2,
                    "min": 0.1,
                    "p25": 0.2,
                    "median": 0.3,
                    "p75": 0.4,
                    "p95": 0.5,
                    "max": 0.6,
                }
                for group in ("all", "cat", "not_visible")
            },
        }
        return {
            "run_kind": "detector",
            "events_processed": 4,
            "cat_events": 2,
            "not_visible_events": 2,
            "clean_events": 4,
            "night_events": 1,
            "sections": {
                section: {"face": values, "side": values}
                for section in ("all", "clean", "night")
            },
            "threshold_sweep": {"face": [], "side": []},
        }

    with factory() as session:
        for miss_rate in (0.5, 0.0):
            session.add(
                EvaluationRun(
                    run_kind="detector",
                    filter_json="{}",
                    status="completed",
                    metrics_json=json.dumps(report(miss_rate)),
                )
            )
        session.commit()

    report_page = client.get("/metrics/runs/1")
    comparison = client.get("/metrics/compare?left=1&right=2")

    assert report_page.status_code == 200
    assert "Сравнение детектора #1" in report_page.text
    assert "False trigger" in report_page.text
    assert comparison.status_code == 200
    assert "Сравниваются miss rate" in comparison.text
    assert "face: Miss rate" in comparison.text
    assert "-0.5000" in comparison.text


def test_detection_settings_can_be_updated(runtime_path: Path, monkeypatch) -> None:
    client = make_client(runtime_path, monkeypatch)
    payload = {
        "face": {
            "confidence_threshold": 0.30,
            "box_min_confidence": 0.15,
            "confirmation_frames": 2,
            "window_frames": 5,
            "sample_fps": 8.0,
            "roi": {"x": 0.1, "y": 0.2, "width": 0.7, "height": 0.6},
        },
        "side": {
            "confidence_threshold": 0.50,
            "box_min_confidence": 0.15,
            "confirmation_frames": 2,
            "window_frames": 4,
            "sample_fps": 3.0,
            "roi": None,
        },
        "candidate_timeout_seconds": 2.0,
        "absence_seconds": 2.5,
        "tail_seconds": 8.0,
        "cooldown_seconds": 4.0,
        "max_event_seconds": 45.0,
    }

    response = client.put("/api/detection/settings", json=payload)

    assert response.status_code == 200
    assert response.json() == payload
    assert client.get("/api/detection/settings").json() == payload
    page = client.get("/cameras")
    assert 'id="detection-face-sample-fps"' in page.text
    assert 'id="detection-face-roi-enabled"' in page.text
    assert 'value="8.0"' in page.text
    assert 'id="detection-max-event-seconds"' in page.text


def test_detection_settings_reject_invalid_window(runtime_path: Path, monkeypatch) -> None:
    client = make_client(runtime_path, monkeypatch)

    response = client.put(
        "/api/detection/settings",
        json={
            "face": {
                "confidence_threshold": 0.30,
                "box_min_confidence": 0.15,
                "confirmation_frames": 5,
                "window_frames": 2,
                "sample_fps": 8.0,
                "roi": None,
            },
            "side": {
                "confidence_threshold": 0.55,
                "box_min_confidence": 0.20,
                "confirmation_frames": 2,
                "window_frames": 5,
                "sample_fps": 3.0,
                "roi": None,
            },
            "candidate_timeout_seconds": 3.0,
            "absence_seconds": 3.0,
            "tail_seconds": 10.0,
            "cooldown_seconds": 5.0,
        },
    )

    assert response.status_code == 422


def test_rejects_unknown_label(runtime_path: Path, monkeypatch) -> None:
    client = make_client(runtime_path, monkeypatch)
    client.post("/api/events", json={"duration_seconds": 5})

    response = client.post(
        "/api/events/1/label",
        json={"label": "not-a-label", "cat_name": None},
    )
    assert response.status_code == 422


def test_training_target_requires_usable_label(runtime_path: Path, monkeypatch) -> None:
    client = make_client(runtime_path, monkeypatch)
    client.post("/api/events", json={"duration_seconds": 5})

    response = client.post(
        "/api/events/1/label",
        json={
            "label": "not_visible",
            "cat_name": None,
            "training_target": "macbook",
        },
    )

    assert response.status_code == 422


def test_rejects_unknown_training_target(runtime_path: Path, monkeypatch) -> None:
    client = make_client(runtime_path, monkeypatch)
    client.post("/api/events", json={"duration_seconds": 5})

    response = client.post(
        "/api/events/1/label",
        json={
            "label": "foreign_cat",
            "cat_name": None,
            "training_target": "cloud",
        },
    )

    assert response.status_code == 422


def test_macbook_export_download(runtime_path: Path, monkeypatch) -> None:
    client = make_client(runtime_path, monkeypatch)
    archive = runtime_path / "exports" / "dataset.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"zip")
    seen_scopes: list[str] = []

    def fake_export(*_args, scope="macbook"):
        seen_scopes.append(scope)
        return archive

    monkeypatch.setattr("src.web.app.create_macbook_export", fake_export)

    response = client.get("/api/training/macbook/export")

    assert response.status_code == 200
    assert response.content == b"zip"
    assert seen_scopes == ["macbook"]

    assert client.get("/api/training/macbook/export?scope=all").status_code == 200
    assert seen_scopes == ["macbook", "all"]

    # Опечатка в scope — отказ, а не молчаливый экспорт не того набора.
    assert client.get("/api/training/macbook/export?scope=everything").status_code == 422


def test_start_local_training_is_detached(runtime_path: Path, monkeypatch) -> None:
    client = make_client(runtime_path, monkeypatch)
    page = client.get("/")
    assert 'id="training-report-state"' in page.text
    assert 'id="training-progress"' in page.text
    assert 'id="training-report-warnings"' in page.text

    class FakeProcess:
        pid = 4321

    monkeypatch.setattr(
        "src.web.app.subprocess.Popen",
        lambda *args, **kwargs: FakeProcess(),
    )

    response = client.post("/api/training/local/start")

    assert response.status_code == 202
    assert response.json()["pid"] == 4321
    status = client.get("/api/training/local/status")
    assert status.json()["state"] == "starting"


def test_stale_local_training_does_not_block_new_run(
    runtime_path: Path,
    monkeypatch,
) -> None:
    client = make_client(runtime_path, monkeypatch)
    stale_status = (
        runtime_path
        / "training"
        / "local"
        / "20000101-000000-stale"
        / "status.json"
    )
    stale_status.parent.mkdir(parents=True)
    stale_status.write_text(
        json.dumps(
            {
                "updated_at": (
                    datetime.now(UTC) - timedelta(minutes=11)
                ).isoformat(),
                "state": "training",
            }
        ),
        encoding="utf-8",
    )

    class FakeProcess:
        pid = 4321

    monkeypatch.setattr(
        "src.web.app.subprocess.Popen",
        lambda *args, **kwargs: FakeProcess(),
    )

    response = client.post("/api/training/local/start")

    assert response.status_code == 202
    stale_payload = json.loads(stale_status.read_text(encoding="utf-8"))
    assert stale_payload["state"] == "failed"
    assert "Прерван без завершения" in stale_payload["message"]


def test_media_path_cannot_escape_data_root(runtime_path: Path, monkeypatch) -> None:
    client = make_client(runtime_path, monkeypatch)
    client.post("/api/events", json={"duration_seconds": 5})
    database = client.app.state.session_factory
    with database() as session:
        event = session.get(Event, 1)
        assert event is not None
        event.media_path = "../outside.mp4"
        session.commit()

    response = client.get("/media/1/face")
    assert response.status_code == 404


def test_delete_event_removes_database_row_and_media(runtime_path: Path, monkeypatch) -> None:
    client = make_client(runtime_path, monkeypatch)
    client.post("/api/events", json={"duration_seconds": 5})
    media = runtime_path / "events" / "00000001" / "face.mp4"
    assert media.is_file()

    response = client.delete("/api/events/1")

    assert response.status_code == 204
    assert not media.parent.exists()
    assert client.get("/media/1/face").status_code == 404
    assert "Событие #1" not in client.get("/").text


def test_delete_missing_event_returns_not_found(runtime_path: Path, monkeypatch) -> None:
    client = make_client(runtime_path, monkeypatch)

    response = client.delete("/api/events/999")

    assert response.status_code == 404


def test_capture_two_cameras_in_parallel(runtime_path: Path, monkeypatch) -> None:
    client = make_client(runtime_path, monkeypatch, side=True)

    response = client.post("/api/events", json={"duration_seconds": 5})

    assert response.json()["status"] == "ready"
    assert (runtime_path / "events" / "00000001" / "face.mp4").is_file()
    assert (runtime_path / "events" / "00000001" / "side.mp4").is_file()
    assert client.get("/media/1/face").status_code == 200
    assert client.get("/media/1/side").status_code == 200
    page = client.get("/").text
    assert 'data-camera-role="face"' in page
    assert 'data-camera-role="side"' in page
    assert 'swapped ? [side, face] : [face, side]' in page


def _collect_first_frame(response) -> bytes:
    for chunk in response.iter_bytes():
        return chunk
    return b""


def test_stream_falls_back_to_the_camera_when_the_recorder_is_idle(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """Aiming a camera must not require a detector model on the machine.

    Only the detector pipeline writes preview files, so without this fallback
    the admin page shows nothing at all until OpenVINO and a model are in place.
    """
    used: dict[str, object] = {}

    def fake_camera_stream(ffmpeg_path: str, rtsp_url: str, fps: int, **options):
        used["ffmpeg_path"] = ffmpeg_path
        used["rtsp_url"] = rtsp_url
        used["rotate_180"] = options.get("rotate_180")
        if "opened" not in used:
            used["opened"] = True
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\ncamera\r\n"

    # Подмена после клиента: make_client ставит свою заглушку камеры, и патч,
    # наложенный раньше, она бы перекрыла.
    client = make_client(runtime_path, monkeypatch)
    monkeypatch.setattr("src.web.app.mjpeg_stream", fake_camera_stream)

    with client.stream("GET", "/stream/face.mjpeg") as response:
        assert response.status_code == 200
        assert b"camera" in _collect_first_frame(response)

    # Порт проставлен явно: инвентарь хранит адрес разобранным и собирает его
    # обратно, а камера на нестандартном порту иначе потеряла бы его.
    assert used["rtsp_url"] == "rtsp://user:password@camera.local:554/media/video1"
    # The role keeps its own orientation flag, which defaults to a rotated camera.
    assert used["rotate_180"] is True


def test_stream_prefers_annotated_frames_while_the_recorder_publishes(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """Detector boxes are more useful than a raw picture, so they win."""

    def fake_camera_stream(ffmpeg_path: str, rtsp_url: str, fps: int, **options):
        raise AssertionError("the camera must not be opened twice")

    published = {"count": 0}

    def fake_published_stream(path: Path, fps: int, idle_timeout_seconds: float = 20.0):
        published["count"] += 1
        if published["count"] == 1:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\nannotated\r\n"

    client = make_client(runtime_path, monkeypatch)
    monkeypatch.setattr("src.web.app.mjpeg_stream", fake_camera_stream)
    monkeypatch.setattr("src.web.app.local_mjpeg_stream", fake_published_stream)
    monkeypatch.setattr("src.web.app.preview_is_fresh", lambda path: True)

    with client.stream("GET", "/stream/face.mjpeg") as response:
        assert b"annotated" in _collect_first_frame(response)


def test_stream_returns_to_detector_boxes_without_a_page_reload(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """Рамки детекции пропадали именно так.

    Поток с живой камеры не кончается сам, поэтому выбор, сделанный в момент
    протухшего превью (перезапуск recorder, загрузка моделей, обрыв питания),
    оставался навсегда: картинка живая, а рамок нет до перезагрузки страницы.
    """
    published = {"fresh": False, "opened": 0}
    clock = {"now": 0.0}

    def tick(seconds: float = 0.5) -> None:
        clock["now"] += seconds

    def frame(payload: bytes) -> bytes:
        return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"

    def fake_camera_stream(ffmpeg_path: str, rtsp_url: str, fps: int, **options):
        # Живая камера кадры не прекращает — оборвать её может только `until`.
        # На третьем кадре recorder оживает и снова публикует превью.
        sent = 0
        while True:
            tick()
            sent += 1
            if sent == 3:
                published["fresh"] = True
            yield frame(b"raw")

    def fake_published_stream(path: Path, fps: int, **options):
        # Кадры отдаёт только первое открытие: ответ обязан закончиться сам,
        # иначе его некому оборвать и тест повиснет на закрытии.
        published["opened"] += 1
        if published["opened"] > 1:
            return
        for _ in range(3):
            tick()
            yield frame(b"annotated")

    client = make_client(runtime_path, monkeypatch)
    monkeypatch.setattr("src.web.app.mjpeg_stream", fake_camera_stream)
    monkeypatch.setattr("src.web.app.local_mjpeg_stream", fake_published_stream)
    monkeypatch.setattr("src.web.app.preview_is_fresh", lambda path: published["fresh"])
    monkeypatch.setattr("src.web.live_stream._now", lambda: clock["now"])
    monkeypatch.setattr("src.web.live_stream._wait", tick)

    with client.stream("GET", "/stream/face.mjpeg") as response:
        seen = b"".join(response.iter_bytes())

    assert b"raw" in seen
    # Главное: рамки вернулись в тот же ответ, без перезагрузки страницы.
    assert b"annotated" in seen
    assert seen.index(b"raw") < seen.index(b"annotated")


def test_stream_reopens_the_camera_at_the_address_it_came_back_on(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """Обрыв питания и возвращение по новому адресу — один и тот же случай.

    Камера, поднявшаяся с другим адресом от DHCP, до этого была потеряна для
    открытой вкладки навсегда: адрес выбирался один раз в начале ответа.
    """
    addresses: list[str] = []
    registry_file = camera_registry_path(runtime_path)

    def move_the_camera() -> None:
        registry = load_registry(registry_file)
        moved = registry.get("face").model_copy(update={"host": "10.0.0.77"})
        save_registry(registry_file, registry.with_camera(moved))

    def fake_camera_stream(ffmpeg_path: str, rtsp_url: str, fps: int, **options):
        addresses.append(rtsp_url)
        # Кадр отдают только две попытки: до переезда камеры и после него.
        # Дальше поток молчит и закрывается по пределу тишины — иначе ответ,
        # который по замыслу не кончается, некому было бы оборвать.
        if len(addresses) == 1:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\ncamera\r\n"
            # Камера вернулась с другим адресом от DHCP, пока вкладка открыта.
            move_the_camera()
        elif len(addresses) == 2:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\ncamera\r\n"

    client = make_client(runtime_path, monkeypatch)
    monkeypatch.setattr("src.web.app.mjpeg_stream", fake_camera_stream)

    with client.stream("GET", "/stream/face.mjpeg") as response:
        assert response.status_code == 200

    assert "camera.local" in addresses[0]
    assert "10.0.0.77" in addresses[1]


def test_cameras_page_lists_every_camera_including_the_ones_without_an_address(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """Камера без адреса — то, что администратору и нужно увидеть.

    Пропадание такой камеры из списка было прежним поведением и означало, что
    после перестановки её негде настроить.
    """
    client = make_client(runtime_path, monkeypatch, side=False)

    page = client.get("/cameras")

    assert page.status_code == 200
    assert "/stream/camera/face.mjpeg" in page.text
    assert 'data-camera-id="side"' in page.text
    assert "Адрес не задан" in page.text
    assert 'id="detection-side-sample-fps"' in page.text


def test_unknown_camera_is_not_confused_with_an_unconfigured_one(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """404 means nobody declared it; 503 means it has no address yet."""
    client = make_client(runtime_path, monkeypatch, side=False)

    assert client.get("/stream/nothing-like-this.mjpeg").status_code == 404
    assert client.get("/stream/side.mjpeg").status_code == 503
    # Выключенная камера тоже не отдаёт поток: recorder её не берёт, и живое
    # изображение обещало бы запись, которой не будет.
    client.put("/api/cameras/face", json={"enabled": False})
    assert client.get("/stream/face.mjpeg").status_code == 503


def test_camera_list_never_returns_the_password(runtime_path: Path, monkeypatch) -> None:
    client = make_client(runtime_path, monkeypatch, side=True)

    cameras = client.get("/api/cameras").json()["cameras"]

    assert [camera["id"] for camera in cameras] == ["face", "side"]
    assert all("password" not in camera for camera in cameras)
    assert cameras[0]["has_password"] is True
    assert cameras[0]["username"] == "user"
    # На странице есть поле «Пароль», но не сам пароль: его значение не
    # выходит за пределы data/runtime/cameras.json.
    assert "user:password@" not in client.get("/cameras").text


def test_editing_a_camera_keeps_the_stored_password(runtime_path: Path, monkeypatch) -> None:
    """Пустое поле пароля означает «оставить прежний».

    Форма не может показать сохранённый пароль, поэтому иначе правка соседнего
    поля стирала бы доступ к камере.
    """
    client = make_client(runtime_path, monkeypatch)

    response = client.put(
        "/api/cameras/face",
        json={"host": "192.168.1.120", "port": 554, "stream_path": "/one", "username": "user"},
    )

    assert response.status_code == 200
    assert response.json()["has_password"] is True
    registry = load_registry(camera_registry_path(runtime_path))
    assert registry.get("face").password == "password"
    assert registry.get("face").host == "192.168.1.120"


def test_a_camera_survives_getting_a_new_address(runtime_path: Path, monkeypatch) -> None:
    """Горячая замена: адрес меняется, камера и её роль остаются на месте."""
    client = make_client(runtime_path, monkeypatch)

    client.put(
        "/api/cameras/face",
        json={"rtsp_url": "rtsp://admin:new-secret@192.168.1.131:554/Streaming/Channels/101"},
    )

    camera = load_registry(camera_registry_path(runtime_path)).by_role("face")
    assert camera.host == "192.168.1.131"
    assert camera.password == "new-secret"
    assert camera.username == "admin"
    # Поток теперь переживает пропажу камеры и не кончается сам, поэтому его
    # читают потоком: обычный GET ждал бы возвращения камеры целиком.
    with client.stream("GET", "/stream/face.mjpeg") as response:
        assert response.status_code != 404


def test_roles_can_be_swapped_without_touching_addresses(
    runtime_path: Path,
    monkeypatch,
) -> None:
    client = make_client(runtime_path, monkeypatch, side=True)
    before = load_registry(camera_registry_path(runtime_path))

    response = client.post("/api/cameras/swap-roles")

    assert response.status_code == 200
    after = load_registry(camera_registry_path(runtime_path))
    assert after.by_role("face").id == "side"
    assert after.by_role("side").id == "face"
    assert after.get("face").host == before.get("face").host
    assert after.get("side").host == before.get("side").host


def test_swapped_roles_do_not_merge_two_cameras_into_one_picture(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """Карточка камеры показывает железку, а не роль.

    Идентификатор камеры вполне может совпасть с именем роли, а `resolve`
    сначала ищет по роли. После перестановки ролей обе карточки страницы камер
    сходились на одной камере: `/stream/face.mjpeg` отдавал ту, что сейчас на
    роли `face`, а не ту, у которой такой идентификатор.
    """
    addresses: list[str] = []

    def fake_camera_stream(ffmpeg_path: str, rtsp_url: str, fps: int, **options):
        addresses.append(rtsp_url)
        return iter(())

    client = make_client(runtime_path, monkeypatch, side=True)
    monkeypatch.setattr("src.web.app.mjpeg_stream", fake_camera_stream)
    client.post("/api/cameras/swap-roles")

    page = client.get("/cameras")
    opened: dict[str, set[str]] = {}
    for camera_id in ("face", "side"):
        addresses.clear()
        with client.stream("GET", f"/stream/camera/{camera_id}.mjpeg") as response:
            assert response.status_code == 200
        opened[camera_id] = set(addresses)

    assert "/stream/camera/face.mjpeg" in page.text
    # Роль `face` теперь у камеры с идентификатором `side`, и именно поэтому
    # проверка адресов: перепутанные карточки иначе выглядели бы исправными.
    assert opened["face"] and all("@camera.local:" in address for address in opened["face"])
    assert opened["side"] and all("@side-camera.local:" in address for address in opened["side"])


def test_assigning_a_taken_role_frees_the_other_camera(runtime_path: Path, monkeypatch) -> None:
    client = make_client(runtime_path, monkeypatch, side=True)

    response = client.put("/api/cameras/side", json={"host": "10.0.0.9", "role": "face"})

    assert response.status_code == 200
    registry = load_registry(camera_registry_path(runtime_path))
    assert registry.by_role("face").id == "side"
    assert registry.get("face").role is None


def test_camera_can_be_added_and_deleted(runtime_path: Path, monkeypatch) -> None:
    client = make_client(runtime_path, monkeypatch)

    created = client.post(
        "/api/cameras",
        json={
            "title": "HiWatch у столба",
            "rtsp_url": "rtsp://admin:secret@10.0.0.15:554/Streaming/Channels/101",
        },
    )

    assert created.status_code == 201
    # Кириллица из названия в идентификатор не переносится, поэтому вторая
    # такая камера получает следующий свободный номер, а не отказ.
    assert created.json()["id"] == "hiwatch"
    assert created.json()["has_password"] is True
    second = client.post("/api/cameras", json={"title": "HiWatch у водостока"})
    assert second.json()["id"] == "hiwatch-2"
    assert client.post("/api/cameras", json={"id": "hiwatch"}).status_code == 409
    assert client.delete("/api/cameras/hiwatch").status_code == 204
    assert client.delete("/api/cameras/hiwatch").status_code == 404


def test_invalid_camera_fields_do_not_echo_the_password(
    runtime_path: Path,
    monkeypatch,
) -> None:
    client = make_client(runtime_path, monkeypatch)

    response = client.put(
        "/api/cameras/face",
        json={"host": "10.0.0.5", "mac": "не-mac", "password": "top-secret"},
    )

    assert response.status_code == 422
    assert "MAC-адрес" in response.json()["detail"]
    assert "top-secret" not in response.text


def test_stream_probe_reports_the_reason(runtime_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "src.web.app.probe_stream",
        lambda ffmpeg_path, url, **options: "401 Unauthorized",
    )
    client = make_client(runtime_path, monkeypatch)

    response = client.post("/api/cameras/face/test")

    assert response.status_code == 200
    assert response.json() == {"ok": False, "reason": "401 Unauthorized"}
    assert client.post("/api/cameras/side/test").status_code == 422


def test_discovery_recognises_a_moved_camera_by_mac(runtime_path: Path, monkeypatch) -> None:
    """Переставленная камера должна узнаваться, а не заводиться заново."""
    client = make_client(runtime_path, monkeypatch)
    client.put("/api/cameras/face", json={"mac": "44:47:cc:11:22:33"})
    monkeypatch.setattr(
        "src.web.app.discover_devices",
        lambda **options: [
            DiscoveredDevice(
                host="192.168.1.131",
                mac="44:47:cc:11:22:33",
                name="HiWatch",
                sources=("dhcp", "onvif"),
            ),
            DiscoveredDevice(host="192.168.1.140", sources=("scan",)),
        ],
    )
    monkeypatch.setattr("src.web.app.port_is_open", lambda host, port, **options: True)
    monkeypatch.setattr(
        "src.web.app.probe_stream",
        lambda ffmpeg_path, url, **options: (
            None if url.endswith("/media/video1") else "404 Not Found"
        ),
    )

    response = client.post("/api/cameras/discover", json={"password": "secret"})

    assert response.status_code == 200
    found = {device["host"]: device for device in response.json()["devices"]}
    moved = found["192.168.1.131"]
    assert moved["matched_camera_id"] == "face"
    assert moved["moved"] is True
    assert moved["stream_path"] == "/media/video1"
    assert moved["sources"] == ["dhcp", "onvif"]
    assert found["192.168.1.140"]["matched_camera_id"] is None
    assert "secret" not in response.text


def test_discovery_reports_a_wrong_password_instead_of_a_silent_miss(
    runtime_path: Path,
    monkeypatch,
) -> None:
    client = make_client(runtime_path, monkeypatch)
    monkeypatch.setattr(
        "src.web.app.discover_devices",
        lambda **options: [DiscoveredDevice(host="192.168.1.131", sources=("scan",))],
    )
    monkeypatch.setattr("src.web.app.port_is_open", lambda host, port, **options: True)
    monkeypatch.setattr(
        "src.web.app.probe_stream",
        lambda ffmpeg_path, url, **options: "401 Unauthorized",
    )

    device = client.post("/api/cameras/discover", json={}).json()["devices"][0]

    assert device["needs_credentials"] is True
    assert device["reason"] == "401 Unauthorized"
    assert device["stream_path"] == ""


def test_discovery_does_not_probe_devices_without_an_rtsp_port(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """В сети камер живёт и сам сервер, и коммутатор — их незачем опрашивать."""
    probed: list[str] = []
    client = make_client(runtime_path, monkeypatch)
    monkeypatch.setattr(
        "src.web.app.discover_devices",
        lambda **options: [DiscoveredDevice(host="192.168.1.1", sources=("arp",))],
    )
    monkeypatch.setattr("src.web.app.port_is_open", lambda host, port, **options: False)
    monkeypatch.setattr(
        "src.web.app.probe_stream",
        lambda ffmpeg_path, url, **options: probed.append(url),
    )

    device = client.post("/api/cameras/discover", json={}).json()["devices"][0]

    assert device["rtsp_open"] is False
    assert probed == []


def test_found_camera_is_applied_through_the_ordinary_edit(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """Обнаружение только предлагает: адрес записывает обычное сохранение."""
    client = make_client(runtime_path, monkeypatch)

    client.put(
        "/api/cameras/face",
        json={
            "host": "192.168.1.131",
            "stream_path": "/media/video1",
            "mac": "44:47:cc:11:22:33",
        },
    )

    camera = load_registry(camera_registry_path(runtime_path)).get("face")
    assert camera.host == "192.168.1.131"
    assert camera.mac == "44:47:cc:11:22:33"
    assert camera.password == "password"


def test_discovery_form_is_on_the_camera_tab(runtime_path: Path, monkeypatch) -> None:
    client = make_client(runtime_path, monkeypatch)

    page = client.get("/cameras")

    assert 'id="discover"' in page.text
    assert 'id="discovery-form"' in page.text
    assert '[data-field="password"]' in page.text
    assert "Добавить как новую" in page.text


def test_capture_mode_is_switched_from_the_admin(runtime_path: Path, monkeypatch) -> None:
    """Включение сбора больше не требует SSH и перезапуска службы."""
    client = make_client(runtime_path, monkeypatch)

    assert client.get("/api/capture-mode").json() == {
        "state": "off",
        "enabled": False,
        "record_events": False,
    }

    response = client.put("/api/capture-mode", json={"state": "record"})

    assert response.status_code == 200
    assert response.json() == {"state": "record", "enabled": True, "record_events": True}
    assert load_capture_mode(
        capture_mode_path(runtime_path),
        CaptureMode(),
    ).state == "record"
    assert client.get("/api/capture-mode").json()["state"] == "record"


def test_capture_mode_rejects_states_that_do_not_exist(runtime_path: Path, monkeypatch) -> None:
    client = make_client(runtime_path, monkeypatch)

    assert client.put("/api/capture-mode", json={"state": "куда-то ещё"}).status_code == 422


def test_switch_shows_the_current_mode(runtime_path: Path, monkeypatch) -> None:
    client = make_client(runtime_path, monkeypatch)
    client.put("/api/capture-mode", json={"state": "observe"})

    page = client.get("/")

    assert 'data-state="observe"' in page.text
    assert page.text.count('class="capture-choice"') == 3
    # Нажатое положение приходит с сервера, а не выставляется скриптом после
    # загрузки: иначе переключатель на секунду показывал бы чужое состояние.
    assert 'data-state="observe"\n        aria-pressed="true"' in page.text
    # Режим сбора имеет смысл, только когда контур разрешён на уровне узла.
    client.put("/api/circuits", json={"circuit": "gate", "enabled": True})
    stopped = client.get("/")

    assert stopped.text.count('class="capture-choice" disabled') == 3
    assert "Контур дверцы выключен на уровне узла" in stopped.text


def test_discovery_shows_one_row_for_a_camera_on_two_addresses(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """Камера после сброса отвечает и по аренде, и по заводскому адресу."""
    client = make_client(runtime_path, monkeypatch)
    monkeypatch.setattr(
        "src.web.app.discover_devices",
        lambda **options: [
            DiscoveredDevice(
                host="192.168.1.108",
                mac="9c:14:63:41:de:15",
                sources=("arp", "scan"),
                other_hosts=("192.168.1.106",),
            )
        ],
    )
    monkeypatch.setattr("src.web.app.port_is_open", lambda host, port, **options: True)
    monkeypatch.setattr("src.web.app.probe_stream", lambda ffmpeg_path, url, **options: None)

    devices = client.post("/api/cameras/discover", json={}).json()["devices"]

    assert len(devices) == 1
    assert devices[0]["host"] == "192.168.1.108"
    assert devices[0]["other_hosts"] == ["192.168.1.106"]


def test_gate_tab_is_honest_about_not_being_deployed(runtime_path: Path, monkeypatch) -> None:
    client = make_client(runtime_path, monkeypatch)

    monkeypatch.setattr(
        "src.web.app.gate_status",
        lambda unit, api_url="": GateStatus("absent", "служба не установлена на узле"),
    )

    page = client.get("/gate")

    assert page.status_code == 200
    # Не развёрнутый контур страница называет прямо, а не делает вид, что его
    # можно включить.
    assert "Служба Frigate на узле не установлена" in page.text
    assert 'data-circuit="gate"' in page.text
    # Верхний уровень навигации — контуры, второй появляется только у котов.
    assert 'href="/gate"' in page.text
    assert 'class="subnav"' not in page.text
    assert 'class="subnav"' in client.get("/").text


def test_node_switch_gives_the_processor_to_one_circuit(runtime_path: Path, monkeypatch) -> None:
    client = make_client(runtime_path, monkeypatch)

    assert client.get("/api/circuits").json()["active"] == ["cats"]

    response = client.put("/api/circuits", json={"active": "gate"})

    assert response.status_code == 200
    assert response.json()["active"] == ["gate"]
    assert load_circuits(circuits_path(runtime_path)).is_active("cats") is False


def test_stopped_cat_circuit_keeps_its_mode(runtime_path: Path, monkeypatch) -> None:
    """Выключение с уровня узла не стирает выбранный режим наблюдения."""
    client = make_client(runtime_path, monkeypatch)
    client.put("/api/capture-mode", json={"state": "record"})

    client.put("/api/circuits", json={"active": "gate"})

    assert client.get("/api/capture-mode").json()["state"] == "record"


def test_gate_recording_defaults_to_events_and_is_switchable(
    runtime_path: Path,
    monkeypatch,
) -> None:
    client = make_client(runtime_path, monkeypatch)

    assert client.get("/api/circuits").json()["gate_recording"] == "events"

    response = client.put("/api/circuits", json={"gate_recording": "continuous"})

    assert response.status_code == 200
    assert response.json()["gate_recording"] == "continuous"
    # Переключение записи не трогает, какому контуру отдан узел.
    assert response.json()["active"] == ["cats"]


def test_cameras_are_grouped_by_circuit(runtime_path: Path, monkeypatch) -> None:
    """Камера калитки не должна лежать вперемешку с кошачьими."""
    client = make_client(runtime_path, monkeypatch)
    client.post("/api/cameras", json={"id": "test-cam", "host": "10.0.0.9", "circuit": "gate"})

    page = client.get("/cameras")

    assert "<h2>Дверца</h2>" in page.text
    assert "<h2>Калитка 1</h2>" in page.text
    # Роль предлагается только камерам дверцы, остальным поле скрыто.
    assert 'class="role-field" hidden' in page.text
    assert 'data-field="circuit"' in page.text


def test_giving_a_role_moves_the_camera_to_the_cat_circuit(
    runtime_path: Path,
    monkeypatch,
) -> None:
    client = make_client(runtime_path, monkeypatch)
    client.post("/api/cameras", json={"id": "test-cam", "host": "10.0.0.9", "circuit": "gate"})

    response = client.put("/api/cameras/test-cam", json={"role": "side"})

    assert response.status_code == 200
    assert response.json()["circuit"] == "cats"
    assert load_registry(camera_registry_path(runtime_path)).get("test-cam").circuit == "cats"


def test_a_gate_camera_is_never_used_by_the_cat_recorder(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """Проверка того, ради чего заводился контур: роль остаётся у своей камеры."""
    client = make_client(runtime_path, monkeypatch, side=True)
    client.post("/api/cameras", json={"id": "test-cam", "host": "10.0.0.9", "circuit": "gate"})

    registry = load_registry(camera_registry_path(runtime_path))

    assert registry.by_role("face").id == "face"
    assert registry.by_role("side").id == "side"
    assert registry.get("test-cam").role is None


def test_the_panel_records_the_choice_without_touching_the_service(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """У веб-процесса нет привилегий: он пишет состояние, а не исполняет его.

    `NoNewPrivileges` в его юните запрещает любое повышение прав, поэтому
    попытка управлять службой отсюда была бы обречена в любом случае.
    """
    monkeypatch.setattr("src.web.app.gate_status", lambda unit, api_url="": GateStatus("stopped"))
    client = make_client(runtime_path, monkeypatch)

    response = client.put("/api/circuits", json={"active": "gate"})

    assert response.status_code == 200
    assert load_circuits(circuits_path(runtime_path)).gate is True
    assert response.json()["gate_state"] == "stopped"


def test_an_undeployed_circuit_is_reported_as_such(runtime_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("src.web.app.gate_status", lambda unit, api_url="": GateStatus("absent"))
    client = make_client(runtime_path, monkeypatch)

    response = client.put("/api/circuits", json={"active": "cats"})

    assert response.json()["gate_deployed"] is False


def test_manual_capture_is_refused_when_the_disk_is_nearly_full(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """Новая запись на почти полном диске добила бы и отчётность о ней."""
    from collections import namedtuple

    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(
        "src.web.app.shutil.disk_usage",
        lambda path: usage(total=100 * 1024**3, used=99 * 1024**3, free=1 * 1024**3),
    )
    client = make_client(runtime_path, monkeypatch)

    response = client.post("/api/events", json={"duration_seconds": 5})

    assert response.status_code == 507
    assert "Освободите место" in response.json()["detail"]


def test_recordings_follow_the_camera_orientation_setting(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """Записи показывались перевёрнутыми на нормально висящей камере.

    Клипы пишутся как есть с камеры, поэтому переворот отыгрывает показ. Но
    константа 180 в разметке осталась с тех пор, когда обе камеры по умолчанию
    висели вверх ногами, и настройку камеры не спрашивала.
    """
    client = make_client(runtime_path, monkeypatch)

    rotated = client.get("/")
    assert '"face": 180' in rotated.text

    client.put("/api/cameras/face", json={"rotate_180": False})

    upright = client.get("/")
    assert '"face": 0' in upright.text
    # Живое превью разворачивает сервер, поэтому второго поворота в CSS быть
    # не должно — иначе картинка встала бы на голову.
    assert "applyCameraRotation" in upright.text
    assert "180 + cameraRotations" not in upright.text


def _record_events(client: TestClient, count: int) -> None:
    for _ in range(count):
        client.post("/api/events", json={"duration_seconds": 5})


def test_passes_are_paginated_by_ten(runtime_path: Path, monkeypatch) -> None:
    """Страница проходов разрасталась до неприличной длины."""
    client = make_client(runtime_path, monkeypatch)
    _record_events(client, 11)

    first = client.get("/")
    second = client.get("/?page=2")

    assert first.text.count('class="pass"') == 10
    assert second.text.count('class="pass"') == 1
    # Самые свежие проходы идут первыми, поэтому на второй странице — самый старый.
    assert 'data-event="1"' in second.text
    assert 'data-event="1"' not in first.text
    assert 'href="/?page=2#labeling"' in first.text
    assert 'href="/?page=1#labeling"' in second.text
    assert "1 из 2" in first.text


def test_page_number_is_clamped_instead_of_failing(runtime_path: Path, monkeypatch) -> None:
    """Ссылка на страницу, которой больше нет, должна вести на ближайшую.

    Иначе сохранённая ссылка ломается ровно тогда, когда с неё удалили
    последний проход.
    """
    client = make_client(runtime_path, monkeypatch)
    _record_events(client, 11)

    assert client.get("/?page=99").status_code == 200
    assert "2 из 2" in client.get("/?page=99").text
    assert "1 из 2" in client.get("/?page=0").text
    assert "1 из 2" in client.get("/?page=-5").text


def test_counters_count_the_whole_base_not_the_page(runtime_path: Path, monkeypatch) -> None:
    """Счётчики описывают всю историю, иначе они меняются от листания.

    Набор для обучения считался питоном по показанному срезу и врал уже на
    сотне проходов, а с пагинацией считал бы по десяти.
    """
    client = make_client(runtime_path, monkeypatch)
    _record_events(client, 11)
    for event_id in (1, 11):
        client.post(
            f"/api/events/{event_id}/label",
            json={"label": "known_cat", "cat_name": "Барсик", "training_target": "macbook"},
        )

    page = client.get("/?page=2")

    assert '<span class="tally-value">11</span>' in page.text
    # Событие №1 лежит на второй странице, №11 — на первой; оба обязаны попасть
    # в счётчик набора для MacBook.
    assert "на MacBook: 2" in page.text


def test_a_single_page_shows_no_pager(runtime_path: Path, monkeypatch) -> None:
    client = make_client(runtime_path, monkeypatch)
    _record_events(client, 3)

    page = client.get("/")

    assert 'class="pager"' not in page.text
    assert "1–3 из 3" in page.text


def test_cameras_live_above_the_circuits_not_inside_the_cats(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """Камеры стали общими для контуров.

    Раздел, спрятанный внутрь дверцы, врал бы про принадлежность: камеру
    регистратора настраивают там же, где кошачью.
    """
    client = make_client(runtime_path, monkeypatch)

    cameras = client.get("/cameras")
    passes = client.get("/")

    assert cameras.status_code == 200
    assert '<a href="/cameras" aria-current="page">Камеры</a>' in cameras.text
    # Внутри кошачьего контура остались только его страницы.
    assert '<nav class="subnav">' in passes.text
    assert 'href="/cameras"' not in passes.text.split('<nav class="subnav">')[1].split("</nav>")[0]


def test_recorder_page_lists_its_cameras_and_switches_the_circuit(
    runtime_path: Path,
    monkeypatch,
) -> None:
    client = make_client(runtime_path, monkeypatch)
    client.put(
        "/api/cameras/face",
        json={"circuit": "nvr", "role": None, "title": "Двор"},
    )

    page = client.get("/nvr")

    assert page.status_code == 200
    assert "Двор" in page.text
    assert '/stream/camera/face.mjpeg' in page.text
    assert 'data-circuit="nvr"' in page.text


def test_recorder_and_gate_run_together_but_not_with_the_cats(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """Калитка и регистратор — один Frigate, а спор за процессор идёт с котами."""
    client = make_client(runtime_path, monkeypatch)

    client.put("/api/circuits", json={"circuit": "gate", "enabled": True})
    both = client.put("/api/circuits", json={"circuit": "nvr", "enabled": True})

    assert both.status_code == 200
    assert both.json()["active"] == ["gate", "nvr"]

    back = client.put("/api/circuits", json={"circuit": "cats", "enabled": True})

    assert back.json()["active"] == ["cats"]


def test_the_node_limit_is_lifted_by_one_switch(runtime_path: Path, monkeypatch) -> None:
    """После переезда на новое железо это единственное, что нужно поменять."""
    client = make_client(runtime_path, monkeypatch)

    client.put("/api/circuits", json={"exclusive": False})
    together = client.put("/api/circuits", json={"circuit": "gate", "enabled": True})

    assert together.json()["active"] == ["cats", "gate"]
    assert together.json()["exclusive"] is False


def test_every_page_shows_which_circuit_is_running(runtime_path: Path, monkeypatch) -> None:
    """Вопрос «что сейчас работает» возникает раньше выбора страницы.

    Узел отдан одному движку, и увидеть это нужно, не уходя со своей вкладки.
    """
    client = make_client(runtime_path, monkeypatch)
    client.put("/api/capture-mode", json={"state": "record"})

    for path in ("/", "/cameras", "/metrics", "/gate", "/nvr"):
        page = client.get(path)
        assert page.status_code == 200, path
        assert 'class="nav-circuit"' in page.text, path
        assert "запись" in page.text, path

    # Состояние следует переключателю сбора, а не только тому, разрешён ли
    # контур на уровне узла.
    client.put("/api/capture-mode", json={"state": "observe"})

    assert '<span class="nav-circuit-state">наблюдение</span>' in client.get("/cameras").text


def test_a_circuit_given_away_reads_as_stopped(runtime_path: Path, monkeypatch) -> None:
    client = make_client(runtime_path, monkeypatch)

    client.put("/api/circuits", json={"circuit": "gate", "enabled": True})
    page = client.get("/cameras")

    assert '<span class="nav-circuit-state">остановлен</span>' in page.text


def test_switching_stays_out_of_the_navigation(runtime_path: Path, monkeypatch) -> None:
    """Ссылка остаётся ссылкой: клик по «Дверце» не должен рвать запись."""
    client = make_client(runtime_path, monkeypatch)

    page = client.get("/cameras")

    assert 'class="nav-circuit" href="/"' in page.text
    # Переключатель живёт на страницах контуров, а не в шапке.
    assert 'id="circuit-toggle"' not in page.text


def test_each_circuit_page_switches_only_itself(runtime_path: Path, monkeypatch) -> None:
    """Страница калитки отвечает за калитку.

    Прежде на каждой из них стоял один и тот же набор из трёх кнопок: он
    повторял в трёх местах то, что и так видно в шапке.
    """
    client = make_client(runtime_path, monkeypatch)

    for path, circuit in (("/", "cats"), ("/gate", "gate"), ("/nvr", "nvr")):
        page = client.get(path)
        assert page.text.count('id="circuit-toggle"') == 1, path
        assert f'data-circuit="{circuit}"' in page.text, path
        # Соседних контуров на странице нет — у каждого своя.
        others = {"cats", "gate", "nvr"} - {circuit}
        for other in others:
            assert f'data-circuit="{other}"' not in page.text, (path, other)


def test_the_cat_circuit_can_be_switched_back_on_from_its_own_page(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """Иначе отданный калитке узел было бы нечем вернуть котам."""
    client = make_client(runtime_path, monkeypatch)
    client.put("/api/circuits", json={"circuit": "gate", "enabled": True})

    assert "Включить контур" in client.get("/").text

    back = client.put("/api/circuits", json={"circuit": "cats", "enabled": True})

    assert back.json()["active"] == ["cats"]
    assert "Выключить контур" in client.get("/").text


def test_the_cameras_page_shows_every_circuit(runtime_path: Path, monkeypatch) -> None:
    """Камера невидимой быть не должна ни при каком контуре.

    Группы страницы перечислялись руками и разошлись со списком контуров:
    регистратор появился, а его группа — нет. Камера при этом жила в инвентаре
    и попадала в конфигурацию Frigate, то есть работала, но поправить или
    удалить её было негде.
    """
    client = make_client(runtime_path, monkeypatch)
    client.post(
        "/api/cameras",
        json={"id": "yard", "host": "192.168.1.60", "circuit": "nvr", "title": "Двор"},
    )

    page = client.get("/cameras")

    assert 'data-camera-id="yard"' in page.text
    assert "Двор" in page.text
    # Заголовок группы — тот же, что у контура в навигации.
    assert page.text.count("Регистратор") >= 1


def test_no_circuit_is_left_without_a_group(runtime_path: Path, monkeypatch) -> None:
    """Список групп выводится из списка контуров, а не повторяет его руками.

    Проверяются заголовки групп, а не любое вхождение названия: контуры
    перечислены ещё и в выпадающем списке карточки, и поиск по тексту прошёл бы
    даже при пропавшей группе.
    """
    client = make_client(runtime_path, monkeypatch)
    for index, circuit in enumerate(CIRCUIT_TITLES):
        client.post(
            "/api/cameras",
            json={"id": f"cam{index}", "host": f"192.168.1.{60 + index}", "circuit": circuit},
        )
    client.post("/api/cameras", json={"id": "loose", "host": "192.168.1.70"})

    page = client.get("/cameras")

    for title in (*CIRCUIT_TITLES.values(), "Без контура"):
        assert f"<h2>{title}</h2>" in page.text, title


def test_gate_tab_shows_its_cameras(runtime_path: Path, monkeypatch) -> None:
    """Камера калитки видна там же, где ей управляют.

    Прежде страница отсылала за картинкой в интерфейс Frigate, хотя живое
    изображение админка отдаёт сама и остальным контурам показывает на месте.
    """
    client = make_client(runtime_path, monkeypatch)
    created = client.post(
        "/api/cameras",
        json={
            "id": "gate-1",
            "title": "На калитке",
            "host": "192.0.2.7",
            "stream_path": "/Streaming/Channels/101",
            "detect_stream_path": "/Streaming/Channels/102",
            "circuit": "gate",
        },
    )
    assert created.status_code == 201

    page = client.get("/gate")

    assert page.status_code == 200
    assert "/stream/camera/gate-1.mjpeg" in page.text
    assert "На калитке" in page.text


def test_previews_are_on_by_default_and_the_header_button_hides_them(
    runtime_path: Path, monkeypatch
) -> None:
    """Превью показываются сразу, а кнопка в шапке их прячет.

    Умолчание перевернулось вместе с переездом на узел, которому живые картинки
    по карману. Разметка при этом не изменилась: адрес потока по-прежнему лежит
    в data-атрибуте, а `src` проставляет скрипт. Иначе браузер открывал бы
    потоки ещё до того, как страница решит, показывать их или нет, и выключение
    перестало бы быть бесплатным для узла.
    """
    client = make_client(runtime_path, monkeypatch, side=True)

    for page in ("/", "/cameras"):
        html = client.get(page).text
        assert ' src="/stream/' not in html, page
        assert 'data-stream-src="/stream/' in html, page
        assert (
            '<button id="preview-toggle" type="button" hidden aria-pressed="true">'
            "Скрыть превью</button>"
        ) in html, page


def test_nvr_tab_links_to_frigate(runtime_path: Path, monkeypatch) -> None:
    """У регистратора та же дорога во Frigate, что и у калитки.

    События и архив живут в интерфейсе Frigate; вкладка без ссылки заставляла
    вспоминать адрес руками.
    """
    monkeypatch.setenv("CATDOOR_GATE_UI_URL", "http://node.example:5000")
    client = make_client(runtime_path, monkeypatch)
    monkeypatch.setattr(
        "src.web.app.gate_status",
        lambda unit, api_url="": GateStatus("running", ""),
    )

    page = client.get("/nvr")

    assert page.status_code == 200
    assert "Открыть Frigate" in page.text
    assert "http://node.example:5000" in page.text


def test_system_stats_endpoint_and_header_widget(runtime_path: Path, monkeypatch) -> None:
    """Шапка показывает загрузку узла, а эндпоинт не выдумывает чисел.

    Первый вызов после старта отдаёт null вместо процента: мгновенного CPU у
    ядра нет, есть только разность двух замеров.
    """
    client = make_client(runtime_path, monkeypatch)

    from src.common.system_stats import CpuSample

    samples = iter([CpuSample(busy=250, total=1400)])
    monkeypatch.setattr("src.web.app.read_cpu_sample", lambda: next(samples, None))
    monkeypatch.setattr("src.web.app.read_memory_mb", lambda: (2100, 7808))

    payload = client.get("/api/system/stats").json()
    assert payload["memory_used_mb"] == 2100
    assert payload["memory_total_mb"] == 7808

    page = client.get("/")
    assert 'id="system-stats"' in page.text


def test_pass_time_is_shown_in_the_local_timezone(runtime_path: Path, monkeypatch) -> None:
    """Проход показывается по местным часам, а в базе остаётся UTC.

    Recorder пишет метку в UTC, sqlite отдаёт её без зоны, и напечатанная как
    есть она отставала бы от часов у дверцы на разницу поясов — под Москвой на
    три часа, да ещё и с прошлой датой у ночного прохода.
    """
    client = make_client(runtime_path, monkeypatch)
    factory = client.app.state.session_factory
    with factory() as session:
        session.add(
            Event(
                created_at=datetime(2026, 8, 6, 21, 30, 5, tzinfo=UTC),
                status="ready",
                duration_seconds=5,
                media_path="events/00000001/face.mp4",
            )
        )
        session.commit()

    page = client.get("/")

    assert '<span class="pass-time">00:30:05</span>' in page.text
    assert "<span>07.08</span>" in page.text
    with factory() as session:
        stored = session.get(Event, 1)
        assert stored is not None
        assert stored.created_at.replace(tzinfo=UTC) == datetime(
            2026, 8, 6, 21, 30, 5, tzinfo=UTC
        )


def test_display_timezone_setting_moves_the_shown_time(runtime_path: Path, monkeypatch) -> None:
    """Пояс задаётся конфигурацией: узел может стоять не в Москве."""
    client = make_client(runtime_path, monkeypatch, timezone="Asia/Novosibirsk")
    factory = client.app.state.session_factory
    with factory() as session:
        session.add(
            Event(
                created_at=datetime(2026, 8, 6, 21, 30, 5, tzinfo=UTC),
                status="ready",
                duration_seconds=5,
                media_path="events/00000001/face.mp4",
            )
        )
        session.commit()

    page = client.get("/")

    assert '<span class="pass-time">04:30:05</span>' in page.text


def test_wide_tables_can_scroll_inside_their_panel() -> None:
    """Таблица шире экрана обязана прокручиваться сама, а не тянуть страницу.

    На телефоне незавёрнутая таблица делала документ шире вьюпорта, и «ехала»
    вся страница, а не только она. Проверка на шаблонах, а не на отрисовке:
    отчёты об оценке требуют завершённого запуска, которого в тестах нет.
    """
    templates = (Path(__file__).resolve().parent.parent / "src" / "web" / "templates").glob("*.html")
    unwrapped: list[str] = []
    for template in templates:
        lines = template.read_text(encoding="utf-8").split("\n")
        for number, line in enumerate(lines):
            if line.strip() != "<table>":
                continue
            previous = next(
                (item for item in reversed(lines[:number]) if item.strip()),
                "",
            )
            if "table-scroll" not in previous:
                unwrapped.append(f"{template.name}:{number + 1}")
    assert unwrapped == []


def test_notification_test_button_reports_what_telegram_answered(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """Неверный токен и чужой чат выглядят одинаково никак — кроме этой кнопки."""
    client = TestClient(
        create_app(
            Settings(
                data_dir=runtime_path,
                database_path=runtime_path / "test.sqlite3",
                telegram_bot_token="8999303876:test",
                telegram_chat_id="12345",
            )
        )
    )
    monkeypatch.setattr(
        "src.common.telegram.TelegramClient.send_message",
        lambda self, text: (_ for _ in ()).throw(TelegramError("Телеграм отказал (401)")),
    )

    response = client.post("/api/notifications/test")

    assert response.status_code == 200
    assert response.json() == {"ok": False, "reason": "Телеграм отказал (401)"}


def test_notification_test_refuses_when_the_bot_is_not_configured(runtime_path: Path) -> None:
    client = TestClient(
        create_app(
            Settings(data_dir=runtime_path, database_path=runtime_path / "test.sqlite3")
        )
    )

    response = client.post("/api/notifications/test")

    assert response.status_code == 422
    assert "не настроен" in response.json()["detail"]


def test_health_page_shows_the_reference_next_to_the_fact(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """Страница состояния отвечает на вопрос «всё ли как в эталоне».

    Настройки на этом узле умышленно отличаются от принятых: клиент поднят на
    умолчаниях, где детектор на процессоре. Страница обязана назвать это
    расхождением, а не промолчать.
    """
    client = make_client(runtime_path, monkeypatch)
    write_recorder_status(
        runtime_path / "runtime" / "recorder-status.json",
        detector_device_actual="CPU",
        detector_model="yolo11n-416.onnx",
        hwaccel_actual="none",
        frames_per_second={"face": 12.0, "side": 6.0},
        inference_ms=13.5,
    )

    page = client.get("/health")
    api = client.get("/api/health")

    assert page.status_code == 200
    assert "Кошачий контур" in page.text
    assert "Контур калитки" in page.text
    assert "Железо узла" in page.text
    assert api.status_code == 200
    payload = api.json()
    assert payload["status"] == "fail"
    checks = {
        check["key"]: check
        for group in payload["groups"]
        for check in group["checks"]
    }
    assert checks["detector_device"]["status"] == "fail"
    assert checks["detector_device"]["actual"] == "CPU"
    assert checks["detector_device"]["expected"] == "GPU"
    assert checks["hwaccel"]["status"] == "fail"
    # Frigate в тестах не поднят, и это «недоступен», а не «сломан».
    assert checks["gate_reachable"]["status"] == "fail"


def test_health_survives_a_node_without_a_recorder(runtime_path: Path, monkeypatch) -> None:
    """Ни статуса, ни датчиков — страница обязана открыться и сказать честно."""
    client = make_client(runtime_path, monkeypatch)

    page = client.get("/health")
    payload = client.get("/api/health").json()

    assert page.status_code == 200
    assert payload["status"] in {"fail", "unknown"}


def test_camera_card_shows_the_decode_mode(runtime_path: Path, monkeypatch) -> None:
    """Настройка, которой не видно в API, обнаруживается только раскопками."""
    client = make_client(runtime_path, monkeypatch)

    updated = client.put("/api/cameras/face", json={"decode": "qsv"})

    assert updated.status_code == 200
    assert updated.json()["decode"] == "qsv"


def test_poster_is_generated_once_and_cached_forever(runtime_path: Path, monkeypatch) -> None:
    """Страница проходов тянула метаданные двадцати роликов разом.

    По каналу до админки в 1.7 МБ/с — при роликах по полсотни мегабайт — превью
    появлялись через секунды, а нажатый play ждал очереди соединений. Постер в
    ~30 КБ отдаёт то же самое одним запросом, и второй раз даже не спрашивается:
    запись прохода неизменяема.
    """
    client = make_client(runtime_path, monkeypatch)
    capture = client.post("/api/events", json={"duration_seconds": 5})
    event_id = capture.json()["id"]
    calls: list[float] = []

    def fake_extract(ffmpeg, video, destination, *, at_seconds=1.0, **kwargs):
        calls.append(at_seconds)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"\xff\xd8jpeg")
        return True

    monkeypatch.setattr("src.web.app.extract_frame", fake_extract)

    first = client.get(f"/media/{event_id}/face/poster")
    second = client.get(f"/media/{event_id}/face/poster")

    assert first.status_code == 200
    assert first.headers["content-type"] == "image/jpeg"
    assert "immutable" in first.headers["cache-control"]
    assert second.status_code == 200
    assert len(calls) == 1, "кадр вынимается один раз, дальше кэш"


def test_poster_extraction_failure_is_a_clean_404(runtime_path: Path, monkeypatch) -> None:
    client = make_client(runtime_path, monkeypatch)
    event_id = client.post("/api/events", json={"duration_seconds": 5}).json()["id"]
    monkeypatch.setattr("src.web.app.extract_frame", lambda *a, **k: False)

    assert client.get(f"/media/{event_id}/face/poster").status_code == 404


def test_deleting_an_event_removes_its_posters(runtime_path: Path, monkeypatch) -> None:
    from src.web.app import poster_path

    client = make_client(runtime_path, monkeypatch)
    event_id = client.post("/api/events", json={"duration_seconds": 5}).json()["id"]
    poster = poster_path(runtime_path, event_id, "face")
    poster.parent.mkdir(parents=True, exist_ok=True)
    poster.write_bytes(b"jpeg")

    assert client.delete(f"/api/events/{event_id}").status_code == 204
    assert not poster.exists()


def test_pass_videos_do_not_preload(runtime_path: Path, monkeypatch) -> None:
    """`preload="metadata"` на двадцати роликах — это и была вся беда."""
    client = make_client(runtime_path, monkeypatch)
    client.post("/api/events", json={"duration_seconds": 5})

    page = client.get("/")

    assert 'preload="none"' in page.text
    assert 'preload="metadata"' not in page.text
    assert "/poster" in page.text


def _gate_pass(runtime_path: Path, **overrides) -> int:
    """Строка журнала калитки прямо в базе: пишет её отдельная служба."""
    from src.common.database import GatePass

    values: dict[str, object] = {
        "frigate_event_id": "1786171119.126838-b1jywn",
        "created_at": datetime.now(UTC),
        "camera": "gate",
        "person_name": None,
        "score": None,
        "zone": None,
        "decision": "denied_mode_off",
        "reason": "Автооткрытие выключено",
        "snapshot_path": None,
    }
    values.update(overrides)
    factory = create_session_factory(runtime_path / "test.sqlite3")
    with factory() as session:
        stored = GatePass(**values)
        session.add(stored)
        session.commit()
        return stored.id


def test_gate_tab_shows_the_journal_of_passes(runtime_path: Path, monkeypatch) -> None:
    """Журнал живёт на вкладке калитки, а не отдельной страницей."""
    client = make_client(runtime_path, monkeypatch)
    _gate_pass(runtime_path, frigate_event_id="a", person_name="Алексей", score=0.94)
    _gate_pass(runtime_path, frigate_event_id="b")

    page = client.get("/gate")

    assert page.status_code == 200
    assert "Журнал проходов" in page.text
    assert "Алексей" in page.text
    assert "неизвестный" in page.text
    assert "автооткрытие выключено" in page.text


def test_the_journal_says_it_is_empty_rather_than_showing_nothing(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """Пустой список и «журнал пуст» — разные сообщения для человека."""
    client = make_client(runtime_path, monkeypatch)

    page = client.get("/gate")

    assert "Журнал пуст" in page.text


def test_gate_counters_count_the_whole_base_not_the_page(
    runtime_path: Path,
    monkeypatch,
) -> None:
    client = make_client(runtime_path, monkeypatch)
    for number in range(12):
        _gate_pass(
            runtime_path,
            frigate_event_id=f"e{number}",
            person_name="Алексей" if number % 2 else None,
        )

    page = client.get("/gate")

    assert '<span class="tally-value">12</span>' in page.text
    assert '<span class="tally-value">6</span>' in page.text
    assert "1 из 2" in page.text


def test_the_journal_filters_by_person_and_by_day(runtime_path: Path, monkeypatch) -> None:
    """Фильтр «за день» человек задаёт по своим часам, а база живёт в UTC.

    Полночь в Москве наступает на три часа раньше: без пересчёта в «сегодня»
    попадали бы три часа вчерашней ночи.
    """
    client = make_client(runtime_path, monkeypatch)
    today = datetime.now(ZoneInfo("Europe/Moscow"))
    _gate_pass(
        runtime_path,
        frigate_event_id="today",
        person_name="Алексей",
        created_at=today.astimezone(UTC),
    )
    _gate_pass(
        runtime_path,
        frigate_event_id="old",
        person_name="Марина",
        created_at=(today - timedelta(days=3)).astimezone(UTC),
    )

    by_name = client.get("/gate?name=Марина")
    by_day = client.get(f"/gate?day={today.strftime('%Y-%m-%d')}")

    assert "Марина" in by_name.text
    assert by_name.text.count("Алексей") == 1  # только чип фильтра
    assert "Алексей" in by_day.text
    assert by_day.text.count("Марина") == 1


def test_an_unreadable_day_filter_falls_back_to_everything(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """Сохранённая ссылка с мусором в параметре не должна отдавать ошибку."""
    client = make_client(runtime_path, monkeypatch)
    _gate_pass(runtime_path, frigate_event_id="a", person_name="Алексей")

    page = client.get("/gate?day=позавчера&who=выдумка")

    assert page.status_code == 200
    assert "Алексей" in page.text


def test_a_gate_snapshot_stays_inside_the_data_directory(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """Путь приходит из базы, а её пишет служба по чужому идентификатору."""
    client = make_client(runtime_path, monkeypatch)
    inside = _gate_pass(
        runtime_path,
        frigate_event_id="a",
        snapshot_path="runtime/gate-passes/a.jpg",
    )
    outside = _gate_pass(
        runtime_path,
        frigate_event_id="b",
        snapshot_path="../../etc/passwd",
    )
    snapshot = runtime_path / "runtime" / "gate-passes" / "a.jpg"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(b"jpeg")

    good = client.get(f"/gate/passes/{inside}/snapshot")

    assert good.status_code == 200
    assert good.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert client.get(f"/gate/passes/{outside}/snapshot").status_code == 404
    assert client.get("/gate/passes/9999/snapshot").status_code == 404


def test_the_gate_tab_still_says_mechanics_are_not_controlled(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """Журнал не даёт права на замок, и страница не должна намекать обратное."""
    client = make_client(runtime_path, monkeypatch)
    _gate_pass(runtime_path, frigate_event_id="a")

    page = client.get("/gate")

    assert "не управляется" in page.text


def test_the_gate_tab_offers_all_three_modes(runtime_path: Path, monkeypatch) -> None:
    """Умолчание — «выключено»: свежая установка не открывает калитку никому."""
    client = make_client(runtime_path, monkeypatch)

    page = client.get("/gate")

    assert "Открытие по лицу" in page.text
    for mode in ("off", "dry", "on"):
        assert f'data-mode="{mode}"' in page.text
    assert client.get("/api/gate/access-settings").json()["mode"] == "off"


def test_access_settings_default_to_off(runtime_path: Path, monkeypatch) -> None:
    client = make_client(runtime_path, monkeypatch)

    settings = client.get("/api/gate/access-settings").json()

    assert settings["mode"] == "off"
    assert settings["require_second_factor"] is False


def test_access_settings_are_saved_for_the_service(runtime_path: Path, monkeypatch) -> None:
    """Веб пишет файл, служба его читает: механикой админка не распоряжается."""
    from src.common.gate_access_config import GateAccessConfig, gate_access_config_path

    client = make_client(runtime_path, monkeypatch)

    response = client.put(
        "/api/gate/access-settings",
        json={
            "mode": "dry",
            "gates": [{"camera": "gate", "relay": 2, "zone": "у калитки"}],
            "allowed_names": ["Алексей"],
            "min_score": 0.95,
        },
    )

    assert response.status_code == 200
    stored = GateAccessConfig.model_validate_json(
        gate_access_config_path(runtime_path).read_text(encoding="utf-8")
    )
    assert stored.mode == "dry"
    assert stored.allowed_names == ["Алексей"]
    assert stored.min_score == 0.95


def test_the_admin_switches_the_mode_and_the_service_reads_it(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """Переключатель режима — единственное, чем человек включает автооткрытие."""
    client = make_client(runtime_path, monkeypatch)

    assert client.put("/api/gate/access-settings", json={"mode": "on"}).status_code == 200
    assert client.get("/api/gate/access-settings").json()["mode"] == "on"


def test_a_typo_in_the_mode_is_refused(runtime_path: Path, monkeypatch) -> None:
    """Опечатка не должна тихо превратиться во что-нибудь работающее."""
    client = make_client(runtime_path, monkeypatch)

    response = client.put("/api/gate/access-settings", json={"mode": "открывать"})

    assert response.status_code == 422
    assert client.get("/api/gate/access-settings").json()["mode"] == "off"


def test_a_phone_mac_never_leaves_the_node_whole(runtime_path: Path, monkeypatch) -> None:
    """MAC — персональные данные семьи: наружу выходит только хвост."""
    client = make_client(runtime_path, monkeypatch)

    client.put(
        "/api/gate/access-settings",
        json={
            "mode": "off",
            "people": [
                {"name": "Алексей", "phone_ip": "192.168.2.50", "phone_mac": "aa:bb:cc:dd:ee:ff"}
            ],
        },
    )
    person = client.get("/api/gate/access-settings").json()["people"][0]

    assert person["phone_mac"] == "··:··:··:··:ee:ff"
    assert person["has_phone_mac"] is True
    assert "aa:bb" not in client.get("/gate").text


def test_a_missing_mac_means_keep_the_stored_one(runtime_path: Path, monkeypatch) -> None:
    """Форма не может вернуть то, чего не получала, — и не должна это стирать."""
    from src.common.gate_access_config import GateAccessConfig, gate_access_config_path

    client = make_client(runtime_path, monkeypatch)
    client.put(
        "/api/gate/access-settings",
        json={"people": [{"name": "Алексей", "phone_mac": "aa:bb:cc:dd:ee:ff"}]},
    )

    client.put(
        "/api/gate/access-settings",
        json={"mode": "dry", "people": [{"name": "Алексей", "phone_ip": "192.168.2.51"}]},
    )

    stored = GateAccessConfig.model_validate_json(
        gate_access_config_path(runtime_path).read_text(encoding="utf-8")
    )
    assert stored.people[0].phone_mac == "aa:bb:cc:dd:ee:ff"
    assert stored.people[0].phone_ip == "192.168.2.51"


def test_a_broken_mac_is_refused_with_a_readable_reason(
    runtime_path: Path,
    monkeypatch,
) -> None:
    client = make_client(runtime_path, monkeypatch)

    response = client.put(
        "/api/gate/access-settings",
        json={"people": [{"name": "Алексей", "phone_mac": "нет"}]},
    )

    assert response.status_code == 422
    assert "MAC" in response.json()["detail"]


def test_names_come_from_the_frigate_face_library(runtime_path: Path, monkeypatch) -> None:
    """Два списка имён разъезжаются и молча перестают совпадать."""
    monkeypatch.setattr(
        "src.web.app.frigate_face_names", lambda api_url, **kwargs: ["Алексей", "Марина"]
    )
    client = make_client(runtime_path, monkeypatch)
    client.put("/api/gate/access-settings", json={"allowed_names": ["Алексей"]})

    page = client.get("/gate")

    assert 'data-name="Марина"' in page.text
    assert 'data-name="Алексей"' in page.text


def test_a_name_that_left_the_library_is_shown_not_hidden(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """Иначе в настройках остаётся правило, которое не сработает никогда."""
    monkeypatch.setattr("src.web.app.frigate_face_names", lambda api_url, **kwargs: ["Марина"])
    client = make_client(runtime_path, monkeypatch)
    client.put("/api/gate/access-settings", json={"allowed_names": ["Алексей"]})

    page = client.get("/gate")

    assert "которых нет в библиотеке лиц" in page.text
    assert "Алексей" in page.text


def test_an_unreachable_face_library_is_not_an_empty_one(
    runtime_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("src.web.app.frigate_face_names", lambda api_url, **kwargs: None)
    client = make_client(runtime_path, monkeypatch)

    page = client.get("/gate")

    assert "спросить не удалось" in page.text


def test_the_gate_is_saved_under_its_frigate_name(runtime_path: Path, monkeypatch) -> None:
    """Инвентарь знает камеру как `test-cam`, а Frigate шлёт события от `test_cam`.

    Дефис в имени камеры Frigate недопустим, и генератор конфигурации меняет
    его на подчёркивание. Сохранив идентификатор инвентаря, панель получила бы
    настройку, которая никогда не совпадёт с событием, — и каждый проход
    получал бы «камера не назначена калиткой». Молча.
    """
    from src.common.camera_registry import CameraRecord, CameraRegistry

    save_registry(
        camera_registry_path(runtime_path),
        CameraRegistry(
            cameras=[
                CameraRecord(id="test-cam", title="на калитке", circuit="gate", host="10.0.0.1")
            ]
        ),
    )
    client = make_client(runtime_path, monkeypatch)

    page = client.get("/gate")

    assert '<option value="test_cam"' in page.text
    assert '<option value="test-cam"' not in page.text


def test_the_admin_asks_the_service_instead_of_writing_to_the_board(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """У веб-процесса нет и не должно быть прав на механику.

    Он кладёт просьбу файлом, а исполняет её `catdoor-gate-access` — та же
    схема, что у переключателя контуров с его примирителем.
    """
    from src.common.gate_open_request import (
        OpenResult,
        load_open_request,
        open_request_path,
        open_result_path,
        save_open_result,
    )

    client = make_client(runtime_path, monkeypatch)
    client.put("/api/gate/access-settings", json={"gates": [{"relay": 1}]})

    def answer(seconds: float) -> None:
        request = load_open_request(open_request_path(runtime_path))
        assert request is not None
        save_open_result(
            open_result_path(runtime_path),
            OpenResult(
                id=request.id, ok=True, detail="Импульс 500 мс на канал 1",
                finished_at=0.0, pulse_ms=500,
            ),
        )

    monkeypatch.setattr("src.web.app.sleep", answer)

    response = client.post("/api/gate/open")

    assert response.status_code == 200
    assert response.json()["pulse_ms"] == 500


def test_a_silent_service_is_reported_not_assumed(runtime_path: Path, monkeypatch) -> None:
    """Сказать «открыл», не дождавшись подтверждения, — та самая ложь.

    Служба смотрит файл раз в секунду; молчание дольше означает, что она не
    работает, и человеку надо сказать именно это.
    """
    client = make_client(runtime_path, monkeypatch)
    client.put("/api/gate/access-settings", json={"gates": [{"relay": 1}]})
    monkeypatch.setattr("src.web.app.OPEN_RESULT_TIMEOUT_SECONDS", 0.05)

    response = client.post("/api/gate/open")

    assert response.status_code == 504
    assert "catdoor-gate-access" in response.json()["detail"]


def test_the_old_answer_is_not_taken_for_this_press(runtime_path: Path, monkeypatch) -> None:
    """Рядом лежит ответ на прошлое нажатие, и он не про эту просьбу."""
    from src.common.gate_open_request import OpenResult, open_result_path, save_open_result

    client = make_client(runtime_path, monkeypatch)
    client.put("/api/gate/access-settings", json={"gates": [{"relay": 1}]})
    save_open_result(
        open_result_path(runtime_path),
        OpenResult(id="прошлое", ok=True, detail="Импульс", finished_at=0.0, pulse_ms=500),
    )
    monkeypatch.setattr("src.web.app.OPEN_RESULT_TIMEOUT_SECONDS", 0.05)

    response = client.post("/api/gate/open")

    assert response.status_code == 504


def test_the_button_needs_a_relay_before_it_opens_anything(
    runtime_path: Path,
    monkeypatch,
) -> None:
    client = make_client(runtime_path, monkeypatch)

    response = client.post("/api/gate/open")

    assert response.status_code == 422
    assert "не настроен" in response.json()["detail"]
    assert "сначала задайте канал реле" in client.get("/gate").text


def test_a_refused_pulse_reaches_the_person_who_pressed(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """Человек у калитки должен узнать, что она не открылась, от админки."""
    from src.common.gate_open_request import (
        OpenResult,
        load_open_request,
        open_request_path,
        open_result_path,
        save_open_result,
    )

    client = make_client(runtime_path, monkeypatch)
    client.put("/api/gate/access-settings", json={"gates": [{"relay": 1}]})

    def answer(seconds: float) -> None:
        request = load_open_request(open_request_path(runtime_path))
        assert request is not None
        save_open_result(
            open_result_path(runtime_path),
            OpenResult(
                id=request.id, ok=False, detail="Плата не ответила", finished_at=0.0
            ),
        )

    monkeypatch.setattr("src.web.app.sleep", answer)

    response = client.post("/api/gate/open")

    assert response.status_code == 502
    assert response.json()["detail"] == "Плата не ответила"


def test_the_admin_offers_to_remember_cats_without_the_macbook(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """Пересчёт эталонов — это запоминание, а не обучение, и живёт он на узле."""
    client = make_client(runtime_path, monkeypatch)

    page = client.get("/")

    assert "Память о котах" in page.text
    assert 'id="rebuild-prototypes"' in page.text
    assert client.get("/api/identity/prototypes/status").json()["state"] == "idle"


def test_rebuilding_prototypes_needs_a_model(runtime_path: Path, monkeypatch) -> None:
    """Без модели опознания пересчитывать нечем, и это надо сказать, а не молчать."""
    client = make_client(runtime_path, monkeypatch)

    response = client.post("/api/identity/prototypes/rebuild")

    assert response.status_code == 422
    assert "модель опознания" in response.json()["detail"]


def test_prototype_rebuild_runs_outside_the_web_process(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """Перезапуск админки не должен обрывать пересчёт — как и обучение.

    Заодно проверяется, что процессу передана модель своего ракурса: эталоны у
    ракурсов разные, и чужие означают, что модель не узнаёт никого.
    """
    from src.common.settings import Settings

    model = runtime_path / "face-identity.onnx"
    model.write_bytes(b"synthetic")
    settings = Settings(
        data_dir=runtime_path,
        database_path=runtime_path / "test.sqlite3",
        identity_face_model_path=model,
    )
    started: dict[str, object] = {}

    def fake_spawn(command, **kwargs):
        started["command"] = command
        return SimpleNamespace(pid=4242)

    monkeypatch.setattr("src.web.app.spawn_detached", fake_spawn)
    client = TestClient(create_app(settings))

    response = client.post("/api/identity/prototypes/rebuild")

    assert response.status_code == 202
    assert response.json()["pid"] == 4242
    command = started["command"]
    assert "training.enrolment" in command
    assert "--face-model" in command
    assert str(model.resolve()) in command


def test_fresh_crops_by_default_and_reuse_on_request(runtime_path: Path, monkeypatch) -> None:
    """Обычный порядок работы — «разметил и нажал», и кадры нужны свежие.

    Взять кропы прошлого прогона дешевле на три четверти часа, но тогда
    свежая разметка молча не попадёт в расчёт. Поэтому по умолчанию нарезаем
    заново, а переиспользование запрашивается явно — оно годится, когда
    разметка не менялась.
    """
    from src.common.settings import Settings

    model = runtime_path / "face-identity.onnx"
    model.write_bytes(b"synthetic")
    previous = runtime_path / "training" / "prototypes" / "20260811-100000-aaaa" / "dataset"
    previous.mkdir(parents=True)
    (previous / "dataset.jsonl").write_text("{}\n", encoding="utf-8")
    (previous.parent / "status.json").write_text('{"state": "completed"}', encoding="utf-8")
    started: dict[str, object] = {}

    def fake_spawn(command, **kwargs):
        started["command"] = command
        return SimpleNamespace(pid=1)

    monkeypatch.setattr("src.web.app.spawn_detached", fake_spawn)
    client = TestClient(
        create_app(
            Settings(
                data_dir=runtime_path,
                database_path=runtime_path / "test.sqlite3",
                identity_face_model_path=model,
            )
        )
    )

    first = client.post("/api/identity/prototypes/rebuild?reuse=true")
    assert first.status_code == 202, first.text
    reused = started["command"]
    started.clear()
    # Первый запуск оставил статус «работает», и второй запрос иначе получил бы
    # честный отказ «пересчёт уже выполняется».
    for status in (runtime_path / "training" / "prototypes").glob("*/status.json"):
        status.write_text('{"state": "completed"}', encoding="utf-8")
    client.post("/api/identity/prototypes/rebuild")
    fresh = started["command"]

    assert "--dataset" in reused
    assert str(previous) in reused
    assert "--dataset" not in fresh


def test_old_crops_are_swept_but_reports_stay(runtime_path: Path) -> None:
    """Каждый прогон — пара гигабайт кадров, и копиться молча они не должны.

    Отчёты остаются: по ним смотрят, что было. Кропы всегда можно нарезать
    заново.

    Считаются прогоны **со своими кадрами**: 2026-08-12 уборка отсчитала два
    последних подряд — прерванный и взявший чужие кропы, — и снесла
    единственный годный набор, ради которого переиспользование и заводилось.
    """
    from training.enrolment import purge_old_runs

    runs = runtime_path / "prototypes"
    for name in ("20260811-100000", "20260811-110000", "20260811-120000"):
        (runs / name / "dataset").mkdir(parents=True)
        (runs / name / "dataset" / "dataset.jsonl").write_text("{}", encoding="utf-8")
        (runs / name / "status.json").write_text('{"state": "completed"}', encoding="utf-8")
    # Два прогона без кадров: прерванный и взявший чужие. Окно они не занимают.
    for name in ("20260811-130000", "20260811-140000"):
        (runs / name).mkdir(parents=True)
        (runs / name / "status.json").write_text('{"state": "completed"}', encoding="utf-8")

    removed = purge_old_runs(runs, keep=2)

    assert removed == ["20260811-100000"]
    assert (runs / "20260811-110000" / "dataset").is_dir()
    assert not (runs / "20260811-100000" / "dataset").exists()
    assert (runs / "20260811-100000" / "status.json").is_file()
    assert (runs / "20260811-120000" / "dataset").is_dir()


def test_the_preset_applies_and_rolls_back_in_one_move(runtime_path: Path, monkeypatch) -> None:
    """Числа пресета подобраны разбором живых проходов, перебирать их незачем.

    Откат должен работать и после перезагрузки узла, поэтому прежние правила
    хранятся в том же файле настроек, а не в памяти процесса.
    """
    from src.common.gate_access_config import GateAccessConfig, gate_access_config_path

    client = make_client(runtime_path, monkeypatch)
    client.put(
        "/api/gate/access-settings",
        json={"movement": {"enabled": False, "return_window_seconds": 5}},
    )

    applied = client.post("/api/gate/access-settings/preset").json()
    stored = GateAccessConfig.model_validate_json(
        gate_access_config_path(runtime_path).read_text(encoding="utf-8")
    )

    assert applied["movement"]["enabled"] is True
    assert applied["movement"]["return_window_seconds"] == 30
    assert applied["preset_applied"] is True
    assert stored.movement.enabled is True

    back = client.post("/api/gate/access-settings/preset?rollback=true").json()

    assert back["movement"]["enabled"] is False
    assert back["movement"]["return_window_seconds"] == 5
    assert back["preset_applied"] is False


def test_saving_other_settings_does_not_disturb_the_rules(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """Панель настроек и переключатель пресета живут рядом и не затирают друг друга."""
    client = make_client(runtime_path, monkeypatch)
    client.post("/api/gate/access-settings/preset")

    client.put("/api/gate/access-settings", json={"mode": "dry", "min_score": 0.9})
    settings = client.get("/api/gate/access-settings").json()

    assert settings["movement"]["enabled"] is True
    assert settings["preset_applied"] is True


def test_the_rules_are_visible_and_switchable_in_the_admin(
    runtime_path: Path,
    monkeypatch,
) -> None:
    client = make_client(runtime_path, monkeypatch)

    page = client.get("/gate")

    assert "Куда идёт человек" in page.text
    assert 'id="movement-enabled"' in page.text
    assert 'id="apply-preset"' in page.text
    # Откатывать нечего, пока пресет не накатили.
    assert 'id="rollback-preset" type="button" disabled' in page.text
