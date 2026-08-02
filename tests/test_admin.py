import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import select

from src.common.camera_discovery import DiscoveredDevice
from src.common.camera_registry import camera_registry_path, load_registry
from src.common.capture_mode import CaptureMode, capture_mode_path, load_capture_mode
from src.common.circuits import circuits_path, load_circuits
from src.common.database import EvaluationRun, Event, create_session_factory
from src.common.gate_service import GateStatus
from src.common.settings import Settings
from src.recorder.status import write_recorder_status
from src.web.app import create_app


def make_client(runtime_path: Path, monkeypatch, *, side: bool = False) -> TestClient:
    def fake_record(
        ffmpeg_path: str,
        rtsp_url: str,
        destination: Path,
        duration_seconds: int,
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"synthetic-video")

    monkeypatch.setattr("src.web.app.record_clip", fake_record)
    settings = Settings(
        camera_face_rtsp="rtsp://user:password@camera.local/media/video1",
        camera_side_rtsp=(
            "rtsp://user:password@side-camera.local/cam/realmonitor?subtype=0" if side else ""
        ),
        data_dir=runtime_path,
        database_path=runtime_path / "test.sqlite3",
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
    assert 'if (camera.getAttribute("src") !== source)' in page.text
    assert "function streamUrl(role)" in page.text
    assert "/stream/face.mjpeg?v=" in page.text
    assert 'cameraRotations[role] = (cameraRotations[role] + 90) % 360' in page.text
    assert 'localStorage.setItem(cameraRotationKey, JSON.stringify(cameraRotations))' in page.text
    assert 'const cameraRotationKey = "catdoor-camera-rotations-v2"' in page.text
    assert 'id="detector-status"' in page.text
    # Разметка сохраняется нажатием по переключателю, отдельной кнопки нет.
    assert 'class="label-choice" data-label="known_cat"' in page.text
    assert "save-label" not in page.text
    # Настройки детекции переехали на вкладку камер: там же список камер и их
    # роли, поэтому редактировать их в двух местах больше не нужно.
    assert 'id="detection-face-confirmation-frames"' not in page.text
    assert '<video preload="metadata"' in page.text
    assert "<video controls" not in page.text
    assert 'class="player-bar"' in page.text
    assert 'class="video-seek"' in page.text
    assert 'document.querySelectorAll(".event-player")' in page.text
    assert 'data-camera-role="face"' in page.text
    assert '<div class="event-recording" data-camera-role="side">' not in page.text
    assert "function applyEventMediaLayout(swapped)" in page.text
    assert "applyEventMediaLayout(swapped);" in page.text
    assert "function applyEventVideoRotations()" in page.text
    assert "(180 + cameraRotations[role]) % 360" in page.text


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
    assert "GPU" not in command
    factory = create_session_factory(runtime_path / "test.sqlite3")
    with factory() as session:
        run = session.get(EvaluationRun, 1)
        assert run is not None
        assert run.status == "starting"
        assert run.detector_model_version_id is not None
        assert run.face_model_version_id is not None
        assert run.side_model_version_id is None


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
    }

    response = client.put("/api/detection/settings", json=payload)

    assert response.status_code == 200
    assert response.json() == payload
    assert client.get("/api/detection/settings").json() == payload
    page = client.get("/cameras")
    assert 'id="detection-face-sample-fps"' in page.text
    assert 'id="detection-face-roi-enabled"' in page.text
    assert 'value="8.0"' in page.text


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
    monkeypatch.setattr(
        "src.web.app.create_macbook_export",
        lambda *_args: archive,
    )

    response = client.get("/api/training/macbook/export")

    assert response.status_code == 200
    assert response.content == b"zip"


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
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\ncamera\r\n"

    monkeypatch.setattr("src.web.app.mjpeg_stream", fake_camera_stream)
    client = make_client(runtime_path, monkeypatch)

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

    def fake_published_stream(path: Path, fps: int):
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\nannotated\r\n"

    monkeypatch.setattr("src.web.app.mjpeg_stream", fake_camera_stream)
    monkeypatch.setattr("src.web.app.local_mjpeg_stream", fake_published_stream)
    monkeypatch.setattr("src.web.app.preview_is_fresh", lambda path: True)
    client = make_client(runtime_path, monkeypatch)

    with client.stream("GET", "/stream/face.mjpeg") as response:
        assert b"annotated" in _collect_first_frame(response)


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
    assert "/stream/face.mjpeg" in page.text
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
    assert client.get("/stream/face.mjpeg").status_code != 404


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

    assert 'class="capture-choice" data-state="observe"' in page.text
    assert page.text.count('class="capture-choice"') == 3
    # Нажатое положение приходит с сервера, а не выставляется скриптом после
    # загрузки: иначе переключатель на секунду показывал бы чужое состояние.
    assert 'data-state="observe"\n        aria-pressed="true"' in page.text


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
        lambda unit: GateStatus("absent", "служба не установлена на узле"),
    )

    page = client.get("/gate")

    assert page.status_code == 200
    # Не развёрнутый контур страница называет прямо, а не делает вид, что его
    # можно включить.
    assert "Служба Frigate на узле не установлена" in page.text
    assert 'class="circuit-choice" data-circuit="gate"' in page.text
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


def test_switching_to_the_gate_starts_its_service(runtime_path: Path, monkeypatch) -> None:
    """Пульт не только запоминает выбор, но и приводит службу к нему."""
    started: list[tuple[str, bool]] = []
    monkeypatch.setattr("src.web.app.gate_status", lambda unit: GateStatus("stopped"))
    monkeypatch.setattr(
        "src.web.app.set_gate_running",
        lambda unit, running: started.append((unit, running)) or None,
    )
    client = make_client(runtime_path, monkeypatch)

    response = client.put("/api/circuits", json={"active": "gate"})

    assert response.status_code == 200
    assert started == [("frigate.service", True)]
    assert response.json()["gate_error"] is None


def test_a_refusal_from_systemd_is_shown_not_swallowed(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """Состояние узла сохранено, служба к нему не приведена — это видно."""
    monkeypatch.setattr("src.web.app.gate_status", lambda unit: GateStatus("stopped"))
    monkeypatch.setattr(
        "src.web.app.set_gate_running",
        lambda unit, running: "sudo: a password is required",
    )
    client = make_client(runtime_path, monkeypatch)

    response = client.put("/api/circuits", json={"active": "gate"})

    assert response.status_code == 200
    assert response.json()["gate_error"] == "sudo: a password is required"
    assert load_circuits(circuits_path(runtime_path)).gate is True


def test_an_undeployed_circuit_is_not_an_error_when_left_alone(
    runtime_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("src.web.app.gate_status", lambda unit: GateStatus("absent"))
    client = make_client(runtime_path, monkeypatch)

    response = client.put("/api/circuits", json={"active": "cats"})

    assert response.json()["gate_error"] is None
    assert response.json()["gate_deployed"] is False
