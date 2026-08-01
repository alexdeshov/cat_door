import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import select

from src.common.database import EvaluationRun, Event, create_session_factory
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
    assert 'class="stream"' in page.text
    assert 'class="event camera-grid"' in page.text
    assert ".camera-grid { grid-template-columns: minmax(0, 1fr) minmax(220px, 360px); }" in (
        page.text
    )
    assert 'id="swap-cameras"' in page.text
    assert 'localStorage.setItem(cameraLayoutKey, swapped ? "swapped" : "default")' in page.text
    assert page.text.count('class="rotate-camera"') == 1
    assert 'if (camera.getAttribute("src") !== source)' in page.text
    assert "function streamUrl(role)" in page.text
    assert "/stream/face.mjpeg?v=" in page.text
    assert 'cameraRotations[role] = (cameraRotations[role] + 90) % 360' in page.text
    assert 'localStorage.setItem(cameraRotationKey, JSON.stringify(cameraRotations))' in page.text
    assert 'const cameraRotationKey = "catdoor-camera-rotations-v2"' in page.text
    assert "Автоматический сбор событий" in page.text
    assert "Настройки детекции" in page.text
    assert 'id="detection-face-confirmation-frames"' in page.text
    assert 'id="detection-side-confirmation-frames"' in page.text
    assert page.text.count('class="setting-help"') == 16
    assert 'aria-label="Подсказка: рабочий confidence"' in page.text
    assert "На запуск события не влияет." in page.text
    assert "Окно не должно быть меньше числа подтверждений." in page.text
    assert 'class="setting-help" tabindex="0" role="note"' in page.text
    assert '<video preload="metadata"' in page.text
    assert "<video controls" not in page.text
    assert 'class="video-controls"' in page.text
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
    assert page.text.index("Автоматический сбор событий") < page.text.index("<h2>Камеры</h2>")
    assert "<strong>2</strong><span class=\"muted\">всего проходов</span>" in page.text
    assert "<strong>1</strong><span class=\"muted\">записано автоматически</span>" in page.text
    assert "<strong>1</strong><span class=\"muted\">ждут разметки</span>" in page.text
    assert "Известные коты: <strong>Барсик</strong>" in page.text
    assert "Независимый CatDoorRecorder ещё не запущен." in page.text


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
    assert "Кадры face/side:" in page.text
    assert "OpenVINO: GPU.0" in page.text
    assert "10/11" in page.text
    assert "0.25/0.5" in page.text

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
    assert "Без предсказаний активной версии" in page.text
    assert "Детекция по камерам" in page.text
    assert "Новая офлайн-оценка" in page.text
    assert 'option value="detector"' in page.text
    assert "Сравнить два завершённых запуска" in page.text
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
    page = client.get("/")
    assert 'id="detection-face-sample-fps"' in page.text
    assert 'id="detection-face-roi-enabled"' in page.text


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
    assert "Отчёт о процессе обучения" in page.text
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

    def fake_camera_stream(ffmpeg_path: str, rtsp_url: str, fps: int):
        used["ffmpeg_path"] = ffmpeg_path
        used["rtsp_url"] = rtsp_url
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\ncamera\r\n"

    monkeypatch.setattr("src.web.app.mjpeg_stream", fake_camera_stream)
    client = make_client(runtime_path, monkeypatch)

    with client.stream("GET", "/stream/face.mjpeg") as response:
        assert response.status_code == 200
        assert b"camera" in _collect_first_frame(response)

    assert used["rtsp_url"] == "rtsp://user:password@camera.local/media/video1"


def test_stream_prefers_annotated_frames_while_the_recorder_publishes(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """Detector boxes are more useful than a raw picture, so they win."""

    def fake_camera_stream(ffmpeg_path: str, rtsp_url: str, fps: int):
        raise AssertionError("the camera must not be opened twice")

    def fake_published_stream(path: Path, fps: int):
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\nannotated\r\n"

    monkeypatch.setattr("src.web.app.mjpeg_stream", fake_camera_stream)
    monkeypatch.setattr("src.web.app.local_mjpeg_stream", fake_published_stream)
    monkeypatch.setattr("src.web.app.preview_is_fresh", lambda path: True)
    client = make_client(runtime_path, monkeypatch)

    with client.stream("GET", "/stream/face.mjpeg") as response:
        assert b"annotated" in _collect_first_frame(response)
