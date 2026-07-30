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
