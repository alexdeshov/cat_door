from pathlib import Path

from fastapi.testclient import TestClient

from src.common.database import Event
from src.common.settings import Settings
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
        json={"label": "known_cat", "cat_name": "Барсик"},
    )
    assert label.status_code == 200
    assert label.json()["cat_name"] == "Барсик"

    page = client.get("/")
    assert page.status_code == 200
    assert "Барсик" in page.text
    assert 'class="stream rotate-180"' in page.text


def test_known_cat_requires_name(runtime_path: Path, monkeypatch) -> None:
    client = make_client(runtime_path, monkeypatch)
    client.post("/api/events", json={"duration_seconds": 5})

    response = client.post(
        "/api/events/1/label",
        json={"label": "known_cat", "cat_name": None},
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
