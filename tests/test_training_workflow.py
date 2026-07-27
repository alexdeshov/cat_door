from datetime import UTC, datetime
from pathlib import Path

from src.common.database import Event, create_session_factory
from training.dataset import load_manifest
from training.workflow import build_dataset


def test_build_dataset_uses_selected_events_only(runtime_path: Path, monkeypatch) -> None:
    database_path = runtime_path / "catdoor.sqlite3"
    event_dir = runtime_path / "events" / "00000001"
    event_dir.mkdir(parents=True)
    (event_dir / "face.mp4").write_bytes(b"video")
    factory = create_session_factory(database_path)
    with factory() as session:
        session.add(
            Event(
                created_at=datetime.now(UTC),
                status="ready",
                duration_seconds=5,
                media_path="events/00000001/face.mp4",
                label="known_cat",
                cat_name="Барсик",
                training_target="local",
            )
        )
        session.add(
            Event(
                created_at=datetime.now(UTC),
                status="ready",
                duration_seconds=5,
                media_path="events/00000002/face.mp4",
                label="known_cat",
                cat_name="Не выбран",
                training_target="macbook",
            )
        )
        session.commit()

    rotations: list[bool] = []

    def fake_extract(
        ffmpeg_path: str,
        source: Path,
        destination_pattern: Path,
        *,
        rotate_180: bool,
    ) -> list[Path]:
        rotations.append(rotate_180)
        destination = destination_pattern.parent / "frame-01.jpg"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"image")
        return [destination]

    monkeypatch.setattr("training.workflow.extract_frames", fake_extract)
    destination = runtime_path / "training" / "dataset"

    manifest, report = build_dataset(
        database_path,
        runtime_path,
        "ffmpeg",
        "local",
        destination,
    )

    items = load_manifest(manifest)
    assert report["events"] == 1
    assert [item.label for item in items] == ["Барсик"]
    assert rotations == [True]
