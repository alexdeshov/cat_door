import os
from pathlib import Path

from sqlalchemy import select

from src.common.database import EventPrediction, create_session_factory
from src.common.model_registry import register_model_version
from src.recorder.events import (
    AutomaticEventPrediction,
    AutomaticEventRequest,
    persist_automatic_event,
)


def make_segment(path: Path, finalized_at: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"segment")
    os.utime(path, (finalized_at, finalized_at))


def test_persists_automatic_event_from_both_ring_buffers(
    runtime_path: Path,
    monkeypatch,
) -> None:
    for role in ("face", "side"):
        make_segment(runtime_path / "runtime" / "ring" / role / "segment.mp4", 102.0)

    def fake_concat(ffmpeg_path: str, segments: list[Path], destination: Path) -> None:
        assert segments
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"combined")

    monkeypatch.setattr("src.recorder.events.concatenate_segments", fake_concat)

    event = persist_automatic_event(
        runtime_path / "catdoor.sqlite3",
        runtime_path,
        "ffmpeg",
        2,
        AutomaticEventRequest(
            started_at=100.0,
            ended_at=103.2,
            initiating_camera="face",
            confidence_face=0.91,
            confidence_side=0.65,
        ),
    )

    assert event.status == "ready"
    assert event.duration_seconds == 4
    assert event.trigger_source == "automatic"
    assert event.initiating_camera == "face"
    assert event.media_path == "events/00000001/face.mp4"
    assert event.side_media_path == "events/00000001/side.mp4"


def test_persists_partial_event_when_one_camera_has_no_segments(
    runtime_path: Path,
    monkeypatch,
) -> None:
    make_segment(runtime_path / "runtime" / "ring" / "face" / "segment.mp4", 102.0)

    def fake_concat(ffmpeg_path: str, segments: list[Path], destination: Path) -> None:
        if not segments:
            from src.recorder.ffmpeg import RecorderError

            raise RecorderError("ring_segments_missing")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"combined")

    monkeypatch.setattr("src.recorder.events.concatenate_segments", fake_concat)

    event = persist_automatic_event(
        runtime_path / "catdoor.sqlite3",
        runtime_path,
        "ffmpeg",
        2,
        AutomaticEventRequest(100.0, 103.0, "face", 0.9, 0.0),
    )

    assert event.status == "partial"
    assert event.media_path is not None
    assert event.side_media_path is None
    assert event.error_code == "automatic_capture_failed:side"


def test_persists_aggregated_online_prediction(
    runtime_path: Path,
    monkeypatch,
) -> None:
    make_segment(runtime_path / "runtime" / "ring" / "face" / "segment.mp4", 102.0)
    model_path = runtime_path / "face.onnx"
    model_path.write_bytes(b"identity-model")
    model_path.with_suffix(".json").write_text(
        '{"labels":["Барсик"],"image_size":160}',
        encoding="utf-8",
    )
    database_path = runtime_path / "catdoor.sqlite3"
    factory = create_session_factory(database_path)
    with factory() as session:
        version = register_model_version(
            session,
            kind="identity_face",
            model_path=model_path,
        )
        session.commit()
        version_id = version.id

    def fake_concat(ffmpeg_path: str, segments: list[Path], destination: Path) -> None:
        if not segments:
            from src.recorder.ffmpeg import RecorderError

            raise RecorderError("ring_segments_missing")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"combined")

    monkeypatch.setattr("src.recorder.events.concatenate_segments", fake_concat)
    event = persist_automatic_event(
        database_path,
        runtime_path,
        "ffmpeg",
        2,
        AutomaticEventRequest(
            100.0,
            103.0,
            "face",
            0.9,
            0.0,
            predictions=(
                AutomaticEventPrediction(
                    model_version_id=version_id,
                    view="face",
                    decision="known:Барсик",
                    predicted_name="Барсик",
                    confidence=0.91,
                    frames_used=2,
                    max_detector_confidence=0.9,
                    mean_inference_ms=8.5,
                    raw_scores=(("Барсик", 0.86), ("Барсик", 0.91)),
                ),
            ),
        ),
    )

    with factory() as session:
        prediction = session.scalar(
            select(EventPrediction).where(EventPrediction.event_id == event.id)
        )
        assert prediction is not None
        assert prediction.origin == "online"
        assert prediction.decision == "known:Барсик"
        assert prediction.frames_used == 2
