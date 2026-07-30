from pathlib import Path

from src.common.database import Event, create_session_factory


def test_automatic_event_metadata_is_persisted(runtime_path: Path) -> None:
    factory = create_session_factory(runtime_path / "catdoor.sqlite3")
    with factory() as session:
        event = Event(
            duration_seconds=18,
            trigger_source="automatic",
            initiating_camera="side",
            detector_confidence_face=0.72,
            detector_confidence_side=0.91,
        )
        session.add(event)
        session.commit()
        event_id = event.id

    with factory() as session:
        stored = session.get(Event, event_id)
        assert stored is not None
        assert stored.trigger_source == "automatic"
        assert stored.initiating_camera == "side"
        assert stored.detector_confidence_face == 0.72
        assert stored.detector_confidence_side == 0.91
