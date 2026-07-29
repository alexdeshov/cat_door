from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from src.common.database import Event, create_session_factory
from src.recorder.ffmpeg import RecorderError
from src.recorder.ring import concatenate_segments, select_segments


@dataclass(frozen=True)
class AutomaticEventRequest:
    started_at: float
    ended_at: float
    initiating_camera: str | None
    confidence_face: float
    confidence_side: float


def persist_automatic_event(
    database_path: Path,
    data_dir: Path,
    ffmpeg_path: str,
    segment_seconds: int,
    request: AutomaticEventRequest,
) -> Event:
    duration = max(1, math.ceil(request.ended_at - request.started_at))
    factory = create_session_factory(database_path)
    with factory() as session:
        event = Event(
            duration_seconds=duration,
            status="recording",
            trigger_source="automatic",
            initiating_camera=request.initiating_camera,
            detector_confidence_face=request.confidence_face,
            detector_confidence_side=request.confidence_side,
        )
        session.add(event)
        session.commit()
        event_id = event.id

    event_dir = data_dir / "events" / f"{event_id:08d}"
    ring_root = data_dir / "runtime" / "ring"
    completed: dict[str, Path] = {}
    errors: list[str] = []
    for role in ("face", "side"):
        segments = select_segments(
            ring_root / role,
            request.started_at,
            request.ended_at,
            segment_seconds,
        )
        destination = event_dir / f"{role}.mp4"
        try:
            concatenate_segments(ffmpeg_path, segments, destination)
            completed[role] = destination
        except RecorderError:
            errors.append(role)

    with factory() as session:
        stored_event = session.get(Event, event_id)
        assert stored_event is not None
        if "face" in completed:
            stored_event.media_path = completed["face"].relative_to(data_dir).as_posix()
        if "side" in completed:
            stored_event.side_media_path = completed["side"].relative_to(data_dir).as_posix()
        stored_event.status = (
            "ready" if not errors else ("partial" if completed else "failed")
        )
        stored_event.error_code = (
            f"automatic_capture_failed:{','.join(errors)}" if errors else None
        )
        session.commit()
        session.refresh(stored_event)
        return stored_event
