from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import select

from src.common.database import Event, EventPrediction, create_session_factory
from src.recorder.ffmpeg import RecorderError
from src.recorder.ring import concatenate_segments, select_segments


@dataclass(frozen=True)
class AutomaticEventRequest:
    started_at: float
    ended_at: float
    initiating_camera: str | None
    confidence_face: float
    confidence_side: float
    predictions: tuple[AutomaticEventPrediction, ...] = ()
    # Момент, когда детектор был увереннее всего в коте. Нужен, чтобы плеер
    # открывал проход с него, а не с предбуфера, где кота ещё нет.
    best_frame_at: float | None = None


@dataclass(frozen=True)
class AutomaticEventPrediction:
    model_version_id: int
    view: str
    decision: str
    predicted_name: str | None
    confidence: float | None
    frames_used: int
    max_detector_confidence: float
    mean_inference_ms: float | None
    raw_scores: tuple[tuple[str, float], ...]


def expire_stale_recordings(
    database_path: Path,
    *,
    older_than: timedelta = timedelta(minutes=10),
    now: datetime | None = None,
) -> int:
    """Пометить отказом записи, которые никогда не завершатся.

    Строка со статусом `recording` создаётся до записи файлов, а финальный
    статус ставится вторым коммитом. Потеря питания или перезапуск между ними
    оставляли строку «идёт запись» навсегда — и после каждого отключения такие
    фантомы копились в ленте. Единственный момент, когда про них известно
    наверняка, — старт recorder: всё, что взялось записываться до него, уже не
    допишется.
    """
    cutoff = (now or datetime.now(UTC)) - older_than
    factory = create_session_factory(database_path)
    with factory() as session:
        stale = session.scalars(
            select(Event).where(Event.status == "recording", Event.created_at < cutoff)
        ).all()
        for event in stale:
            event.status = "failed"
            event.error_code = "interrupted"
        if stale:
            session.commit()
    return len(stale)


def _preview_offset(
    first_segment: Path,
    segment_seconds: float,
    best_frame_at: float,
) -> float | None:
    """Секунда записи, на которой детектор увидел кота лучше всего.

    Отсчёт от начала первого сегмента: сегмент считается завершённым по времени
    правки файла, а начался он на свою длину раньше — так же, как их отбирает
    `select_segments`.
    """
    try:
        segment_started_at = first_segment.stat().st_mtime - segment_seconds
    except OSError:
        return None
    offset = best_frame_at - segment_started_at
    # Отрицательное смещение означало бы кадр раньше записи — такого не бывает,
    # но округление часов и подмена сегментов теоретически могут его дать.
    return round(offset, 1) if offset >= 0 else None


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
    preview_offset: float | None = None
    for role in ("face", "side"):
        segments = select_segments(
            ring_root / role,
            request.started_at,
            request.ended_at,
            segment_seconds,
        )
        # Запись склеивается из сегментов кольцевого буфера, и её нулевая
        # секунда — начало первого из них, а не начало события. Отсчитывать
        # смещение надо именно отсюда, иначе плеер промахнётся на длину
        # предбуфера.
        if role == "face" and segments and request.best_frame_at is not None:
            preview_offset = _preview_offset(segments[0], segment_seconds, request.best_frame_at)
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
        stored_event.preview_offset_seconds = preview_offset
        for prediction in request.predictions:
            session.add(
                EventPrediction(
                    event_id=stored_event.id,
                    model_version_id=prediction.model_version_id,
                    view=prediction.view,
                    origin="online",
                    evaluation_run_id=None,
                    decision=prediction.decision,
                    predicted_name=prediction.predicted_name,
                    confidence=prediction.confidence,
                    frames_used=prediction.frames_used,
                    max_detector_confidence=prediction.max_detector_confidence,
                    mean_inference_ms=prediction.mean_inference_ms,
                    raw_scores_json=json.dumps(
                        [
                            {"name": name, "confidence": confidence}
                            for name, confidence in prediction.raw_scores
                        ],
                        ensure_ascii=False,
                    ),
                    created_at=datetime.now(UTC),
                )
            )
        session.commit()
        session.refresh(stored_event)
        return stored_event
