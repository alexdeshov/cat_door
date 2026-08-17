from __future__ import annotations

from collections.abc import Collection
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from statistics import mean
from typing import Any

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from src.common.database import Event, EventPrediction
from src.common.metrics import (
    FOREIGN_LABEL,
    NO_PREDICTION,
    REJECT_LABELS,
    UNCERTAIN,
    as_expected_label,
    calculate_metrics,
    decision_name,
)


def combine_event_decisions(predictions: list[EventPrediction]) -> str:
    named = {
        name
        for prediction in predictions
        if (name := decision_name(prediction.decision)) is not None
        and name not in REJECT_LABELS
    }
    if len(named) == 1:
        return f"known:{next(iter(named))}"
    if len(named) > 1:
        return UNCERTAIN
    # Явный отказ «это не наш кот» — не молчание: камера посмотрела и ответила.
    if any(
        decision_name(prediction.decision) in REJECT_LABELS
        for prediction in predictions
    ):
        return f"known:{FOREIGN_LABEL}"
    if any(prediction.decision == UNCERTAIN for prediction in predictions):
        return UNCERTAIN
    return NO_PREDICTION


def _latest_online_model_versions(session: Session) -> dict[str, int]:
    predictions = session.scalars(
        select(EventPrediction)
        .where(EventPrediction.origin == "online")
        .order_by(EventPrediction.created_at.desc(), EventPrediction.id.desc())
    ).all()
    versions: dict[str, int] = {}
    for prediction in predictions:
        versions.setdefault(prediction.view, prediction.model_version_id)
    return versions


def _window_report(
    events: list[Event],
    predictions: list[EventPrediction],
    known_names: Collection[str] | None = None,
) -> dict[str, Any]:
    predictions_by_event: dict[int, list[EventPrediction]] = {}
    for prediction in predictions:
        predictions_by_event.setdefault(prediction.event_id, []).append(prediction)
    labelled = [
        event
        for event in events
        if event.label in {"known_cat", "foreign_cat"}
    ]
    expected: list[str] = []
    decisions: list[str] = []
    inference_ms: list[float] = []
    labelled_events = 0
    for event in labelled:
        expected_name = (
            as_expected_label(event.cat_name, known_names)
            if event.label == "known_cat"
            else FOREIGN_LABEL
        )
        if event.label == "known_cat" and not event.cat_name:
            # Метка «свой» без имени: разметка не закончена, судить не о чем.
            continue
        event_predictions = predictions_by_event.get(event.id, [])
        labelled_events += 1
        if not event_predictions:
            continue
        expected.append(expected_name)
        decisions.append(combine_event_decisions(event_predictions))
        timings = [
            prediction.mean_inference_ms
            for prediction in event_predictions
            if prediction.mean_inference_ms is not None
        ]
        inference_ms.append(mean(timings) if timings else 0.0)
    metrics = (
        asdict(
            calculate_metrics(
                expected,
                decisions,
                inference_ms,
                max_far=1.0,
                max_frr=1.0,
            )
        )
        if expected
        else None
    )
    progress: dict[str, dict[str, int]] = {}
    for event in events:
        if event.label != "known_cat" or not event.cat_name:
            continue
        item = progress.setdefault(
            event.cat_name,
            {"events": 0, "identity_frames": 0},
        )
        item["events"] += 1
        item["identity_frames"] += sum(
            prediction.frames_used
            for prediction in predictions_by_event.get(event.id, [])
        )
    return {
        "events": len(events),
        "pending": sum(event.label == "pending" for event in events),
        "labelled_identity_events": labelled_events,
        "coverage": {
            "with_active_model_predictions": len(expected),
            "without_active_model_predictions": labelled_events - len(expected),
            "rate": len(expected) / labelled_events if labelled_events else None,
        },
        "metrics": metrics,
        "dataset_progress": dict(sorted(progress.items())),
    }


def build_online_metrics(
    session: Session,
    *,
    now: datetime | None = None,
    known_names: Collection[str] | None = None,
) -> dict[str, Any]:
    """Как установленная модель ведёт себя на размеченных проходах узла.

    `known_names` — коты, которых модель обучалась узнавать (метки из её
    метаданных). Все прочие считаются чужими: см. `as_expected_label`.
    """
    current_time = now or datetime.now(UTC)
    active_versions = _latest_online_model_versions(session)
    events = session.scalars(select(Event).order_by(Event.created_at)).all()
    if active_versions:
        version_filters = [
            and_(
                EventPrediction.view == view,
                EventPrediction.model_version_id == version_id,
            )
            for view, version_id in active_versions.items()
        ]
        predictions = session.scalars(
            select(EventPrediction).where(
                EventPrediction.origin == "online",
                or_(*version_filters),
            )
        ).all()
    else:
        predictions = []
    windows: dict[str, int | None] = {"7_days": 7, "30_days": 30, "all": None}
    reports: dict[str, dict[str, Any]] = {}
    for name, days in windows.items():
        cutoff = current_time - timedelta(days=days) if days is not None else None
        selected_events = [
            event
            for event in events
            if cutoff is None or _as_utc(event.created_at) >= cutoff
        ]
        selected_ids = {event.id for event in selected_events}
        selected_predictions = [
            prediction
            for prediction in predictions
            if prediction.event_id in selected_ids
        ]
        reports[name] = _window_report(
            selected_events,
            selected_predictions,
            known_names,
        )
    return {
        "generated_at": current_time.isoformat(),
        "active_model_versions": active_versions,
        # Пусто — метаданные моделей не прочитались; тогда доли считаются
        # по-старому, и это видно снаружи, а не только в коде.
        "known_names": sorted(known_names) if known_names is not None else None,
        "windows": reports,
    }


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
