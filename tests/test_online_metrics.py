from datetime import UTC, datetime, timedelta
from pathlib import Path

from src.common.database import (
    Event,
    EventPrediction,
    ModelVersion,
    create_session_factory,
)
from src.common.online_metrics import build_online_metrics, combine_event_decisions


def _prediction(
    *,
    event_id: int,
    version_id: int,
    view: str,
    decision: str,
    frames: int = 2,
) -> EventPrediction:
    return EventPrediction(
        event_id=event_id,
        model_version_id=version_id,
        view=view,
        origin="online",
        decision=decision,
        frames_used=frames,
    )


def test_conflicting_camera_names_are_uncertain() -> None:
    assert (
        combine_event_decisions(
            [
                _prediction(
                    event_id=1,
                    version_id=1,
                    view="face",
                    decision="known:Барсик",
                ),
                _prediction(
                    event_id=1,
                    version_id=2,
                    view="side",
                    decision="known:Мурка",
                ),
            ]
        )
        == "uncertain"
    )


def test_online_metrics_use_current_models_and_rolling_windows(
    runtime_path: Path,
) -> None:
    now = datetime(2026, 7, 30, 12, tzinfo=UTC)
    factory = create_session_factory(runtime_path / "catdoor.sqlite3")
    with factory() as session:
        face = ModelVersion(
            kind="identity_face",
            file_name="face.onnx",
            sha256="a" * 64,
            metadata_json="{}",
        )
        side = ModelVersion(
            kind="identity_side",
            file_name="side.onnx",
            sha256="b" * 64,
            metadata_json="{}",
        )
        session.add_all([face, side])
        session.flush()
        known = Event(
            created_at=now - timedelta(days=2),
            duration_seconds=5,
            status="ready",
            label="known_cat",
            cat_name="Барсик",
        )
        foreign = Event(
            created_at=now - timedelta(days=10),
            duration_seconds=5,
            status="ready",
            label="foreign_cat",
        )
        pending = Event(
            created_at=now - timedelta(days=1),
            duration_seconds=5,
            status="ready",
            label="pending",
        )
        session.add_all([known, foreign, pending])
        session.flush()
        session.add_all(
            [
                _prediction(
                    event_id=known.id,
                    version_id=face.id,
                    view="face",
                    decision="known:Барсик",
                    frames=3,
                ),
                _prediction(
                    event_id=foreign.id,
                    version_id=face.id,
                    view="face",
                    decision="known:Барсик",
                ),
                _prediction(
                    event_id=known.id,
                    version_id=side.id,
                    view="side",
                    decision="uncertain",
                ),
            ]
        )
        session.commit()

        report = build_online_metrics(session, now=now)

    seven_days = report["windows"]["7_days"]
    thirty_days = report["windows"]["30_days"]
    assert seven_days["events"] == 2
    assert seven_days["pending"] == 1
    assert seven_days["metrics"]["false_reject_rate"] == 0.0
    assert seven_days["metrics"]["false_accept_rate"] is None
    assert seven_days["coverage"] == {
        "with_active_model_predictions": 1,
        "without_active_model_predictions": 0,
        "rate": 1.0,
    }
    assert seven_days["dataset_progress"]["Барсик"]["identity_frames"] == 5
    assert thirty_days["metrics"]["false_accept_rate"] == 1.0
    assert report["active_model_versions"] == {"face": face.id, "side": side.id}


def test_old_events_without_active_version_predictions_are_not_false_rejects(
    runtime_path: Path,
) -> None:
    now = datetime(2026, 7, 30, 12, tzinfo=UTC)
    factory = create_session_factory(runtime_path / "catdoor.sqlite3")
    with factory() as session:
        old_version = ModelVersion(
            kind="identity_face",
            file_name="face-old.onnx",
            sha256="c" * 64,
            metadata_json="{}",
        )
        active_version = ModelVersion(
            kind="identity_face",
            file_name="face-new.onnx",
            sha256="d" * 64,
            metadata_json="{}",
        )
        session.add_all([old_version, active_version])
        session.flush()
        old_event = Event(
            created_at=now - timedelta(days=2),
            duration_seconds=5,
            status="ready",
            label="known_cat",
            cat_name="Barsik",
        )
        current_event = Event(
            created_at=now - timedelta(days=1),
            duration_seconds=5,
            status="ready",
            label="known_cat",
            cat_name="Barsik",
        )
        session.add_all([old_event, current_event])
        session.flush()
        session.add_all(
            [
                _prediction(
                    event_id=old_event.id,
                    version_id=old_version.id,
                    view="face",
                    decision="known:Barsik",
                ),
                _prediction(
                    event_id=current_event.id,
                    version_id=active_version.id,
                    view="face",
                    decision="no_prediction",
                ),
            ]
        )
        session.commit()

        report = build_online_metrics(session, now=now)

    window = report["windows"]["7_days"]
    assert report["active_model_versions"] == {"face": active_version.id}
    assert window["labelled_identity_events"] == 2
    assert window["coverage"] == {
        "with_active_model_predictions": 1,
        "without_active_model_predictions": 1,
        "rate": 0.5,
    }
    assert window["metrics"]["samples"] == 1
    assert window["metrics"]["false_rejects"] == 1
