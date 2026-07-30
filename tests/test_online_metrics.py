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
    assert seven_days["dataset_progress"]["Барсик"]["identity_frames"] == 5
    assert thirty_days["metrics"]["false_accept_rate"] == 1.0
    assert report["active_model_versions"] == {"face": face.id, "side": side.id}
