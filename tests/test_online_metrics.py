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


def test_a_cat_the_model_never_learned_counts_as_a_stranger(
    runtime_path: Path,
) -> None:
    """Разбор 17 августа: четыре прохода Киры, Ксюши и Джуна получили имена своих.

    Метрика их не показывала: чужими она считала только метку «чужой кот», а
    эти коты размечены своими, с именами, которых модель выдать не умеет. Их
    ошибки уходили в «перепутала своих», а доля пропущенных чужих оставалась
    `null` — ровно тот класс ошибок, ради которого метрика и заведена.
    """
    now = datetime(2026, 8, 17, 21, tzinfo=UTC)
    factory = create_session_factory(runtime_path / "catdoor.sqlite3")
    with factory() as session:
        face = ModelVersion(
            kind="identity_face",
            file_name="face.onnx",
            sha256="c" * 64,
            metadata_json="{}",
        )
        session.add(face)
        session.flush()
        stranger = Event(
            created_at=now - timedelta(hours=2),
            duration_seconds=5,
            status="ready",
            label="known_cat",
            # Свой кот для дома, чужой для модели: её обучали на двух других.
            cat_name="Ксюша",
        )
        session.add(stranger)
        session.flush()
        session.add(
            _prediction(
                event_id=stranger.id,
                version_id=face.id,
                view="face",
                decision="known:Яся",
            )
        )
        session.commit()

        blind = build_online_metrics(session, now=now)
        seeing = build_online_metrics(
            session,
            now=now,
            known_names={"Яся", "Черныш"},
        )

    assert blind["windows"]["7_days"]["metrics"]["false_accept_rate"] is None
    assert blind["windows"]["7_days"]["metrics"]["misidentified_known"] == 1
    assert seeing["windows"]["7_days"]["metrics"]["false_accept_rate"] == 1.0
    assert seeing["windows"]["7_days"]["metrics"]["false_accepts"] == 1
    assert seeing["known_names"] == ["Черныш", "Яся"]


def test_an_explicit_refusal_is_an_answer_not_silence() -> None:
    """«Не наш кот» — это решение камеры, а не молчание модели."""
    assert (
        combine_event_decisions(
            [
                _prediction(
                    event_id=1,
                    version_id=1,
                    view="face",
                    decision="known:__foreign__",
                ),
                _prediction(
                    event_id=1,
                    version_id=2,
                    view="side",
                    decision="uncertain",
                ),
            ]
        )
        == "known:__foreign__"
    )
