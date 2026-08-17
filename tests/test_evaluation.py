from datetime import UTC, datetime

import numpy as np

from src.common.detection_config import CameraDetectionConfig
from src.common.evaluation import (
    DetectorViewEvaluation,
    EvaluatedDetectorEvent,
    EvaluatedEvent,
    build_detector_evaluation_report,
    build_identity_evaluation_report,
    evaluate_detector_frames,
    evaluate_view_frames,
)
from src.inference.cat_detector import CatDetection, CatDetectionResult
from src.inference.cat_identity import IdentityPrediction


class FakeDetector:
    def __init__(self, confidences: list[float]) -> None:
        self.confidences = iter(confidences)

    def detect(self, frame: np.ndarray) -> CatDetectionResult:
        assert frame.shape == (320, 320, 3)
        confidence = next(self.confidences)
        return CatDetectionResult(
            confidence=confidence,
            detections=(
                CatDetection(
                    box=(80.0, 100.0, 240.0, 220.0),
                    confidence=confidence,
                ),
            ),
        )


class FakeIdentity:
    def __init__(self, predictions: list[tuple[str, float]]) -> None:
        self.predictions = iter(predictions)

    def predict(self, frame: np.ndarray, threshold: float) -> IdentityPrediction:
        assert frame.size > 0
        name, confidence = next(self.predictions)
        return IdentityPrediction(
            name=name if confidence >= threshold else None,
            confidence=confidence,
            candidate_name=name,
        )


def test_evaluate_view_frames_uses_shared_crop_and_event_aggregation() -> None:
    config = CameraDetectionConfig(
        confidence_threshold=0.30,
        box_min_confidence=0.15,
        confirmation_frames=2,
        window_frames=5,
    )
    # Кадров четыре, а не два: решение по проходу принимает кворум, и на паре
    # кадров его не собрать никаким согласием — см. `MIN_QUORUM_FRAMES`.
    frames = [
        np.full((360, 640, 3), value, dtype=np.uint8) for value in (40, 80, 60, 70)
    ]

    result = evaluate_view_frames(
        frames,
        FakeDetector([0.70, 0.75, 0.72, 0.71]),
        FakeIdentity(
            [("Барсик", 0.91), ("Барсик", 0.86), ("Барсик", 0.88), ("Барсик", 0.90)]
        ),
        config,
        0.80,
    )

    assert result.aggregate.decision == "known:Барсик"
    assert result.aggregate.frames_used == 4
    assert result.aggregate.votes == 4
    assert result.max_detector_confidence == 0.75
    assert len(result.raw_scores) == 4


def test_evaluation_report_separates_clean_and_training_seen_events() -> None:
    config = CameraDetectionConfig(
        confidence_threshold=0.30,
        box_min_confidence=0.15,
        confirmation_frames=2,
        window_frames=5,
    )
    known = evaluate_view_frames(
        [np.full((360, 640, 3), 40, dtype=np.uint8)],
        FakeDetector([0.80]),
        FakeIdentity([("Барсик", 0.90)]),
        config,
        0.80,
    )
    false_accept = evaluate_view_frames(
        [np.full((360, 640, 3), 60, dtype=np.uint8)],
        FakeDetector([0.20]),
        FakeIdentity([]),
        config,
        0.80,
    )
    events = [
        EvaluatedEvent(
            event_id=1,
            label="known_cat",
            cat_name="Барсик",
            training_target="excluded",
            created_at=datetime(2026, 7, 30, 12, tzinfo=UTC),
            views={"face": known},
        ),
        EvaluatedEvent(
            event_id=2,
            label="foreign_cat",
            cat_name=None,
            training_target="macbook",
            created_at=datetime(2026, 7, 30, 23, tzinfo=UTC),
            views={"face": false_accept},
        ),
    ]

    report = build_identity_evaluation_report(
        events,
        ["face"],
        {"face": 0.30},
        identity_threshold=0.80,
    )

    assert report["events_processed"] == 2
    assert report["training_seen_events"] == 1
    assert report["clean_events"] == 1
    assert report["detection"]["face"]["misses"] == 1
    assert report["metrics"]["face"]["false_accept_rate"] == 0.0
    assert report["clean_metrics"]["face"]["false_accept_rate"] is None


def test_detector_report_separates_misses_false_triggers_and_night() -> None:
    config = CameraDetectionConfig(
        confidence_threshold=0.30,
        box_min_confidence=0.15,
        confirmation_frames=2,
        window_frames=5,
    )
    frame = np.full((360, 640, 3), 40, dtype=np.uint8)
    cat_seen = evaluate_detector_frames([frame], FakeDetector([0.80]), config)
    cat_missed = evaluate_detector_frames([frame], FakeDetector([0.20]), config)
    false_trigger = evaluate_detector_frames([frame], FakeDetector([0.60]), config)
    events = [
        EvaluatedDetectorEvent(
            event_id=1,
            label="known_cat",
            training_target="excluded",
            created_at=datetime(2026, 7, 30, 12, tzinfo=UTC),
            views={"face": cat_seen},
        ),
        EvaluatedDetectorEvent(
            event_id=2,
            label="foreign_cat",
            training_target="excluded",
            created_at=datetime(2026, 7, 30, 23, tzinfo=UTC),
            views={"face": cat_missed},
        ),
        EvaluatedDetectorEvent(
            event_id=3,
            label="not_visible",
            training_target="excluded",
            created_at=datetime(2026, 7, 30, 23, tzinfo=UTC),
            views={"face": false_trigger},
        ),
    ]

    report = build_detector_evaluation_report(
        events,
        ["face"],
        {"face": 0.30},
    )
    all_metrics = report["sections"]["all"]["face"]
    night_metrics = report["sections"]["night"]["face"]

    assert report["run_kind"] == "detector"
    assert all_metrics["miss_rate"] == 0.5
    assert all_metrics["false_trigger_rate"] == 1.0
    assert all_metrics["confidence_distribution"]["cat"]["median"] == 0.5
    assert night_metrics["miss_rate"] == 1.0
    assert report["threshold_sweep"]["face"][5]["threshold"] == 0.3


def test_detector_sweep_counts_every_pass_not_only_the_clean_ones() -> None:
    """Развёртка порогов детектора строится по всем проходам.

    `training_contaminated` помечает события, ушедшие в обучение **модели
    опознания**; детектор — сток YOLO11 на COCO, нашего материала не видел.
    Цена ошибки измерена 2026-08-09: из 189 проходов с котами «чистыми» были
    два, и таблица «пропуски на равной доле ложных» держалась на одном коте.
    """
    events = [
        EvaluatedDetectorEvent(
            event_id=index,
            label="known_cat" if index % 2 else "not_visible",
            # Почти всё ушло в обучение опознания — как на живом узле.
            training_target="macbook" if index > 1 else "excluded",
            created_at=datetime(2026, 8, 9, 12, tzinfo=UTC),
            views={
                "face": DetectorViewEvaluation(
                    frame_confidences=(0.9,),
                    frames_seen=1,
                    max_detector_confidence=0.9 if index % 2 else 0.1,
                    mean_detector_inference_ms=10.0,
                )
            },
        )
        for index in range(20)
    ]

    report = build_detector_evaluation_report(events, ["face"], {"face": 0.5})

    point = report["threshold_sweep"]["face"][0]
    assert point["cat_events"] == 10
    assert point["not_visible_events"] == 10
    # «Чистых» всего два — если бы развёртка считалась по ним, было бы 1 и 1.
    assert report["clean_events"] == 2
