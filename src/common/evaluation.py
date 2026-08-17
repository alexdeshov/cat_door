from __future__ import annotations

import time
from collections.abc import Collection
from dataclasses import asdict, dataclass
from datetime import datetime
from statistics import mean
from typing import Protocol

import numpy as np

from src.common.detection_config import CameraDetectionConfig
from src.common.metrics import (
    DEFAULT_IDENTITY_QUORUM,
    FOREIGN_LABEL,
    NOCAT_LABEL,
    AggregatedPrediction,
    ScoredPrediction,
    aggregate_frame_predictions,
    as_expected_label,
    calculate_metrics,
    confusion_matrix,
    threshold_sweep,
)
from src.inference.cat_detector import CatDetectionResult
from src.inference.cat_identity import IdentityPrediction
from src.inference.preprocessing import (
    DETECTOR_INPUT_SIZE,
    crop_normalized_roi,
    crop_strongest_cat,
    prepare_detector_frame,
)


class CatDetector(Protocol):
    def detect(self, rgb_frame: np.ndarray) -> CatDetectionResult: ...


class IdentityModel(Protocol):
    def predict(self, frame: np.ndarray, threshold: float) -> IdentityPrediction: ...


@dataclass(frozen=True)
class ViewEvaluation:
    aggregate: AggregatedPrediction
    raw_scores: tuple[ScoredPrediction, ...]
    frames_seen: int
    max_detector_confidence: float
    mean_detector_inference_ms: float | None
    mean_identity_inference_ms: float | None


@dataclass(frozen=True)
class EvaluatedEvent:
    event_id: int
    label: str
    cat_name: str | None
    training_target: str
    created_at: datetime
    views: dict[str, ViewEvaluation]

    def expected_identity(
        self,
        known_names: Collection[str] | None = None,
    ) -> str | None:
        """Чего ждать от модели на этом проходе.

        `known_names` — коты, которых модель умеет называть. Кот, размеченный
        своим, но модели незнакомый, для неё чужой: считать иначе — значит
        прятать пропущенных чужих в «перепутала своих». См. `as_expected_label`.
        """
        if self.label == "known_cat":
            return as_expected_label(self.cat_name, known_names) if self.cat_name else None
        if self.label == "foreign_cat":
            return FOREIGN_LABEL
        if self.label == "not_visible":
            return NOCAT_LABEL
        return None

    @property
    def training_contaminated(self) -> bool:
        return self.training_target in {"macbook", "local"}

    @property
    def is_night(self) -> bool:
        return self.created_at.hour >= 20 or self.created_at.hour < 7


@dataclass(frozen=True)
class DetectorViewEvaluation:
    frame_confidences: tuple[float, ...]
    frames_seen: int
    max_detector_confidence: float
    mean_detector_inference_ms: float | None


@dataclass(frozen=True)
class EvaluatedDetectorEvent:
    event_id: int
    label: str
    training_target: str
    created_at: datetime
    views: dict[str, DetectorViewEvaluation]

    @property
    def training_contaminated(self) -> bool:
        return self.training_target in {"macbook", "local"}

    @property
    def is_night(self) -> bool:
        return self.created_at.hour >= 20 or self.created_at.hour < 7


def empty_view_evaluation() -> ViewEvaluation:
    return ViewEvaluation(
        aggregate=aggregate_frame_predictions([], 1.0),
        raw_scores=(),
        frames_seen=0,
        max_detector_confidence=0.0,
        mean_detector_inference_ms=None,
        mean_identity_inference_ms=None,
    )


def evaluate_view_frames(
    frames: list[np.ndarray],
    detector: CatDetector,
    identity: IdentityModel,
    camera_config: CameraDetectionConfig,
    identity_threshold: float,
    quorum: float = DEFAULT_IDENTITY_QUORUM,
) -> ViewEvaluation:
    raw_scores: list[ScoredPrediction] = []
    detector_timings: list[float] = []
    identity_timings: list[float] = []
    max_detector_confidence = 0.0
    detector_input_size = int(
        getattr(detector, "input_size", DETECTOR_INPUT_SIZE)
    )
    detector_letterbox_value = int(getattr(detector, "letterbox_value", 0))
    detector_resize_mode = str(getattr(detector, "resize_mode", "letterbox"))
    detector_letterbox_alignment = str(
        getattr(detector, "letterbox_alignment", "center")
    )
    for frame in frames:
        detector_source = crop_normalized_roi(frame, camera_config.roi)
        detector_frame = prepare_detector_frame(
            detector_source,
            detector_input_size,
            detector_letterbox_value,
            detector_resize_mode,
            detector_letterbox_alignment,
        )
        started_at = time.perf_counter()
        detection = detector.detect(detector_frame)
        detector_timings.append((time.perf_counter() - started_at) * 1000)
        max_detector_confidence = max(
            max_detector_confidence,
            detection.confidence,
        )
        crop = crop_strongest_cat(
            frame,
            detection.detections,
            min_confidence=camera_config.confidence_threshold,
            detector_frame=detector_source,
            roi=camera_config.roi,
            input_size=detector_input_size,
            resize_mode=detector_resize_mode,
            letterbox_alignment=detector_letterbox_alignment,
        )
        if crop is None:
            continue
        started_at = time.perf_counter()
        prediction = identity.predict(crop, identity_threshold)
        identity_timings.append((time.perf_counter() - started_at) * 1000)
        candidate_name = prediction.candidate_name or prediction.name
        if candidate_name is not None:
            raw_scores.append(
                ScoredPrediction(
                    name=candidate_name,
                    confidence=prediction.confidence,
                )
            )
    return ViewEvaluation(
        aggregate=aggregate_frame_predictions(raw_scores, identity_threshold, quorum=quorum),
        raw_scores=tuple(raw_scores),
        frames_seen=len(frames),
        max_detector_confidence=max_detector_confidence,
        mean_detector_inference_ms=(
            mean(detector_timings) if detector_timings else None
        ),
        mean_identity_inference_ms=(
            mean(identity_timings) if identity_timings else None
        ),
    )


def evaluate_detector_frames(
    frames: list[np.ndarray],
    detector: CatDetector,
    camera_config: CameraDetectionConfig,
) -> DetectorViewEvaluation:
    confidences: list[float] = []
    timings: list[float] = []
    detector_input_size = int(
        getattr(detector, "input_size", DETECTOR_INPUT_SIZE)
    )
    detector_letterbox_value = int(getattr(detector, "letterbox_value", 0))
    detector_resize_mode = str(getattr(detector, "resize_mode", "letterbox"))
    detector_letterbox_alignment = str(
        getattr(detector, "letterbox_alignment", "center")
    )
    for frame in frames:
        detector_source = crop_normalized_roi(frame, camera_config.roi)
        detector_frame = prepare_detector_frame(
            detector_source,
            detector_input_size,
            detector_letterbox_value,
            detector_resize_mode,
            detector_letterbox_alignment,
        )
        started_at = time.perf_counter()
        detection = detector.detect(detector_frame)
        timings.append((time.perf_counter() - started_at) * 1000)
        confidences.append(detection.confidence)
    return DetectorViewEvaluation(
        frame_confidences=tuple(confidences),
        frames_seen=len(frames),
        max_detector_confidence=max(confidences, default=0.0),
        mean_detector_inference_ms=mean(timings) if timings else None,
    )


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "samples": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "max": None,
        }
    return {
        "samples": len(values),
        "min": min(values),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": max(values),
    }


def _detector_view_metrics(
    events: list[EvaluatedDetectorEvent],
    view: str,
    threshold: float,
) -> dict[str, object]:
    with_media = [
        event
        for event in events
        if view in event.views and event.views[view].frames_seen > 0
    ]
    cat_events = [
        event
        for event in with_media
        if event.label in {"known_cat", "foreign_cat"}
    ]
    not_visible_events = [
        event for event in with_media if event.label == "not_visible"
    ]
    cat_confidences = [
        event.views[view].max_detector_confidence for event in cat_events
    ]
    not_visible_confidences = [
        event.views[view].max_detector_confidence
        for event in not_visible_events
    ]
    all_confidences = [
        event.views[view].max_detector_confidence for event in with_media
    ]
    misses = sum(confidence < threshold for confidence in cat_confidences)
    false_triggers = sum(
        confidence >= threshold for confidence in not_visible_confidences
    )
    timings = [
        timing
        for event in with_media
        if (timing := event.views[view].mean_detector_inference_ms) is not None
    ]
    return {
        "events": len(with_media),
        "missing_media": len(events) - len(with_media),
        "cat_events": len(cat_events),
        "misses": misses,
        "miss_rate": misses / len(cat_events) if cat_events else None,
        "not_visible_events": len(not_visible_events),
        "false_triggers": false_triggers,
        "false_trigger_rate": (
            false_triggers / len(not_visible_events)
            if not_visible_events
            else None
        ),
        "threshold": threshold,
        "median_inference_ms": float(np.median(timings)) if timings else None,
        "p95_inference_ms": float(np.percentile(timings, 95)) if timings else None,
        "confidence_distribution": {
            "all": _distribution(all_confidences),
            "cat": _distribution(cat_confidences),
            "not_visible": _distribution(not_visible_confidences),
        },
    }


def build_detector_evaluation_report(
    events: list[EvaluatedDetectorEvent],
    evaluated_views: list[str],
    camera_thresholds: dict[str, float],
) -> dict[str, object]:
    sections = {
        "all": events,
        "clean": [event for event in events if not event.training_contaminated],
        "night": [event for event in events if event.is_night],
    }
    section_metrics = {
        section: {
            view: _detector_view_metrics(selected, view, camera_thresholds[view])
            for view in evaluated_views
        }
        for section, selected in sections.items()
    }
    sweep_thresholds = [round(value / 100, 2) for value in range(5, 96, 5)]
    threshold_sweep: dict[str, list[dict[str, float | int | None]]] = {}
    # Развёртка строится по ВСЕМ проходам, а не по «чистым». `training_contaminated`
    # помечает события, ушедшие в обучение **модели опознания**; детектор же —
    # сток YOLO11 на COCO, он нашего материала не видел никогда, и исключать эти
    # события из его оценки незачем.
    #
    # Цена ошибки измерена 2026-08-09: из 189 проходов с котами «чистыми» были
    # два. Развёртка честно считалась по ним, и таблица «пропуски на равной доле
    # ложных» выглядела осмысленно, будучи одним котом из двух. Такое не
    # проявляется ничем, кроме подозрительно круглых 50% и 0%.
    for view in evaluated_views:
        valid = [
            event
            for event in events
            if view in event.views and event.views[view].frames_seen > 0
        ]
        cat_confidences = [
            event.views[view].max_detector_confidence
            for event in valid
            if event.label in {"known_cat", "foreign_cat"}
        ]
        not_visible_confidences = [
            event.views[view].max_detector_confidence
            for event in valid
            if event.label == "not_visible"
        ]
        threshold_sweep[view] = [
            {
                "threshold": threshold,
                "miss_rate": (
                    sum(value < threshold for value in cat_confidences)
                    / len(cat_confidences)
                    if cat_confidences
                    else None
                ),
                "false_trigger_rate": (
                    sum(value >= threshold for value in not_visible_confidences)
                    / len(not_visible_confidences)
                    if not_visible_confidences
                    else None
                ),
                "cat_events": len(cat_confidences),
                "not_visible_events": len(not_visible_confidences),
            }
            for threshold in sweep_thresholds
        ]
    return {
        "run_kind": "detector",
        "events_processed": len(events),
        "cat_events": sum(
            event.label in {"known_cat", "foreign_cat"} for event in events
        ),
        "not_visible_events": sum(
            event.label == "not_visible" for event in events
        ),
        "clean_events": len(sections["clean"]),
        "night_events": len(sections["night"]),
        "sections": section_metrics,
        "threshold_sweep": threshold_sweep,
    }


def combine_view_evaluations(
    views: list[ViewEvaluation],
    identity_threshold: float,
    *,
    quorum: float = DEFAULT_IDENTITY_QUORUM,
) -> ViewEvaluation:
    scores = [score for view in views for score in view.raw_scores]
    detector_timings = [
        view.mean_detector_inference_ms
        for view in views
        if view.mean_detector_inference_ms is not None
    ]
    identity_timings = [
        view.mean_identity_inference_ms
        for view in views
        if view.mean_identity_inference_ms is not None
    ]
    return ViewEvaluation(
        aggregate=aggregate_frame_predictions(scores, identity_threshold, quorum=quorum),
        raw_scores=tuple(scores),
        frames_seen=sum(view.frames_seen for view in views),
        max_detector_confidence=max(
            (view.max_detector_confidence for view in views),
            default=0.0,
        ),
        mean_detector_inference_ms=(
            mean(detector_timings) if detector_timings else None
        ),
        mean_identity_inference_ms=(
            mean(identity_timings) if identity_timings else None
        ),
    )


def _identity_metrics(
    events: list[EvaluatedEvent],
    view: str,
    thresholds: list[float],
    known_names: Collection[str] | None = None,
) -> dict[str, object] | None:
    labelled = [
        event
        for event in events
        if event.expected_identity(known_names) is not None and view in event.views
    ]
    if not labelled:
        return None
    expected = [
        expectation
        for event in labelled
        if (expectation := event.expected_identity(known_names)) is not None
    ]
    decisions = [event.views[view].aggregate.decision for event in labelled]
    inference_ms = [
        event.views[view].mean_identity_inference_ms or 0.0
        for event in labelled
    ]
    metrics = calculate_metrics(
        expected,
        decisions,
        inference_ms,
        max_far=1.0,
        max_frr=1.0,
    )
    scores = [
        (
            ScoredPrediction(
                name=result.aggregate.predicted_name,
                confidence=result.aggregate.confidence,
            )
            if result.aggregate.predicted_name is not None
            and result.aggregate.confidence is not None
            else None
        )
        for result in (event.views[view] for event in labelled)
    ]
    return {
        **asdict(metrics),
        "confusion": confusion_matrix(expected, decisions),
        "threshold_sweep": [
            {"threshold": point.threshold, **asdict(point.metrics)}
            for point in threshold_sweep(scores, expected, thresholds)
        ],
    }


def build_identity_evaluation_report(
    events: list[EvaluatedEvent],
    evaluated_views: list[str],
    camera_thresholds: dict[str, float],
    *,
    identity_threshold: float,
    quorum: float = DEFAULT_IDENTITY_QUORUM,
    known_names: Collection[str] | None = None,
) -> dict[str, object]:
    enriched: list[EvaluatedEvent] = []
    for event in events:
        existing = [event.views[view] for view in evaluated_views if view in event.views]
        views = dict(event.views)
        views["combined"] = combine_view_evaluations(
            existing,
            identity_threshold,
            quorum=quorum,
        )
        enriched.append(
            EvaluatedEvent(
                event_id=event.event_id,
                label=event.label,
                cat_name=event.cat_name,
                training_target=event.training_target,
                created_at=event.created_at,
                views=views,
            )
        )
    sweep_thresholds = [round(value / 100, 2) for value in range(50, 96, 5)]
    metrics_by_view = {
        view: _identity_metrics(enriched, view, sweep_thresholds, known_names)
        for view in [*evaluated_views, "combined"]
    }
    clean_events = [event for event in enriched if not event.training_contaminated]
    clean_metrics = {
        view: _identity_metrics(clean_events, view, sweep_thresholds, known_names)
        for view in [*evaluated_views, "combined"]
    }
    night_events = [event for event in enriched if event.is_night]
    night_metrics = {
        view: _identity_metrics(night_events, view, sweep_thresholds, known_names)
        for view in [*evaluated_views, "combined"]
    }
    detection: dict[str, dict[str, float | int | None]] = {}
    cat_events = [event for event in enriched if event.label != "not_visible"]
    for view in evaluated_views:
        threshold = camera_thresholds[view]
        seen = [
            event.views.get(view, empty_view_evaluation()).max_detector_confidence
            >= threshold
            for event in cat_events
        ]
        timings = [
            result.mean_detector_inference_ms
            for event in enriched
            if (result := event.views.get(view)) is not None
            and result.mean_detector_inference_ms is not None
        ]
        detection[view] = {
            "events": len(seen),
            "misses": seen.count(False),
            "miss_rate": seen.count(False) / len(seen) if seen else None,
            "mean_inference_ms": mean(timings) if timings else None,
            "threshold": threshold,
        }
    return {
        "run_kind": "identity",
        "events_processed": len(enriched),
        "identity_events": sum(
            event.expected_identity(known_names) is not None for event in enriched
        ),
        "training_seen_events": sum(
            event.training_contaminated for event in enriched
        ),
        "clean_events": len(clean_events),
        "identity_threshold": identity_threshold,
        "metrics": metrics_by_view,
        "clean_metrics": clean_metrics,
        "night_metrics": night_metrics,
        "detection": detection,
    }
