from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from statistics import mean
from typing import TypeAlias

FOREIGN_LABEL = "__foreign__"
UNCERTAIN = "uncertain"
NO_PREDICTION = "no_prediction"

PredictionValue: TypeAlias = str | None


@dataclass(frozen=True)
class ScoredPrediction:
    name: str
    confidence: float


@dataclass(frozen=True)
class AggregatedPrediction:
    decision: str
    predicted_name: str | None
    confidence: float | None
    frames_used: int


@dataclass(frozen=True)
class AcceptanceMetrics:
    samples: int
    known_samples: int
    foreign_samples: int
    correct: int
    false_accepts: int
    false_rejects: int
    misidentified_known: int
    uncertain: int
    no_prediction: int
    accuracy: float
    false_accept_rate: float | None
    false_reject_rate: float | None
    uncertain_rate: float
    mean_inference_ms: float
    latency_passed: bool
    passed: bool


@dataclass(frozen=True)
class ThresholdPoint:
    threshold: float
    metrics: AcceptanceMetrics


def normalize_decision(prediction: PredictionValue) -> str:
    if prediction is None:
        return UNCERTAIN
    if prediction in {UNCERTAIN, NO_PREDICTION} or prediction.startswith("known:"):
        return prediction
    return f"known:{prediction}"


def decision_name(decision: str) -> str | None:
    return decision.removeprefix("known:") if decision.startswith("known:") else None


def aggregate_frame_predictions(
    predictions: list[ScoredPrediction],
    threshold: float,
) -> AggregatedPrediction:
    if not predictions:
        return AggregatedPrediction(
            decision="no_prediction",
            predicted_name=None,
            confidence=None,
            frames_used=0,
        )
    accepted = [
        prediction
        for prediction in predictions
        if prediction.confidence >= threshold
    ]
    if not accepted:
        strongest = max(predictions, key=lambda item: item.confidence)
        return AggregatedPrediction(
            decision="uncertain",
            predicted_name=strongest.name,
            confidence=strongest.confidence,
            frames_used=len(predictions),
        )
    counts = Counter(prediction.name for prediction in accepted)
    winner = max(
        counts,
        key=lambda name: (
            counts[name],
            max(item.confidence for item in accepted if item.name == name),
            name,
        ),
    )
    winner_scores = [
        prediction.confidence
        for prediction in accepted
        if prediction.name == winner
    ]
    return AggregatedPrediction(
        decision=f"known:{winner}",
        predicted_name=winner,
        confidence=max(winner_scores),
        frames_used=len(predictions),
    )


def calculate_metrics(
    expected: list[str],
    predicted: list[PredictionValue],
    inference_ms: list[float],
    *,
    max_far: float,
    max_frr: float,
    max_mean_inference_ms: float = float("inf"),
) -> AcceptanceMetrics:
    if not expected or len(expected) != len(predicted) or len(expected) != len(inference_ms):
        raise ValueError("expected, predicted and inference_ms must have the same non-zero length")
    decisions = [normalize_decision(value) for value in predicted]
    known_samples = sum(label != FOREIGN_LABEL for label in expected)
    foreign_samples = len(expected) - known_samples
    correct = 0
    false_accepts = 0
    false_rejects = 0
    misidentified_known = 0
    for label, decision in zip(expected, decisions, strict=True):
        predicted_name = decision_name(decision)
        if label == FOREIGN_LABEL:
            if predicted_name is None:
                correct += 1
            else:
                false_accepts += 1
            continue
        if predicted_name == label:
            correct += 1
        else:
            false_rejects += 1
            if predicted_name is not None:
                misidentified_known += 1
    uncertain = decisions.count(UNCERTAIN)
    no_prediction = decisions.count(NO_PREDICTION)
    far = false_accepts / foreign_samples if foreign_samples else None
    frr = false_rejects / known_samples if known_samples else None
    mean_inference = mean(inference_ms)
    latency_passed = mean_inference <= max_mean_inference_ms
    passed = (
        far is not None
        and frr is not None
        and far <= max_far
        and frr <= max_frr
        and latency_passed
    )
    return AcceptanceMetrics(
        samples=len(expected),
        known_samples=known_samples,
        foreign_samples=foreign_samples,
        correct=correct,
        false_accepts=false_accepts,
        false_rejects=false_rejects,
        misidentified_known=misidentified_known,
        uncertain=uncertain,
        no_prediction=no_prediction,
        accuracy=correct / len(expected),
        false_accept_rate=far,
        false_reject_rate=frr,
        uncertain_rate=uncertain / len(expected),
        mean_inference_ms=mean_inference,
        latency_passed=latency_passed,
        passed=passed,
    )


def threshold_sweep(
    scores: list[ScoredPrediction | None],
    labels: list[str],
    thresholds: list[float],
    *,
    max_far: float = 1.0,
    max_frr: float = 1.0,
) -> list[ThresholdPoint]:
    if len(scores) != len(labels) or not scores:
        raise ValueError("scores and labels must have the same non-zero length")
    points: list[ThresholdPoint] = []
    for threshold in thresholds:
        decisions: list[PredictionValue] = [
            (
                NO_PREDICTION
                if score is None
                else f"known:{score.name}"
                if score.confidence >= threshold
                else UNCERTAIN
            )
            for score in scores
        ]
        metrics = calculate_metrics(
            labels,
            decisions,
            [0.0] * len(labels),
            max_far=max_far,
            max_frr=max_frr,
        )
        points.append(
            ThresholdPoint(
                threshold=threshold,
                metrics=replace(metrics, mean_inference_ms=0.0),
            )
        )
    return points


def confusion_matrix(
    expected: list[str],
    predicted: list[PredictionValue],
) -> dict[str, dict[str, int]]:
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted must have the same length")
    result: dict[str, dict[str, int]] = {}
    for label, prediction in zip(expected, predicted, strict=True):
        decision = normalize_decision(prediction)
        row = result.setdefault(label, {})
        row[decision] = row.get(decision, 0) + 1
    return result


def mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None
