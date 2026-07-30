from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import mean


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


def mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None
