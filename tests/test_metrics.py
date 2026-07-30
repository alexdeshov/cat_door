from src.common.metrics import ScoredPrediction, aggregate_frame_predictions


def test_aggregate_frame_predictions_uses_majority_then_confidence() -> None:
    result = aggregate_frame_predictions(
        [
            ScoredPrediction("Барсик", 0.91),
            ScoredPrediction("Мурка", 0.95),
            ScoredPrediction("Барсик", 0.84),
        ],
        0.80,
    )

    assert result.decision == "known:Барсик"
    assert result.predicted_name == "Барсик"
    assert result.confidence == 0.91
    assert result.frames_used == 3


def test_aggregate_frame_predictions_distinguishes_uncertain_and_missing() -> None:
    uncertain = aggregate_frame_predictions(
        [ScoredPrediction("Барсик", 0.62)],
        0.80,
    )
    missing = aggregate_frame_predictions([], 0.80)

    assert uncertain.decision == "uncertain"
    assert uncertain.predicted_name == "Барсик"
    assert uncertain.confidence == 0.62
    assert missing.decision == "no_prediction"
    assert missing.confidence is None
