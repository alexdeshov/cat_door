from src.common.metrics import (
    ScoredPrediction,
    aggregate_frame_predictions,
    confusion_matrix,
    threshold_sweep,
)


def test_aggregate_frame_predictions_uses_majority_then_confidence() -> None:
    result = aggregate_frame_predictions(
        [
            ScoredPrediction("Барсик", 0.91),
            ScoredPrediction("Мурка", 0.95),
            ScoredPrediction("Барсик", 0.84),
            ScoredPrediction("Барсик", 0.88),
            ScoredPrediction("Барсик", 0.90),
        ],
        0.80,
    )

    assert result.decision == "known:Барсик"
    assert result.predicted_name == "Барсик"
    # Медиана Барсика, а не его лучший кадр: максимум — это самый удачный кроп
    # из всех, и именно он выдавал чужих котов за своих.
    assert result.confidence == 0.89
    assert result.frames_used == 5
    assert result.votes == 4


def test_aggregate_frame_predictions_needs_a_quorum_not_one_lucky_frame() -> None:
    """Разбор 17 августа: чужой кот получил имя по 56 кадрам из 154.

    Прежнее правило называло кота, если порог перешагнул хоть один кроп. При
    сотне кропов за проход это делало редкую ошибку почти неизбежной.
    """
    scores = [ScoredPrediction("Яся", 0.95)] * 4 + [ScoredPrediction("Яся", 0.10)] * 8

    result = aggregate_frame_predictions(scores, 0.80)

    assert result.decision == "uncertain"
    assert result.votes == 4
    assert result.frames_used == 12
    # Показываем самый сильный кадр: «не уверена (0.95)» объясняет отказ.
    assert result.confidence == 0.95


def test_aggregate_frame_predictions_keeps_a_floor_under_the_quorum() -> None:
    """Половина от двух кадров — это один кадр, то есть прежнее правило."""
    result = aggregate_frame_predictions(
        [ScoredPrediction("Яся", 0.95), ScoredPrediction("Яся", 0.10)],
        0.80,
    )

    assert result.decision == "uncertain"


def test_aggregate_frame_predictions_does_not_hand_names_to_reject_labels() -> None:
    """«Чужой» — это отказ, а не кот, которого можно назвать."""
    result = aggregate_frame_predictions(
        [ScoredPrediction("__foreign__", 0.95)] * 5,
        0.80,
    )

    assert result.decision == "uncertain"
    assert result.votes == 0


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


def test_threshold_sweep_reuses_raw_scores_without_inference() -> None:
    points = threshold_sweep(
        [
            ScoredPrediction("Барсик", 0.92),
            ScoredPrediction("Барсик", 0.64),
            ScoredPrediction("Барсик", 0.70),
        ],
        ["Барсик", "Барсик", "__foreign__"],
        [0.60, 0.80],
    )

    assert points[0].metrics.false_accept_rate == 1.0
    assert points[0].metrics.false_reject_rate == 0.0
    assert points[1].metrics.false_accept_rate == 0.0
    assert points[1].metrics.false_reject_rate == 0.5
    assert points[1].metrics.uncertain == 2


def test_confusion_matrix_uses_unified_decision_names() -> None:
    matrix = confusion_matrix(
        ["Барсик", "Барсик", "__foreign__"],
        ["Барсик", "known:Мурка", "no_prediction"],
    )

    assert matrix == {
        "Барсик": {"known:Барсик": 1, "known:Мурка": 1},
        "__foreign__": {"no_prediction": 1},
    }
