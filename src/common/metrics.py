from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass, replace
from math import ceil
from statistics import mean, median
from typing import TypeAlias

FOREIGN_LABEL = "__foreign__"
# Кадры проходов, размеченных как «кота не видно»: тени, пустая терраса,
# качнувшаяся ветка. В отличие от чужих котов этот класс **обучается**, и в этом
# весь смысл. Без него модель закрыта по своим котам, softmax обязан
# распределить единицу между ними, и отказаться ей нечем — тень неизбежно
# получает чьё-то имя, а порог уверенности спасает не всегда: на камере подхода
# ложные срабатывания идут с медианной уверенностью 0.91 против 0.93 у
# настоящих проходов.
NOCAT_LABEL = "__nocat__"
# Метки, которые означают «имени быть не должно»: и ожидаемые в наборе, и
# предсказанные моделью. Предсказать `__nocat__` — это отказ, а не ошибка.
REJECT_LABELS = frozenset({FOREIGN_LABEL, NOCAT_LABEL})
UNCERTAIN = "uncertain"
NO_PREDICTION = "no_prediction"

PredictionValue: TypeAlias = str | None


@dataclass(frozen=True)
class ScoredPrediction:
    name: str
    confidence: float


# Какая доля кадров с котом должна назвать одно и то же имя, чтобы проход
# получил его.
#
# Прежнее правило — «хватит одного кадра выше порога» — стоило разбора
# 17 августа: чужой рыжий кот получил имя «Яся 0.93» и уехал в уведомление.
#
# Замер на записях 23 проходов (12 своих, 11 чужих для модели), прогнанных
# установленными детектором и моделью на рабочих 12 к/с:
#
#   кворум   свои названы   чужие пропущены
#     нет        12 / 12          7 / 11
#     0.30       10 / 12          1 / 11
#     0.50        9 / 12          1 / 11
#     0.60        7 / 12          1 / 11
#
# Взято 0.50, хотя 0.30 на этой выборке узнаёт на один проход больше: полосу
# между «свой» и «чужой» держит один-единственный проход (#688 набрал 25%
# кадров, ближайший свой — 31%), и ставить рабочую точку в пяти пунктах от
# ошибки, ради которой всё это разбиралось, — значит назначать порог хвостом.
# Цена несимметрична в нужную сторону: не узнать своего — неудобство, пустить
# чужого — ошибка. Число правится одним `CATDOOR_IDENTITY_QUORUM`.
#
# Единственный чужой, которого не отсекает никакой кворум, — #508: чёрная
# Ксюша против чёрного Черныша. Это случай не для агрегатора, а для обучения.
DEFAULT_IDENTITY_QUORUM = 0.5
# Нижняя граница кворума в кадрах. На коротком проходе половина — это один
# кадр, то есть ровно прежнее правило под новым именем.
MIN_QUORUM_FRAMES = 3


def required_votes(frames_used: int, quorum: float) -> int:
    """Сколько кадров должны сойтись на одном имени."""
    return max(MIN_QUORUM_FRAMES, ceil(quorum * frames_used))


@dataclass(frozen=True)
class AggregatedPrediction:
    decision: str
    predicted_name: str | None
    confidence: float | None
    frames_used: int
    # Сколько кадров назвали это имя. Отдельно от `frames_used`, потому что
    # «56 из 154» и «56 из 56» — разные ответы, а уверенность у них одна.
    votes: int = 0


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


def as_expected_label(
    cat_name: str | None,
    known_names: Collection[str] | None,
) -> str:
    """Чего ждать от модели на этом коте: имя или отказ.

    Кот, которого модель не обучалась узнавать, для неё **чужой** — как бы он
    ни был свой для дома. Пока метрика считала чужими только события с меткой
    «чужой кот», Кира, Ксюша, Джун и Бася приходили в неё своими, с именами,
    которых модель физически выдать не может. Их ошибки уходили в
    `misidentified_known`, доля пропущенных чужих оставалась `null`, и класс
    ошибок, ради которого метрика заведена, ею не измерялся: разбор 17 августа
    нашёл четыре таких прохода из 33 названных.

    `known_names=None` — «неизвестно, кого знает модель» (метаданные не
    прочитались). Тогда считаем по-старому: соврать про доли хуже, чем не
    посчитать.
    """
    if not cat_name:
        return FOREIGN_LABEL
    if known_names is None:
        return cat_name
    return cat_name if cat_name in known_names else FOREIGN_LABEL


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
    *,
    quorum: float = DEFAULT_IDENTITY_QUORUM,
) -> AggregatedPrediction:
    """Решение по проходу из покадровых оценок.

    Имя получает тот, за кого сошлось не меньше кворума кадров. Уверенностью
    прохода считается **медиана** его кадров, а не максимум: максимум — это
    самый удачный кадр из сотни, и именно он выдавал чужих котов за своих с
    цифрой 0.93. Медиана отвечает на вопрос, который человек и задаёт, глядя
    на строку журнала: насколько похоже было **обычно**.
    """
    if not predictions:
        return AggregatedPrediction(
            decision="no_prediction",
            predicted_name=None,
            confidence=None,
            frames_used=0,
            votes=0,
        )
    frames_used = len(predictions)
    # Кадры, где ближайшим оказался класс отказа, в голосовании не участвуют:
    # «чужой» — не кандидат в имена. Но из знаменателя они не выпадают, иначе
    # проход, где кота девять раз не узнали и один раз узнали, получил бы
    # единогласный кворум.
    accepted = [
        prediction
        for prediction in predictions
        if prediction.confidence >= threshold and prediction.name not in REJECT_LABELS
    ]
    counts = Counter(prediction.name for prediction in accepted)
    if counts:
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
        if len(winner_scores) >= required_votes(frames_used, quorum):
            return AggregatedPrediction(
                decision=f"known:{winner}",
                predicted_name=winner,
                confidence=median(winner_scores),
                frames_used=frames_used,
                votes=len(winner_scores),
            )
    # Кворум не собрался. Показываем самый сильный кадр: «не уверена (0.93)»
    # объясняет отказ, а голое «не уверена» — нет.
    strongest = max(predictions, key=lambda item: item.confidence)
    return AggregatedPrediction(
        decision="uncertain",
        predicted_name=strongest.name,
        confidence=strongest.confidence,
        frames_used=frames_used,
        votes=counts.get(strongest.name, 0),
    )


def calculate_metrics(
    expected: Sequence[str],
    predicted: Sequence[PredictionValue],
    inference_ms: Sequence[float],
    *,
    max_far: float,
    max_frr: float,
    max_mean_inference_ms: float = float("inf"),
) -> AcceptanceMetrics:
    if not expected or len(expected) != len(predicted) or len(expected) != len(inference_ms):
        raise ValueError("expected, predicted and inference_ms must have the same non-zero length")
    decisions = [normalize_decision(value) for value in predicted]
    known_samples = sum(label not in REJECT_LABELS for label in expected)
    foreign_samples = len(expected) - known_samples
    correct = 0
    false_accepts = 0
    false_rejects = 0
    misidentified_known = 0
    for label, decision in zip(expected, decisions, strict=True):
        predicted_name = decision_name(decision)
        # Отказом считается и молчание модели, и явное «не кот»: для дверцы это
        # одно и то же решение — имени не присваивать.
        rejected = predicted_name is None or predicted_name in REJECT_LABELS
        if label in REJECT_LABELS:
            if rejected:
                correct += 1
            else:
                false_accepts += 1
            continue
        if predicted_name == label:
            correct += 1
        else:
            false_rejects += 1
            # Назвать кота другим котом и не узнать его вовсе — разные ошибки:
            # первая опаснее, потому что выглядит как уверенный ответ.
            if not rejected:
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
    scores: Sequence[ScoredPrediction | None],
    labels: Sequence[str],
    thresholds: Sequence[float],
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
    expected: Sequence[str],
    predicted: Sequence[PredictionValue],
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
