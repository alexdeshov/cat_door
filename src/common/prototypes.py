"""Эталоны котов: как их считать, где ставить порог и как сравнивать наборы.

Живёт в `common`, потому что нужен по обе стороны и на разных машинах. На
макбуке этим считаются эталоны сразу после обучения; на узле — при пересчёте по
свежей разметке, где ни torch, ни макбука нет вовсе. Ничего тяжелее numpy здесь
не появляется намеренно: узел должен уметь это сам.

Эталон — среднее нормированных векторов кота, снова нормированное. Опознание —
косинус до ближайшего эталона. «Чужой» получается порогом, а не классом: это
свойство пространства, и потому новый кот заводится одним эталоном, без
переобучения сети.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

# Доля чужих, которую согласны принять за своих. Тот же один процент, что
# записан в критериях приёмки: цена ошибки здесь не меняется от того, кто
# считает — макбук или узел.
TARGET_FALSE_ACCEPT_RATE = 0.01
# Порог, когда чужих в наборе нет вовсе. Назначать его не из чего, и честнее
# взять осторожное число, чем выдать удобное.
FALLBACK_THRESHOLD = 0.55
# Насколько лучший эталон должен опережать второй, чтобы это было решением, а
# не монеткой.
DEFAULT_MARGIN = 0.05


def unit(vector: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(values))
    return values if norm == 0.0 else values / norm


def centroid(vectors: Sequence[Sequence[float]]) -> list[float]:
    """Эталон по набору векторов кота.

    Сперва нормируются слагаемые, потом результат: без первого длинные векторы
    перетягивали бы среднее на себя, без второго косинус перестал бы быть
    косинусом.
    """
    if not len(vectors):
        raise ValueError("cannot build a prototype from nothing")
    stacked = np.stack([unit(vector) for vector in vectors])
    return [float(value) for value in unit(stacked.mean(axis=0))]


@dataclass(frozen=True)
class MatchQuality:
    """Как набор эталонов ведёт себя на отложенной разметке."""

    recognised: float
    false_accepts: float
    samples: int
    strangers: int

    def not_worse_than(self, other: MatchQuality, *, slack: float = 0.02) -> bool:
        """Годится ли новый набор взамен старого.

        Ложные пропуски не прощаются вовсе: принять чужого за своего — та самая
        ошибка, ради которой всё это разбиралось. Узнаваемость своих может
        просесть на пару процентов — это шум выборки в пару сотен кропов, а не
        ухудшение.
        """
        return (
            self.false_accepts <= other.false_accepts
            and self.recognised >= other.recognised - slack
        )


def choose_threshold(
    same: Sequence[float],
    other: Sequence[float],
) -> tuple[float, dict[str, float]]:
    """Порог по отложенному набору: там, где чужие ещё не проходят.

    Отдельно считается, какая доля своих проходит выбранный порог. Порог,
    отсекающий чужих идеально и своих заодно, — это не строгая модель, а
    неработающая, и уехать на узел незамеченной она не должна.
    """
    stats = {
        "same_mean": float(np.mean(same)) if len(same) else 0.0,
        "other_mean": float(np.mean(other)) if len(other) else 0.0,
        "same_min": float(np.min(same)) if len(same) else 0.0,
        "other_max": float(np.max(other)) if len(other) else 0.0,
    }
    if not len(other):
        threshold = FALLBACK_THRESHOLD
    else:
        ranked = sorted(other, reverse=True)
        allowed = int(len(ranked) * TARGET_FALSE_ACCEPT_RATE)
        threshold = round(min(ranked[allowed] + 0.01, 0.95), 3)
    stats["threshold"] = threshold
    stats["same_above_threshold"] = (
        float(np.mean([score >= threshold for score in same])) if len(same) else 0.0
    )
    return threshold, stats


def match(
    embedding: Sequence[float] | np.ndarray,
    prototypes: Mapping[str, Sequence[float] | np.ndarray],
    *,
    threshold: float,
    margin: float = DEFAULT_MARGIN,
) -> tuple[str | None, str | None, float]:
    """Имя, кандидат и близость. Имя пусто, если далеко или неотличимо.

    Кандидат отдаётся всегда: в журнале «был похож на Черныша, но не дотянул»
    объясняет отказ, а голое «не узнан» — нет.
    """
    if not prototypes:
        return (None, None, 0.0)
    vector = unit(embedding)
    scored = sorted(
        ((float(np.dot(unit(value), vector)), name) for name, value in prototypes.items()),
        reverse=True,
    )
    best_score, best_name = scored[0]
    distinct = len(scored) < 2 or (best_score - scored[1][0]) >= margin
    named = best_name if best_score >= threshold and distinct else None
    return (named, best_name, best_score)


def evaluate(
    prototypes: Mapping[str, Sequence[float] | np.ndarray],
    samples: Sequence[tuple[str, Sequence[float]]],
    *,
    threshold: float,
    margin: float = DEFAULT_MARGIN,
) -> MatchQuality:
    """Прогнать размеченные векторы через эталоны.

    `samples` — пары «настоящее имя, вектор»; имя, которого нет среди эталонов,
    считается чужим. Так один прогон отвечает сразу на оба вопроса: узнаём ли
    своих и не пускаем ли чужих.
    """
    known = 0
    recognised = 0
    strangers = 0
    accepted_strangers = 0
    for label, vector in samples:
        named, _, _ = match(vector, prototypes, threshold=threshold, margin=margin)
        if label in prototypes:
            known += 1
            recognised += int(named == label)
        else:
            strangers += 1
            accepted_strangers += int(named is not None)
    return MatchQuality(
        recognised=recognised / known if known else 0.0,
        false_accepts=accepted_strangers / strangers if strangers else 0.0,
        samples=known,
        strangers=strangers,
    )
