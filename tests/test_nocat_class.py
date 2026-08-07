"""Класс «не кот»: право модели отказаться, а не выбирать из шести котов."""

from __future__ import annotations

from src.common.metrics import (
    FOREIGN_LABEL,
    NOCAT_LABEL,
    calculate_metrics,
)

THRESHOLDS = {"max_far": 0.2, "max_frr": 0.3}


def metrics(expected: list[str], predicted: list[str | None]):
    return calculate_metrics(
        expected,
        predicted,
        [1.0] * len(expected),
        **THRESHOLDS,
    )


def test_saying_not_a_cat_counts_as_refusal_not_as_a_name() -> None:
    """Предсказать «не кот» — это отказ, а не ошибочное имя.

    Без этого явный отказ модели считался бы ложным принятием наравне с
    «тень зовут Ясей», и класс, заведённый ради отказа, ухудшал бы метрику,
    которую он призван улучшить.
    """
    result = metrics([NOCAT_LABEL, NOCAT_LABEL], [NOCAT_LABEL, None])

    assert result.false_accepts == 0
    assert result.correct == 2
    # Тени не входят в число проходов своих котов: доля нераспознанных считается
    # по котам, а не по кадрам пустой террасы.
    assert result.known_samples == 0


def test_calling_a_shadow_by_a_cat_name_is_a_false_accept() -> None:
    result = metrics([NOCAT_LABEL], ["Яся"])

    assert result.false_accepts == 1
    assert result.correct == 0


def test_refusing_a_real_cat_is_a_miss_but_not_a_misidentification() -> None:
    """Не узнать кота и назвать его другим котом — разные ошибки.

    Вторая опаснее: она выглядит как уверенный ответ, и дверца по ней пустит
    не того. Считать их вместе значит потерять именно ту, что дороже.
    """
    result = metrics(["Яся", "Яся"], [NOCAT_LABEL, "Черныш"])

    assert result.false_rejects == 2
    assert result.misidentified_known == 1


def test_foreign_cats_still_measured_the_old_way() -> None:
    result = metrics([FOREIGN_LABEL, FOREIGN_LABEL], [None, "Яся"])

    assert result.false_accepts == 1
    assert result.false_accept_rate == 0.5
