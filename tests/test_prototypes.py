"""Эталоны и порог: та часть, которую узел считает сам, без torch и макбука."""

from __future__ import annotations

import numpy as np

from src.common.prototypes import (
    FALLBACK_THRESHOLD,
    centroid,
    choose_threshold,
    evaluate,
    match,
    unit,
)


def test_a_prototype_is_a_direction_not_a_length() -> None:
    """Косинус считается скалярным произведением, и длина эталона его исказит."""
    prototype = centroid([[3.0, 0.0], [0.0, 4.0]])

    assert np.isclose(np.linalg.norm(prototype), 1.0)
    assert np.allclose(prototype, [0.7071, 0.7071], atol=1e-3)


def test_long_vectors_do_not_outvote_short_ones() -> None:
    """Слагаемые нормируются до усреднения: иначе один яркий кадр решал бы всё."""
    honest = centroid([[1.0, 0.0], [0.0, 1.0]])
    with_a_loud_one = centroid([[100.0, 0.0], [0.0, 1.0]])

    assert np.allclose(honest, with_a_loud_one, atol=1e-5)


def test_the_nearest_prototype_wins_if_it_is_close_and_alone() -> None:
    prototypes = {"Черныш": [1.0, 0.0], "Яся": [0.0, 1.0]}

    named, candidate, score = match([0.99, 0.14], prototypes, threshold=0.7)

    assert named == "Черныш"
    assert candidate == "Черныш"
    assert score > 0.9


def test_a_candidate_is_reported_even_when_refused() -> None:
    """«Был похож на Черныша, но не дотянул» объясняет отказ, «не узнан» — нет."""
    named, candidate, _ = match([0.7, 0.7], {"Черныш": [1.0, 0.0]}, threshold=0.9)

    assert named is None
    assert candidate == "Черныш"


def test_two_equally_close_cats_are_not_a_decision() -> None:
    prototypes = {"Черныш": [1.0, 0.0], "Яся": [0.99, 0.14]}

    named, _, _ = match([1.0, 0.07], prototypes, threshold=0.5, margin=0.05)

    assert named is None


def test_the_threshold_keeps_strangers_out() -> None:
    same = [0.9, 0.85, 0.8]
    other = [0.6, 0.55, 0.4]

    threshold, stats = choose_threshold(same, other)

    assert threshold > max(other)
    assert stats["same_above_threshold"] == 1.0


def test_a_set_without_strangers_cannot_choose_a_threshold() -> None:
    """Назначать порог не из чего — честнее взять осторожный, чем удобный."""
    threshold, stats = choose_threshold([0.9, 0.8], [])

    assert threshold == FALLBACK_THRESHOLD
    assert stats["other_max"] == 0.0


def test_a_threshold_that_rejects_everyone_is_visible() -> None:
    """Порог, отсекающий чужих идеально и своих заодно, — неработающая модель.

    Уехать на узел незамеченной она не должна, поэтому доля прошедших своих
    считается всегда.
    """
    _, stats = choose_threshold([0.3, 0.35], [0.9, 0.88])

    assert stats["same_above_threshold"] == 0.0


def test_evaluation_answers_both_questions_at_once() -> None:
    prototypes = {"Черныш": [1.0, 0.0]}
    samples = [
        ("Черныш", [0.98, 0.2]),
        ("Черныш", [0.1, 0.99]),
        ("__foreign__", [0.0, 1.0]),
        ("__foreign__", [0.99, 0.1]),
    ]

    quality = evaluate(prototypes, samples, threshold=0.8)

    assert quality.samples == 2
    assert quality.strangers == 2
    assert quality.recognised == 0.5
    assert quality.false_accepts == 0.5


def test_a_new_set_may_not_let_more_strangers_in() -> None:
    """Ложные пропуски не прощаются: это та ошибка, ради которой всё считалось."""
    from src.common.prototypes import MatchQuality

    old = MatchQuality(recognised=0.8, false_accepts=0.01, samples=100, strangers=100)
    stricter = MatchQuality(recognised=0.79, false_accepts=0.0, samples=100, strangers=100)
    looser = MatchQuality(recognised=0.95, false_accepts=0.05, samples=100, strangers=100)
    forgetful = MatchQuality(recognised=0.5, false_accepts=0.0, samples=100, strangers=100)

    assert stricter.not_worse_than(old)
    assert not looser.not_worse_than(old)
    assert not forgetful.not_worse_than(old)


def test_a_zero_vector_does_not_explode() -> None:
    assert np.allclose(unit([0.0, 0.0]), [0.0, 0.0])


def test_an_indistinguishable_cat_poisons_the_threshold() -> None:
    """Кот, которого пространство не различает, рушит опознание остальных.

    Измерено на живом узле 2026-08-11: пересчёт завёл эталоны всем шести
    размеченным котам, хотя пространство обучено на двух. Четверо остальных при
    обучении были «чужими» — их векторы лежат в одной области, и эталон такого
    кота оказывается похож на всех чужих сразу. Порог, вынужденный отсечь
    чужих, поднимается почти до единицы и отсекает заодно своих: узнаваемость
    упала с 0.72 до 0.34, и защита от ухудшения не дала это применить.

    Отсюда правило: по умолчанию обновляются только те коты, которых модель уже
    знает.
    """
    known = {"Черныш": [1.0, 0.0, 0.0], "Яся": [0.0, 1.0, 0.0]}
    # Эталон кота, которого пространство не различает: он похож на всё сразу.
    muddled = {**known, "Бася": [0.577, 0.577, 0.577]}
    mine = [("Черныш", [0.9, 0.436, 0.0]), ("Яся", [0.436, 0.9, 0.0])]
    strangers = [("__foreign__", [0.6, 0.55, 0.58]), ("__foreign__", [0.55, 0.6, 0.58])]
    samples = mine + strangers

    def with_chosen_threshold(prototypes: dict[str, list[float]]):
        same = [
            match(vector, prototypes, threshold=0.0)[2]
            for label, vector in mine
        ]
        other = [
            match(vector, prototypes, threshold=0.0)[2]
            for label, vector in strangers
        ]
        threshold, _ = choose_threshold(same, other)
        return evaluate(prototypes, samples, threshold=threshold), threshold

    before, before_threshold = with_chosen_threshold(known)
    after, after_threshold = with_chosen_threshold(muddled)

    assert before.recognised == 1.0
    assert after_threshold > before_threshold
    assert after.recognised < before.recognised
    assert not after.not_worse_than(before)
