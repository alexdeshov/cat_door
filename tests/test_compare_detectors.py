from scripts.compare_detectors import miss_at_false_rate

# Развёртка «уверенной» модели: она метит всё подряд сильнее, и на низком пороге
# ложных у неё много — но на высоком она всё ещё видит котов.
CONFIDENT = [
    {"threshold": 0.30, "miss_rate": 0.10, "false_trigger_rate": 0.60},
    {"threshold": 0.50, "miss_rate": 0.15, "false_trigger_rate": 0.35},
    {"threshold": 0.70, "miss_rate": 0.22, "false_trigger_rate": 0.09},
    {"threshold": 0.85, "miss_rate": 0.40, "false_trigger_rate": 0.03},
]

# Осторожная модель: на том же пороге 0.50 ложных у неё вчетверо меньше.
CAUTIOUS = [
    {"threshold": 0.30, "miss_rate": 0.30, "false_trigger_rate": 0.20},
    {"threshold": 0.50, "miss_rate": 0.45, "false_trigger_rate": 0.08},
    {"threshold": 0.70, "miss_rate": 0.60, "false_trigger_rate": 0.02},
    {"threshold": 0.85, "miss_rate": 0.80, "false_trigger_rate": 0.00},
]


def test_comparison_on_one_threshold_would_reverse_the_answer() -> None:
    """Ровно та ошибка, которую проект дважды чуть не сделал.

    На общем пороге 0.50 «уверенная» модель выглядит хуже: 35% ложных против
    8%. На равной доле ложных она же оказывается лучше по пропускам. Тест
    существует, чтобы это расхождение было видно числами, а не на память.
    """
    on_common_threshold = (CONFIDENT[1]["false_trigger_rate"], CAUTIOUS[1]["false_trigger_rate"])
    assert on_common_threshold[0] > on_common_threshold[1]

    _, confident_miss = miss_at_false_rate(CONFIDENT, 0.10)
    _, cautious_miss = miss_at_false_rate(CAUTIOUS, 0.10)

    assert confident_miss == 0.22
    assert cautious_miss == 0.45
    assert confident_miss < cautious_miss


def test_the_softest_qualifying_threshold_wins() -> None:
    """Из порогов, укладывающихся в ограничение, берётся самый мягкий.

    Он даёт меньше всего пропусков — именно так модель настраивали бы вручную.
    """
    threshold, miss = miss_at_false_rate(CONFIDENT, 0.10)

    assert threshold == 0.70
    assert miss == 0.22


def test_an_unreachable_constraint_is_named_as_such() -> None:
    """Ноль ложных достижим не всегда, и выдумывать порог нельзя."""
    assert miss_at_false_rate(CONFIDENT, 0.0) == (None, None)


def test_points_without_numbers_are_skipped() -> None:
    """Развёртка без котов или без «кота не видно» даёт None вместо доли."""
    sweep = [
        {"threshold": 0.30, "miss_rate": None, "false_trigger_rate": None},
        {"threshold": 0.50, "miss_rate": 0.20, "false_trigger_rate": 0.05},
    ]

    assert miss_at_false_rate(sweep, 0.10) == (0.50, 0.20)
