"""Ветки движения — на числах живых проходов 12 августа."""

from __future__ import annotations

import pytest

from src.gate.movement import APPROACHING, LEAVING, STILL, UNKNOWN, area_ratio, classify


@pytest.mark.parametrize(
    ("first", "current", "expected"),
    [
        # Уходы: площадь падает в разы.
        (0.432, 0.003, LEAVING),
        (0.369, 0.003, LEAVING),
        (0.199, 0.001, LEAVING),
        (0.059, 0.027, LEAVING),
        # Приходы: растёт в разы.
        (0.060, 0.456, APPROACHING),
        (0.003, 0.159, APPROACHING),
        (0.002, 0.134, APPROACHING),
        # Владелец стоял у калитки и никуда не шёл.
        (0.44, 0.70, STILL),
    ],
)
def test_live_passes_are_classified_as_they_happened(
    first: float, current: float, expected: str
) -> None:
    assert classify(first, current) == expected


def test_nothing_to_compare_is_not_a_guess() -> None:
    """Первое обновление события приходит через секунды.

    Выдавать это за «стоит» значило бы открыть калитку тому, кто уходит.
    """
    assert classify(None, 0.4) == UNKNOWN
    assert classify(0.4, None) == UNKNOWN
    assert classify(0.0, 0.4) == UNKNOWN


def test_thresholds_are_adjustable() -> None:
    """Камеру подвинут — пороги придётся перемерить, и это должно быть можно."""
    assert classify(0.1, 0.15, approaching_ratio=1.4) == APPROACHING
    assert classify(0.1, 0.08, leaving_ratio=0.9) == LEAVING


def test_the_ratio_reaches_the_journal() -> None:
    """По нему пороги и перемеривают, когда геометрия изменится."""
    assert area_ratio(0.06, 0.456) == 7.6
    assert area_ratio(None, 0.4) is None
