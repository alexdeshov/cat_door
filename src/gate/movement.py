"""Куда идёт человек: к калитке, от неё или никуда.

Признак — во сколько раз изменилась площадь рамки от первого кадра события к
текущему. Измерения 2026-08-12 (`docs/gate-direction-experiments.md`): уходы
дают от ÷2.2 до ÷199, приходы от ×7.6 до ×58, а человек, стоящий у калитки, —
×1.6. Пороги вдвое в обе стороны лежат в пустых полосах между этими группами, а
не подогнаны к краю.

Третья ветка появилась из наблюдения, которое легко упустить: когда человек не
идёт, **направления не существует**. Правило, знающее только «к» и «от»,
обязано отнести стоящего к одному из них — и ошибётся в половине случаев.
"""

from __future__ import annotations

APPROACHING = "approaching"
LEAVING = "leaving"
STILL = "still"
UNKNOWN = "unknown"

# Во сколько раз должна вырасти площадь, чтобы считать «идёт к калитке».
DEFAULT_APPROACHING_RATIO = 2.0
# И во сколько упасть, чтобы считать «уходит».
DEFAULT_LEAVING_RATIO = 0.5


def classify(
    first_area: float | None,
    current_area: float | None,
    *,
    approaching_ratio: float = DEFAULT_APPROACHING_RATIO,
    leaving_ratio: float = DEFAULT_LEAVING_RATIO,
) -> str:
    """Ветка движения по изменению площади рамки.

    `unknown` — пока не с чем сравнивать: первое обновление события приходит
    через несколько секунд. Это состояние короткое и честное; выдавать за него
    «стоит» значило бы открыть калитку тому, кто в этот момент уходит.
    """
    if not first_area or not current_area or first_area <= 0:
        return UNKNOWN
    ratio = current_area / first_area
    if ratio >= approaching_ratio:
        return APPROACHING
    if ratio <= leaving_ratio:
        return LEAVING
    return STILL


def area_ratio(first_area: float | None, current_area: float | None) -> float | None:
    """Во сколько раз выросла рамка. Пишется в журнал, чтобы пороги можно было перемерить."""
    if not first_area or not current_area or first_area <= 0:
        return None
    return round(current_area / first_area, 3)
