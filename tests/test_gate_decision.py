"""Таблица случаев для решения об открытии калитки.

Функция чистая, поэтому проверяется без железа, без сети и без Frigate — ровно
ради этого она такой и написана.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from src.common.gate_access_config import GateAccessConfig
from src.gate.decision import (
    DECISION_COOLDOWN,
    DECISION_DRY_RUN,
    DECISION_LEAVING,
    DECISION_LOW_SCORE,
    DECISION_MODE_OFF,
    DECISION_MOVEMENT_UNKNOWN,
    DECISION_NO_GATE,
    DECISION_NOT_ALLOWED,
    DECISION_NOT_APPROACHING,
    DECISION_SECOND_FACTOR,
    DECISION_WINDOW,
    DECISION_ZONE,
    PHONE_FACTOR,
    Factor,
    decide,
)

MOSCOW = ZoneInfo("Europe/Moscow")
NOON = datetime(2026, 8, 11, 12, 0, tzinfo=MOSCOW)


def _config(**overrides: object) -> GateAccessConfig:
    values: dict[str, object] = {
        "mode": "dry",
        "gates": [{"camera": "gate", "relay": 2, "zone": "у калитки"}],
        "allowed_names": ["Алексей"],
        "min_score": 0.93,
        "cooldown_seconds": 20,
        "open_from": "07:00",
        "open_to": "23:00",
    }
    values.update(overrides)
    return GateAccessConfig.model_validate(values)


def _decide(**overrides: object):  # type: ignore[no-untyped-def]
    values: dict[str, object] = {
        "camera": "gate",
        "person_name": "Алексей",
        "score": 0.96,
        "zones": ("у калитки",),
        "config": _config(),
        "now": NOON,
    }
    values.update(overrides)
    return decide(**values)  # type: ignore[arg-type]


def test_a_known_face_in_the_zone_would_open() -> None:
    verdict = _decide()

    assert verdict.decision == DECISION_DRY_RUN
    assert verdict.would_open
    assert "Алексей" in verdict.reason


def test_the_off_mode_decides_nothing() -> None:
    """`off` — это «ни одной команды», а не «решить и не делать»."""
    verdict = _decide(config=_config(mode="off"))

    assert verdict.decision == DECISION_MODE_OFF


def test_a_camera_that_is_not_a_gate_opens_nothing() -> None:
    """Иначе узнанное лицо на террасе открыло бы калитку с другой стороны участка."""
    verdict = _decide(camera="terrace")

    assert verdict.decision == DECISION_NO_GATE


def test_a_person_outside_the_zone_is_refused() -> None:
    verdict = _decide(zones=("двор",))

    assert verdict.decision == DECISION_ZONE


def test_an_unrecognised_face_is_refused_and_says_so() -> None:
    verdict = _decide(person_name=None, score=None)

    assert verdict.decision == DECISION_NOT_ALLOWED
    assert verdict.reason == "Лицо не узнано"


def test_a_recognised_stranger_is_refused() -> None:
    """Белый список, а не «любой узнанный»: в библиотеке лиц есть и соседи."""
    verdict = _decide(person_name="Марина")

    assert verdict.decision == DECISION_NOT_ALLOWED


def test_an_uncertain_recognition_is_refused() -> None:
    """Frigate ставит имя при 0.9, а открываем от 0.93: имя «на троечку» не повод."""
    verdict = _decide(score=0.91)

    assert verdict.decision == DECISION_LOW_SCORE
    assert "0.91" in verdict.reason


def test_a_missing_score_is_not_a_high_one() -> None:
    verdict = _decide(score=None)

    assert verdict.decision == DECISION_LOW_SCORE


def test_night_is_outside_the_window() -> None:
    verdict = _decide(now=NOON.replace(hour=3))

    assert verdict.decision == DECISION_WINDOW


def test_a_window_across_midnight_stays_open() -> None:
    """Окно «с вечера до утра» наивное сравнение закрыло бы целиком."""
    config = _config(open_from="22:00", open_to="06:00")

    assert _decide(config=config, now=NOON.replace(hour=23)).decision == DECISION_DRY_RUN
    assert _decide(config=config, now=NOON.replace(hour=5)).decision == DECISION_DRY_RUN
    assert _decide(config=config, now=NOON.replace(hour=12)).decision == DECISION_WINDOW


def test_the_pause_holds_after_a_recent_opening() -> None:
    """Обратной связи от геркона нет: пауза — единственная защита от очереди импульсов."""
    verdict = _decide(last_opened_at=NOON - timedelta(seconds=5))

    assert verdict.decision == DECISION_COOLDOWN
    assert "20" in verdict.reason


def test_the_pause_expires() -> None:
    verdict = _decide(last_opened_at=NOON - timedelta(seconds=30))

    assert verdict.decision == DECISION_DRY_RUN


def test_the_phone_factor_is_recorded_but_does_not_decide() -> None:
    """По умолчанию телефон только наблюдается: главный риск — ложные отказы."""
    verdict = _decide(factors=[Factor(PHONE_FACTOR, "against", "телефона нет в сети")])

    assert verdict.decision == DECISION_DRY_RUN
    assert [factor.name for factor in verdict.factors] == ["face", PHONE_FACTOR]


def test_a_required_phone_must_say_yes() -> None:
    config = _config(require_second_factor=True)

    assert _decide(config=config, factors=[Factor(PHONE_FACTOR, "for", "")]).decision == (
        DECISION_DRY_RUN
    )
    assert _decide(
        config=config, factors=[Factor(PHONE_FACTOR, "against", "")]
    ).decision == DECISION_SECOND_FACTOR


def test_an_unknown_phone_is_a_refusal_not_a_permission() -> None:
    """Недоступность домашней сети — известная болячка узла, и она не открывает."""
    verdict = _decide(
        config=_config(require_second_factor=True),
        factors=[Factor(PHONE_FACTOR, "unknown", "проверка не выполнилась")],
    )

    assert verdict.decision == DECISION_SECOND_FACTOR


def test_a_required_factor_that_was_never_checked_refuses() -> None:
    """Молчание — не согласие: фактор, которого нет в списке, не разрешает."""
    verdict = _decide(config=_config(require_second_factor=True), factors=[])

    assert verdict.decision == DECISION_SECOND_FACTOR


def test_every_decision_carries_the_face_factor() -> None:
    """В журнал пишутся все факторы: иначе разбирать отказы невозможно."""
    verdict = _decide(score=0.5)

    assert verdict.factors[0].name == "face"
    assert verdict.factors[0].verdict == "for"


@pytest.mark.parametrize("mode", ["on", "ON", " on "])
def test_the_mode_survives_a_hand_edited_file(mode: str) -> None:
    """Файл правят и руками, а «ON» с пробелом — тот же режим, что «on»."""
    assert _config(mode=mode).mode == "on"


def test_an_unknown_mode_is_refused() -> None:
    """Опечатка в режиме не должна тихо превратиться во что-нибудь работающее."""
    with pytest.raises(ValueError, match="mode"):
        _config(mode="открывать")


def test_the_open_mode_actually_opens() -> None:
    """В `on` решение то же, что в `dry`: разница — в исполнении, не в логике.

    Функция решения о железе не знает вовсе, и это намеренно: одна и та же
    таблица случаев проверяет оба режима.
    """
    assert _decide(config=_config(mode="on")).decision == DECISION_DRY_RUN


def _moving(**rules: object):  # type: ignore[no-untyped-def]
    values = {"enabled": True, **rules}
    return _config(movement=values)


def test_rules_change_nothing_until_switched_on() -> None:
    """Умолчание — как было до 12 августа: свежая установка ведёт себя прежне."""
    assert _decide(movement="leaving").decision == DECISION_DRY_RUN


def test_a_leaving_person_never_opens_the_gate() -> None:
    """Та самая жалоба: калитка открывалась вслед выходящему."""
    verdict = _decide(config=_moving(), movement="leaving")

    assert verdict.decision == DECISION_LEAVING
    assert ("movement", "against") in [(f.name, f.verdict) for f in verdict.factors]


def test_an_approaching_person_is_let_in_as_before() -> None:
    assert _decide(config=_moving(), movement="approaching").decision == DECISION_DRY_RUN


def test_standing_still_opens_only_right_after_the_gate_closed() -> None:
    """Возвращение: вышел, закрыл за собой, развернулся — и хочет обратно."""
    standing = {"config": _moving(), "movement": "still"}

    assert _decide(**standing, gate_just_closed=True).decision == DECISION_DRY_RUN
    assert _decide(**standing).decision == DECISION_NOT_APPROACHING


def test_the_return_exception_can_be_switched_off() -> None:
    verdict = _decide(
        config=_moving(open_when_still=False), movement="still", gate_just_closed=True
    )

    assert verdict.decision == DECISION_NOT_APPROACHING


def test_unknown_movement_waits_instead_of_opening() -> None:
    """Событие обновляется раз в несколько секунд.

    Считать незнание разрешением значило бы открывать на первом же опросе —
    ровно та ошибка, из-за которой правила и появились.
    """
    verdict = _decide(config=_moving(), movement="unknown")

    assert verdict.decision == DECISION_MOVEMENT_UNKNOWN


def test_movement_is_checked_after_the_face_not_before() -> None:
    """Чужому отказывают по имени, а не по походке: причина должна быть той.

    Иначе в журнале «человек уходит» встречалось бы у прохожих, которых мы и не
    собирались пускать, и разбирать отказы стало бы не по чему.
    """
    verdict = _decide(config=_moving(), person_name="Марина", movement="leaving")

    assert verdict.decision == DECISION_NOT_ALLOWED
