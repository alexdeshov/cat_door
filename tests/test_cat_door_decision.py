"""Таблица случаев для решения об открытии дверцы.

Условие владельца: дверца открывается только тогда, когда опознанный кот
появился **на камере дверцы**. Всё остальное — предохранители к нему.
"""

from __future__ import annotations

import pytest

from src.common.cat_door_config import CatDoorConfig
from src.common.cat_door_decision import decide


def _config(**overrides: object) -> CatDoorConfig:
    values: dict[str, object] = {
        "mode": "on",
        "allowed_cats": ["Черныш", "Яся"],
        "min_confidence": 0.9,
        "cooldown_seconds": 30,
    }
    values.update(overrides)
    return CatDoorConfig.model_validate(values)


def test_a_known_cat_at_the_door_opens_it() -> None:
    decision = decide(
        name="Черныш", confidence=0.95, role="face", config=_config(), now=1000.0
    )

    assert decision.opens
    assert decision.code == "open"
    assert "Черныш" in decision.reason


def test_the_terrace_camera_never_opens_the_door() -> None:
    """Тот же кот на террасе виден, но он не у дверцы.

    Открыв по этой камере, мы отпускаем защёлку неизвестно для кого и на пять
    секунд раньше, чем нужно.
    """
    decision = decide(
        name="Черныш", confidence=0.99, role="side", config=_config(), now=1000.0
    )

    assert not decision.opens
    assert decision.code == "denied_wrong_camera"


@pytest.mark.parametrize(
    ("name", "confidence", "config", "code"),
    [
        pytest.param("Черныш", 0.95, _config(mode="off"), "denied_mode_off", id="выключено"),
        pytest.param("Черныш", 0.95, _config(mode="dry"), "dry_run", id="наблюдение"),
        pytest.param(None, 0.95, _config(), "denied_unknown_cat", id="кот не опознан"),
        pytest.param("Бася", 0.99, _config(), "denied_not_allowed", id="чужой кот"),
        pytest.param("Черныш", 0.5, _config(), "denied_low_confidence", id="узнан неуверенно"),
        pytest.param("Черныш", None, _config(), "denied_low_confidence", id="без уверенности"),
        pytest.param(
            "Черныш", 0.95, _config(allowed_cats=[]), "denied_not_allowed", id="список пуст"
        ),
    ],
)
def test_the_reasons_are_specific(
    name: str | None, confidence: float | None, config: CatDoorConfig, code: str
) -> None:
    """«Дверца не открылась» без причины разбирается пересмотром записи."""
    decision = decide(
        name=name, confidence=confidence, role="face", config=config, now=1000.0
    )

    assert decision.code == code
    assert not decision.opens


def test_an_empty_list_of_cats_opens_for_nobody() -> None:
    """Пустой список — «никого», как и у калитки.

    Опознание здесь удобство, а не пропуск: открывать незнакомому коту дверцу
    в дом по умолчанию нельзя. Причина при этом говорит именно про пустой
    список, а не про кота — иначе владелец будет искать ошибку в опознании.
    """
    decision = decide(
        name="Кто-то", confidence=0.95, role="face", config=_config(allowed_cats=[]), now=1000.0
    )

    assert not decision.opens
    assert decision.code == "denied_not_allowed"
    assert "пуст" in decision.reason


def test_the_pause_holds_the_latch() -> None:
    """Кот, гуляющий у дверцы, иначе получит очередь удержаний по пять секунд."""
    config = _config(cooldown_seconds=30)

    soon = decide(
        name="Яся", confidence=0.95, role="face", config=config, now=1010.0,
        last_opened_at=1000.0,
    )
    later = decide(
        name="Яся", confidence=0.95, role="face", config=config, now=1040.0,
        last_opened_at=1000.0,
    )

    assert soon.code == "denied_cooldown"
    assert "прошло 10 с из 30 с" in soon.reason
    assert later.opens


def test_a_clock_that_jumped_back_waits_the_full_pause() -> None:
    """После обрыва питания часы идут позади: незнание времени — не повод открывать."""
    decision = decide(
        name="Яся", confidence=0.95, role="face", config=_config(), now=900.0,
        last_opened_at=1000.0,
    )

    assert decision.code == "denied_cooldown"
    assert "сместились" in decision.reason
