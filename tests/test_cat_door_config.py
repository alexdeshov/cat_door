"""Настройки дверцы: умолчания, границы и отказ от догадок."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from src.common.cat_door_config import (
    CatDoorConfig,
    cat_door_config_path,
    load_cat_door_config,
    save_cat_door_config,
)


def test_the_door_opens_to_nobody_by_default() -> None:
    """Свежая установка не открывает дверцу никому, пока не выбрали кому."""
    config = CatDoorConfig()

    assert config.mode == "off"
    assert config.allowed_cats == []
    assert config.allows("Черныш") is False


def test_relay_two_because_one_is_the_gate() -> None:
    """Реле 1 занято калиткой, вход DI1 — герконом дверцы."""
    assert CatDoorConfig().relay == 2


def test_holding_the_latch_is_bounded() -> None:
    """Защёлка на 12 В обычно кратковременного включения: долгое удержание греет."""
    with pytest.raises(ValueError, match="hold_ms"):
        CatDoorConfig(hold_ms=60_000)


def test_settings_survive_a_round_trip(runtime_path: Path) -> None:
    path = cat_door_config_path(runtime_path)
    config = CatDoorConfig(mode="dry", allowed_cats=["Черныш", "Яся"], hold_ms=4000)

    save_cat_door_config(path, config)

    assert load_cat_door_config(path, CatDoorConfig()) == config


def test_a_broken_file_leaves_a_trace(runtime_path: Path, caplog) -> None:
    """Молчаливый откат к умолчанию открыл бы дверцу не тем или никому."""
    path = cat_door_config_path(runtime_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{это не json", encoding="utf-8")
    defaults = CatDoorConfig(mode="dry")

    with caplog.at_level(logging.WARNING):
        assert load_cat_door_config(path, defaults) == defaults

    assert "не прочитаны" in caplog.text


def test_an_unknown_mode_falls_back_loudly(runtime_path: Path, caplog) -> None:
    path = cat_door_config_path(runtime_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mode": "открывать"}), encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        config = load_cat_door_config(path, CatDoorConfig(mode="dry"))

    assert config.mode == "dry"
    assert "не приняты" in caplog.text


def test_only_the_named_cats_are_allowed() -> None:
    """Список имён, а не «любой узнанный»: чужие коты во дворе не редкость."""
    config = CatDoorConfig(allowed_cats=["Черныш"])

    assert config.allows("Черныш") is True
    assert config.allows("Бася") is False
    assert config.allows(None) is False
