from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from src.common.gate_access_config import (
    GateAccessConfig,
    gate_access_config_path,
    load_gate_access_config,
    mask_mac,
    save_gate_access_config,
)


def test_the_default_is_silence() -> None:
    """Первый запуск не должен ничего решать и тем более открывать."""
    config = GateAccessConfig()

    assert config.mode == "off"
    assert config.require_second_factor is False
    assert config.gates == []


def test_settings_survive_a_round_trip(runtime_path: Path) -> None:
    path = gate_access_config_path(runtime_path)
    config = GateAccessConfig.model_validate(
        {
            "mode": "dry",
            "gates": [{"camera": "gate", "relay": 2, "zone": "у калитки"}],
            "allowed_names": ["Алексей"],
            "people": [{"name": "Алексей", "phone_ip": "192.168.2.50", "phone_mac": "AA-BB-CC-DD-EE-FF"}],
        }
    )

    save_gate_access_config(path, config)

    assert load_gate_access_config(path, GateAccessConfig()) == config
    # MAC приводится к одному виду: два написания одного адреса не должны
    # выглядеть как два разных телефона.
    assert config.people[0].phone_mac == "aa:bb:cc:dd:ee:ff"


def test_the_settings_file_is_readable_only_by_its_owner(runtime_path: Path) -> None:
    """Внутри персональные данные семьи: адреса и MAC телефонов."""
    path = gate_access_config_path(runtime_path)

    save_gate_access_config(path, GateAccessConfig())

    assert path.stat().st_mode & 0o077 == 0


def test_a_missing_file_is_the_first_run_not_a_failure(runtime_path: Path) -> None:
    defaults = GateAccessConfig(mode="dry")

    assert load_gate_access_config(gate_access_config_path(runtime_path), defaults) == defaults


def test_a_broken_file_leaves_a_trace(runtime_path: Path, caplog) -> None:
    """Молчаливый откат тут дороже, чем в детекции: режим человек переключает руками."""
    path = gate_access_config_path(runtime_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{это не json", encoding="utf-8")
    defaults = GateAccessConfig(mode="dry")

    with caplog.at_level(logging.WARNING):
        assert load_gate_access_config(path, defaults) == defaults

    assert "не прочитаны" in caplog.text


def test_rejected_settings_leave_a_trace(runtime_path: Path, caplog) -> None:
    path = gate_access_config_path(runtime_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mode": "dry", "min_score": 12}), encoding="utf-8")
    defaults = GateAccessConfig()

    with caplog.at_level(logging.WARNING):
        assert load_gate_access_config(path, defaults) == defaults

    assert "не приняты" in caplog.text


def test_the_open_mode_never_arrives_through_the_file(runtime_path: Path, caplog) -> None:
    """Правку файла руками правило владельца связывает так же, как и админку."""
    path = gate_access_config_path(runtime_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mode": "on"}), encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        config = load_gate_access_config(path, GateAccessConfig())

    assert config.mode == "off"
    assert "письменного разрешения" in caplog.text


def test_the_cat_door_relay_is_not_available_to_the_gate() -> None:
    """Реле 1 закреплено за кошачьей дверцей — это отдельное разрешение."""
    with pytest.raises(ValueError, match="relay"):
        GateAccessConfig.model_validate({"gates": [{"camera": "gate", "relay": 1}]})


def test_a_broken_mac_is_refused() -> None:
    with pytest.raises(ValueError, match="MAC"):
        GateAccessConfig.model_validate({"people": [{"name": "Алексей", "phone_mac": "нет"}]})


def test_one_person_one_record() -> None:
    with pytest.raises(ValueError, match="повторяются"):
        GateAccessConfig.model_validate(
            {"people": [{"name": "Алексей"}, {"name": "Алексей"}]}
        )


def test_time_is_written_as_hours_and_minutes() -> None:
    with pytest.raises(ValueError, match="ЧЧ:ММ"):
        GateAccessConfig.model_validate({"open_from": "семь утра"})


def test_a_mac_is_shown_by_its_tail_only() -> None:
    """Отличить свою запись от чужой можно, снять адрес с чужого экрана — нет."""
    assert mask_mac("aa:bb:cc:dd:ee:ff") == "··:··:··:··:ee:ff"
    assert mask_mac("") == ""
