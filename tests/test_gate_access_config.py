from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from src.common.gate_access_config import (
    MOVEMENT_PRESET,
    SECOND_CHANCE_PRESET,
    GateAccessConfig,
    apply_movement_preset,
    apply_second_chance_preset,
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


def test_an_unknown_mode_falls_back_to_the_previous_one(runtime_path: Path, caplog) -> None:
    """Опечатка в режиме не должна ни открыть калитку, ни закрыть её молча."""
    path = gate_access_config_path(runtime_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mode": "открывать"}), encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        config = load_gate_access_config(path, GateAccessConfig(mode="dry"))

    assert config.mode == "dry"
    assert "не приняты" in caplog.text


def test_the_relay_must_exist_on_the_board() -> None:
    """Каналов у платы восемь. Ноль или девятый — опечатка, а не настройка.

    Реле 1 план резервировал за кошачьей дверцей, но на объекте калитку
    подключили именно к нему: механики у дверцы нет, канал ей подберут отдельно.
    """
    assert GateAccessConfig.model_validate({"gates": [{"relay": 1}]}).gates[0].relay == 1
    with pytest.raises(ValueError, match="relay"):
        GateAccessConfig.model_validate({"gates": [{"relay": 0}]})
    with pytest.raises(ValueError, match="relay"):
        GateAccessConfig.model_validate({"gates": [{"relay": 9}]})


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


def test_both_presets_survive_a_round_trip(tmp_path: Path) -> None:
    """Пресеты и их резервные копии обязаны пережить перезапуск службы.

    Настройки живут в одном файле, и служба перечитывает его на ходу. Потеря
    любой из двух резервных копий означала бы, что кнопка «вернуть как было»
    после перезагрузки узла возвращает не то, что было.
    """
    path = tmp_path / "gate-access.json"
    config = apply_second_chance_preset(
        apply_movement_preset(GateAccessConfig(mode="on", cooldown_seconds=10))
    )

    save_gate_access_config(path, config)
    restored = load_gate_access_config(path, GateAccessConfig())

    assert restored == config
    assert restored.movement == MOVEMENT_PRESET
    assert restored.second_chance == SECOND_CHANCE_PRESET
    assert restored.cooldown_seconds == 10
    # Резервные копии — то, к чему возвращает откат. Без них кнопка отката
    # молча выключала бы правила вместо возврата прежних.
    assert restored.replaced_by_preset is not None
    assert restored.replaced_by_chance_preset is not None


def test_a_torn_file_leaves_the_running_settings_alone(tmp_path: Path) -> None:
    """Обрыв питания посреди записи не должен обнулить настройки.

    Сама запись атомарна, и увидеть половину файла служба не может. Но если
    файл всё же испорчен — диском, чужой рукой, чем угодно, — служба обязана
    работать на прежних значениях и сказать об этом, а не откатиться к
    умолчаниям молча.
    """
    path = tmp_path / "gate-access.json"
    running = apply_second_chance_preset(GateAccessConfig(mode="on", cooldown_seconds=10))
    save_gate_access_config(path, running)
    text = path.read_text(encoding="utf-8")
    path.write_text(text[: len(text) // 2], encoding="utf-8")

    restored = load_gate_access_config(path, running)

    assert restored == running
    assert restored.second_chance.enabled is True


def test_saving_leaves_no_debris_behind(tmp_path: Path) -> None:
    """Временный файл записи не должен пережить саму запись.

    Иначе каталог настроек обрастал бы обломками после каждого сохранения, а
    разбирать, какой из них настоящий, пришлось бы человеку.
    """
    path = tmp_path / "gate-access.json"

    for cooldown in (10, 20, 30):
        save_gate_access_config(path, GateAccessConfig(cooldown_seconds=cooldown))

    assert [item.name for item in tmp_path.iterdir()] == ["gate-access.json"]
    assert load_gate_access_config(path, GateAccessConfig()).cooldown_seconds == 30
