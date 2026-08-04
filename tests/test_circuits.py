from pathlib import Path

import pytest

from src.common.circuits import (
    CircuitConflict,
    NodeCircuits,
    circuits_path,
    ensure_allowed,
    load_circuits,
    save_circuits,
)


def test_node_starts_with_the_cat_circuit_alone() -> None:
    """Умолчание — то, что работало до появления второго контура."""
    circuits = NodeCircuits()

    assert circuits.cats is True
    assert circuits.gate is False
    assert circuits.exclusive is True
    assert circuits.gate_recording == "events"
    assert circuits.active_names == ("cats",)


def test_switching_a_circuit_stops_the_other() -> None:
    switched = NodeCircuits().with_only("gate")

    assert switched.active_names == ("gate",)
    assert switched.cats is False


def test_both_circuits_are_refused_until_the_node_is_replaced() -> None:
    """Отказ, а не тихая деградация: одновременная работа портит оба контура."""
    with pytest.raises(CircuitConflict):
        ensure_allowed(NodeCircuits(cats=True, gate=True))


def test_a_hand_edited_conflict_is_defused_not_fatal() -> None:
    """Опечатка в файле не должна останавливать запись проходов."""
    resolved = NodeCircuits(cats=True, gate=True).resolved()

    assert resolved.active_names == ("cats",)


def test_lifting_the_limit_is_one_field() -> None:
    """После переезда режим `both` включается снятием признака, не переделкой."""
    both = NodeCircuits(cats=True, gate=True, exclusive=False)

    assert both.active_names == ("cats", "gate")


def test_recording_mode_is_kept_across_switching(runtime_path: Path) -> None:
    path = circuits_path(runtime_path)
    save_circuits(path, NodeCircuits(gate_recording="continuous").with_only("gate"))

    restored = load_circuits(path)

    assert restored.gate_recording == "continuous"
    assert restored.active_names == ("gate",)


def test_broken_file_leaves_the_cat_circuit_running(runtime_path: Path) -> None:
    """Испорченный файл не должен молча останавливать запись проходов."""
    path = circuits_path(runtime_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ это не json", encoding="utf-8")

    assert load_circuits(path).is_active("cats") is True


def test_the_recorder_shares_frigate_with_the_gate() -> None:
    """Калитка и регистратор — один и тот же процесс, а не два конкурента.

    Ограничение узла про движки: запрещать их друг другу значило бы запрещать
    Frigate работать вместе с самим собой.
    """
    both = NodeCircuits(cats=False, gate=True, nvr=True)

    ensure_allowed(both)
    assert both.active_names == ("gate", "nvr")
    assert both.frigate_wanted is True


def test_enabling_the_recorder_stops_the_cats_but_not_the_gate() -> None:
    """Посторониться нужно перед чужим движком, а не перед соседом по своему."""
    switched = NodeCircuits(cats=False, gate=True).with_circuit("nvr", True)

    assert switched.gate is True
    assert switched.nvr is True

    from_cats = NodeCircuits(cats=True).with_circuit("nvr", True)

    assert from_cats.cats is False
    assert from_cats.nvr is True


def test_lifting_the_limit_lets_every_circuit_run() -> None:
    """Кнопка на вкладке узла — единственное, что меняется после переезда."""
    free = NodeCircuits(cats=True, exclusive=False).with_circuit("nvr", True)

    assert free.cats is True
    assert free.nvr is True
    ensure_allowed(free)


def test_enabling_the_cats_stops_every_frigate_circuit() -> None:
    running = NodeCircuits(cats=False, gate=True, nvr=True)

    switched = running.with_circuit("cats", True)

    assert switched.active_names == ("cats",)


def test_turning_a_circuit_off_never_conflicts() -> None:
    """Выключение — всегда разрешённое действие, что бы ни стояло рядом."""
    running = NodeCircuits(cats=False, gate=True, nvr=True)

    assert running.with_circuit("nvr", False).active_names == ("gate",)


def test_a_hand_edited_conflict_stops_frigate_not_the_cats() -> None:
    resolved = NodeCircuits(cats=True, gate=True, nvr=True).resolved()

    assert resolved.active_names == ("cats",)
    assert resolved.frigate_wanted is False
