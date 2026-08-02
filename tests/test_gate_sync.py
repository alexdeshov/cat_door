from pathlib import Path

from src.common.circuits import NodeCircuits, circuits_path, save_circuits
from src.common.gate_service import GateStatus
from src.common.gate_sync import reconcile
from src.common.settings import Settings


def test_stopped_service_is_started_when_the_node_is_given_to_the_gate(
    runtime_path: Path,
    monkeypatch,
) -> None:
    actions: list[tuple[str, bool]] = []
    monkeypatch.setattr("src.common.gate_sync.gate_status", lambda unit: GateStatus("stopped"))
    monkeypatch.setattr(
        "src.common.gate_sync.set_gate_running",
        lambda unit, running: actions.append((unit, running)) or None,
    )
    save_circuits(circuits_path(runtime_path), NodeCircuits().with_only("gate"))

    message = reconcile(Settings(data_dir=runtime_path))

    assert actions == [("frigate.service", True)]
    assert "запущена" in message


def test_running_service_is_stopped_when_the_node_returns_to_the_cats(
    runtime_path: Path,
    monkeypatch,
) -> None:
    actions: list[tuple[str, bool]] = []
    monkeypatch.setattr("src.common.gate_sync.gate_status", lambda unit: GateStatus("running"))
    monkeypatch.setattr(
        "src.common.gate_sync.set_gate_running",
        lambda unit, running: actions.append((unit, running)) or None,
    )
    save_circuits(circuits_path(runtime_path), NodeCircuits().with_only("cats"))

    reconcile(Settings(data_dir=runtime_path))

    assert actions == [("frigate.service", False)]


def test_nothing_is_done_when_the_state_already_matches(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """Примиритель срабатывает на каждое изменение файла, в том числе чужое."""
    actions: list[tuple[str, bool]] = []
    monkeypatch.setattr("src.common.gate_sync.gate_status", lambda unit: GateStatus("running"))
    monkeypatch.setattr(
        "src.common.gate_sync.set_gate_running",
        lambda unit, running: actions.append((unit, running)) or None,
    )
    save_circuits(circuits_path(runtime_path), NodeCircuits().with_only("gate"))

    reconcile(Settings(data_dir=runtime_path))

    assert actions == []


def test_an_undeployed_service_is_not_an_error(runtime_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("src.common.gate_sync.gate_status", lambda unit: GateStatus("absent"))
    monkeypatch.setattr(
        "src.common.gate_sync.set_gate_running",
        lambda unit, running: "не должно вызываться",
    )

    message = reconcile(Settings(data_dir=runtime_path))

    assert "не установлена" in message
