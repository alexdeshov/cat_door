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
    # Проверка контейнера мокается обязательно: без мока тест зовёт настоящий
    # Docker и на узле с остановленным контейнером Frigate честно его
    # «перезапускает». Ровно так тест и упал на сервере, пройдя на маке,
    # где Docker просто нет.
    monkeypatch.setattr("src.common.gate_sync._container_running", lambda name: True)
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


def test_a_dead_container_under_an_active_unit_is_restarted(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """Oneshot с RemainAfterExit не видит смерть контейнера — видит примиритель.

    Иначе админка и systemd хором говорили бы «работает» при мёртвом Frigate
    неделями, до пустого журнала проходов.
    """
    actions: list[tuple[str, bool]] = []
    monkeypatch.setattr("src.common.gate_sync.gate_status", lambda unit: GateStatus("running"))
    monkeypatch.setattr("src.common.gate_sync._container_running", lambda name: False)
    monkeypatch.setattr(
        "src.common.gate_sync.set_gate_running",
        lambda unit, running: actions.append((unit, running)) or None,
    )
    save_circuits(circuits_path(runtime_path), NodeCircuits().with_only("gate"))

    message = reconcile(Settings(data_dir=runtime_path))

    assert actions == [("frigate.service", False), ("frigate.service", True)]
    assert "перезапущена" in message


def test_a_live_container_is_left_alone(runtime_path: Path, monkeypatch) -> None:
    actions: list[tuple[str, bool]] = []
    monkeypatch.setattr("src.common.gate_sync.gate_status", lambda unit: GateStatus("running"))
    monkeypatch.setattr("src.common.gate_sync._container_running", lambda name: True)
    monkeypatch.setattr(
        "src.common.gate_sync.set_gate_running",
        lambda unit, running: actions.append((unit, running)) or None,
    )
    save_circuits(circuits_path(runtime_path), NodeCircuits().with_only("gate"))

    reconcile(Settings(data_dir=runtime_path))

    assert actions == []


def test_unanswered_docker_is_not_treated_as_a_dead_container(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """Docker не ответил — судить не по чему; перезапуск наугад вреднее."""
    actions: list[tuple[str, bool]] = []
    monkeypatch.setattr("src.common.gate_sync.gate_status", lambda unit: GateStatus("running"))
    monkeypatch.setattr("src.common.gate_sync._container_running", lambda name: None)
    monkeypatch.setattr(
        "src.common.gate_sync.set_gate_running",
        lambda unit, running: actions.append((unit, running)) or None,
    )
    save_circuits(circuits_path(runtime_path), NodeCircuits().with_only("gate"))

    reconcile(Settings(data_dir=runtime_path))

    assert actions == []


def test_a_starting_service_is_given_time(runtime_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "src.common.gate_sync.gate_status",
        lambda unit: GateStatus("starting", "служба запускается"),
    )
    monkeypatch.setattr(
        "src.common.gate_sync.set_gate_running",
        lambda unit, running: (_ for _ in ()).throw(AssertionError("не должно вызываться")),
    )
    save_circuits(circuits_path(runtime_path), NodeCircuits().with_only("gate"))

    assert "запускается" in reconcile(Settings(data_dir=runtime_path))
