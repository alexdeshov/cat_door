from types import SimpleNamespace

from src.common.gate_service import GateStatus, gate_status, set_gate_running


def _fake_run(monkeypatch, answers: dict[tuple[str, ...], tuple[int, str]]) -> list[list[str]]:
    calls: list[list[str]] = []

    def run(command, **options):
        calls.append(command)
        code, output = answers.get(tuple(command), (1, ""))
        return SimpleNamespace(returncode=code, stdout=output, stderr="")

    monkeypatch.setattr("src.common.gate_service.subprocess.run", run)
    return calls


def test_running_service_is_reported_running(monkeypatch) -> None:
    _fake_run(monkeypatch, {("systemctl", "is-active", "frigate.service"): (0, "active")})

    assert gate_status("frigate.service") == GateStatus("running")


def test_absent_service_is_not_the_same_as_stopped(monkeypatch) -> None:
    """Отсутствующую службу нельзя включить, и обещать это пользователю нельзя."""
    _fake_run(
        monkeypatch,
        {
            ("systemctl", "is-active", "frigate.service"): (3, "inactive"),
            ("systemctl", "show", "frigate.service", "-p", "LoadState", "--value"): (
                0,
                "not-found",
            ),
        },
    )

    status = gate_status("frigate.service")

    assert status.state == "absent"
    assert status.deployed is False


def test_stopped_service_can_be_started(monkeypatch) -> None:
    _fake_run(
        monkeypatch,
        {
            ("systemctl", "is-active", "frigate.service"): (3, "inactive"),
            ("systemctl", "show", "frigate.service", "-p", "LoadState", "--value"): (0, "loaded"),
        },
    )

    status = gate_status("frigate.service")

    assert status.state == "stopped"
    assert status.deployed is True


def test_failed_service_is_visible(monkeypatch) -> None:
    _fake_run(monkeypatch, {("systemctl", "is-active", "frigate.service"): (3, "failed")})

    assert gate_status("frigate.service").state == "failed"


def test_starting_calls_systemctl_directly(monkeypatch) -> None:
    """Без sudo: команду исполняет примиритель, который и так работает от root.

    Через веб-процесс это не сработало бы в принципе — `NoNewPrivileges` в его
    юните запрещает любое повышение прав, включая setuid у sudo.
    """
    calls = _fake_run(
        monkeypatch,
        {("systemctl", "start", "--no-block", "frigate.service"): (0, "")},
    )

    assert set_gate_running("frigate.service", True) is None
    assert calls == [["systemctl", "start", "--no-block", "frigate.service"]]


def test_refused_sudo_is_reported_not_swallowed(monkeypatch) -> None:
    _fake_run(monkeypatch, {})

    reason = set_gate_running("frigate.service", False)

    assert reason is not None
    assert "systemctl stop" in reason


def test_activating_is_reported_as_starting_not_stopped(monkeypatch) -> None:
    """Первый запуск тянет образ минуты; «остановлена» в это время — ложь.

    Она же и вредна: примиритель пытался бы запустить ещё раз, а владелец —
    щёлкать переключателем.
    """
    _fake_run(monkeypatch, {("systemctl", "is-active", "frigate.service"): (3, "activating")})

    status = gate_status("frigate.service")

    assert status.state == "starting"
    assert status.running is True


def test_start_does_not_block_on_container_readiness(monkeypatch) -> None:
    calls = _fake_run(
        monkeypatch,
        {("systemctl", "start", "--no-block", "frigate.service"): (0, "")},
    )

    assert set_gate_running("frigate.service", True) is None
    assert calls == [["systemctl", "start", "--no-block", "frigate.service"]]
