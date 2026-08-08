from types import SimpleNamespace

from src.common.gate_service import (
    GateStatus,
    frigate_safe_mode,
    gate_status,
    set_gate_running,
)


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


def test_frigate_without_cameras_is_not_reported_running(monkeypatch) -> None:
    """Служба жива, а камер в ней нет — для systemd это здоровый контур.

    2026-08-08 узел простоял так час: камера, названная цифрой, сделала
    конфигурацию негодной, Frigate поднялся в safe mode, а пульт показывал
    «работает».
    """
    monkeypatch.setattr("src.common.gate_service._run", lambda command: (0, "active"))
    monkeypatch.setattr("src.common.gate_service.frigate_safe_mode", lambda url, **kw: True)

    status = gate_status("frigate.service", "http://127.0.0.1:5000")

    assert status.state == "degraded"
    assert "камер нет" in status.detail
    # Примирителю с ней делать нечего: беда в конфигурации, а не в запуске.
    assert status.running is True


def test_unreachable_frigate_does_not_invent_a_breakage(monkeypatch) -> None:
    """Неизвестность и «всё сломано» — разные вещи."""
    monkeypatch.setattr("src.common.gate_service._run", lambda command: (0, "active"))
    monkeypatch.setattr("src.common.gate_service.frigate_safe_mode", lambda url, **kw: None)

    assert gate_status("frigate.service", "http://127.0.0.1:5000").state == "running"


def test_safe_mode_is_read_without_credentials(monkeypatch) -> None:
    """Внутренний порт Frigate отвечает без пароля — тем и пользуемся."""
    import io

    class _Answer:
        def __init__(self, body: bytes) -> None:
            self._body = io.BytesIO(body)

        def read(self) -> bytes:
            return self._body.read()

        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            return None

    monkeypatch.setattr(
        "src.common.gate_service.urllib.request.urlopen",
        lambda url, timeout: _Answer(b'{"safe_mode": true, "cameras": {}}'),
    )

    assert frigate_safe_mode("http://127.0.0.1:5000") is True
