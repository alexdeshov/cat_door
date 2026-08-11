from __future__ import annotations

import logging
import subprocess

from src.gate.presence import PhonePresence, phone_factor

MAC = "aa:bb:cc:dd:ee:ff"


def _runner(neighbour_output: str, *, calls: list[list[str]] | None = None):  # type: ignore[no-untyped-def]
    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        if calls is not None:
            calls.append(command)
        output = neighbour_output if command[0] == "ip" else ""
        return subprocess.CompletedProcess(command, 0, output, "")

    return run


def test_a_phone_on_the_home_network_says_yes() -> None:
    run = _runner(f"192.168.2.50 dev wlp2s0 lladdr {MAC} REACHABLE\n")

    assert phone_factor("192.168.2.50", MAC, run=run).verdict == "for"


def test_a_sleeping_phone_still_answers_arp() -> None:
    """Телефоны игнорируют ping, а на «кто держит адрес» отвечает прошивка Wi-Fi.

    Ядро помечает такого соседа `STALE`: адрес известен, свежесть не
    подтверждена. Это присутствие, а не его отсутствие.
    """
    run = _runner(f"192.168.2.50 dev wlp2s0 lladdr {MAC} STALE\n")

    assert phone_factor("192.168.2.50", MAC, run=run).verdict == "for"


def test_a_phone_that_did_not_answer_says_no() -> None:
    run = _runner("192.168.2.50 dev wlp2s0 FAILED\n")

    assert phone_factor("192.168.2.50", MAC, run=run).verdict == "against"


def test_an_unresolved_address_is_not_an_answer() -> None:
    """Пустая таблица соседей — «не знаю»: ядро могло и не спрашивать."""
    run = _runner("")

    assert phone_factor("192.168.2.50", MAC, run=run).verdict == "unknown"


def test_a_changed_mac_is_noticed_out_loud(caplog) -> None:
    """iOS и Android умеют менять приватный MAC, и привязка рвётся молча.

    Молчание — худший исход: система просто перестала бы открывать, и никто не
    понял бы почему.
    """
    run = _runner("192.168.2.50 dev wlp2s0 lladdr 11:22:33:44:55:66 REACHABLE\n")

    with caplog.at_level(logging.WARNING):
        factor = phone_factor("192.168.2.50", MAC, run=run)

    assert factor.verdict == "against"
    assert "не совпал" in caplog.text


def test_a_missing_tool_never_opens_the_gate() -> None:
    """Недоступность сети или отсутствие `ip` — это «не знаю», а не «за»."""

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        raise OSError("ip: команда не найдена")

    assert phone_factor("192.168.2.50", MAC, run=run).verdict == "unknown"


def test_a_phone_without_settings_is_unknown() -> None:
    assert phone_factor("", "", run=_runner("")).verdict == "unknown"


def test_the_same_address_is_not_polled_every_second() -> None:
    """Служба смотрит события раз в секунду: без памяти это шквал процессов."""
    calls: list[list[str]] = []
    clock = {"now": 0.0}
    presence = PhonePresence(
        run=_runner(f"192.168.2.50 dev wlp2s0 lladdr {MAC} REACHABLE\n", calls=calls),
        clock=lambda: clock["now"],
    )

    presence.factor("192.168.2.50", MAC)
    presence.factor("192.168.2.50", MAC)
    clock["now"] = 60.0
    presence.factor("192.168.2.50", MAC)

    assert [command[0] for command in calls] == ["ping", "ip", "ping", "ip"]
