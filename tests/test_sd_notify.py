import os
import socket
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from src.common.sd_notify import Notifier


@pytest.fixture
def listener() -> Iterator[socket.socket]:
    """A datagram socket standing in for the one systemd would provide.

    Bound under /tmp rather than pytest's tmp_path: a unix socket path cannot
    exceed roughly one hundred bytes, and the per-test directory on macOS is
    already longer than that.
    """
    server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        server.bind(str(Path(directory) / "notify.sock"))
        server.settimeout(1)
        yield server
        server.close()


def _address(listener: socket.socket) -> str:
    return str(listener.getsockname())


def test_notifier_is_inert_without_a_socket() -> None:
    """Running from a shell must stay exactly as it was."""
    notifier = Notifier.from_environment({})

    assert not notifier.enabled
    assert notifier.watchdog_interval_seconds is None
    notifier.ready("ignored")
    notifier.watchdog()


def test_ready_carries_the_status_line(listener: socket.socket) -> None:
    notifier = Notifier(_address(listener))

    notifier.ready("Запись идёт")

    assert listener.recv(1024).decode("utf-8") == "READY=1\nSTATUS=Запись идёт"


def test_watchdog_ping_is_sent(listener: socket.socket) -> None:
    notifier = Notifier(_address(listener))

    notifier.watchdog()

    assert listener.recv(1024) == b"WATCHDOG=1"


def test_status_never_spans_two_lines(listener: socket.socket) -> None:
    """Newlines separate fields, so a multi-line message would forge one."""
    notifier = Notifier(_address(listener))

    notifier.status("first\nREADY=1")

    assert listener.recv(1024) == b"STATUS=first READY=1"


def test_watchdog_interval_is_half_the_timeout(listener: socket.socket) -> None:
    notifier = Notifier(_address(listener), watchdog_usec=60_000_000)

    assert notifier.watchdog_interval_seconds == 30.0


def test_watchdog_period_is_ignored_when_it_belongs_to_another_process() -> None:
    """WATCHDOG_USEC is inherited; WATCHDOG_PID says whose timeout it is."""
    notifier = Notifier.from_environment(
        {
            "NOTIFY_SOCKET": "/run/systemd/notify",
            "WATCHDOG_USEC": "60000000",
            "WATCHDOG_PID": str(os.getpid() + 1),
        }
    )

    assert notifier.enabled
    assert notifier.watchdog_interval_seconds is None


def test_watchdog_period_is_kept_when_it_is_ours() -> None:
    notifier = Notifier.from_environment(
        {
            "NOTIFY_SOCKET": "/run/systemd/notify",
            "WATCHDOG_USEC": "60000000",
            "WATCHDOG_PID": str(os.getpid()),
        }
    )

    assert notifier.watchdog_interval_seconds == 30.0


def test_abstract_socket_address_is_translated() -> None:
    """A leading @ stands for the NUL byte of the abstract namespace."""
    notifier = Notifier("@catdoor-notify")

    assert notifier.enabled
    assert notifier._address == "\0catdoor-notify"


def test_delivery_failure_does_not_propagate(tmp_path: Path) -> None:
    """Losing a ping costs a restart; raising would cost the recording."""
    notifier = Notifier(str(tmp_path / "nobody-is-listening.sock"))

    notifier.ready()
    notifier.watchdog()
