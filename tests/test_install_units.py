"""Checks on the systemd units, which nothing else in the suite executes.

A typo in a unit file surfaces only on the runtime host, usually as a service
that silently never starts, so the invariants that tie the units to the code
are asserted here instead.
"""

import configparser
import stat
from pathlib import Path

import pytest

from src.recorder.app import RECORDER_RETRY_SECONDS

INSTALL_DIR = Path(__file__).resolve().parents[1] / "install"
UNITS = ("catdoor-recorder.service", "catdoor-web.service")


def _unit(name: str) -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    # systemd keys are case sensitive; configparser lowercases them by default.
    parser.optionxform = str  # type: ignore[method-assign]
    parser.read(INSTALL_DIR / name, encoding="utf-8")
    return parser


@pytest.mark.parametrize("name", UNITS)
def test_unit_waits_for_readiness(name: str) -> None:
    """Type=notify keeps the unit in `activating` until the process is up.

    With the default Type=simple systemd would call a service that dies on a
    bad configuration `active` for as long as the process takes to exit.
    """
    unit = _unit(name)

    assert unit["Service"]["Type"] == "notify"
    assert unit["Service"]["NotifyAccess"] == "main"


@pytest.mark.parametrize("name", UNITS)
def test_watchdog_outlives_the_status_cadence(name: str) -> None:
    """The ping period must exceed the interval between pings by a wide margin.

    The recorder pings whenever it publishes status, at least once per retry
    interval. A watchdog tighter than that would restart a healthy process.
    """
    watchdog_seconds = int(_unit(name)["Service"]["WatchdogSec"])

    assert watchdog_seconds > RECORDER_RETRY_SECONDS * 4


@pytest.mark.parametrize("name", UNITS)
def test_unit_runs_unprivileged_from_the_virtualenv(name: str) -> None:
    service = _unit(name)["Service"]

    assert service["User"] == "catdoor"
    assert service["ExecStart"].startswith("/opt/catdoor/.venv/bin/python -m ")
    assert service["WorkingDirectory"] == "/opt/catdoor"
    assert service["NoNewPrivileges"] == "true"


@pytest.mark.parametrize("name", UNITS)
def test_only_the_data_volume_is_writable(name: str) -> None:
    """ProtectSystem=strict makes everything read-only, including /var/lib.

    Without the explicit exception the services cannot write a single event.
    """
    service = _unit(name)["Service"]

    assert service["ProtectSystem"] == "strict"
    assert service["ReadWritePaths"] == "/var/lib/catdoor"


@pytest.mark.parametrize("name", UNITS)
def test_service_never_stops_retrying(name: str) -> None:
    """No start limit: an unattended node must keep trying after a bad night."""
    service = _unit(name)["Service"]

    assert service["Restart"] == "always"
    assert int(service["RestartMaxDelaySec"]) == 1800
    assert "StartLimitBurst" not in service


def test_installer_is_executable() -> None:
    mode = (INSTALL_DIR / "install.sh").stat().st_mode

    assert mode & stat.S_IXUSR
