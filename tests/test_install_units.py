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
    """No start limit: an unattended node must keep trying after a bad night.

    Отключение лимита задаётся явно: умолчание systemd — пять запусков за
    десять секунд, после чего юнит остаётся в `failed` и узел молча перестаёт
    подниматься.
    """
    unit = _unit(name)
    service = unit["Service"]

    assert service["Restart"] == "always"
    assert int(service["RestartMaxDelaySec"]) == 1800
    assert "StartLimitBurst" not in service
    assert int(unit["Unit"]["StartLimitIntervalSec"]) == 0


def test_installer_is_executable() -> None:
    mode = (INSTALL_DIR / "install.sh").stat().st_mode

    assert mode & stat.S_IXUSR


def _dnsmasq_directives() -> dict[str, list[str]]:
    lines = (INSTALL_DIR / "dnsmasq-catdoor-cameras.conf").read_text(encoding="utf-8")
    directives: dict[str, list[str]] = {}
    for line in lines.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, _, value = stripped.partition("=")
        directives.setdefault(key, []).append(value)
    return directives


def test_dhcp_serves_only_the_camera_interface() -> None:
    """Bound to eno1 alone: the home LAN already has a DHCP server on it.

    Именно `bind-dynamic`, а не `bind-interfaces`: второй требует, чтобы
    интерфейс существовал в момент старта, и узел не поднимался после
    перезагрузки — dnsmasq стартует раньше, чем netplan настраивает eno1.
    """
    directives = _dnsmasq_directives()

    assert directives["interface"] == ["eno1"]
    assert "bind-dynamic" in directives
    assert "bind-interfaces" not in directives


def test_dhcp_does_not_answer_dns() -> None:
    """systemd-resolved already holds port 53, and cameras need no names."""
    assert _dnsmasq_directives()["port"] == ["0"]


def test_cameras_are_given_no_way_out() -> None:
    """Options 3 and 6 carry no address, which tells dnsmasq to omit them.

    Option 3 is the default gateway and option 6 the resolver. Naming an option
    without a value suppresses it, so a camera gets an address and no route
    anywhere — docs/cameras.md requires the camera network to have no internet
    access.
    """
    assert _dnsmasq_directives()["dhcp-option"] == ["3", "6"]


def test_dhcp_pool_avoids_the_factory_addresses() -> None:
    """Пул не должен накрывать адреса, на которые камеры встают после сброса.

    `.10` — узел в netplan, `.64` — заводской у Hikvision и HiWatch, `.108` — у
    Dahua, а на этой платформе сделаны и Nobelic, и часть камер Ivideon. Камера
    с заводской статикой внутри пула перехватывает адрес соседки молча: RTSP
    отвечает 401, и со стороны узла это неотличимо от неверного пароля.
    """
    start, end, _lease = _dnsmasq_directives()["dhcp-range"][0].split(",")

    assert int(start.rsplit(".", 1)[1]) > 108
    assert int(end.rsplit(".", 1)[1]) < 255
