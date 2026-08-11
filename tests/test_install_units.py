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


def test_every_unit_is_copied_by_the_installer() -> None:
    """Юнит, не попавший в установщик, обновляется только руками.

    Так уже было с примирителем контура калитки: он жил отдельной ручной
    установкой, и правки его юнита молча не доезжали на узел.
    """
    installer = (INSTALL_DIR / "install.sh").read_text(encoding="utf-8")
    missing = [
        path.name
        for path in sorted(INSTALL_DIR.glob("catdoor-*.*"))
        if path.suffix in {".service", ".path", ".timer"} and path.name not in installer
    ]

    assert missing == []


def test_the_gate_journal_needs_no_token_to_be_enabled() -> None:
    """Журнал проходов включается безусловно, в отличие от уведомлений.

    Уведомлениям нужен бот, и без токена служба падала бы каждые пять секунд.
    Журналу не нужно ничего: он читает ленту Frigate и пишет свою таблицу.
    """
    installer = (INSTALL_DIR / "install.sh").read_text(encoding="utf-8")

    assert "systemctl enable catdoor-gate-access.service" in installer
    assert "-m src.gate.access" in (
        (INSTALL_DIR / "catdoor-gate-access.service").read_text(encoding="utf-8")
    )


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


def test_cameras_are_given_a_gateway_and_a_clock_but_no_resolver() -> None:
    """Опция 3 — шлюз, 42 — сервер времени, 6 — DNS без значения, то есть молчок.

    Шлюз выдаётся не для дороги наружу, а потому что аренду без него прошивки
    отвергают: HiWatch уходит в link-local, Dahua — на запасной адрес. Время
    камерам взять больше неоткуда — заводской NTP смотрит в интернет, которого
    у них нет. Имена по-прежнему не нужны, и DNS на узле отключён.
    """
    assert _dnsmasq_directives()["dhcp-option"] == [
        "3,192.168.1.10",
        "6",
        "42,192.168.1.10",
    ]


def test_time_is_served_only_to_the_camera_segment() -> None:
    """Узел раздаёт время камерам и никому больше.

    `allow` без адреса открыл бы NTP всей домашней сети, а `local stratum`
    нужен, чтобы камеры получали время и в час, когда сам узел не дотянулся до
    upstream: сеть камер изолирована, и ждать интернета там некому.
    """
    chrony = (INSTALL_DIR / "chrony-catdoor-cameras.conf").read_text(encoding="utf-8")
    directives = [
        line.strip()
        for line in chrony.splitlines()
        if line.strip() and not line.startswith("#")
    ]

    assert "allow 192.168.1.0/24" in directives
    assert not any(line == "allow" or line.startswith("allow all") for line in directives)
    assert any(line.startswith("local stratum") for line in directives)


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


DATA_UNITS = (
    "catdoor-recorder.service",
    "catdoor-web.service",
    "catdoor-gate-notify.service",
    "catdoor-gate-access.service",
    "catdoor-gate-sync.service",
    "catdoor-backup.service",
)


@pytest.mark.parametrize("name", DATA_UNITS)
def test_unit_waits_for_the_data_volume(name: str) -> None:
    """Каталог данных лежит на отдельном томе, и systemd про него не знает.

    Путь приходит из окружения, а не из юнита, поэтому неявной зависимости
    systemd не выводит. Служба, стартовавшая раньше монтирования, пишет в
    пустой каталог под точкой монтирования — и делает это молча.

    2026-08-09 ровно это случилось с Frigate: контейнер поднялся на 2.3 секунды
    раньше тома, четыре часа писал записи не на тот диск и не видел библиотеки
    лиц. Гонка проигрывалась и при чистой загрузке — её маскировали перезапуски
    по другим поводам.
    """
    text = (INSTALL_DIR / name).read_text(encoding="utf-8")

    line = next(
        (l for l in text.splitlines() if l.startswith("RequiresMountsFor=")),
        None,
    )
    assert line is not None, f"{name} не ждёт своего тома"
    assert "/var/lib/catdoor" in line
