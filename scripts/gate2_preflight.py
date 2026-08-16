#!/usr/bin/env python3
"""Приёмка второго узла: одной командой видно, готов ли он к перевозке.

Задуман по образцу `scripts/acceptance.py`, но проверяет не контуры, а
живучесть самой машины у чужого дома: чем она найдётся в сети, что поднимется
после загрузки и что обязано молчать, пока камеры не подключены.

Запуск на узле: sudo python3 scripts/gate2_preflight.py
Коды возврата: 0 — всё сошлось, 1 — есть строки «беда».
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

WIFI_IFACE = "wlp0s20f3"
CAMERA_IFACE = "eno1"
CAMERA_SUBNET = "192.168.3."
CAMERA_ADDRESS = "192.168.3.1/24"
HOSTNAME = "gate2"

OK, WARN, BAD = "норма", "внимание", "беда"


@dataclass
class Check:
    name: str
    status: str
    detail: str


def run(*args: str) -> str:
    """Выполнить команду и вернуть вывод; пустая строка вместо исключения."""
    try:
        done = subprocess.run(
            args, capture_output=True, text=True, timeout=20, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (done.stdout + done.stderr).strip()


def unit_state(unit: str) -> tuple[str, str]:
    return run("systemctl", "is-active", unit), run("systemctl", "is-enabled", unit)


def check_identity() -> list[Check]:
    host = run("hostname")
    hosts_file = Path("/etc/hosts").read_text(encoding="utf-8")
    ok = host == HOSTNAME and f"127.0.1.1 {HOSTNAME}" in hosts_file
    return [
        Check(
            "имя узла",
            OK if ok else BAD,
            f"hostname={host}, запись в /etc/hosts "
            f"{'есть' if f'127.0.1.1 {HOSTNAME}' in hosts_file else 'отсутствует'}",
        )
    ]


def check_mdns() -> list[Check]:
    active, enabled = unit_state("avahi-daemon")
    resolved = run("avahi-resolve", "-n", f"{HOSTNAME}.local")
    address = resolved.split()[-1] if resolved else ""
    # Свежий avahi объявлял узел адресом docker-моста: спасательный кабель был
    # бы мёртв ровно тогда, когда нужен. Отсюда проверка не факта ответа, а
    # того, каким адресом узел себя называет.
    plausible = address.startswith(("192.168.", "169.254."))
    status = OK if active == "active" and enabled == "enabled" and plausible else BAD
    return [
        Check(
            "mDNS",
            status,
            f"avahi {active}/{enabled}, {HOSTNAME}.local → {address or 'нет ответа'}",
        )
    ]


def check_network() -> list[Check]:
    checks: list[Check] = []
    # Разбираем YAML, а не отступы: первая версия ловила регулярным выражением
    # уровень `auth:` и бодро сообщала «две точки доступа: auth, auth».
    merged = yaml.safe_load(run("netplan", "get", "wifis")) or {}
    points = list(merged.get(WIFI_IFACE, {}).get("access-points", {}))
    checks.append(
        Check(
            "точки доступа",
            OK if len(points) >= 2 else BAD,
            f"{len(points)} шт.: {', '.join(points) if points else 'не найдены'}",
        )
    )

    eno1 = yaml.safe_load(run("netplan", "get", f"ethernets.{CAMERA_IFACE}")) or {}
    has_address = CAMERA_ADDRESS in (eno1.get("addresses") or [])
    has_linklocal = "ipv4" in (eno1.get("link-local") or [])
    checks.append(
        Check(
            "сеть камер",
            OK if has_address and has_linklocal else BAD,
            f"{CAMERA_IFACE}: адрес {'есть' if has_address else 'нет'}, "
            f"link-local {'есть' if has_linklocal else 'нет'}",
        )
    )

    disabled = Path("/etc/cloud/cloud.cfg.d/99-disable-network-config.cfg")
    checks.append(
        Check(
            "cloud-init не трогает сеть",
            OK if disabled.exists() else BAD,
            "запрет на месте" if disabled.exists() else "запрета нет — сеть перепишут",
        )
    )
    return checks


def check_wifi_health() -> list[Check]:
    checks: list[Check] = []
    active, enabled = unit_state("wifi-keepalive")
    checks.append(
        Check(
            "keepalive Wi-Fi",
            OK if active == "active" and enabled == "enabled" else BAD,
            f"{active}/{enabled}",
        )
    )

    power = run("iw", "dev", WIFI_IFACE, "get", "power_save")
    scheme = Path("/sys/module/iwlmvm/parameters/power_scheme")
    scheme_value = scheme.read_text(encoding="utf-8").strip() if scheme.exists() else "?"
    # power_scheme=1 применяется только при загрузке модуля: пока узел не
    # перезагружали после правки modprobe, здесь честно будет 2.
    checks.append(
        Check(
            "энергосбережение",
            OK if "off" in power and scheme_value == "1" else WARN,
            f"{power or 'iw молчит'}, power_scheme={scheme_value}",
        )
    )
    return checks


def check_silent_services() -> list[Check]:
    """То, что обязано молчать, пока узел не приехал на место."""
    checks: list[Check] = []
    for unit, why in (
        ("dnsmasq", "DHCP камер: авторитетный сервер в чужой сети опасен"),
        ("hub", "брокер и push включаются на месте"),
        ("frigate", "камер ещё нет"),
    ):
        active, enabled = unit_state(unit)
        quiet = active != "active" and enabled != "enabled"
        checks.append(
            Check(f"молчит: {unit}", OK if quiet else BAD, f"{active}/{enabled} — {why}")
        )
    return checks


def check_time() -> list[Check]:
    active, _ = unit_state("chrony")
    conf = Path("/etc/chrony/conf.d/catdoor-cameras.conf")
    text = conf.read_text(encoding="utf-8") if conf.exists() else ""
    ready = f"allow {CAMERA_SUBNET}0/24" in text and "local stratum" in text
    return [
        Check(
            "время для камер",
            OK if active == "active" and ready else BAD,
            f"chrony {active}, раздача в {CAMERA_SUBNET}0/24 "
            f"{'настроена' if ready else 'не настроена'}",
        )
    ]


def check_payload() -> list[Check]:
    images = run("docker", "images", "--format", "{{.Repository}}")
    wanted = {
        "ghcr.io/blakeblackshear/frigate": "Frigate",
        "eclipse-mosquitto": "брокер",
        "binwiederhier/ntfy": "push",
    }
    missing = [name for repo, name in wanted.items() if repo not in images]
    free_gb = shutil.disk_usage("/").free // 2**30
    return [
        Check(
            "образы скачаны",
            OK if not missing else BAD,
            "все три" if not missing else f"нет: {', '.join(missing)}",
        ),
        Check(
            "место на диске",
            OK if free_gb >= 20 else WARN,
            f"{free_gb} ГБ свободно",
        ),
    ]


def check_permissions() -> list[Check]:
    prep = Path("/etc/sudoers.d/50-claude-prep")
    if not prep.exists():
        return [Check("права sudo", OK, "временных прав нет")]
    wide = "NOPASSWD:ALL" in prep.read_text(encoding="utf-8").replace(" ", "")
    return [
        Check(
            "права sudo",
            WARN if wide else OK,
            "временный NOPASSWD:ALL ещё стоит — сузить перед погрузкой"
            if wide
            else "права сужены",
        )
    ]


def main() -> int:
    checks: list[Check] = []
    for step in (
        check_identity,
        check_mdns,
        check_network,
        check_wifi_health,
        check_silent_services,
        check_time,
        check_payload,
        check_permissions,
    ):
        checks.extend(step())

    width = max(len(c.name) for c in checks)
    for check in checks:
        print(f"{check.status:9} {check.name:{width}}  {check.detail}")

    bad = sum(1 for c in checks if c.status == BAD)
    warn = sum(1 for c in checks if c.status == WARN)
    print(f"\nитог: бед {bad}, предупреждений {warn}, всего проверок {len(checks)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
