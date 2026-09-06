"""Сторож здоровья узла: беды уходят в Телеграм, а не ждут, пока их заметят.

Появился после 2026-08-21. Ночью моргнул свет, узел перезагрузился, служба
калитки поднялась раньше брокера и три часа прожила без створки, а камера
калитки вовсе не вернулась после обрыва — и всё это не сказало владельцу ни
слова: вкладка «Состояние» знает многое, но смотрят на неё глазами. Этот
сторож раз в минуту сверяет всё то же самое и говорит вслух.

Что проверяется:

- `/api/health` админки — весь эталон узла: детектор, камеры (включая «камера
  не отдаёт кадры»), служба калитки (створка слушается), железо, копии;
- сами службы узла: `systemctl is-active` по каждой из `WATCHED_UNITS`;
- недоступность самой админки — тоже беда, а не молчание;
- синхронизация часов (после обрыва питания они врут);
- мины, стреляющие только при перезагрузке: негодная конфигурация dnsmasq
  (`dnsmasq --test`) и наблюдаемые службы вне автозагрузки (`is-enabled`);
- перезагрузка узла — отдельное разовое сообщение.

Как уведомляется. Беда должна продержаться `CONFIRM_CYCLES` циклов подряд —
мигание на границе порога не будит владельца. Появившаяся беда — одно
сообщение; длящаяся — напоминание раз в `REMIND_SECONDS`; ушедшая —
«выздоровело», тоже после `CONFIRM_CYCLES` чистых циклов. Всё, что случилось
за цикл, склеивается в одно сообщение. После загрузки узла первые
`BOOT_GRACE_SECONDS` беды не считаются: службам положено время подняться.

Машина состояний — чистые функции над словарём состояния; файл состояния
переживает перезапуск сторожа, поэтому рестарт не повторяет старые тревоги.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from src.common.sd_notify import Notifier
from src.common.settings import ConfigurationError, Settings, validate_runtime_paths
from src.common.telegram import TelegramClient

logger = logging.getLogger("catdoor.healthwatch")

POLL_SECONDS = 60.0
CONFIRM_CYCLES = 2
REMIND_SECONDS = 6 * 3600.0
BOOT_GRACE_SECONDS = 180.0
HEALTH_TIMEOUT_SECONDS = 15.0

# Дочерним процессам сокет systemd не полагается: systemctl, унаследовав
# NOTIFY_SOCKET службы, шлёт в него свои уведомления, и журнал засоряется
# «Got notification message from PID …, but reception only permitted for main
# PID» на каждый цикл (найдено при первом же запуске 2026-08-21).
def _clean_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key != "NOTIFY_SOCKET"}


WATCHED_UNITS = (
    "catdoor-recorder",
    "catdoor-web",
    "catdoor-gate-access",
    "catdoor-gate-notify",
    "catdoor-cloud-backup",
    # Кошачья копия появилась 30 августа, а в этот список попала только 31-го,
    # аудитом: новая служба без строки здесь — слепое пятно, о падении которого
    # никто не скажет. Тест `test_every_installed_service_is_watched` не даст
    # повториться.
    "catdoor-cloud-backup-cats",
    # Инфраструктура узла — не catdoor, но без неё контуры слепнут.
    # 2026-08-21 dnsmasq не поднялся после обрыва питания из-за забытого
    # .bak-файла в /etc/dnsmasq.d, камеры на DHCP остались без адресов —
    # и никто не сказал ни слова: список наблюдал только свои службы.
    "dnsmasq",
    "frigate",
)

# Статусы чеков, которые считаются бедой. «unknown» — не беда, а незнание:
# алерт по нему будил бы владельца от каждого недособранного показания.
ALERT_STATUSES = {"fail", "warn"}


def state_path(data_dir: Path) -> Path:
    return data_dir / "runtime" / "health-watch.json"


def load_state(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_state(path: Path, state: dict[str, Any]) -> None:
    try:
        scratch = path.with_suffix(".tmp")
        scratch.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        scratch.replace(path)
    except OSError as exc:
        logger.warning("Состояние сторожа не записано: %s", exc)


@dataclass(frozen=True)
class Trouble:
    """Одна наблюдаемая беда этого цикла."""

    id: str
    title: str
    # «warn» — есть время подумать, «fail» — уже сломано. В чате они должны
    # различаться значком: хроника про место на диске неотличима была от
    # упавшей копии.
    severity: str = "fail"


def reconcile(
    observed: list[Trouble],
    state: dict[str, Any],
    *,
    now: float,
    timezone: ZoneInfo,
) -> tuple[list[str], dict[str, Any]]:
    """Свести наблюдения цикла с памятью и решить, о чём говорить.

    Возвращает строки для сообщения и новое состояние. Чистая функция: вся
    дребезго- и спам-защита проверяется таблицей случаев, а не ночёвкой у
    калитки.
    """
    seen = {trouble.id: trouble for trouble in observed}
    problems: dict[str, dict[str, Any]] = dict(state.get("problems") or {})
    pending: dict[str, int] = dict(state.get("pending") or {})
    lines: list[str] = []

    def moment(epoch: float) -> str:
        return datetime.fromtimestamp(epoch, timezone).strftime("%H:%M")

    for trouble in observed:
        if trouble.id in problems:
            record = problems[trouble.id]
            record["misses"] = 0
            if now - float(record.get("notified_at") or 0) >= REMIND_SECONDS:
                lines.append(
                    f"⚠️ Всё ещё: {trouble.title} (с {moment(float(record['since']))})"
                )
                record["notified_at"] = now
            continue
        count = pending.get(trouble.id, 0) + 1
        if count >= CONFIRM_CYCLES:
            problems[trouble.id] = {
                "title": trouble.title,
                "since": now,
                "notified_at": now,
                "misses": 0,
            }
            pending.pop(trouble.id, None)
            icon = "🟡" if trouble.severity == "warn" else "🔴"
            lines.append(f"{icon} {trouble.title}")
        else:
            pending[trouble.id] = count
    for missing in [key for key in pending if key not in seen]:
        # Не подтвердилось — забываем без слов: это и есть защита от дребезга.
        pending.pop(missing)
    for key in list(problems):
        if key in seen:
            continue
        record = problems[key]
        record["misses"] = int(record.get("misses") or 0) + 1
        if record["misses"] >= CONFIRM_CYCLES:
            lasted = now - float(record["since"])
            shown = f"{lasted / 3600:.1f} ч" if lasted >= 3600 else f"{lasted / 60:.0f} мин"
            lines.append(f"✅ Прошло: {record['title']} (длилось {shown})")
            problems.pop(key)

    fresh = dict(state)
    fresh["problems"] = problems
    fresh["pending"] = pending
    return lines, fresh


# --- Сборщики наблюдений -----------------------------------------------------


def fetch_health(url: str) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(url, timeout=HEALTH_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 — любой отказ означает одно: админка не отвечает
        return None


def health_troubles(payload: dict[str, Any] | None) -> list[Trouble]:
    """Беды из `/api/health`: каждый чек не-ok — отдельная беда со своим id.

    Отдельная, а не «узел нездоров» одной строкой: владельцу в чате нужно имя
    камеры и суть, а не приглашение открыть вкладку.
    """
    if payload is None:
        return [Trouble("health-api", "Админка не отвечает — здоровье узла не видно")]
    troubles = []
    for group in payload.get("groups") or []:
        for check in group.get("checks") or []:
            status = str(check.get("status"))
            if status not in ALERT_STATUSES:
                continue
            actual = str(check.get("actual") or "—")
            troubles.append(
                Trouble(
                    f"check:{check.get('key')}",
                    f"{group.get('title')}: {check.get('title')} — {actual}",
                    severity=status,
                )
            )
    return troubles


def unit_troubles(units: tuple[str, ...] = WATCHED_UNITS) -> list[Trouble]:
    troubles = []
    for unit in units:
        done = subprocess.run(
            ["systemctl", "is-active", f"{unit}.service"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            env=_clean_env(),
        )
        status = done.stdout.strip() or "unknown"
        if status != "active":
            troubles.append(
                Trouble(f"unit:{unit}", f"Служба {unit} не работает ({status})")
            )
    return troubles


def latent_troubles(units: tuple[str, ...] = WATCHED_UNITS) -> list[Trouble]:
    """Мины, взведённые сейчас, но стреляющие только при перезагрузке.

    Обе — с натуры 2026-08-21. Забытый `.bak` в `/etc/dnsmasq.d` дублировал
    dhcp-host: работающему dnsmasq он был безразличен, а первый же ребут после
    обрыва питания уронил старт — и камеры остались без адресов. Юнит frigate
    оказался disabled: сегодня его подняли зависимости, а полагаться на это
    нельзя. Ждать следующего ребута, чтобы узнать, — поздно; проверять надо,
    пока узел жив.
    """
    troubles = []
    done = subprocess.run(
        ["/usr/sbin/dnsmasq", "--test"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
        env=_clean_env(),
    )
    if done.returncode != 0:
        reason = (done.stderr or done.stdout).strip().splitlines()
        troubles.append(
            Trouble(
                "latent:dnsmasq-config",
                "Конфигурация dnsmasq не переживёт перезагрузку: "
                + (reason[-1][:140] if reason else "dnsmasq --test провалился"),
            )
        )
    for unit in units:
        done = subprocess.run(
            ["systemctl", "is-enabled", f"{unit}.service"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            env=_clean_env(),
        )
        status = done.stdout.strip()
        if status and status not in ("enabled", "static", "alias", "indirect"):
            troubles.append(
                Trouble(
                    f"latent:enable-{unit}",
                    f"Служба {unit} не в автозагрузке ({status}) — "
                    "после перезагрузки не поднимется",
                )
            )
    return troubles


def clock_trouble() -> list[Trouble]:
    """Часы после обрыва питания идут назад — и тогда решения по времени врут."""
    done = subprocess.run(
        ["timedatectl", "show", "--property=NTPSynchronized", "--value"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
        env=_clean_env(),
    )
    if done.returncode != 0:
        return []
    if done.stdout.strip() == "yes":
        return []
    return [Trouble("clock", "Часы узла не синхронизированы — время решений врёт")]


def boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except OSError:
        return ""


def uptime_seconds() -> float:
    try:
        return float(Path("/proc/uptime").read_text(encoding="ascii").split()[0])
    except (OSError, ValueError, IndexError):
        return BOOT_GRACE_SECONDS + 1


def boot_message(state: dict[str, Any], *, timezone: ZoneInfo) -> tuple[str | None, dict[str, Any]]:
    """Разовое сообщение о перезагрузке узла.

    2026-08-21 узел перезагрузил обрыв питания — владелец узнал об этом от
    аудита через три часа. Первая запись состояния сообщение не шлёт: сторож
    только что установлен, а не узел только что упал.
    """
    current = boot_id()
    if not current:
        return None, state
    stored = state.get("boot_id")
    fresh = dict(state)
    fresh["boot_id"] = current
    if stored is None or stored == current:
        # Недоставленная заметка о прошлой перезагрузке остаётся в состоянии
        # и уходит со следующим циклом: учебный ребут 2026-08-21 показал, что
        # сторож просыпается раньше Wi-Fi и DNS, и «🔄» первой попытки терялся
        # навсегда — при настоящем обрыве света владелец снова остался бы без
        # главного сообщения.
        return fresh.get("boot_note"), fresh
    started = datetime.fromtimestamp(time.time() - uptime_seconds(), timezone)
    fresh["boot_note"] = (
        f"🔄 Узел перезагружался: загрузка в {started.strftime('%H:%M:%S')}. "
        "Обычные причины — свет или обновление ядра; камеры и брокер проверяю следом."
    )
    return fresh["boot_note"], fresh


def run(*, once: bool = False) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    settings = Settings()
    validate_runtime_paths(settings)
    timezone = ZoneInfo(settings.display_timezone)
    telegram = TelegramClient(
        settings.telegram_bot_token,
        settings.telegram_chat_id,
        api_base=settings.telegram_api_base,
    )
    if not telegram.configured:
        # Сторож без голоса бесполезен, но падать не должен: журнал остаётся.
        logger.warning("Бот не настроен: беды узла попадут только в журнал сторожа")
    health_url = "http://127.0.0.1:8080/api/health"
    path = state_path(settings.data_dir)
    notifier = Notifier.from_environment()
    notifier.ready("Сторож здоровья на связи")
    while True:
        state = load_state(path)
        lines: list[str] = []
        booted, state = boot_message(state, timezone=timezone)
        if booted:
            lines.append(booted)
        if uptime_seconds() >= BOOT_GRACE_SECONDS:
            observed = (
                health_troubles(fetch_health(health_url))
                + unit_troubles()
                + latent_troubles()
                + clock_trouble()
            )
            spoken, state = reconcile(
                observed, state, now=time.time(), timezone=timezone
            )
            lines.extend(spoken)
        for line in lines:
            logger.info("%s", line)
        delivered = True
        if lines and telegram.configured:
            try:
                telegram.send_message("Здоровье узла\n" + "\n".join(lines))
            except Exception as exc:  # noqa: BLE001
                # Телеграм недоступен — беда не отменяется: напоминание
                # повторит её, а заметка о перезагрузке останется в состоянии
                # и уйдёт следующим циклом, когда сеть поднимется.
                delivered = False
                logger.warning("Сообщение не ушло: %s", exc)
        if delivered:
            # Без бота заметка тоже снимается: вечный повтор в журнал — шум.
            state.pop("boot_note", None)
        # Отметка круга — чтобы страница «Состояние» видела, жив ли сам сторож:
        # за сторожем смотрит только systemd, а он о молчании не скажет.
        state["checked_at"] = time.time()
        save_state(path, state)
        notifier.watchdog()
        if once:
            return
        time.sleep(POLL_SECONDS)


def main() -> None:
    try:
        run()
    except ConfigurationError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
