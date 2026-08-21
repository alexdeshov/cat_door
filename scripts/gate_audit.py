"""Аудит калитки: перекрёстная сверка решений, створки и ленты Frigate.

Владелец не всегда дома, и некорректный проход или фантомный щелчок легко
пропустить. Скрипт раз в несколько дней (скилл `/gate-audit`) сверяет три
независимых источника за окно аудита:

- журнал решений службы (`/api/gate/passes` админки): кто, решение, импульсы;
- журнал самой службы (journald): переходы створки по классификатору,
  предупреждения о плате и брокере;
- ленту Frigate (`/api/events`): все события людей и машин на камерах калитки.

Каждая проверка ищет расхождение, которое глазами не поймать: импульс без
последующего открытия створки, открытие створки без импульса и без человека,
событие без строки в журнале, регрессию правила стороны. Скрипт только
находит и раскладывает — выводы и исправления остаются человеку и сессии
разбора.

Запуск с рабочей машины: `python scripts/gate_audit.py --days 3`.
Узлу нужен только ssh; всё остальное скрипт достаёт сам.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from urllib.parse import quote
from zoneinfo import ZoneInfo

NODE = "alex@192.168.2.137"
ADMIN = "http://localhost:8080"
FRIGATE = "http://localhost:5000"
MOSCOW = ZoneInfo("Europe/Moscow")

# Сколько секунд после импульса ждать открытия створки, прежде чем счесть
# щелчок «в пустоту». Замерено 2026-08-12: импульс → открыта ≈ 6 с; минута
# покрывает и человека, который копается с сумками.
OPEN_AFTER_PULSE_SECONDS = 60.0
# Окно, в котором перед открытием створки обязан быть либо импульс, либо
# человек в кадре. Открытие без обоих — ключ, рука изнутри или фантом.
EXPLAIN_OPEN_SECONDS = 30.0
IMMORTAL_EVENT_SECONDS = 3600.0

LEAF_LINE = re.compile(r"Створка: (открыта|закрыта)")


@dataclass
class Finding:
    severity: str  # "high" | "medium" | "low"
    check: str
    at: float
    title: str
    detail: str = ""
    event_id: str | None = None
    pass_id: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "severity": self.severity,
            "check": self.check,
            "at": stamp(self.at),
            "title": self.title,
            "detail": self.detail,
            "event_id": self.event_id,
            "pass_id": self.pass_id,
        }


@dataclass
class Audit:
    """Всё собранное с узла за окно аудита."""

    since: float
    until: float
    passes: list[dict] = field(default_factory=list)
    settings: dict = field(default_factory=dict)
    leaf: list[tuple[float, str]] = field(default_factory=list)  # (время, open|closed)
    warnings: list[tuple[float, str]] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    ongoing: list[dict] = field(default_factory=list)


def stamp(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, MOSCOW).strftime("%Y-%m-%d %H:%M:%S")


def ssh(node: str, command: str) -> str:
    done = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=10", node, command],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if done.returncode != 0:
        raise RuntimeError(f"ssh не удался: {command[:80]}… → {done.stderr.strip()[:200]}")
    return done.stdout


def collect(node: str, since: float, until: float) -> Audit:
    audit = Audit(since=since, until=until)
    # `+00:00` в незакодированном URL превращается в пробел, и сервер (теперь
    # громко) отвергает фильтр. Первый прогон поймал это на живых данных.
    since_iso = quote(datetime.fromtimestamp(since, UTC).isoformat())
    until_iso = quote(datetime.fromtimestamp(until, UTC).isoformat())
    audit.passes = json.loads(
        ssh(
            node,
            f"curl -sf '{ADMIN}/api/gate/passes?since={since_iso}&until={until_iso}&limit=5000'",
        )
    )["passes"]
    audit.settings = json.loads(ssh(node, f"curl -sf '{ADMIN}/api/gate/access-settings'"))
    # Журнал службы: переходы створки и предупреждения. journald — единственное
    # место, где живут строки классификатора; база их не хранит.
    raw = ssh(
        node,
        f"journalctl -u catdoor-gate-access --since '@{int(since)}' "
        f"--until '@{int(until)}' --no-pager -o json 2>/dev/null",
    )
    for line in raw.splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        moment = int(entry.get("__REALTIME_TIMESTAMP", 0)) / 1_000_000
        message = entry.get("MESSAGE", "") or ""
        leaf = LEAF_LINE.search(message)
        if leaf:
            audit.leaf.append((moment, "open" if leaf.group(1) == "открыта" else "closed"))
        elif entry.get("PRIORITY") in ("3", "4") or "WARNING" in message:
            audit.warnings.append((moment, message))
    cameras = ",".join(
        gate["camera"] for gate in audit.settings.get("gates", []) if gate.get("camera")
    ) or "test_cam"
    audit.events = json.loads(
        ssh(
            node,
            f"curl -sf '{FRIGATE}/api/events?cameras={cameras}"
            f"&after={since}&before={until}&limit=2000&include_thumbnails=0'",
        )
    )
    audit.ongoing = json.loads(
        ssh(
            node,
            f"curl -sf '{FRIGATE}/api/events?cameras={cameras}"
            f"&in_progress=1&limit=100&include_thumbnails=0'",
        )
    )
    return audit


def leaf_between(audit: Audit, start: float, end: float, state: str) -> bool:
    return any(start < at <= end and value == state for at, value in audit.leaf)


def event_started(row: dict) -> float | None:
    """Начало события из его идентификатора Frigate.

    У ручных открытий идентификатор `manual-<uuid>` — начала события у них
    нет, и это не ошибка данных.
    """
    head = str(row.get("frigate_event_id", "")).split("-")[0]
    try:
        return float(head)
    except ValueError:
        return None


def row_moment(row: dict, audit: Audit) -> float:
    """Лучшее известное время строки — для отметки находки в отчёте."""
    started = event_started(row)
    if started is not None:
        return started
    if row.get("last_pulse_at"):
        return float(row["last_pulse_at"])
    try:
        return datetime.fromisoformat(row["created_at"]).timestamp()
    except (KeyError, ValueError):
        return audit.since


def check_pulses(audit: Audit) -> list[Finding]:
    """Каждый импульс обязан объясняться: створка после него открылась."""
    findings = []
    for row in audit.passes:
        pulse_at = row.get("last_pulse_at")
        if not row.get("pulses") or pulse_at is None or row.get("modbus_ok") is False:
            continue
        if row.get("pulse_gate_state") == "open":
            # Импульс в уже открытую створку: «не открылась после» — потому
            # что открыта была до. Разбор 2026-08-21 показал: три из пяти
            # находок первого прогона были ровно этим (семья шла друг за
            # другом). Под правилом стороны такие импульсы больше не уходят,
            # а исторические — объяснены и не тревога.
            continue
        if not leaf_between(audit, pulse_at, pulse_at + OPEN_AFTER_PULSE_SECONDS, "open"):
            findings.append(
                Finding(
                    "high",
                    "impulse-no-open",
                    pulse_at,
                    f"Импульс без последующего открытия створки ({row.get('person_name')})",
                    "За минуту после щелчка классификатор не увидел «открыта»: "
                    "либо фантомный щелчок в пустоту (замок мог вытолкнуть "
                    "створку незаметно для классификатора), либо человек не "
                    "вошёл, либо слепое окно классификатора. Смотреть запись.",
                    row.get("frigate_event_id"),
                    row.get("id"),
                )
            )
        if row.get("pulse_gate_state") in (None, "unknown"):
            findings.append(
                Finding(
                    "medium",
                    "blind-pulse",
                    pulse_at,
                    f"Импульс при неизвестной створке ({row.get('person_name')})",
                    "Правило стороны в слепоте разоружено намеренно; каждый "
                    "такой импульс — кандидат в призраки, смотреть запись.",
                    row.get("frigate_event_id"),
                    row.get("id"),
                )
            )
        started = event_started(row)
        if started and leaf_between(audit, started, pulse_at, "closed"):
            findings.append(
                Finding(
                    "high",
                    "side-rule-regression",
                    pulse_at,
                    f"Импульс после хода створки во время события ({row.get('person_name')})",
                    "Створка закрывалась между началом события и щелчком — "
                    "правило стороны обязано было отказать. Регрессия логики "
                    "или расхождение часов; разбирать обязательно.",
                    row.get("frigate_event_id"),
                    row.get("id"),
                )
            )
        if (row.get("pulses") or 0) > 1:
            findings.append(
                Finding(
                    "high",
                    "multi-pulse",
                    pulse_at,
                    f"Больше одного импульса на событие ({row.get('person_name')})",
                    "Повторы структурно мертвы под правилом стороны — второй "
                    "импульс означает регрессию.",
                    row.get("frigate_event_id"),
                    row.get("id"),
                )
            )
    return findings


def check_openings(audit: Audit) -> list[Finding]:
    """Каждое открытие створки обязано объясняться: импульс или человек рядом."""
    findings = []
    pulse_times = [
        row["last_pulse_at"]
        for row in audit.passes
        if row.get("pulses") and row.get("last_pulse_at") is not None
    ]
    for at, state in audit.leaf:
        if state != "open":
            continue
        explained = any(0 <= at - pulse <= OPEN_AFTER_PULSE_SECONDS for pulse in pulse_times)
        person_near = any(
            event.get("label") == "person"
            and event.get("start_time", 0) - EXPLAIN_OPEN_SECONDS <= at
            and (event.get("end_time") or audit.until) + EXPLAIN_OPEN_SECONDS >= at
            for event in audit.events + audit.ongoing
        )
        if not explained and not person_near:
            findings.append(
                Finding(
                    "medium",
                    "unexplained-open",
                    at,
                    "Створка открылась без импульса и без человека в кадре",
                    "Ключ, рука изнутри, фантом реле или проспавшая детекция — "
                    "по журналу не различить, смотреть запись nvr_2.",
                )
            )
    return findings


def check_journal_completeness(audit: Audit) -> list[Finding]:
    """Каждое событие человека у калитки обязано иметь строку в журнале."""
    known = {row.get("frigate_event_id") for row in audit.passes}
    findings = []
    for event in audit.events:
        if event.get("label") not in ("person", "car"):
            continue
        if event["id"] not in known:
            findings.append(
                Finding(
                    "high",
                    "missing-journal-row",
                    event.get("start_time", audit.since),
                    f"Событие {event.get('label')} без строки в журнале калитки",
                    "Служба его не записала: лежала, не видела камеру или "
                    "упёрлась в инвентарь. Полнота журнала — основа всех "
                    "остальных проверок.",
                    event["id"],
                )
            )
    return findings


def check_denied_own(audit: Audit) -> list[Finding]:
    """Свои, не получившие открытия: сырьё для разбора ложных отказов."""
    allowed = set(audit.settings.get("allowed_names") or [])
    findings = []
    for row in audit.passes:
        name = row.get("person_name")
        if name not in allowed or row.get("pulses"):
            continue
        if row.get("decision") in ("opened", "dry_run", "logged"):
            continue
        findings.append(
            Finding(
                "low",
                "own-denied",
                row_moment(row, audit),
                f"{name}: {row.get('decision')}",
                (row.get("reason") or "")[:160],
                row.get("frigate_event_id"),
                row.get("id"),
            )
        )
    return findings


def check_health(audit: Audit) -> list[Finding]:
    """Железо и датчики: отказы платы, предупреждения, бессмертные события."""
    findings = []
    for row in audit.passes:
        if row.get("modbus_ok") is False or row.get("decision") == "error":
            findings.append(
                Finding(
                    "medium",
                    "board-error",
                    row_moment(row, audit),
                    f"Отказ платы ({row.get('person_name')})",
                    (row.get("reason") or "")[:160],
                    row.get("frigate_event_id"),
                    row.get("id"),
                )
            )
    # Одинаковые предупреждения сворачиваются в одну строку со счётчиком:
    # загрузка узла даёт десятки «Frigate не ответил» подряд, и по одному они
    # хоронят в отчёте всё важное. Ключ — текст после уровня и логгера.
    grouped: dict[str, list[float]] = {}
    for at, message in audit.warnings:
        key = message.split(":", 2)[-1][:120]
        grouped.setdefault(key, []).append(at)
    for key, moments in grouped.items():
        count = len(moments)
        span = f" ×{count}, до {stamp(max(moments))}" if count > 1 else ""
        findings.append(
            Finding(
                "low",
                "service-warning",
                min(moments),
                f"Предупреждение службы{span}",
                key,
            )
        )
    for event in audit.ongoing:
        age = audit.until - event.get("start_time", audit.until)
        if age > IMMORTAL_EVENT_SECONDS:
            findings.append(
                Finding(
                    "low",
                    "immortal-event",
                    event.get("start_time", audit.since),
                    f"Незавершённое событие возрастом {age / 3600:.1f} ч",
                    "Похоже на огрызок после перезапуска Frigate.",
                    event["id"],
                )
            )
    return findings


def report(audit: Audit, findings: list[Finding]) -> str:
    order = {"high": 0, "medium": 1, "low": 2}
    findings = sorted(findings, key=lambda item: (order[item.severity], item.at))
    lines = [
        f"# Аудит калитки: {stamp(audit.since)} — {stamp(audit.until)}",
        "",
        (
            f"Строк журнала: {len(audit.passes)} · переходов створки: {len(audit.leaf)} · "
            f"событий Frigate: {len(audit.events)} · находок: {len(findings)}"
        ),
        (
            f"Режим: {audit.settings.get('mode')} · разрешены: "
            f"{', '.join(audit.settings.get('allowed_names') or []) or '—'}"
        ),
        "",
    ]
    if not findings:
        lines.append("Расхождений не найдено.")
    label = {"high": "## Разбирать обязательно", "medium": "## Посмотреть", "low": "## Для сведения"}
    current = None
    for item in findings:
        if item.severity != current:
            current = item.severity
            lines += [label[current], ""]
        where = f" · событие `{item.event_id}`" if item.event_id else ""
        snap = f" · снимок: /gate/passes/{item.pass_id}/snapshot" if item.pass_id else ""
        lines.append(f"- **{stamp(item.at)}** [{item.check}] {item.title}{where}{snap}")
        if item.detail:
            lines.append(f"  {item.detail}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Аудит логики калитки по трём источникам")
    parser.add_argument("--node", default=NODE)
    parser.add_argument("--days", type=float, default=3.0, help="окно аудита, суток назад")
    parser.add_argument("--since", help="начало окна ISO (перекрывает --days)")
    parser.add_argument("--json", help="куда положить находки машинно-читаемо")
    args = parser.parse_args()
    until = datetime.now(UTC).timestamp()
    since = (
        datetime.fromisoformat(args.since).replace(tzinfo=MOSCOW).timestamp()
        if args.since and datetime.fromisoformat(args.since).tzinfo is None
        else datetime.fromisoformat(args.since).timestamp()
        if args.since
        else until - args.days * 86400
    )
    audit = collect(args.node, since, until)
    findings = (
        check_pulses(audit)
        + check_openings(audit)
        + check_journal_completeness(audit)
        + check_denied_own(audit)
        + check_health(audit)
    )
    print(report(audit, findings))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "since": stamp(since),
                    "until": stamp(until),
                    "findings": [item.as_dict() for item in findings],
                },
                handle,
                ensure_ascii=False,
                indent=1,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
