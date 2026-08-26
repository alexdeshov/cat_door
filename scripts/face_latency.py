"""Замер задержки узнавания лица у калитки: откуда взялись секунды ожидания.

Человек стоит у закрытой створки не потому, что служба медленная, а потому что
Frigate ещё не набрал кворум: имя событию он даёт не по лучшему кадру, а по
`min_faces` совпадениям подряд идущих попыток. Разобрать это по журналу нельзя
— там видно только «имя появилось», — зато Frigate сохраняет каждую попытку
распознавания отдельным файлом:

    <событие>-<время>-<имя|unknown>-<балл>.webp

Скрипт сшивает эти файлы с журналом проходов админки и раскладывает ожидание на
три отрезка, у каждого из которых свой виновник:

- **старт → первое совпадение** — сколько человек шёл, прежде чем его лицо
  вообще стало узнаваемым (свет, ракурс, расстояние, разрешение потока);
- **первое → кворум** — пауза между годными кадрами, цена `min_faces` и низкой
  частоты детекции;
- **кворум → импульс** — наша часть: опрос событий раз в секунду. Если она
  больше пары секунд, чинить надо у себя, а не у Frigate.

Этим же замером сравнивается «до» и «после» любой правки частоты, порогов или
потока детекции: числа из рассуждений здесь стабильно врут (docs/cameras.md).

Запуск с рабочей машины: `python scripts/face_latency.py --days 3`.
Узлу нужен только ssh; попытки лежат у root в каталоге данных Frigate и читаются
без sudo. С `--sizes` скрипт заодно померяет размеры кропов — это ответ на
вопрос «поможет ли более крупный поток детекции».
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from urllib.parse import quote
from zoneinfo import ZoneInfo

NODE = "alex@192.168.2.137"
ADMIN = "http://localhost:8080"
TRAIN_DIR = "/var/lib/frigate/media/clips/faces/train"
MOSCOW = ZoneInfo("Europe/Moscow")

# Столько совпадений Frigate ждёт, прежде чем назвать событие именем
# (`face_recognition.min_faces` в его конфигурации). Здесь это не настройка, а
# то, что скрипт обязан знать, чтобы посчитать момент кворума.
DEFAULT_MIN_FACES = 2

# Имя, которым Frigate метит попытку, не дотянувшую до `unknown_score`.
UNKNOWN = "unknown"

# Попытка распознавания: `<событие>-<время>-<имя>-<балл>.webp`. Имя берётся
# жадно — в нём попадаются дефисы, а балл в конце ни одного не содержит.
ATTEMPT = re.compile(
    r"^(?P<event>\d+\.\d+-[A-Za-z0-9]+)-(?P<at>\d+\.\d+)-(?P<name>.+)-(?P<score>\d(?:\.\d+)?)\.webp$"
)


@dataclass(frozen=True)
class Attempt:
    """Один кадр лица, который Frigate попробовал узнать."""

    event_id: str
    at: float
    name: str
    score: float
    filename: str

    @property
    def matched(self) -> bool:
        return self.name != UNKNOWN


@dataclass
class EventLatency:
    """Как одно событие шло от появления человека до импульса."""

    event_id: str
    started: float
    attempts: list[Attempt] = field(default_factory=list)
    person: str | None = None
    pulse_at: float | None = None

    @property
    def matches(self) -> list[Attempt]:
        return [item for item in self.attempts if item.matched]

    @property
    def first_attempt(self) -> float | None:
        return self.attempts[0].at if self.attempts else None

    @property
    def first_match(self) -> Attempt | None:
        matches = self.matches
        return matches[0] if matches else None

    def quorum(self, min_faces: int) -> Attempt | None:
        """Попытка, на которой имя набрало кворум.

        Совпадения считаются по каждому имени отдельно: два кадра, узнавших
        разных людей, кворумом не являются ни для кого.
        """
        seen: dict[str, int] = defaultdict(int)
        for attempt in self.matches:
            seen[attempt.name] += 1
            if seen[attempt.name] >= min_faces:
                return attempt
        return None

    def as_dict(self, min_faces: int) -> dict[str, object]:
        quorum = self.quorum(min_faces)
        first = self.first_match
        return {
            "event_id": self.event_id,
            "started": stamp(self.started),
            "person": self.person,
            "attempts": len(self.attempts),
            "matched": len(self.matches),
            "first_attempt": _delta(self.first_attempt, self.started),
            "first_match": _delta(first.at if first else None, self.started),
            "quorum": _delta(quorum.at if quorum else None, self.started),
            "pulse": _delta(self.pulse_at, self.started),
            "quorum_wait": _delta(quorum.at if quorum else None, first.at if first else None),
            "our_lag": _delta(self.pulse_at, quorum.at if quorum else None),
        }


def stamp(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, MOSCOW).strftime("%d.%m %H:%M:%S")


def _delta(later: float | None, earlier: float | None) -> float | None:
    if later is None or earlier is None:
        return None
    return round(later - earlier, 1)


def ssh(node: str, command: str) -> str:
    done = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=10", node, command],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if done.returncode != 0:
        raise RuntimeError(f"ssh не удался: {command[:80]}… → {done.stderr.strip()[:200]}")
    return done.stdout


def parse_attempt(filename: str) -> Attempt | None:
    """Попытка из имени файла. `None` — файл не наш, и это не ошибка."""
    found = ATTEMPT.match(filename.strip())
    if not found:
        return None
    return Attempt(
        event_id=found["event"],
        at=float(found["at"]),
        name=found["name"],
        score=float(found["score"]),
        filename=filename.strip(),
    )


def event_started(event_id: str) -> float | None:
    """Начало события из его идентификатора Frigate.

    У ручных открытий идентификатор `manual-<uuid>`: начала события у них нет, и
    это не ошибка данных.
    """
    head = event_id.split("-")[0]
    try:
        return float(head)
    except ValueError:
        return None


def measure(
    attempts: list[Attempt],
    passes: list[dict],
    *,
    min_faces: int = DEFAULT_MIN_FACES,
    since: float | None = None,
) -> list[EventLatency]:
    """Разложить каждое событие на отрезки ожидания."""
    pulses = {
        str(row.get("frigate_event_id")): row for row in passes if row.get("last_pulse_at")
    }
    grouped: dict[str, list[Attempt]] = defaultdict(list)
    for attempt in attempts:
        if since is not None and attempt.at < since:
            continue
        grouped[attempt.event_id].append(attempt)

    rows: list[EventLatency] = []
    for event_id, items in grouped.items():
        started = event_started(event_id)
        if started is None:
            continue
        row = EventLatency(event_id=event_id, started=started)
        row.attempts = sorted(items, key=lambda item: item.at)
        opened = pulses.get(event_id)
        if opened:
            row.person = opened.get("person_name")
            row.pulse_at = float(opened["last_pulse_at"])
        rows.append(row)
    return sorted(rows, key=lambda row: row.started)


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 1) if values else None


def summarise(rows: list[EventLatency], *, min_faces: int = DEFAULT_MIN_FACES) -> dict[str, object]:
    """Итог по всем событиям: где именно живут секунды."""
    first: list[float] = []
    waits: list[float] = []
    lags: list[float] = []
    totals: list[float] = []
    matched_share: list[float] = []
    for row in rows:
        start_to_first = _delta(row.first_match.at if row.first_match else None, row.started)
        if start_to_first is not None:
            first.append(start_to_first)
        quorum = row.quorum(min_faces)
        wait = _delta(quorum.at if quorum else None, row.first_match.at if row.first_match else None)
        if wait is not None:
            waits.append(wait)
        lag = _delta(row.pulse_at, quorum.at if quorum else None)
        if lag is not None:
            lags.append(lag)
        total = _delta(row.pulse_at, row.started)
        if total is not None:
            totals.append(total)
        if row.attempts:
            matched_share.append(len(row.matches) / len(row.attempts))
    return {
        "events": len(rows),
        "with_quorum": sum(1 for row in rows if row.quorum(min_faces)),
        "with_pulse": len(totals),
        "median_first_match": _median(first),
        "median_quorum_wait": _median(waits),
        "worst_quorum_wait": max(waits) if waits else None,
        "median_our_lag": _median(lags),
        "worst_our_lag": max(lags) if lags else None,
        "median_total": _median(totals),
        "worst_total": max(totals) if totals else None,
        "median_matched_share": _median([round(value, 2) for value in matched_share]),
    }


def size_report(
    attempts: list[Attempt], sizes: dict[str, tuple[int, int]]
) -> dict[str, object] | None:
    """Крупнее ли кропы, которые удалось узнать.

    Вопрос практический: стоит ли поднимать разрешение потока детекции. Если
    узнанные кропы не крупнее неузнанных, дело не в пикселях, а в резкости.
    """
    matched = [sizes[item.filename] for item in attempts if item.matched and item.filename in sizes]
    missed = [
        sizes[item.filename] for item in attempts if not item.matched and item.filename in sizes
    ]
    if not matched or not missed:
        return None
    return {
        "matched": len(matched),
        "missed": len(missed),
        "median_matched_area": _median([float(w * h) for w, h in matched]),
        "median_missed_area": _median([float(w * h) for w, h in missed]),
        "smallest_matched_area": min(w * h for w, h in matched),
        "largest_missed_area": max(w * h for w, h in missed),
    }


def _cell(values: dict[str, object], key: str) -> str:
    value = values[key]
    return "—" if value is None else str(value)


def render(
    rows: list[EventLatency],
    summary: dict[str, object],
    *,
    min_faces: int,
    sizes: dict[str, object] | None = None,
) -> str:
    head = (
        f"Событий с попытками: {summary['events']}, "
        f"кворум набран у {summary['with_quorum']}, "
        f"импульс ушёл у {summary['with_pulse']}. Кворум = {min_faces} совпадения."
    )
    columns = (
        f"{'начало':>14}  {'имя':<10} {'попыток':>8} {'первое':>7} "
        f"{'кворум':>7} {'ожидание':>9} {'импульс':>8} {'наша доля':>10}"
    )
    lines = [head, "", columns]
    for row in rows:
        values = row.as_dict(min_faces)
        counts = f"{len(row.matches)}/{len(row.attempts)}"
        lines.append(
            f"{stamp(row.started):>14}  {(row.person or '—'):<10} {counts:>8} "
            f"{_cell(values, 'first_match'):>7} {_cell(values, 'quorum'):>7} "
            f"{_cell(values, 'quorum_wait'):>9} {_cell(values, 'pulse'):>8} "
            f"{_cell(values, 'our_lag'):>10}"
        )
    lines += [
        "",
        "Медианы (секунды):",
        f"  старт → первое совпадение: {summary['median_first_match']}",
        (
            f"  первое → кворум:           {summary['median_quorum_wait']} "
            f"(худшее {summary['worst_quorum_wait']})"
        ),
        (
            f"  кворум → импульс (наше):   {summary['median_our_lag']} "
            f"(худшее {summary['worst_our_lag']})"
        ),
        (
            f"  весь путь до импульса:     {summary['median_total']} "
            f"(худшее {summary['worst_total']})"
        ),
        f"  доля узнанных попыток:     {summary['median_matched_share']}",
    ]
    if sizes:
        lines += [
            "",
            "Размеры кропов (пикселей площади):",
            f"  медиана узнанных:   {sizes['median_matched_area']} ({sizes['matched']} шт.)",
            f"  медиана неузнанных: {sizes['median_missed_area']} ({sizes['missed']} шт.)",
            (
                f"  самый мелкий узнанный: {sizes['smallest_matched_area']}, "
                f"самый крупный неузнанный: {sizes['largest_missed_area']}"
            ),
        ]
    return "\n".join(lines)


def collect_attempts(node: str, train_dir: str) -> list[Attempt]:
    listing = ssh(node, f"ls {train_dir} 2>/dev/null")
    attempts = [parse_attempt(line) for line in listing.splitlines() if line.strip()]
    return [item for item in attempts if item is not None]


def collect_passes(node: str, since: float, until: float) -> list[dict]:
    # `+00:00` в незакодированном URL превращается в пробел, и сервер отвергает
    # фильтр — та же грабля, что и в `gate_audit.py`.
    since_iso = quote(datetime.fromtimestamp(since, UTC).isoformat())
    until_iso = quote(datetime.fromtimestamp(until, UTC).isoformat())
    raw = ssh(
        node,
        f"curl -sf '{ADMIN}/api/gate/passes?since={since_iso}&until={until_iso}&limit=5000'",
    )
    return list(json.loads(raw)["passes"])


def collect_sizes(node: str, train_dir: str) -> dict[str, tuple[int, int]]:
    """Размеры кропов через ffprobe на узле: один заход, а не файл за файлом."""
    raw = ssh(
        node,
        f"cd {train_dir} && for f in *.webp; do printf '%s\\t' \"$f\"; "
        'ffprobe -v error -select_streams v:0 -show_entries stream=width,height '
        '-of csv=p=0 "$f"; done',
    )
    sizes: dict[str, tuple[int, int]] = {}
    for line in raw.splitlines():
        name, _, dimensions = line.partition("\t")
        width, _, height = dimensions.strip().partition(",")
        try:
            sizes[name] = (int(width), int(height))
        except ValueError:
            continue
    return sizes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=float, default=3.0, help="окно замера в сутках")
    parser.add_argument("--node", default=NODE, help="узел для ssh")
    parser.add_argument("--train-dir", default=TRAIN_DIR, help="каталог попыток распознавания")
    parser.add_argument(
        "--min-faces",
        type=int,
        default=DEFAULT_MIN_FACES,
        help="кворум Frigate: сколько совпадений нужно для имени",
    )
    parser.add_argument("--sizes", action="store_true", help="померить размеры кропов (ffprobe)")
    parser.add_argument("--json", action="store_true", help="выдать разбор машинам, а не глазам")
    args = parser.parse_args(argv)

    until = datetime.now(UTC)
    since = until - timedelta(days=args.days)
    attempts = collect_attempts(args.node, args.train_dir)
    passes = collect_passes(args.node, since.timestamp(), until.timestamp())
    rows = measure(attempts, passes, min_faces=args.min_faces, since=since.timestamp())
    summary = summarise(rows, min_faces=args.min_faces)

    sizes: dict[str, object] | None = None
    if args.sizes:
        measured = collect_sizes(args.node, args.train_dir)
        kept = [item for row in rows for item in row.attempts]
        sizes = size_report(kept, measured)

    if args.json:
        print(
            json.dumps(
                {
                    "summary": summary,
                    "sizes": sizes,
                    "events": [row.as_dict(args.min_faces) for row in rows],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(render(rows, summary, min_faces=args.min_faces, sizes=sizes))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
