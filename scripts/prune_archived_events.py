"""Подрезка зеркала кошачьих проходов в архиве по возрасту.

Зеркало событий росло без предела с самого начала: `rsync` в
`backup_to_archive.sh` идёт без `--delete` и без отсечки по возрасту. Так было
задумано — «архив чистится руками», чтобы ошибка в разметке не уносила
обучающий материал, — но руки не дошли ни разу, и к 30 августа зеркало весило
84 ГБ на диске, который Frigate держит полным.

Это та самая ручная чистка, только с двумя предохранителями:

- **возраст**: удаляется то, что старше заданного срока;
- **подтверждение облаком**: проход, которого нет в журнале залитого у
  `catdoor-cloud-backup-cats`, не трогается никогда.

Второй важнее первого. Клипы кошачьих проходов существуют в единственном
экземпляре — камеры контура котов Frigate не пишет вовсе, — и «удалили по
возрасту, а копия не уехала» означает потерю навсегда. Поэтому любая неясность
(нет журнала, не читается база, пустой журнал) трактуется как «не удалять
ничего»: несделанная уборка стоит гигабайтов, лишнее удаление — записи.

Живой каталог `$DATA/events` не трогается: он лежит на просторном томе, и его
читает админка, когда показывает проход.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PruneReport:
    removed: int = 0
    freed_bytes: int = 0
    kept_not_in_cloud: int = 0
    reason: str = ""

    def line(self, days: int) -> str:
        if self.reason:
            return f"архив: {self.reason} — не удаляю ничего"
        tail = (
            f"; оставлено {self.kept_not_in_cloud} — их ещё нет в облаке"
            if self.kept_not_in_cloud
            else ""
        )
        return (
            f"архив: подрезка старше {days} дн — удалено проходов {self.removed}, "
            f"освобождено {self.freed_bytes / 1e9:.1f} ГБ{tail}"
        )


def uploaded_ids(state_path: Path) -> set[str] | None:
    """Журнал залитого. `None` — прочитать не удалось, значит удалять нельзя."""
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    return {str(item) for item in payload.get("uploaded", [])}


def _directory_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def prune(
    *,
    events_dir: Path,
    database: Path,
    state_path: Path,
    days: int,
    now: float | None = None,
) -> PruneReport:
    if days <= 0:
        return PruneReport(reason="подрезка выключена")
    uploaded = uploaded_ids(state_path)
    if uploaded is None:
        return PruneReport(reason="журнал облачной копии не прочитан")
    if not uploaded:
        return PruneReport(reason="в облаке пока ни одного прохода")

    cutoff = (now if now is not None else time.time()) - days * 86400
    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        rows = connection.execute(
            "SELECT id FROM events WHERE created_at < datetime(?, 'unixepoch')",
            (int(cutoff),),
        ).fetchall()
        connection.close()
    except sqlite3.Error:
        return PruneReport(reason="база проходов не прочитана")

    removed = freed = kept = 0
    for (event_id,) in rows:
        if str(event_id) not in uploaded:
            kept += 1
            continue
        path = events_dir / f"{int(event_id):08d}"
        if not path.is_dir():
            continue
        size = _directory_size(path)
        try:
            shutil.rmtree(path)
        except OSError:
            continue
        removed += 1
        freed += size
    return PruneReport(removed=removed, freed_bytes=freed, kept_not_in_cloud=kept)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-dir", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--days", type=int, required=True)
    args = parser.parse_args()
    report = prune(
        events_dir=args.events_dir,
        database=args.database,
        state_path=args.state,
        days=args.days,
    )
    print(report.line(args.days))


if __name__ == "__main__":
    main()
