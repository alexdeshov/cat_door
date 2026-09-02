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

Из этого следует третья обязанность: отдавать `rsync` список исключений
(`--exclude-file`). Подрезанный проход остаётся в живом каталоге, и `rsync`
без исключений копирует его в архив заново — первые дни после включения
подрезки (31.08–02.09) копия каждое утро полчаса возила ~77 ГБ туда, а
подрезка тут же удаляла их обратно, продавливая свободное место архивного
диска до алерта. Список строится тем же предикатом, что и удаление (старше
окна **и** подтверждён облаком), поэтому исключить лишнего он не может.
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
    # Имена каталогов, попавших под предикат подрезки, — включая уже
    # отсутствующие в архиве: они всё ещё лежат в живом каталоге, и без
    # исключения rsync привезёт их обратно.
    prunable: tuple[str, ...] = ()

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
    prunable: list[str] = []
    for (event_id,) in rows:
        if str(event_id) not in uploaded:
            kept += 1
            continue
        name = f"{int(event_id):08d}"
        prunable.append(name)
        path = events_dir / name
        if not path.is_dir():
            continue
        size = _directory_size(path)
        try:
            shutil.rmtree(path)
        except OSError:
            continue
        removed += 1
        freed += size
    return PruneReport(
        removed=removed,
        freed_bytes=freed,
        kept_not_in_cloud=kept,
        prunable=tuple(prunable),
    )


def write_exclude_file(report: PruneReport, path: Path) -> None:
    """Список исключений для rsync: по строке `/имя/` на подрезанный каталог.

    Пишется через переименование: оборванный на середине файл исключил бы
    непонятно что. Пустой список — тоже результат (подрезка выключена или не
    состоялась): rsync с пустым файлом копирует всё, как и раньше.
    """
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        "".join(f"/{name}/\n" for name in report.prunable), encoding="utf-8"
    )
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-dir", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--days", type=int, required=True)
    parser.add_argument("--exclude-file", type=Path, default=None)
    args = parser.parse_args()
    report = prune(
        events_dir=args.events_dir,
        database=args.database,
        state_path=args.state,
        days=args.days,
    )
    if args.exclude_file is not None:
        write_exclude_file(report, args.exclude_file)
    print(report.line(args.days))


if __name__ == "__main__":
    main()
