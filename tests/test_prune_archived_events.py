"""Подрезка зеркала проходов: удаляет только старое и только то, что в облаке."""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prune_archived_events import prune, uploaded_ids, write_exclude_file

NOW = time.mktime((2026, 8, 31, 12, 0, 0, 0, 0, -1))


def _database(path: Path, events: list[tuple[int, str]]) -> Path:
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, created_at TEXT)")
    connection.executemany("INSERT INTO events VALUES (?, ?)", events)
    connection.commit()
    connection.close()
    return path


def _archive(root: Path, ids: list[int], payload: bytes = b"x" * 1000) -> Path:
    for event_id in ids:
        directory = root / f"{event_id:08d}"
        directory.mkdir(parents=True)
        (directory / "face.mp4").write_bytes(payload)
        (directory / "side.mp4").write_bytes(payload)
    return root


def _state(path: Path, uploaded: list[int]) -> Path:
    path.write_text(json.dumps({"uploaded": [str(x) for x in uploaded]}), encoding="utf-8")
    return path


def test_old_and_uploaded_is_removed(tmp_path: Path) -> None:
    events = _archive(tmp_path / "events", [1, 2])
    database = _database(
        tmp_path / "db.sqlite3",
        [(1, "2026-08-01 10:00:00"), (2, "2026-08-30 10:00:00")],
    )
    state = _state(tmp_path / "state.json", [1, 2])

    report = prune(
        events_dir=events, database=database, state_path=state, days=7, now=NOW
    )

    assert report.removed == 1
    assert not (events / "00000001").exists()  # старый — удалён
    assert (events / "00000002").exists()  # свежий — остался
    assert report.freed_bytes == 2000


def test_a_pass_missing_from_the_cloud_is_never_touched(tmp_path: Path) -> None:
    """Главный предохранитель. Клип прохода существует в единственном
    экземпляре — камеры контура котов Frigate не пишет, — и удаление до заливки
    означает потерю навсегда."""
    events = _archive(tmp_path / "events", [1, 2])
    database = _database(
        tmp_path / "db.sqlite3",
        [(1, "2026-08-01 10:00:00"), (2, "2026-08-02 10:00:00")],
    )
    state = _state(tmp_path / "state.json", [1])  # второго в облаке нет

    report = prune(
        events_dir=events, database=database, state_path=state, days=7, now=NOW
    )

    assert report.removed == 1
    assert report.kept_not_in_cloud == 1
    assert (events / "00000002").exists()
    assert "оставлено 1" in report.line(7)


def test_unreadable_journal_stops_everything(tmp_path: Path) -> None:
    """Нечитаемый журнал — это неизвестность, а не «ничего не залито»."""
    events = _archive(tmp_path / "events", [1])
    database = _database(tmp_path / "db.sqlite3", [(1, "2026-08-01 10:00:00")])
    broken = tmp_path / "state.json"
    broken.write_text("{не json", encoding="utf-8")

    report = prune(
        events_dir=events, database=database, state_path=broken, days=7, now=NOW
    )

    assert report.removed == 0
    assert (events / "00000001").exists()
    assert "не удаляю ничего" in report.line(7)


def test_missing_journal_stops_everything(tmp_path: Path) -> None:
    events = _archive(tmp_path / "events", [1])
    database = _database(tmp_path / "db.sqlite3", [(1, "2026-08-01 10:00:00")])

    report = prune(
        events_dir=events,
        database=database,
        state_path=tmp_path / "нет-такого.json",
        days=7,
        now=NOW,
    )

    assert report.removed == 0
    assert (events / "00000001").exists()


def test_empty_journal_stops_everything(tmp_path: Path) -> None:
    """Пустой журнал бывает у только что поднятой службы: заливка ещё не шла,
    а все проходы уже старые. Удалить их значило бы стереть архив целиком."""
    events = _archive(tmp_path / "events", [1])
    database = _database(tmp_path / "db.sqlite3", [(1, "2026-08-01 10:00:00")])
    state = _state(tmp_path / "state.json", [])

    report = prune(
        events_dir=events, database=database, state_path=state, days=7, now=NOW
    )

    assert report.removed == 0
    assert (events / "00000001").exists()


def test_zero_days_means_off(tmp_path: Path) -> None:
    """Умолчание: обновление кода не должно начать удалять само по себе."""
    events = _archive(tmp_path / "events", [1])
    database = _database(tmp_path / "db.sqlite3", [(1, "2026-08-01 10:00:00")])
    state = _state(tmp_path / "state.json", [1])

    report = prune(
        events_dir=events, database=database, state_path=state, days=0, now=NOW
    )

    assert report.removed == 0
    assert (events / "00000001").exists()
    assert "выключена" in report.line(0)


def test_unreadable_database_stops_everything(tmp_path: Path) -> None:
    events = _archive(tmp_path / "events", [1])
    state = _state(tmp_path / "state.json", [1])

    report = prune(
        events_dir=events,
        database=tmp_path / "нет-базы.sqlite3",
        state_path=state,
        days=7,
        now=NOW,
    )

    assert report.removed == 0
    assert (events / "00000001").exists()


def test_prunable_covers_already_missing_directories(tmp_path: Path) -> None:
    """Защита от круговорота. Подрезанный вчера проход всё ещё лежит в живом
    каталоге, и rsync без исключения привозил его в архив обратно — ~77 ГБ
    каждое утро (31.08–02.09). Поэтому в список исключений входит всё, что
    попадает под предикат подрезки, даже если в архиве каталога уже нет."""
    events = _archive(tmp_path / "events", [2])  # первого в архиве уже нет
    database = _database(
        tmp_path / "db.sqlite3",
        [(1, "2026-08-01 10:00:00"), (2, "2026-08-02 10:00:00"), (3, "2026-08-03 10:00:00"), (4, "2026-08-30 10:00:00")],
    )
    state = _state(tmp_path / "state.json", [1, 2, 4])  # третий не залит

    report = prune(
        events_dir=events, database=database, state_path=state, days=7, now=NOW
    )

    assert report.removed == 1
    assert report.prunable == ("00000001", "00000002")  # без свежего и незалитого


def test_exclude_file_lists_anchored_directories(tmp_path: Path) -> None:
    events = _archive(tmp_path / "events", [1])
    database = _database(tmp_path / "db.sqlite3", [(1, "2026-08-01 10:00:00")])
    state = _state(tmp_path / "state.json", [1])
    exclude = tmp_path / "exclude.list"

    report = prune(
        events_dir=events, database=database, state_path=state, days=7, now=NOW
    )
    write_exclude_file(report, exclude)

    assert exclude.read_text(encoding="utf-8") == "/00000001/\n"


def test_exclude_file_is_empty_when_pruning_declined(tmp_path: Path) -> None:
    """Не состоялась подрезка — не должно быть и исключений: rsync копирует
    всё, как до этой правки."""
    events = _archive(tmp_path / "events", [1])
    database = _database(tmp_path / "db.sqlite3", [(1, "2026-08-01 10:00:00")])
    exclude = tmp_path / "exclude.list"

    report = prune(
        events_dir=events,
        database=database,
        state_path=tmp_path / "нет-журнала.json",
        days=7,
        now=NOW,
    )
    write_exclude_file(report, exclude)

    assert report.reason
    assert exclude.read_text(encoding="utf-8") == ""


def test_journal_reader_tells_unreadable_from_empty(tmp_path: Path) -> None:
    """Разница принципиальна: пустой журнал — это факт, нечитаемый — незнание."""
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"uploaded": []}), encoding="utf-8")
    broken = tmp_path / "broken.json"
    broken.write_text("не json", encoding="utf-8")

    assert uploaded_ids(empty) == set()
    assert uploaded_ids(broken) is None
    assert uploaded_ids(tmp_path / "нет.json") is None
