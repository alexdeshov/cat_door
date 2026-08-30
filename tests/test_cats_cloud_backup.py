"""Облачная копия кошачьих проходов: отбор, порядок, пропуски, ключи."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.cloud_backup.cats import (
    object_key,
    pending_events,
)
from src.common.cloud_backup_config import CloudBackupConfig
from src.common.database import Event, create_session_factory


def _factory(tmp_path: Path) -> Any:
    return create_session_factory(tmp_path / "catdoor.sqlite3")


def _add(
    factory: Any,
    *,
    status: str = "ready",
    face: str | None = "events/00000001/face.mp4",
    side: str | None = "events/00000001/side.mp4",
    day: int = 1,
) -> int:
    with factory() as session:
        event = Event(
            created_at=datetime(2026, 8, day, 12, 0, tzinfo=UTC),
            status=status,
            duration_seconds=30,
            label="known_cat",
            media_path=face,
            side_media_path=side,
        )
        session.add(event)
        session.commit()
        return int(event.id)


def test_key_is_deterministic_so_a_lost_journal_costs_a_version_not_a_double() -> None:
    """Потерянный журнал не должен плодить объекты: ключ считается от прохода.

    Иначе повторная заливка после потери состояния оплачивалась бы вторым
    экземпляром каждого клипа, а не перезаписью того же объекта.
    """
    moment = datetime(2026, 8, 30, 21, 15, tzinfo=UTC)

    first = object_key("cat-events", 1143, moment, "face")
    second = object_key("cat-events/", 1143, moment, "face")

    assert first == "cat-events/2026-08-30/00001143/face.mp4"
    assert first == second


def test_pending_goes_oldest_first(tmp_path: Path) -> None:
    """Порядок не косметика: локальное окно хранения съедает самое старое,
    поэтому спасать в первую очередь надо его, а не свежие проходы."""
    factory = _factory(tmp_path)
    older = _add(factory, day=1)
    newer = _add(factory, day=20)

    queue = pending_events(factory, set(), set())

    assert [item["id"] for item in queue] == [older, newer]


def test_uploaded_and_deferred_are_skipped(tmp_path: Path) -> None:
    factory = _factory(tmp_path)
    first = _add(factory, day=1)
    second = _add(factory, day=2)
    third = _add(factory, day=3)

    queue = pending_events(factory, {str(first)}, {str(second)})

    assert [item["id"] for item in queue] == [third]


def test_event_without_any_clip_never_enters_the_queue(tmp_path: Path) -> None:
    """Проход без файлов заливать нечем, и держать его в очереди — значит
    вечно показывать «очередь не пуста» при исправной службе."""
    factory = _factory(tmp_path)
    _add(factory, face=None, side=None)

    assert pending_events(factory, set(), set()) == []


def test_partial_pass_is_taken_for_the_view_that_recorded(tmp_path: Path) -> None:
    """`partial` — это проход, где один ракурс не записался. Второй ракурс всё
    равно единственная видеозапись, и он должен уехать."""
    factory = _factory(tmp_path)
    _add(factory, status="partial", side=None)

    queue = pending_events(factory, set(), set())

    assert len(queue) == 1
    assert set(queue[0]["views"]) == {"face"}


def test_unfinished_recording_is_not_taken(tmp_path: Path) -> None:
    """Клип ещё пишется: залить его сейчас значило бы увезти огрызок и
    отметить проход сделанным."""
    factory = _factory(tmp_path)
    _add(factory, status="recording")

    assert pending_events(factory, set(), set()) == []


def test_limit_bounds_the_batch(tmp_path: Path) -> None:
    factory = _factory(tmp_path)
    for day in (1, 2, 3):
        _add(factory, day=day)

    assert len(pending_events(factory, set(), set(), 2)) == 2


def test_cats_bucket_falls_back_to_the_shared_one() -> None:
    """Пустой кошачий бакет означает «тот же, что у калитки, только префикс» —
    иначе включение копии без отдельного бакета молча писало бы в никуда."""
    config = CloudBackupConfig(bucket="peski26", access_key_id="id", secret_key="s")

    assert config.cats_target_bucket == "peski26"
    assert config.cats_ready is True

    own = config.model_copy(update={"cats_bucket": "peski26-cats"})
    assert own.cats_target_bucket == "peski26-cats"


def test_cats_not_ready_without_keys() -> None:
    config = CloudBackupConfig(cats_bucket="peski26-cats")

    assert config.cats_ready is False


def test_missing_file_is_reported_as_a_skip_not_a_failure(tmp_path: Path) -> None:
    """Окно хранения могло съесть файл раньше, чем он уехал. Такой проход
    отмечается залитым: иначе служба спотыкалась бы об него вечно."""
    from src.cloud_backup.cats import _upload_event

    sent: list[tuple[str, str]] = []
    client = SimpleNamespace(
        upload_file=lambda path, bucket, key: sent.append((bucket, key))
    )
    data_dir = tmp_path
    (data_dir / "events" / "00000007").mkdir(parents=True)
    (data_dir / "events" / "00000007" / "face.mp4").write_bytes(b"x" * 10)
    event = {
        "id": 7,
        "created_at": datetime(2026, 8, 30, tzinfo=UTC),
        "views": {
            "face": "events/00000007/face.mp4",
            "side": "events/00000007/side.mp4",
        },
    }
    notifier = SimpleNamespace(watchdog_interval_seconds=None, watchdog=lambda: None)

    size, missing = _upload_event(client, "peski26-cats", "cat-events", data_dir, event, notifier)

    assert size == 10
    assert missing == ["side"]
    assert sent == [("peski26-cats", "cat-events/2026-08-30/00000007/face.mp4")]
