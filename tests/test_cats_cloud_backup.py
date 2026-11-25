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


def test_failure_defers_the_pass_instead_of_retrying_at_once() -> None:
    """Отказ откладывает проход: `deferred_ids` читает `last_at`.

    До 25 ноября отказ писал поле `at`, отсрочка считалась от нуля, и упёршийся
    в предел бакет повторялся десять раз в секунду — ~780 тысяч платных POST.
    """
    from src.cloud_backup.cats import record_failure
    from src.cloud_backup.uploader import _attempt_entry, deferred_ids

    attempts: dict[str, dict[str, float]] = {}
    assert record_failure(attempts, "4665", 1000.0) == 1
    assert deferred_ids(attempts, 1001.0) == {"4665"}
    assert deferred_ids(attempts, 1000.0 + 61) == set()

    assert record_failure(attempts, "4665", 1100.0) == 2
    assert deferred_ids(attempts, 1100.0 + 100) == {"4665"}

    # Запись прежнего формата чинится на ходу, а не остаётся вечно без отсрочки.
    legacy: dict[str, dict[str, float]] = {"7": {"count": 3, "at": 500.0}}
    legacy = {k: _attempt_entry(v) for k, v in legacy.items()}
    assert record_failure(legacy, "7", 2000.0) == 4
    assert deferred_ids(legacy, 2001.0) == {"7"}


def test_round_without_uploads_sleeps_even_with_a_queue() -> None:
    """Подряд крутится только догон, который что-то заливает. Круг с отказом
    при непустой очереди обязан спать — иначе он и есть горячий цикл."""
    from src.cloud_backup.cats import POLL_SECONDS, pause_after_round

    assert pause_after_round(done=3, remaining=40) == 0.0
    assert pause_after_round(done=0, remaining=40) == POLL_SECONDS
    assert pause_after_round(done=3, remaining=0) == POLL_SECONDS


def test_full_bucket_is_recognised_through_the_boto_wrapper() -> None:
    """`upload_file` заворачивает отказ в свою ошибку — код ищется по цепочке."""
    from src.cloud_backup.uploader import is_bucket_full

    inner = RuntimeError(
        "An error occurred (BucketMaxSizeExceeded) when calling the UploadPart operation"
    )
    try:
        try:
            raise inner
        except RuntimeError as exc:
            raise ValueError("Failed to upload face.mp4") from exc
    except ValueError as wrapped:
        assert is_bucket_full(wrapped)
    assert not is_bucket_full(RuntimeError("AccessDenied"))


def test_full_bucket_pauses_for_an_hour_and_alerts_sparingly() -> None:
    """Полный бакет — час тишины без единого запроса; тревога при первом упоре
    и раз в шесть часов, «снова принимает» — один раз."""
    from src.cloud_backup.uploader import (
        BUCKET_FULL_PAUSE_SECONDS,
        bucket_full_wait,
        note_bucket_free,
        note_bucket_full,
    )

    state: dict[str, Any] = {}
    assert bucket_full_wait(state, 1000.0) == 0.0

    assert note_bucket_full(state, 1000.0) is True
    assert bucket_full_wait(state, 1000.0 + 60) == BUCKET_FULL_PAUSE_SECONDS - 60
    assert bucket_full_wait(state, 1000.0 + BUCKET_FULL_PAUSE_SECONDS) == 0.0

    # Через час проверка снова упёрлась — молча, до шести часов с тревоги.
    assert note_bucket_full(state, 1000.0 + 3600) is False
    assert note_bucket_full(state, 1000.0 + 6 * 3600) is True

    assert note_bucket_free(state) is True
    assert note_bucket_free(state) is False
    assert bucket_full_wait(state, 1000.0) == 0.0


def test_clock_jump_back_does_not_freeze_the_upload() -> None:
    """После обрыва питания часы идут назад: отметка «до» уезжает далеко в
    будущее. Пауза длиннее часа невозможна — значит, пробовать сразу."""
    from src.cloud_backup.uploader import bucket_full_wait

    assert bucket_full_wait({"bucket_full_until": 1_000_000.0}, 100.0) == 0.0
