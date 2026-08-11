"""Журнал проходов калитки: кто подошёл, узнала ли его система, что она решила.

Отдельная служба, а не поток веб-службы, по той же причине, что и уведомления:
перезапуск админки не должен оставлять дыру в журнале, а падение журнала не
должно ронять админку.

Механики здесь нет и в этой версии быть не может. Единственное решение, которое
служба умеет записать, — `denied_mode_off`: человек у калитки был, автооткрытие
выключено. Настройки и функция решения появятся отдельно, импульс на реле —
только после письменного разрешения владельца (`AGENTS.md`).

Источник — HTTP API Frigate, как у уведомлений. Опрос чаще: сообщению
безразлично, придёт оно секундой позже, а решению об открытии — нет.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.common.database import GatePass, create_session_factory
from src.common.sd_notify import Notifier
from src.common.settings import ConfigurationError, Settings, validate_runtime_paths
from src.gate.notify import (
    WATCH_WINDOW_SECONDS,
    GateEvent,
    fetch_events,
    fetch_snapshot,
    gate_cameras,
    recognised_name,
)

logger = logging.getLogger("catdoor.gate-access")

# Журнал ведётся про людей: машина у калитки лицом её не откроет, а строка о
# ней утопила бы в шуме то, ради чего журнал заведён.
PASS_LABELS = frozenset({"person"})
# Пока настроек нет, решение одно и честное: автооткрытие выключено.
DECISION_MODE_OFF = "denied_mode_off"
REASON_MODE_OFF = "Автооткрытие выключено"
# Уборка старых строк — раз в час: чаще незачем, а раз в сутки означало бы, что
# после долгого простоя чистка не случится вовсе.
PURGE_INTERVAL_SECONDS = 3600.0
_UNSAFE_IN_NAME = re.compile(r"[^A-Za-z0-9._-]")


@dataclass(frozen=True)
class GatePassRecord:
    """Строка журнала до того, как она попала в базу."""

    frigate_event_id: str
    created_at: datetime
    camera: str
    person_name: str | None
    score: float | None
    zone: str | None
    decision: str
    reason: str | None
    has_snapshot: bool


def pass_snapshot_path(data_dir: Path, frigate_event_id: str) -> Path:
    """Куда положить снимок прохода.

    Не в каталог уведомлений: там снимок удаляется сразу после отправки, а
    здесь он живёт столько же, сколько строка журнала. Идентификатор приходит
    из чужого API, поэтому в имя файла попадают только безопасные символы —
    иначе он выбрался бы из каталога.
    """
    safe = _UNSAFE_IN_NAME.sub("_", frigate_event_id) or "unknown"
    return data_dir / "runtime" / "gate-passes" / f"{safe}.jpg"


def passes_from_events(
    events: list[GateEvent],
    *,
    cameras: dict[str, str],
) -> list[GatePassRecord]:
    """Что из ленты Frigate попадает в журнал калитки.

    Frigate обслуживает оба контура узла разом, и лента у него общая: без
    отбора по камерам в журнал калитки поехали бы проходы кошачьей дверцы.

    Незнакомые лица записываются тоже, с пустым именем. Строка «в 02:40 у
    калитки был кто-то неузнанный» стоит дороже большинства узнанных.
    """
    records = []
    for event in events:
        if event.label not in PASS_LABELS:
            continue
        if event.camera not in cameras:
            continue
        if not event.id:
            continue
        records.append(
            GatePassRecord(
                frigate_event_id=event.id,
                created_at=datetime.fromtimestamp(event.started_at, UTC),
                camera=event.camera,
                person_name=recognised_name(event),
                score=event.name_score,
                zone=", ".join(event.zones)[:100] or None,
                decision=DECISION_MODE_OFF,
                reason=REASON_MODE_OFF,
                has_snapshot=event.has_snapshot,
            )
        )
    return records


def record_passes(
    session: Session,
    records: list[GatePassRecord],
    *,
    snapshot: Callable[[GatePassRecord], str | None] | None = None,
    heartbeat: Callable[[], None] | None = None,
) -> tuple[int, int]:
    """Записать проходы. Возвращает «добавлено» и «уточнено».

    Одно и то же событие приходит несколько опросов подряд: имя лицу Frigate
    приписывает не сразу, а через секунды после появления человека. Поэтому
    строка обновляется по идентификатору события, а не добавляется заново — и
    поэтому же имя, появившееся поздно, всё-таки попадает в журнал.
    """
    if not records:
        return (0, 0)
    beat = heartbeat or (lambda: None)
    known = {
        row.frigate_event_id: row
        for row in session.execute(
            select(GatePass).where(
                GatePass.frigate_event_id.in_([record.frigate_event_id for record in records])
            )
        ).scalars()
    }
    added = 0
    refined = 0
    for record in records:
        # Сторож кормится на каждой строке: забор снимка стоит до пятнадцати
        # секунд, и пачки из двадцати событий хватило бы, чтобы систему убили
        # посреди работы.
        beat()
        stored = known.get(record.frigate_event_id)
        if stored is None:
            stored = GatePass(
                frigate_event_id=record.frigate_event_id,
                created_at=record.created_at,
                camera=record.camera,
                person_name=record.person_name,
                score=record.score,
                zone=record.zone,
                decision=record.decision,
                reason=record.reason,
                snapshot_path=snapshot(record) if snapshot is not None else None,
            )
            session.add(stored)
            added += 1
            continue
        changed = False
        if record.person_name is not None and record.person_name != stored.person_name:
            logger.info(
                "Проход %s опознан: %s", record.frigate_event_id, record.person_name
            )
            stored.person_name = record.person_name
            changed = True
        if record.score is not None and record.score != stored.score:
            stored.score = record.score
            changed = True
        if record.zone is not None and record.zone != stored.zone:
            stored.zone = record.zone
            changed = True
        if stored.snapshot_path is None and snapshot is not None:
            stored.snapshot_path = snapshot(record)
            changed = stored.snapshot_path is not None or changed
        if changed:
            refined += 1
    return (added, refined)


def purge_old_passes(session: Session, *, data_dir: Path, older_than: datetime) -> int:
    """Убрать строки старше срока хранения вместе с их снимками.

    Без этого каталог снимков рос бы вечно: у калитки людно, и каждый проход —
    файл. Снимок удаляется первым: строка без снимка — это «кадра не осталось»,
    а снимок без строки не найдёт уже никто.
    """
    stale = list(
        session.execute(select(GatePass).where(GatePass.created_at < older_than)).scalars()
    )
    for row in stale:
        if row.snapshot_path:
            (data_dir / row.snapshot_path).unlink(missing_ok=True)
        session.delete(row)
    return len(stale)


def _snapshot_writer(
    settings: Settings, events: dict[str, GateEvent]
) -> Callable[[GatePassRecord], str | None]:
    """Забрать снимок события и вернуть путь к нему относительно `data_dir`.

    Относительный путь, а не полный: том с данными переезжает между узлами, и
    записанный в базу `/srv/data/...` пережил бы переезд неправильно.
    """

    def store(record: GatePassRecord) -> str | None:
        event = events.get(record.frigate_event_id)
        if event is None or not event.has_snapshot:
            return None
        destination = pass_snapshot_path(settings.data_dir, record.frigate_event_id)
        if not destination.exists() and not fetch_snapshot(
            settings.gate_api_url, event, destination
        ):
            return None
        return destination.relative_to(settings.data_dir).as_posix()

    return store


def run(*, once: bool = False) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    settings = Settings()
    validate_runtime_paths(settings)
    factory = create_session_factory(settings.resolved_database_path)
    notifier = Notifier.from_environment()
    notifier.ready("Журнал калитки на связи")
    last_purge_at: float | None = None
    while True:
        # Окно наблюдения отсчитывается от текущего времени, а не от старта
        # службы: перезапуск не должен оставлять в журнале дыру. Повторов это
        # не даёт — строка обновляется по идентификатору события.
        window_start = time.time() - WATCH_WINDOW_SECONDS
        try:
            events = fetch_events(settings.gate_api_url, window_start)
        except RuntimeError as exc:
            # Frigate перезапускают; это не повод падать.
            logger.warning("%s", exc)
            events = []
        cameras = gate_cameras(settings.data_dir)
        if events and not cameras:
            # Пустой журнал из-за нечитаемого инвентаря выглядит точно так же,
            # как «к калитке никто не подходил», а это разные вещи.
            logger.warning("Ни одной камеры контура калитки — журнал пуст не потому, что тихо")
        records = passes_from_events(events, cameras=cameras)
        if records:
            with factory() as session:
                added, refined = record_passes(
                    session,
                    records,
                    snapshot=_snapshot_writer(settings, {event.id: event for event in events}),
                    heartbeat=notifier.watchdog,
                )
                session.commit()
            if added or refined:
                logger.info("Журнал калитки: добавлено %d, уточнено %d", added, refined)
        notifier.watchdog()
        now = time.monotonic()
        if settings.gate_access_retention_days and (
            last_purge_at is None or now - last_purge_at >= PURGE_INTERVAL_SECONDS
        ):
            with factory() as session:
                removed = purge_old_passes(
                    session,
                    data_dir=settings.data_dir,
                    older_than=datetime.now(UTC)
                    - timedelta(days=settings.gate_access_retention_days),
                )
                session.commit()
            if removed:
                logger.info("Из журнала калитки убрано %d старых проходов", removed)
            last_purge_at = now
        if once:
            return
        time.sleep(settings.gate_access_poll_seconds)


def main() -> None:
    try:
        run()
    except ConfigurationError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
