"""Журнал проходов калитки: кто подошёл, узнала ли его система, что она решила.

Отдельная служба, а не поток веб-службы, по той же причине, что и уведомления:
перезапуск админки не должен оставлять дыру в журнале, а падение журнала не
должно ронять админку.

**Реле трогает только режим `on`.** В `off` не уходит ни одной команды, в
`dry` решение заканчивается записью «открыл бы» — это тот же приём, которым
проверялась кошачья дверца, и он даёт настоящую статистику ложных
распознаваний. Кнопка в админке работает во всех трёх режимах: они про
автоматику, а не про человека, нажавшего сам.

Кнопка живёт в вебе, а плата — здесь, потому что у веб-процесса нет и не должно
быть прав на механику. Веб кладёт просьбу файлом, служба её исполняет и кладёт
рядом ответ — та же схема, что у переключателя контуров.

Решение принимает чистая функция `src/gate/decision.py`, ничего не знающая ни о
базе, ни о сети. Служба лишь собирает для неё факты и раскладывает результат по
столбцам.

Источник — HTTP API Frigate, как у уведомлений. Опрос чаще: сообщению
безразлично, придёт оно секундой позже, а решению об открытии — нет.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.common.cat_door_config import (
    CatDoorConfig,
    cat_door_config_path,
    load_cat_door_config,
)
from src.common.database import GatePass, create_session_factory
from src.common.gate_access_config import (
    GateAccessConfig,
    gate_access_config_path,
    load_gate_access_config,
)
from src.common.gate_open_request import (
    REQUEST_TTL_SECONDS,
    TARGET_CAT_DOOR,
    OpenRequest,
    OpenResult,
    boot_id,
    heartbeat_path,
    load_open_request,
    load_open_result,
    open_request_path,
    open_result_path,
    save_open_result,
    touch_heartbeat,
)
from src.common.sd_notify import Notifier
from src.common.settings import ConfigurationError, Settings, validate_runtime_paths
from src.common.telegram import TelegramClient
from src.gate.decision import DECISION_DRY_RUN, Factor, decide
from src.gate.gate_state import UNKNOWN as GATE_STATE_UNKNOWN
from src.gate.gate_state import GateStateWatcher
from src.gate.movement import UNKNOWN as MOVEMENT_UNKNOWN
from src.gate.movement import area_ratio as movement_ratio
from src.gate.movement import classify as classify_movement
from src.gate.notify import (
    APPROACH_UNKNOWN,
    WATCH_WINDOW_SECONDS,
    GateEvent,
    approach_from_box,
    fetch_events,
    fetch_first_area,
    fetch_first_box,
    fetch_snapshot,
    gate_cameras,
    recognised_name,
)
from src.gate.opener import RelayOpener
from src.gate.presence import PhonePresence

logger = logging.getLogger("catdoor.gate-access")

# Кого пишем в журнал. Решения принимаются только по людям: калитку по номеру
# не открывают, и машина здесь — строка отчёта, а не участник разговора.
PASS_LABELS = frozenset({"person", "car"})
DECIDED_LABELS = frozenset({"person"})
# Что записывается про машину. Не отказ: отказывать ей не в чем, её просто
# никто ни о чём не спрашивал.
DECISION_LOGGED = "logged"
# Решения, после которых считается пауза.
OPENING_DECISIONS = frozenset({DECISION_DRY_RUN, "opened"})
# Решения, за которыми стоит уже случившееся действие: импульс ушёл или не
# ушёл, и это факт о физическом мире. Переоценка того же события на следующем
# опросе фактом не является и переписать его не может.
TERMINAL_DECISIONS = frozenset({"opened", "error"})
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
    label: str
    person_name: str | None
    plate: str | None
    score: float | None
    zone: str | None
    decision: str
    reason: str | None
    has_snapshot: bool
    # Обстоятельства прохода. Пока только наблюдаются и пишутся в журнал: на
    # решение не влияют, пока не наберётся неделя записей.
    approach: str = APPROACH_UNKNOWN
    first_box_bottom: float | None = None
    movement: str = MOVEMENT_UNKNOWN
    area_ratio: float | None = None
    gate_state_before: str | None = None
    gate_state_after: str | None = None
    # Какие факторы сказали «за», «против» и «не знаю». Без этого разбирать
    # отказы невозможно: видно решение, но не видно, чем оно вызвано.
    factors_json: str | None = None
    # Что реально ушло на плату. Пусто у всего, кроме исполненных открытий.
    pulse_ms: int | None = None
    modbus_ok: bool | None = None


def pass_snapshot_path(data_dir: Path, frigate_event_id: str) -> Path:
    """Куда положить снимок прохода.

    Не в каталог уведомлений: там снимок удаляется сразу после отправки, а
    здесь он живёт столько же, сколько строка журнала. Идентификатор приходит
    из чужого API, поэтому в имя файла попадают только безопасные символы —
    иначе он выбрался бы из каталога.
    """
    safe = _UNSAFE_IN_NAME.sub("_", frigate_event_id) or "unknown"
    return data_dir / "runtime" / "gate-passes" / f"{safe}.jpg"


@dataclass(frozen=True)
class LastOpening:
    """Когда и по какому событию система в последний раз сказала «открыл бы».

    Идентификатор нужен из-за окна наблюдения: событие перечитывается каждый
    опрос, и без него собственное решение о человеке через секунду становилось
    бы для него же паузой — «прошло 1 с из 20».
    """

    frigate_event_id: str
    at: datetime


def passes_from_events(
    events: list[GateEvent],
    *,
    cameras: dict[str, str],
    config: GateAccessConfig,
    timezone: ZoneInfo,
    last_opening: LastOpening | None = None,
    factors_for: Callable[[str], list[Factor]] | None = None,
    approach_for: Callable[[str], tuple[str, float | None]] | None = None,
    geometry_for: Callable[[GateEvent], tuple[str, float | None]] | None = None,
    gate_state: str | None = None,
    gate_just_closed: bool = False,
) -> tuple[list[GatePassRecord], LastOpening | None]:
    """Что из ленты Frigate попадает в журнал и что система об этом решила.

    Frigate обслуживает оба контура узла разом, и лента у него общая: без
    отбора по камерам в журнал калитки поехали бы проходы кошачьей дверцы.

    Незнакомые лица записываются тоже, с пустым именем. Строка «в 02:40 у
    калитки был кто-то неузнанный» стоит дороже большинства узнанных.

    Возвращает строки и обновлённую отметку последнего «открыл бы»: пауза
    считается по ней, а не по часам службы.
    """
    records = []
    opening = last_opening
    for event in events:
        if event.label not in PASS_LABELS:
            continue
        if event.camera not in cameras:
            continue
        if not event.id:
            continue
        moment = datetime.fromtimestamp(event.started_at, UTC)
        if event.label not in DECIDED_LABELS:
            # Машина: пишем и идём дальше. Ни решения, ни паузы, ни факторов —
            # иначе она съедала бы паузу у людей и мешала разбирать отказы.
            records.append(
                GatePassRecord(
                    frigate_event_id=event.id,
                    created_at=moment,
                    camera=event.camera,
                    label=event.label,
                    person_name=None,
                    plate=recognised_name(event),
                    score=event.name_score,
                    zone=", ".join(event.zones)[:100] or None,
                    decision=DECISION_LOGGED,
                    reason="Машина у калитки — только запись",
                    has_snapshot=event.has_snapshot,
                    gate_state_before=gate_state,
                )
            )
            continue
        approach, bottom = (
            approach_for(event.id) if approach_for is not None else (APPROACH_UNKNOWN, None)
        )
        movement, ratio = (
            geometry_for(event) if geometry_for is not None else (MOVEMENT_UNKNOWN, None)
        )
        name = recognised_name(event)
        extra = factors_for(name) if factors_for is not None and name is not None else []
        verdict = decide(
            camera=event.camera,
            person_name=name,
            score=event.name_score,
            zones=event.zones,
            config=config,
            now=moment.astimezone(timezone),
            # Собственное прошлое решение по этому же событию паузой не считается.
            last_opened_at=(
                opening.at
                if opening is not None and opening.frigate_event_id != event.id
                else None
            ),
            factors=extra,
            movement=movement,
            gate_just_closed=gate_just_closed,
        )
        if verdict.would_open:
            opening = LastOpening(event.id, moment)
        records.append(
            GatePassRecord(
                frigate_event_id=event.id,
                created_at=moment,
                camera=event.camera,
                label=event.label,
                person_name=name,
                plate=None,
                score=event.name_score,
                zone=", ".join(event.zones)[:100] or None,
                decision=verdict.decision,
                reason=verdict.reason,
                has_snapshot=event.has_snapshot,
                factors_json=json.dumps(
                    [factor.as_dict() for factor in verdict.factors], ensure_ascii=False
                ),
                approach=approach,
                first_box_bottom=bottom,
                movement=movement,
                area_ratio=ratio,
                gate_state_before=gate_state,
            )
        )
    return records, opening


def execute_openings(
    records: list[GatePassRecord],
    *,
    config: GateAccessConfig,
    opener: RelayOpener,
    already: dict[str, str],
    notify: Callable[[str], None] | None = None,
) -> list[GatePassRecord]:
    """Довести решение «открыл бы» до импульса, если режим это позволяет.

    Отделено от самого решения намеренно: решение — чистая функция, которую
    проверяют таблицей случаев, а исполнение трогает железо. Смешав их, нельзя
    было бы проверить ни то ни другое.

    **Повтор исключается по журналу, а не по времени.** Событие перечитывается
    каждый опрос всё окно наблюдения: без этой проверки один человек у калитки
    получал бы импульс каждую секунду, а после отказа платы служба долбила бы её
    без остановки. Строка, у которой уже есть исход, второй раз не исполняется.
    """
    if config.mode != "on" or not config.gates:
        return records
    relay = config.gates[0].relay
    executed = []
    for record in records:
        if record.decision != DECISION_DRY_RUN:
            executed.append(record)
            continue
        outcome = already.get(record.frigate_event_id)
        if outcome in TERMINAL_DECISIONS:
            # Исход уже есть — несём его дальше, а не прежнее «открыл бы».
            executed.append(replace(record, decision=outcome))
            continue
        pulse = opener.pulse(relay, config.pulse_ms)
        executed.append(
            replace(
                record,
                decision="opened" if pulse.ok else "error",
                reason=(
                    f"{record.person_name}: открыто по лицу. {pulse.detail}"
                    if pulse.ok
                    else f"{record.person_name}: открыть не удалось. {pulse.detail}"
                ),
                pulse_ms=pulse.pulse_ms,
                modbus_ok=pulse.ok,
            )
        )
        if notify is not None:
            # Уведомление на каждое открытие — условие разрешения владельца.
            notify(
                f"Калитка: {'открыта' if pulse.ok else 'открыть не удалось'} "
                f"по лицу — {record.person_name}"
            )
    return executed


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
                label=record.label,
                person_name=record.person_name,
                plate=record.plate,
                score=record.score,
                zone=record.zone,
                decision=record.decision,
                reason=record.reason,
                factors_json=record.factors_json,
                approach=record.approach,
                first_box_bottom=record.first_box_bottom,
                movement=record.movement,
                area_ratio=record.area_ratio,
                gate_state_before=record.gate_state_before,
                gate_state_after=record.gate_state_after,
                pulse_ms=record.pulse_ms,
                modbus_ok=record.modbus_ok,
                snapshot_path=snapshot(record) if snapshot is not None else None,
            )
            session.add(stored)
            added += 1
            continue
        changed = False
        if record.plate and record.plate != stored.plate:
            # Номер Frigate читает не сразу, как и имя.
            stored.plate = record.plate
            changed = True
        if record.person_name is not None and record.person_name != stored.person_name:
            logger.info(
                "Проход %s опознан: %s", record.frigate_event_id, record.person_name
            )
            stored.person_name = record.person_name
            changed = True
        if stored.decision in TERMINAL_DECISIONS and record.decision not in TERMINAL_DECISIONS:
            # Строку, за которой стоит ушедший импульс, переоценка не трогает.
            #
            # 2026-08-11 на живом прогоне без этой защиты вышло качание:
            # следующий опрос переписывал «открыл» обратно в «открыл бы», ещё
            # следующий видел незавершённое событие и открывал снова — калитка
            # щёлкала каждые две секунды всё окно наблюдения.
            continue
        if record.decision != stored.decision:
            # Решение обязано меняться вместе с именем: первый опрос видит
            # человека без имени и отказывает, а через две секунды это уже
            # свой. Строка, застрявшая на первом решении, врала бы в самой
            # интересной части журнала.
            logger.info(
                "Проход %s: решение %s → %s",
                record.frigate_event_id,
                stored.decision,
                record.decision,
            )
            stored.decision = record.decision
            stored.reason = record.reason
            stored.factors_json = record.factors_json
            stored.pulse_ms = record.pulse_ms
            stored.modbus_ok = record.modbus_ok
            changed = True
        if record.score is not None and record.score != stored.score:
            stored.score = record.score
            changed = True
        if record.zone is not None and record.zone != stored.zone:
            stored.zone = record.zone
            changed = True
        if record.gate_state_after and record.gate_state_after != stored.gate_state_after:
            # Створка могла шевельнуться уже после того, как строка появилась.
            stored.gate_state_after = record.gate_state_after
            changed = True
        if record.movement != MOVEMENT_UNKNOWN and record.movement != stored.movement:
            # Движение уточняется с каждым обновлением события: человек, который
            # только что стоял, мог пойти.
            stored.movement = record.movement
            stored.area_ratio = record.area_ratio
            changed = True
        if stored.approach in (None, APPROACH_UNKNOWN) and record.approach != APPROACH_UNKNOWN:
            stored.approach = record.approach
            stored.first_box_bottom = record.first_box_bottom
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


def handle_open_request(
    session: Session,
    *,
    request: OpenRequest,
    config: GateAccessConfig,
    opener: RelayOpener,
    now: float,
    timezone: ZoneInfo,
    cat_door: CatDoorConfig | None = None,
    notify: Callable[[str], None] | None = None,
) -> OpenResult:
    """Исполнить нажатие кнопки «Открыть калитку».

    Единственный путь, которым в этой версии на плату уходит команда. Автоматика
    до неё не доходит: решение по лицу заканчивается записью «открыл бы».

    Просьба стареет намеренно. Нажатие, до которого служба не добралась за
    десяток секунд, человеку у калитки уже не нужно, а сработать оно успело бы
    ровно тогда, когда он ушёл.
    """
    current_boot = boot_id()
    if request.boot_id and current_boot and request.boot_id != current_boot:
        # Узел перезагружался между нажатием и этой секундой. Часам здесь
        # верить нельзя: после обрыва питания RTC отстаёт, chrony подтягивает
        # время не сразу, и просьба из прошлой жизни выглядит свежей.
        logger.warning("Просьба открыть калитку осталась от прошлой загрузки — не исполняю")
        return OpenResult(
            id=request.id,
            ok=False,
            detail="Просьба осталась от прошлой загрузки узла",
            finished_at=now,
        )
    age = now - request.requested_at
    if not 0 <= age <= REQUEST_TTL_SECONDS:
        # Отрицательный возраст — часы шагнули назад. Это не «свежая просьба»,
        # а неизвестно что; человек нажмёт ещё раз, и это дешевле, чем открыть
        # калитку неизвестно когда.
        logger.warning("Просьба открыть калитку негодна по времени (возраст %.1f с)", age)
        return OpenResult(
            id=request.id,
            ok=False,
            detail="Просьба устарела: служба не успела её увидеть",
            finished_at=now,
        )
    if request.target == TARGET_CAT_DOOR:
        if cat_door is None:
            return OpenResult(
                id=request.id, ok=False, detail="Дверца не настроена", finished_at=now
            )
        channel, milliseconds, what = cat_door.relay, cat_door.hold_ms, "дверца"
    else:
        gate = config.gates[0] if config.gates else None
        if gate is None:
            return OpenResult(
                id=request.id,
                ok=False,
                detail="Канал реле калитки не настроен",
                finished_at=now,
            )
        channel, milliseconds, what = gate.relay, config.pulse_ms, "калитка"
    pulse = opener.pulse(channel, milliseconds)
    moment = datetime.fromtimestamp(now, UTC)
    session.add(
        GatePass(
            # Своего события у ручного открытия нет, а ключ уникален: берём
            # идентификатор просьбы — он же защищает от двойного исполнения.
            frigate_event_id=f"manual-{request.id}",
            created_at=moment,
            camera="",
            label=request.target,
            decision="opened" if pulse.ok else "error",
            reason=(
                f"{what.capitalize()}: открыто вручную из админки. {pulse.detail}"
                if pulse.ok
                else f"{what.capitalize()}: открыть вручную не удалось. {pulse.detail}"
            ),
            factors_json=json.dumps(
                [{"name": "manual", "verdict": "for", "detail": request.source}],
                ensure_ascii=False,
            ),
            pulse_ms=pulse.pulse_ms,
            modbus_ok=pulse.ok,
        )
    )
    if notify is not None:
        # Уведомление на каждое срабатывание — условие разрешения владельца, а
        # не удобство. Отказ отправки не отменяет открытия и не роняет службу.
        local = moment.astimezone(timezone)
        notify(
            f"{what.capitalize()}: {'открыта' if pulse.ok else 'открыть не удалось'} "
            f"по кнопке в админке · {local.strftime('%H:%M')}"
        )
    return OpenResult(
        id=request.id,
        ok=pulse.ok,
        detail=pulse.detail,
        finished_at=now,
        pulse_ms=pulse.pulse_ms,
    )


def last_opening(session: Session) -> LastOpening | None:
    """Последнее решение «открыл бы» из журнала. Нужно, чтобы пауза пережила перезапуск."""
    row = session.execute(
        select(GatePass)
        .where(GatePass.decision.in_(OPENING_DECISIONS))
        .order_by(GatePass.created_at.desc(), GatePass.id.desc())
        .limit(1)
    ).scalar_one_or_none()
    if row is None:
        return None
    moment = row.created_at
    # sqlite отдаёт метки без зоны, а записаны они в UTC.
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return LastOpening(row.frigate_event_id, moment)


def _approach_for(
    settings: Settings, cache: dict[str, tuple[str, float | None]]
) -> Callable[[str], tuple[str, float | None]]:
    """Откуда пришёл человек. Ответ на событие неизменен, поэтому запоминается."""

    def approach(event_id: str) -> tuple[str, float | None]:
        if event_id in cache:
            return cache[event_id]
        bottom = fetch_first_box(settings.gate_api_url, event_id)
        result = (approach_from_box(bottom), bottom)
        if bottom is None:
            # Не запоминаем неудачу. Frigate заводит запись таймлайна с
            # задержкой, а мы спрашиваем на первом же опросе — на живом узле
            # 2026-08-12 из-за этого все проходы получили «откуда неизвестно»
            # навсегда. Событие живёт в окне наблюдения пять минут, и
            # следующий опрос спросит снова.
            return result
        cache[event_id] = result
        if len(cache) > 200:
            # Окно наблюдения — пять минут; двести событий с запасом
            # перекрывают его, а расти без предела памяти незачем.
            cache.pop(next(iter(cache)))
        return result

    return approach


def _geometry_for(
    settings: Settings, first_areas: dict[str, float | None], config: GateAccessConfig
) -> Callable[[GateEvent], tuple[str, float | None]]:
    """Куда идёт человек, по изменению площади рамки.

    Первая площадь берётся из таймлайна и не меняется, поэтому запоминается.
    Текущая приходит в самом событии и обновляется раз в несколько секунд —
    это и есть предел разрешения при опросе по HTTP.
    """

    def geometry(event: GateEvent) -> tuple[str, float | None]:
        if event.id not in first_areas:
            first = fetch_first_area(settings.gate_api_url, event.id)
            if first is None:
                # Не запоминаем незнание: таймлайн заводится с задержкой.
                return (MOVEMENT_UNKNOWN, None)
            first_areas[event.id] = first
            if len(first_areas) > 200:
                first_areas.pop(next(iter(first_areas)))
        first = first_areas[event.id]
        rules = config.movement
        return (
            classify_movement(
                first,
                event.area,
                approaching_ratio=rules.approaching_ratio,
                leaving_ratio=rules.leaving_ratio,
            ),
            movement_ratio(first, event.area),
        )

    return geometry


def _factors_for(
    config: GateAccessConfig, presence: PhonePresence
) -> Callable[[str], list[Factor]]:
    """Что известно о человеке, кроме его лица.

    Фактор телефона считается всегда, когда телефон задан, — даже при
    `require_second_factor: false`. Иначе включать его пришлось бы вслепую, а
    план требует сперва посмотреть, сколько раз телефон отвечал в момент
    прихода: главный риск этой схемы — ложные отказы, а не подделка.
    """

    def factors(name: str) -> list[Factor]:
        if config.mode == "off":
            # В выключенном режиме решение всё равно одно, а проверка стоит
            # процессов и сетевых пакетов на каждое событие.
            return []
        person = config.person(name)
        if person is None or not person.phone_ip or not person.phone_mac:
            return []
        return [presence.factor(person.phone_ip, person.phone_mac)]

    return factors


def _notify_all(send: Callable[[str], None] | None, messages: list[str]) -> None:
    """Разослать накопленные сообщения. Отказ отправки ничего не отменяет."""
    if send is None:
        return
    for message in messages:
        try:
            send(message)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Уведомление не ушло: %s", exc)


class OpenRequestWatcher:
    """Отдельный поток, который слушает просьбы открыть.

    Раньше просьбы разбирались в общем цикле, между опросом Frigate и записью
    журнала. Опрос может занять секунды — таймаут HTTP десять, забор снимка
    пятнадцать, — и человек у двери успевал получить «служба не ответила»,
    хотя служба работала и вот-вот бы ответила.

    Поток проверяет файл пять раз в секунду и ничем тяжёлым не занят. Команды
    на плату идут под общим замком: писать в один порт шлюза с двух сторон —
    верный способ однажды получить отказ соединения.
    """

    def __init__(
        self,
        settings: Settings,
        factory: Callable[[], Session],
        opener: RelayOpener,
        board_lock: threading.Lock,
        *,
        timezone: ZoneInfo,
        notify: Callable[[str], None] | None,
        interval: float = 0.2,
    ) -> None:
        self._settings = settings
        self._factory = factory
        self._opener = opener
        self._board = board_lock
        self._timezone = timezone
        self._notify = notify
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="catdoor-open-requests", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        request_path = open_request_path(self._settings.data_dir)
        result_path = open_result_path(self._settings.data_dir)
        alive_path = heartbeat_path(self._settings.data_dir)
        while not self._stop.wait(self._interval):
            touch_heartbeat(alive_path)
            try:
                self._once(request_path, result_path)
            except Exception:
                # Поток обязан пережить что угодно: он единственный, кто
                # отвечает человеку у двери.
                logger.exception("Разбор просьбы открыть не удался")

    def _once(self, request_path: Path, result_path: Path) -> None:
        request = load_open_request(request_path)
        if request is None:
            return
        done = load_open_result(result_path)
        if done is not None and done.id == request.id:
            return
        config = load_gate_access_config(
            gate_access_config_path(self._settings.data_dir), GateAccessConfig()
        )
        cat_door = load_cat_door_config(
            cat_door_config_path(self._settings.data_dir), CatDoorConfig()
        )
        pending: list[str] = []
        with self._board, self._factory() as session:
            result = handle_open_request(
                session,
                request=request,
                config=config,
                opener=self._opener,
                now=time.time(),
                timezone=self._timezone,
                cat_door=cat_door,
                notify=pending.append,
            )
            session.commit()
        # Ответ — сразу после журнала и до уведомлений: тот, кто стоит перед
        # дверью, не должен ждать чата.
        save_open_result(result_path, result)
        _notify_all(self._notify, pending)


def _telegram_sender(settings: Settings) -> Callable[[str], None] | None:
    """Отправитель уведомлений об открытии. `None` — бот не настроен.

    Уведомление на каждое срабатывание — условие разрешения владельца. Если
    бота нет, служба всё равно работает: отказаться открывать из-за ненастроенного
    телеграма значило бы запереть человека у калитки, — но говорит об этом вслух.
    """
    client = TelegramClient(
        settings.telegram_bot_token,
        settings.telegram_chat_id,
        api_base=settings.telegram_api_base,
    )
    if not client.configured:
        logger.warning(
            "Бот не настроен: открытия калитки не попадут в Телеграм, только в журнал"
        )
        return None
    return client.send_message


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
    timezone = ZoneInfo(settings.display_timezone)
    config_path = gate_access_config_path(settings.data_dir)
    config = load_gate_access_config(config_path, GateAccessConfig())
    presence = PhonePresence()
    # Створка: пока только наблюдаем и пишем в журнал. Решения это не трогает —
    # сперва неделя записей, потом разговор о правилах.
    gate_watcher = GateStateWatcher(settings.mqtt_host, settings.mqtt_port)
    if not gate_watcher.start():
        logger.info("Состояние створки писаться не будет")
    # Плата у калитки и у дверцы одна и та же, поэтому адрес берётся из уже
    # существующей настройки контроллера.
    opener = RelayOpener(
        host=settings.door_controller_host,
        port=settings.door_controller_port,
    )
    telegram = _telegram_sender(settings)
    # Плату держит один процесс, но потоков в нём теперь два: замок делает так,
    # чтобы в порт шлюза одновременно не полезли оба.
    board_lock = threading.Lock()
    requests = OpenRequestWatcher(
        settings, factory, opener, board_lock, timezone=timezone, notify=telegram
    )
    requests.start()
    notifier = Notifier.from_environment()
    notifier.ready("Журнал калитки на связи")
    last_purge_at: float | None = None
    # Первая рамка события неизменна, и спрашивать её у Frigate повторно на
    # каждом опросе — лишний запрос на каждое событие в окне наблюдения.
    first_boxes: dict[str, tuple[str, float | None]] = {}
    first_areas: dict[str, float | None] = {}
    record_states: dict[str, str] = {}
    with factory() as session:
        # Пауза переживает перезапуск: иначе обновление админки среди дня
        # обнуляло бы её, и человек, которого камера видит снова и снова,
        # получал бы очередь решений.
        opening = last_opening(session)
    while True:
        config = load_gate_access_config(config_path, config)
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
        state = gate_watcher.snapshot()
        records, opening = passes_from_events(
            events,
            cameras=cameras,
            config=config,
            timezone=timezone,
            last_opening=opening,
            factors_for=_factors_for(config, presence),
            approach_for=_approach_for(settings, first_boxes),
            geometry_for=_geometry_for(settings, first_areas, config),
            gate_state=state.value if state.known else GATE_STATE_UNKNOWN,
            gate_just_closed=state.closed_within(
                config.movement.return_window_seconds, now=time.time()
            ),
        )
        # Состояние после прохода уточняется на следующих опросах, пока событие
        # в окне наблюдения: створка часто шевелится уже после появления строки.
        for record in records:
            record_states.setdefault(record.frigate_event_id, state.value)
        records = [
            replace(record, gate_state_after=state.value if state.known else GATE_STATE_UNKNOWN)
            for record in records
        ]
        if records:
            with factory() as session:
                # Исход прошлых опросов по этим же событиям: он решает, не
                # исполнено ли открытие уже.
                already = {
                    row.frigate_event_id: row.decision
                    for row in session.execute(
                        select(GatePass).where(
                            GatePass.frigate_event_id.in_(
                                [record.frigate_event_id for record in records]
                            )
                        )
                    ).scalars()
                }
                automatic: list[str] = []
                with board_lock:
                    records = execute_openings(
                        records,
                        config=config,
                        opener=opener,
                        already=already,
                        notify=automatic.append,
                    )
                added, refined = record_passes(
                    session,
                    records,
                    snapshot=_snapshot_writer(settings, {event.id: event for event in events}),
                    heartbeat=notifier.watchdog,
                )
                session.commit()
            if added or refined:
                logger.info("Журнал калитки: добавлено %d, уточнено %d", added, refined)
            # Сообщения уходят после записи в журнал по той же причине:
            # недоступный Телеграм не должен задерживать то, ради чего служба
            # работает.
            _notify_all(telegram, automatic)
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
