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
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.common.database import GatePass, create_session_factory
from src.common.gate_access_config import (
    GateAccessConfig,
    gate_access_config_path,
    load_gate_access_config,
)
from src.common.gate_open_request import (
    REQUEST_TTL_SECONDS,
    OpenRequest,
    OpenResult,
    boot_id,
    load_open_request,
    load_open_result,
    open_request_path,
    open_result_path,
    save_open_result,
)
from src.common.sd_notify import Notifier
from src.common.settings import ConfigurationError, Settings, validate_runtime_paths
from src.common.telegram import TelegramClient
from src.gate.decision import DECISION_DRY_RUN, Factor, decide
from src.gate.notify import (
    WATCH_WINDOW_SECONDS,
    GateEvent,
    fetch_events,
    fetch_snapshot,
    gate_cameras,
    recognised_name,
)
from src.gate.opener import RelayOpener
from src.gate.presence import PhonePresence

logger = logging.getLogger("catdoor.gate-access")

# Журнал ведётся про людей: машина у калитки лицом её не откроет, а строка о
# ней утопила бы в шуме то, ради чего журнал заведён.
PASS_LABELS = frozenset({"person"})
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
    person_name: str | None
    score: float | None
    zone: str | None
    decision: str
    reason: str | None
    has_snapshot: bool
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
        )
        if verdict.would_open:
            opening = LastOpening(event.id, moment)
        records.append(
            GatePassRecord(
                frigate_event_id=event.id,
                created_at=moment,
                camera=event.camera,
                person_name=name,
                score=event.name_score,
                zone=", ".join(event.zones)[:100] or None,
                decision=verdict.decision,
                reason=verdict.reason,
                has_snapshot=event.has_snapshot,
                factors_json=json.dumps(
                    [factor.as_dict() for factor in verdict.factors], ensure_ascii=False
                ),
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
            try:
                notify(
                    f"Калитка: {'открыта' if pulse.ok else 'открыть не удалось'} "
                    f"по лицу — {record.person_name}"
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Уведомление об открытии не ушло: %s", exc)
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
                person_name=record.person_name,
                score=record.score,
                zone=record.zone,
                decision=record.decision,
                reason=record.reason,
                factors_json=record.factors_json,
                pulse_ms=record.pulse_ms,
                modbus_ok=record.modbus_ok,
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
    gate = config.gates[0] if config.gates else None
    if gate is None:
        return OpenResult(
            id=request.id,
            ok=False,
            detail="Канал реле калитки не настроен",
            finished_at=now,
        )
    pulse = opener.pulse(gate.relay, config.pulse_ms)
    moment = datetime.fromtimestamp(now, UTC)
    session.add(
        GatePass(
            # Своего события у ручного открытия нет, а ключ уникален: берём
            # идентификатор просьбы — он же защищает от двойного исполнения.
            frigate_event_id=f"manual-{request.id}",
            created_at=moment,
            camera="",
            decision="opened" if pulse.ok else "error",
            reason=(
                f"Открыто вручную из админки. {pulse.detail}"
                if pulse.ok
                else f"Открыть вручную не удалось. {pulse.detail}"
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
        try:
            notify(
                f"Калитка: {'открыта' if pulse.ok else 'открыть не удалось'} "
                f"по кнопке в админке · {local.strftime('%H:%M')}"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Уведомление об открытии не ушло: %s", exc)
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
    request_path = open_request_path(settings.data_dir)
    result_path = open_result_path(settings.data_dir)
    # Плата у калитки и у дверцы одна и та же, поэтому адрес берётся из уже
    # существующей настройки контроллера.
    opener = RelayOpener(
        host=settings.door_controller_host,
        port=settings.door_controller_port,
    )
    telegram = _telegram_sender(settings)
    notifier = Notifier.from_environment()
    notifier.ready("Журнал калитки на связи")
    last_purge_at: float | None = None
    with factory() as session:
        # Пауза переживает перезапуск: иначе обновление админки среди дня
        # обнуляло бы её, и человек, которого камера видит снова и снова,
        # получал бы очередь решений.
        opening = last_opening(session)
    while True:
        # Нажатие кнопки разбирается первым и до опроса Frigate: человек стоит у
        # калитки, а неотвеченный запрос к Frigate стоит десяти секунд таймаута.
        config = load_gate_access_config(config_path, config)
        request = load_open_request(request_path)
        done = load_open_result(result_path)
        if request is not None and (done is None or done.id != request.id):
            with factory() as session:
                result = handle_open_request(
                    session,
                    request=request,
                    config=config,
                    opener=opener,
                    now=time.time(),
                    timezone=timezone,
                    notify=telegram,
                )
                session.commit()
            # Ответ пишется после записи в журнал: админка, увидев «открыл»,
            # должна найти строку на месте.
            save_open_result(result_path, result)
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
        records, opening = passes_from_events(
            events,
            cameras=cameras,
            config=config,
            timezone=timezone,
            last_opening=opening,
            factors_for=_factors_for(config, presence),
        )
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
                records = execute_openings(
                    records,
                    config=config,
                    opener=opener,
                    already=already,
                    notify=telegram,
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
