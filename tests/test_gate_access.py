from __future__ import annotations

import inspect
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Self
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy.exc import IntegrityError

from src.common.database import GatePass, create_session_factory
from src.common.gate_access_config import GateAccessConfig
from src.common.gate_open_request import REQUEST_TTL_SECONDS, OpenRequest
from src.gate.access import (
    ACT_WITHIN_SECONDS,
    GatePassRecord,
    LastOpening,
    PriorPass,
    execute_openings,
    handle_open_request,
    last_opening,
    pass_snapshot_path,
    passes_from_events,
    purge_old_passes,
    record_passes,
)
from src.gate.decision import DECISION_DRY_RUN, DECISION_MODE_OFF, PHONE_FACTOR, Factor
from src.gate.notify import GateEvent, fetch_events
from src.gate.opener import PulseResult

MOSCOW = ZoneInfo("Europe/Moscow")


class _FakeOpener:
    """Открывалка, которая ничего не открывает и помнит, о чём её просили."""

    def __init__(self, result: PulseResult) -> None:
        self._result = result
        self.calls: list[tuple[int, int]] = []

    def pulse(self, channel: int, milliseconds: int) -> PulseResult:
        self.calls.append((channel, milliseconds))
        return self._result


def _config(**overrides: object) -> GateAccessConfig:
    values: dict[str, object] = {
        "mode": "dry",
        "gates": [{"camera": "gate", "relay": 2}],
        "allowed_names": ["Алексей"],
    }
    values.update(overrides)
    return GateAccessConfig.model_validate(values)


def _passes(events, **overrides):  # type: ignore[no-untyped-def]
    values: dict[str, object] = {
        "cameras": {"gate": "Калитка"},
        "config": _config(mode="off"),
        "timezone": MOSCOW,
        # «Сейчас» фиксировано и дневное: без него решения брали настоящие
        # часы, и вся пачка тестов падала каждый день с 23:00 до 07:00 МСК —
        # окно автопроходов по умолчанию закрыто ночью (denied_window).
        "now": datetime(2026, 8, 14, 12, 0, tzinfo=MOSCOW),
    }
    values.update(overrides)
    return passes_from_events(events, **values)  # type: ignore[arg-type]


class _Response:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        return None


def _event(**overrides: object) -> GateEvent:
    values: dict[str, object] = {
        "id": "1786171119.126838-b1jywn",
        "camera": "gate",
        "label": "person",
        "sub_label": None,
        "started_at": 1786171119.126838,
        "has_snapshot": True,
    }
    values.update(overrides)
    return GateEvent(**values)  # type: ignore[arg-type]


def _record(**overrides: object) -> GatePassRecord:
    values: dict[str, object] = {
        "frigate_event_id": "1786171119.126838-b1jywn",
        "created_at": datetime(2026, 8, 11, 9, 0, tzinfo=UTC),
        "camera": "gate",
        "label": "person",
        "person_name": None,
        "plate": None,
        "score": None,
        "zone": None,
        "decision": DECISION_MODE_OFF,
        "reason": "Автооткрытие выключено",
        "has_snapshot": True,
    }
    values.update(overrides)
    return GatePassRecord(**values)  # type: ignore[arg-type]


def _session(runtime_path: Path):  # type: ignore[no-untyped-def]
    return create_session_factory(runtime_path / "catdoor.sqlite3")


def test_only_the_gate_cameras_reach_the_journal() -> None:
    """Лента Frigate общая на оба контура узла.

    Без отбора в журнал калитки поехали бы проходы кошачьей дверцы — и его
    цифры перестали бы означать «люди у калитки».
    """
    records, _ = _passes([_event(id="a", camera="gate"), _event(id="b", camera="face")])

    assert [record.frigate_event_id for record in records] == ["a"]


def test_a_car_is_logged_but_never_decided_about() -> None:
    """Машины нужны для отчёта, а не для решений: калитку по номеру не открывают.

    Отдельная ветка нужна не для порядка: пройди машина обычным путём, она
    съедала бы паузу у людей и засоряла разбор отказов причинами, которых к ней
    никто не предъявлял.
    """
    records, opening = _passes(
        [_event(id="a", label="car", sub_label="А123ВС"), _event(id="b", label="person")]
    )
    car = next(r for r in records if r.frigate_event_id == "a")

    assert [record.frigate_event_id for record in records] == ["a", "b"]
    assert car.label == "car"
    assert car.decision == "logged"
    assert car.plate == "А123ВС"
    assert car.person_name is None
    # Машина не двигает паузу: она никого не открывала.
    assert opening is None


def test_a_stranger_is_written_down_too() -> None:
    """«В 02:40 у калитки был кто-то неузнанный» — самая ценная строка журнала."""
    records, _ = _passes([_event(sub_label=None)])

    assert records[0].person_name is None
    assert records[0].decision == DECISION_MODE_OFF


def test_the_journal_records_who_and_how_certainly() -> None:
    records, _ = _passes(
        [_event(sub_label="Алексей", name_score=0.94, zones=("двор", "у калитки"))]
    )

    assert records[0].person_name == "Алексей"
    assert records[0].score == pytest.approx(0.94)
    assert records[0].zone == "двор, у калитки"


def test_frigate_confidence_is_read_from_both_places(monkeypatch: pytest.MonkeyPatch) -> None:
    """Frigate переносил уверенность распознавания между версиями.

    Она лежит то в корне ответа, то во вложенном `data`. Узел обновляют, и
    журнал не должен молча остаться без цифры.
    """
    payload = [
        {
            "id": "a",
            "camera": "gate",
            "label": "person",
            "sub_label": "Алексей",
            "start_time": 100.0,
            "has_snapshot": True,
            "data": {"sub_label_score": 0.97},
            "zones": ["у калитки"],
            "end_time": 112.0,
        },
        {
            "id": "b",
            "camera": "gate",
            "label": "person",
            "sub_label": "Алексей",
            "start_time": 200.0,
            "has_snapshot": False,
            "sub_label_score": 0.91,
        },
    ]
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: _Response(json.dumps(payload).encode("utf-8")),
    )

    events = fetch_events("http://127.0.0.1:5000", 0.0)

    assert [event.name_score for event in events] == [pytest.approx(0.97), pytest.approx(0.91)]
    assert events[0].zones == ("у калитки",)
    assert events[0].ended_at == pytest.approx(112.0)
    assert events[1].ended_at is None


def test_a_pass_is_written_once_and_then_refined(runtime_path: Path) -> None:
    """Имя Frigate приписывает лицу не сразу, а через секунды после события.

    Событие перечитывается каждый опрос всё окно наблюдения. Без обновления по
    идентификатору один гость дал бы десяток строк, а имя не попало бы ни в
    одну из них — оно приходит позже первой записи.
    """
    factory = _session(runtime_path)

    with factory() as session:
        first = record_passes(session, [_record()])
        session.commit()
    with factory() as session:
        second = record_passes(session, [_record(person_name="Алексей", score=0.95)])
        session.commit()
    with factory() as session:
        rows = list(session.query(GatePass))

    assert first == (1, 0)
    assert second == (0, 1)
    assert len(rows) == 1
    assert rows[0].person_name == "Алексей"
    assert rows[0].score == pytest.approx(0.95)


def test_an_unchanged_pass_is_not_touched_again(runtime_path: Path) -> None:
    """Иначе каждый опрос считался бы работой и журнал службы стал бы бесполезен."""
    factory = _session(runtime_path)

    with factory() as session:
        record_passes(session, [_record(person_name="Алексей")])
        session.commit()
    with factory() as session:
        again = record_passes(session, [_record(person_name="Алексей")])
        session.commit()

    assert again == (0, 0)


def test_the_snapshot_is_fetched_once(runtime_path: Path) -> None:
    """Забор снимка стоит до пятнадцати секунд — повторять его нечего."""
    factory = _session(runtime_path)
    calls: list[str] = []

    def snapshot(record: GatePassRecord) -> str | None:
        calls.append(record.frigate_event_id)
        return "runtime/gate-passes/a.jpg"

    with factory() as session:
        record_passes(session, [_record(frigate_event_id="a")], snapshot=snapshot)
        session.commit()
    with factory() as session:
        record_passes(
            session,
            [_record(frigate_event_id="a", person_name="Алексей")],
            snapshot=snapshot,
        )
        session.commit()
        stored = session.query(GatePass).one()

    assert calls == ["a"]
    assert stored.snapshot_path == "runtime/gate-passes/a.jpg"


def test_mechanics_columns_stay_empty(runtime_path: Path) -> None:
    """Механики в этой службе нет, и журнал не должен утверждать обратное.

    «Открыл» без импульса — самая дорогая ложь, какую может рассказать журнал
    калитки: по нему потом разбирают, кто входил.
    """
    factory = _session(runtime_path)

    with factory() as session:
        record_passes(session, [_record()])
        session.commit()
        stored = session.query(GatePass).one()

    assert stored.pulse_ms is None
    assert stored.modbus_ok is None
    assert stored.reed_before is None
    assert stored.reed_after is None


def test_old_passes_leave_with_their_snapshots(runtime_path: Path) -> None:
    """Иначе каталог снимков растёт вечно: у калитки людно."""
    factory = _session(runtime_path)
    snapshot = runtime_path / "runtime" / "gate-passes" / "old.jpg"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(b"jpeg")

    with factory() as session:
        record_passes(
            session,
            [
                _record(
                    frigate_event_id="old",
                    created_at=datetime.now(UTC) - timedelta(days=40),
                ),
                _record(
                    frigate_event_id="fresh",
                    created_at=datetime.now(UTC) - timedelta(hours=1),
                ),
            ],
            snapshot=lambda record: (
                "runtime/gate-passes/old.jpg" if record.frigate_event_id == "old" else None
            ),
        )
        session.commit()
    with factory() as session:
        removed = purge_old_passes(
            session,
            data_dir=runtime_path,
            older_than=datetime.now(UTC) - timedelta(days=30),
        )
        session.commit()
    with factory() as session:
        left = [row.frigate_event_id for row in session.query(GatePass)]

    assert removed == 1
    assert left == ["fresh"]
    assert not snapshot.exists()


def test_a_snapshot_name_cannot_escape_its_directory(runtime_path: Path) -> None:
    """Идентификатор приходит из чужого API и попадает в имя файла."""
    path = pass_snapshot_path(runtime_path, "../../etc/passwd")

    assert path.resolve().parent == (runtime_path / "runtime" / "gate-passes").resolve()
    assert "/" not in path.name


def test_the_dry_mode_decides_and_writes_it_down() -> None:
    """Ради этого режим и заведён: настоящая статистика на реальных людях."""
    records, opening = _passes(
        [_event(sub_label="Алексей", name_score=0.96)],
        config=_config(),
    )

    assert records[0].decision == DECISION_DRY_RUN
    assert opening is not None
    assert opening.frigate_event_id == records[0].frigate_event_id


def test_a_late_name_changes_the_decision(runtime_path: Path) -> None:
    """Имя приходит через секунды после события, и решение обязано его догнать.

    Строка, застрявшая на «лицо не узнано», врала бы в самой интересной части
    журнала — там, где разбирают, кого система не пустила.
    """
    factory = _session(runtime_path)
    first, _ = _passes([_event(sub_label=None)], config=_config())
    second, _ = _passes([_event(sub_label="Алексей", name_score=0.96)], config=_config())

    with factory() as session:
        record_passes(session, first)
        session.commit()
    with factory() as session:
        record_passes(session, second)
        session.commit()
        stored = session.query(GatePass).one()

    assert first[0].decision != DECISION_DRY_RUN
    assert stored.decision == DECISION_DRY_RUN
    assert stored.person_name == "Алексей"


def test_the_pause_does_not_fire_on_the_events_own_decision() -> None:
    """Событие перечитывается каждый опрос всё окно наблюдения.

    Без этого собственное «открыл бы» через секунду становилось бы для того же
    человека паузой — «прошло 1 с из 20», — и журнал наполнялся бы отказами на
    ровном месте.
    """
    event = _event(sub_label="Алексей", name_score=0.96)

    first, opening = _passes([event], config=_config())
    second, _ = _passes([event], config=_config(), last_opening=opening)

    assert first[0].decision == DECISION_DRY_RUN
    assert second[0].decision == DECISION_DRY_RUN


def test_the_pause_fires_on_the_next_person() -> None:
    started = _event().started_at
    first = _event(id="a", sub_label="Алексей", name_score=0.96, started_at=started)
    second = _event(id="b", sub_label="Алексей", name_score=0.96, started_at=started + 5)

    records, _ = _passes([first, second], config=_config(cooldown_seconds=20))

    assert records[0].decision == DECISION_DRY_RUN
    assert records[1].decision == "denied_cooldown"


def test_the_pause_survives_a_restart(runtime_path: Path) -> None:
    """Обновление админки среди дня не должно обнулять паузу."""
    factory = _session(runtime_path)
    records, _ = _passes(
        [_event(sub_label="Алексей", name_score=0.96)],
        config=_config(),
    )

    with factory() as session:
        record_passes(session, records)
        session.commit()
    with factory() as session:
        remembered = last_opening(session)

    assert isinstance(remembered, LastOpening)
    assert remembered.frigate_event_id == records[0].frigate_event_id


def test_the_phone_factor_reaches_the_journal() -> None:
    """Фактор пишется всегда, даже когда на решение не влияет.

    Иначе включать его пришлось бы вслепую: главный риск этой схемы — ложные
    отказы, и увидеть их можно только по накопленным строкам.
    """
    records, _ = _passes(
        [_event(sub_label="Алексей", name_score=0.96)],
        config=_config(),
        factors_for=lambda name: [Factor(PHONE_FACTOR, "against", "телефона нет в сети")],
    )
    factors = json.loads(records[0].factors_json or "[]")

    assert records[0].decision == DECISION_DRY_RUN
    assert {factor["name"]: factor["verdict"] for factor in factors} == {
        "face": "for",
        PHONE_FACTOR: "against",
    }


def test_only_one_module_can_write_to_the_board() -> None:
    """Сторожевой тест, который не удаляется, а сужается.

    Разрешение владельца от 2026-08-11 позволяет ровно одно действие — импульс
    на реле калитки, — и вся запись на плату собрана в одном модуле. Если
    функции записи Modbus заведутся где-то ещё, это уже не то, что разрешали:
    ни автоматика решения, ни монитор дверцы писать не должны.
    """
    silent = [
        Path("src/gate/access.py"),
        Path("src/gate/decision.py"),
        Path("src/gate/presence.py"),
        Path("src/web/door_monitor.py"),
        Path("src/web/app.py"),
    ]
    # Modbus пишет функциями 0x05, 0x06 и 0x0F. Ни одной из них не должно быть
    # там, где записи не место, — как и самой сборки кадра.
    forbidden = ("flash_frame", "write_coil", "write_register", "0x05", "0x06", "0x0F")

    for path in silent:
        text = path.read_text(encoding="utf-8")
        assert not any(marker in text for marker in forbidden), f"{path} умеет писать на плату"
    assert "flash_frame" in Path("src/gate/opener.py").read_text(encoding="utf-8")


def test_the_automatic_path_never_touches_the_board() -> None:
    """Решение по лицу заканчивается записью «открыл бы» и ничем больше.

    Автооткрытие остаётся в `dry` до четырёх условий выхода, и разрешение
    владельца его не включает. Проверяется не только решение, но и то, что
    открывалку в этот путь невозможно передать: пока её нет в подписи, никакая
    правка тела функции случайно до платы не дотянется.
    """
    records, _ = _passes([_event(sub_label="Алексей", name_score=0.96)], config=_config())

    assert records[0].decision == DECISION_DRY_RUN
    assert "opener" not in inspect.signature(passes_from_events).parameters


def test_the_button_opens_and_leaves_a_trace(runtime_path: Path) -> None:
    """Условие разрешения владельца: журнал каждого срабатывания и уведомление."""
    factory = _session(runtime_path)
    sent: list[str] = []

    with factory() as session:
        result = handle_open_request(
            session,
            request=OpenRequest(id="a", requested_at=1000.0),
            config=_config(mode="off"),
            opener=_FakeOpener(PulseResult(True, "Импульс 500 мс на канал 1", 500)),
            now=1000.5,
            timezone=MOSCOW,
            notify=sent.append,
        )
        session.commit()
        stored = session.query(GatePass).one()

    assert result.ok
    assert stored.decision == "opened"
    assert stored.pulse_ms == 500
    assert stored.modbus_ok is True
    assert "вручную" in (stored.reason or "")
    assert sent and "Калитка" in sent[0]


def test_the_button_works_with_automation_off(runtime_path: Path) -> None:
    """`off` и `dry` — про автоматику. Человек, нажавший кнопку, решил сам."""
    factory = _session(runtime_path)

    with factory() as session:
        result = handle_open_request(
            session,
            request=OpenRequest(id="a", requested_at=1000.0),
            config=_config(mode="off"),
            opener=_FakeOpener(PulseResult(True, "ок", 500)),
            now=1000.1,
            timezone=MOSCOW,
        )
        session.commit()

    assert result.ok


def test_a_failed_pulse_is_not_written_as_an_opening(runtime_path: Path) -> None:
    """«Открыл» без импульса — самая дорогая ложь, какую может рассказать журнал."""
    factory = _session(runtime_path)

    with factory() as session:
        result = handle_open_request(
            session,
            request=OpenRequest(id="a", requested_at=1000.0),
            config=_config(mode="off"),
            opener=_FakeOpener(PulseResult(False, "Плата не ответила", 500)),
            now=1000.1,
            timezone=MOSCOW,
        )
        session.commit()
        stored = session.query(GatePass).one()

    assert result.ok is False
    assert stored.decision == "error"
    assert stored.modbus_ok is False


def test_a_stale_request_never_opens_the_gate(runtime_path: Path) -> None:
    """Нажатие, до которого служба не добралась, сработало бы, когда человек ушёл.

    Это же защищает от худшего сценария перезапуска: файл, оставшийся на диске,
    не должен открыть калитку при следующем старте службы.
    """
    factory = _session(runtime_path)
    opener = _FakeOpener(PulseResult(True, "ок", 500))

    with factory() as session:
        result = handle_open_request(
            session,
            request=OpenRequest(id="a", requested_at=1000.0),
            config=_config(mode="off"),
            opener=opener,
            now=1000.0 + REQUEST_TTL_SECONDS + 1,
            timezone=MOSCOW,
        )
        session.commit()
        rows = list(session.query(GatePass))

    assert result.ok is False
    assert "устарела" in result.detail
    assert opener.calls == []
    assert rows == []


def test_an_unconfigured_relay_says_so_instead_of_guessing(runtime_path: Path) -> None:
    factory = _session(runtime_path)
    opener = _FakeOpener(PulseResult(True, "ок", 500))

    with factory() as session:
        result = handle_open_request(
            session,
            request=OpenRequest(id="a", requested_at=1000.0),
            config=GateAccessConfig(),
            opener=opener,
            now=1000.1,
            timezone=MOSCOW,
        )

    assert result.ok is False
    assert "не настроен" in result.detail
    assert opener.calls == []


def test_the_request_is_executed_once(runtime_path: Path) -> None:
    """Уникальный ключ журнала — он же защита от двойного исполнения."""
    factory = _session(runtime_path)
    request = OpenRequest(id="a", requested_at=1000.0)

    with factory() as session:
        handle_open_request(
            session,
            request=request,
            config=_config(mode="off"),
            opener=_FakeOpener(PulseResult(True, "ок", 500)),
            now=1000.1,
            timezone=MOSCOW,
        )
        session.commit()
    with factory() as session, pytest.raises(IntegrityError):
        handle_open_request(
            session,
            request=request,
            config=_config(mode="off"),
            opener=_FakeOpener(PulseResult(True, "ок", 500)),
            now=1000.2,
            timezone=MOSCOW,
        )
        session.commit()


def test_the_on_mode_turns_a_decision_into_a_pulse() -> None:
    """Разница между `dry` и `on` — только в исполнении, решение то же."""
    opener = _FakeOpener(PulseResult(True, "Импульс 500 мс на канал 1", 500))
    records, _ = _passes(
        [_event(sub_label="Алексей", name_score=0.96)],
        config=_config(mode="on"),
    )
    sent: list[str] = []

    executed, _ = execute_openings(
        records,
        config=_config(mode="on"),
        opener=opener,
        already={},
        notify=sent.append,
    )

    assert executed[0].decision == "opened"
    assert executed[0].pulse_ms == 500
    assert executed[0].modbus_ok is True
    assert opener.calls == [(2, 500)]
    assert sent and "Алексей" in sent[0]


def test_the_dry_mode_still_touches_nothing() -> None:
    opener = _FakeOpener(PulseResult(True, "ок", 500))
    records, _ = _passes([_event(sub_label="Алексей", name_score=0.96)], config=_config())

    executed, _ = execute_openings(records, config=_config(), opener=opener, already={})

    assert executed[0].decision == DECISION_DRY_RUN
    assert opener.calls == []


def test_an_event_is_opened_once_however_often_it_is_reread() -> None:
    """Событие перечитывается каждый опрос всё окно наблюдения.

    Без проверки по журналу один человек у калитки получал бы импульс каждую
    секунду в течение пяти минут.
    """
    opener = _FakeOpener(PulseResult(True, "ок", 500))
    records, _ = _passes(
        [_event(sub_label="Алексей", name_score=0.96)],
        config=_config(mode="on"),
    )

    execute_openings(records, config=_config(mode="on"), opener=opener, already={})
    execute_openings(
        records,
        config=_config(mode="on"),
        opener=opener,
        already={records[0].frigate_event_id: PriorPass("opened", 1, 1000.0)},
    )

    assert opener.calls == [(2, 500)]


def test_a_failed_opening_is_not_retried_every_second() -> None:
    """Отказ платы не повод долбить её без остановки: у человека есть кнопка."""
    opener = _FakeOpener(PulseResult(False, "Плата не ответила", 500))
    records, _ = _passes(
        [_event(sub_label="Алексей", name_score=0.96)],
        config=_config(mode="on"),
    )

    executed, _ = execute_openings(records, config=_config(mode="on"), opener=opener, already={})
    execute_openings(
        records,
        config=_config(mode="on"),
        opener=opener,
        already={records[0].frigate_event_id: PriorPass("error")},
    )

    assert executed[0].decision == "error"
    assert executed[0].modbus_ok is False
    assert opener.calls == [(2, 500)]


def test_a_refused_decision_never_reaches_the_board() -> None:
    """Всё, кроме «открыл бы», до платы не доходит даже в режиме `on`."""
    opener = _FakeOpener(PulseResult(True, "ок", 500))
    records, _ = _passes([_event(sub_label=None)], config=_config(mode="on"))

    executed, _ = execute_openings(records, config=_config(mode="on"), opener=opener, already={})

    assert executed[0].decision != "opened"
    assert opener.calls == []


def test_what_went_to_the_board_reaches_the_journal(runtime_path: Path) -> None:
    factory = _session(runtime_path)
    records, _ = _passes(
        [_event(sub_label="Алексей", name_score=0.96)],
        config=_config(mode="on"),
    )
    executed, _ = execute_openings(
        records,
        config=_config(mode="on"),
        opener=_FakeOpener(PulseResult(True, "ок", 500)),
        already={},
    )

    with factory() as session:
        record_passes(session, executed)
        session.commit()
        stored = session.query(GatePass).one()

    assert stored.decision == "opened"
    assert stored.pulse_ms == 500
    assert stored.modbus_ok is True


def test_a_reread_event_does_not_reopen_the_gate(runtime_path: Path) -> None:
    """Самый дорогой дефект этой работы, пойманный живым прогоном 2026-08-11.

    Событие перечитывается каждый опрос всё окно наблюдения. Исполнение
    пропускалось по журналу правильно, но строка при этом несла прежнее «открыл
    бы» — и следующая запись переписывала «открыл» обратно. Ещё через опрос
    служба видела незавершённое событие и открывала снова: калитка щёлкала
    каждые две секунды пять минут подряд.

    Отсюда правило: за «открыл» и «отказ» стоит уже случившееся действие, и
    переоценка того же события их не трогает.
    """
    factory = _session(runtime_path)
    opener = _FakeOpener(PulseResult(True, "Импульс 500 мс на канал 2", 500))
    config = _config(mode="on")

    for _ in range(4):
        records, _ = _passes([_event(sub_label="Алексей", name_score=0.96)], config=config)
        with factory() as session:
            already = {
                row.frigate_event_id: PriorPass(row.decision, row.pulses or 0, row.last_pulse_at)
                for row in session.query(GatePass)
            }
            record_passes(
                session,
                execute_openings(records, config=config, opener=opener, already=already)[0],
            )
            session.commit()

    with factory() as session:
        stored = session.query(GatePass).one()

    assert opener.calls == [(2, 500)]
    assert stored.decision == "opened"
    assert stored.pulse_ms == 500
    assert stored.modbus_ok is True


def test_a_failed_opening_is_not_downgraded_either(runtime_path: Path) -> None:
    """«Отказ» — тоже факт: импульс пробовали и он не ушёл."""
    factory = _session(runtime_path)
    opener = _FakeOpener(PulseResult(False, "Плата не ответила", 500))
    config = _config(mode="on")

    for _ in range(3):
        records, _ = _passes([_event(sub_label="Алексей", name_score=0.96)], config=config)
        with factory() as session:
            already = {
                row.frigate_event_id: PriorPass(row.decision, row.pulses or 0, row.last_pulse_at)
                for row in session.query(GatePass)
            }
            record_passes(
                session,
                execute_openings(records, config=config, opener=opener, already=already)[0],
            )
            session.commit()

    with factory() as session:
        stored = session.query(GatePass).one()

    assert opener.calls == [(2, 500)]
    assert stored.decision == "error"


def test_a_request_from_a_previous_boot_never_opens_the_gate(
    runtime_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Обрыв питания не должен открывать калитку при следующей загрузке.

    Возраст по часам защищает не всегда: после обрыва RTC отстаёт, chrony
    подтягивает время не сразу, и просьба из прошлой жизни выглядит свежей.
    Проверено 2026-08-11 — в этом случае калитка открывалась сама, ночью, без
    человека. Идентификатор загрузки от часов не зависит вовсе.
    """
    factory = _session(runtime_path)
    opener = _FakeOpener(PulseResult(True, "ок", 500))
    monkeypatch.setattr("src.gate.access.boot_id", lambda: "загрузка-нынешняя")

    with factory() as session:
        result = handle_open_request(
            session,
            request=OpenRequest(
                id="a", requested_at=1000.0, boot_id="загрузка-прошлая"
            ),
            config=_config(mode="off"),
            opener=opener,
            now=1000.5,
            timezone=MOSCOW,
        )
        rows = list(session.query(GatePass))

    assert result.ok is False
    assert "прошлой загрузки" in result.detail
    assert opener.calls == []
    assert rows == []


def test_a_clock_that_stepped_backwards_is_not_freshness(runtime_path: Path) -> None:
    """Отрицательный возраст — это не «свежая просьба», а неизвестно что.

    Человек нажмёт ещё раз, и это дешевле, чем открыть калитку неизвестно
    когда.
    """
    factory = _session(runtime_path)
    opener = _FakeOpener(PulseResult(True, "ок", 500))

    with factory() as session:
        result = handle_open_request(
            session,
            request=OpenRequest(id="a", requested_at=1000.0),
            config=_config(mode="off"),
            opener=opener,
            now=1000.0 - 3600,
            timezone=MOSCOW,
        )

    assert result.ok is False
    assert opener.calls == []


def test_the_same_boot_is_not_an_obstacle(
    runtime_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Обычное нажатие проверку загрузки не замечает."""
    factory = _session(runtime_path)
    opener = _FakeOpener(PulseResult(True, "ок", 500))
    monkeypatch.setattr("src.gate.access.boot_id", lambda: "загрузка-нынешняя")

    with factory() as session:
        result = handle_open_request(
            session,
            request=OpenRequest(
                id="a", requested_at=1000.0, boot_id="загрузка-нынешняя"
            ),
            config=_config(mode="off"),
            opener=opener,
            now=1000.4,
            timezone=MOSCOW,
        )
        session.commit()

    assert result.ok
    assert opener.calls == [(2, 500)]


def test_the_first_box_says_where_the_person_came_from() -> None:
    """Рамка, упирающаяся в нижний край, — человек уже вплотную.

    Замер на живых проходах 2026-08-12: у выхода, разворота у калитки и
    открытия кодом низ первой рамки был ровно 1.00, у настоящих приходов —
    0.23–0.91. Одиннадцать событий разделились без ошибок.
    """
    from src.gate.notify import approach_from_box

    assert approach_from_box(0.23) == "far"
    assert approach_from_box(0.91) == "far"
    assert approach_from_box(1.0) == "near"
    assert approach_from_box(None) == "unknown"


def test_the_circumstances_reach_the_journal(runtime_path: Path) -> None:
    """Наблюдение без вмешательства: обстоятельства пишутся, решения не трогают."""
    factory = _session(runtime_path)
    records, _ = _passes(
        [_event(sub_label="Алексей", name_score=0.96)],
        config=_config(),
        approach_for=lambda event_id: ("far", 0.31),
        gate_state="closed",
    )

    with factory() as session:
        record_passes(session, records)
        session.commit()
        stored = session.query(GatePass).one()

    assert stored.approach == "far"
    assert stored.first_box_bottom == pytest.approx(0.31)
    assert stored.gate_state_before == "closed"
    # Решение осталось прежним: обстоятельства пока только наблюдаются.
    assert stored.decision == DECISION_DRY_RUN


def test_the_gate_state_after_is_refined_later(runtime_path: Path) -> None:
    """Створка часто шевелится уже после того, как строка появилась."""
    factory = _session(runtime_path)
    first, _ = _passes([_event(sub_label="Алексей", name_score=0.96)], config=_config())

    with factory() as session:
        record_passes(session, [replace(first[0], gate_state_after="closed")])
        session.commit()
    with factory() as session:
        record_passes(session, [replace(first[0], gate_state_after="open")])
        session.commit()
        stored = session.query(GatePass).one()

    assert stored.gate_state_after == "open"


def test_an_unknown_approach_is_asked_again(runtime_path: Path) -> None:
    """Frigate заводит запись таймлайна с задержкой, а мы спрашиваем сразу.

    На живом узле 2026-08-12 из-за запомненной неудачи все проходы получили
    «откуда неизвестно» навсегда, хотя через секунду ответ уже был. Событие
    живёт в окне наблюдения пять минут — значит следующий опрос обязан спросить
    снова.
    """
    from src.common.settings import Settings
    from src.gate.access import _approach_for

    answers = [None, 0.31]
    calls: list[str] = []

    def fake_fetch(api_url: str, event_id: str, **kwargs: object) -> float | None:
        calls.append(event_id)
        return answers[len(calls) - 1]

    from src.gate import access

    original = access.fetch_first_box
    access.fetch_first_box = fake_fetch  # type: ignore[assignment]
    try:
        approach = _approach_for(Settings(data_dir=runtime_path), {})
        first = approach("a")
        second = approach("a")
        third = approach("a")
    finally:
        access.fetch_first_box = original  # type: ignore[assignment]

    assert first == ("unknown", None)
    assert second == ("far", pytest.approx(0.31))
    # Удачный ответ запоминается: он про первую рамку и меняться не может.
    assert third == ("far", pytest.approx(0.31))
    assert calls == ["a", "a"]


def test_the_cat_door_is_opened_on_its_own_relay(runtime_path: Path) -> None:
    """Одна служба пишет на плату — просьба адресная, исполнитель общий.

    Два процесса полезли бы в один TCP-порт шлюза, а он в прозрачном режиме
    второго соединения может и не принять.
    """
    from src.common.cat_door_config import CatDoorConfig
    from src.common.gate_open_request import TARGET_CAT_DOOR

    factory = _session(runtime_path)
    opener = _FakeOpener(PulseResult(True, "Открыто 5000 мс на канал 2", 5000))

    with factory() as session:
        result = handle_open_request(
            session,
            request=OpenRequest(id="a", requested_at=1000.0, target=TARGET_CAT_DOOR),
            config=_config(mode="off"),
            opener=opener,
            now=1000.1,
            timezone=MOSCOW,
            cat_door=CatDoorConfig(relay=2, hold_ms=5000),
        )
        session.commit()
        stored = session.query(GatePass).one()

    assert result.ok
    assert opener.calls == [(2, 5000)]
    assert stored.label == "cat_door"
    assert "Дверца" in (stored.reason or "")


def test_without_settings_the_door_is_not_guessed_at(runtime_path: Path) -> None:
    """Нет настроек — нет и канала: угадывать, куда подать питание, нельзя."""
    from src.common.gate_open_request import TARGET_CAT_DOOR

    factory = _session(runtime_path)
    opener = _FakeOpener(PulseResult(True, "ок", 5000))

    with factory() as session:
        result = handle_open_request(
            session,
            request=OpenRequest(id="a", requested_at=1000.0, target=TARGET_CAT_DOOR),
            config=_config(mode="off"),
            opener=opener,
            now=1000.1,
            timezone=MOSCOW,
        )

    assert result.ok is False
    assert opener.calls == []


def test_a_slow_telegram_does_not_delay_the_answer(runtime_path: Path) -> None:
    """Тот, кто стоит перед дверью, не должен ждать чата.

    2026-08-12: защёлка щёлкнула мгновенно, а служба пятнадцать секунд ждала
    недоступный Телеграм — и человек увидел «служба не ответила», хотя дверца
    уже открылась. Порядок теперь такой: сработало → записали → ответили → и
    только потом сообщаем.
    """
    from src.gate.access import _notify_all

    factory = _session(runtime_path)
    collected: list[str] = []

    with factory() as session:
        result = handle_open_request(
            session,
            request=OpenRequest(id="a", requested_at=1000.0),
            config=_config(mode="off"),
            opener=_FakeOpener(PulseResult(True, "ок", 500)),
            now=1000.1,
            timezone=MOSCOW,
            notify=collected.append,
        )
        session.commit()

    # Ответ готов, а сообщение ещё даже не отправлялось.
    assert result.ok
    assert collected == [
        "Калитка: открыта по кнопке в админке · " + collected[0].split("· ")[1]
    ]

    def unreachable(_message: str) -> None:
        raise OSError("Телеграм недоступен")

    # И отказ отправки ничего не отменяет.
    _notify_all(unreachable, collected)


def test_a_pulse_never_chases_someone_who_has_left() -> None:
    """Решение дозревает после ухода — калитке от этого открываться незачем.

    12 августа так и вышло: человек подошёл в 16:41:14, получил отказ по паузе,
    а когда пауза истекла, решение пересчиталось в «открыл» — и калитка
    щёлкнула через пятьдесят секунд в пустоту.
    """
    opener = _FakeOpener(PulseResult(True, "ок", 500))
    records, _ = _passes(
        [_event(sub_label="Алексей", name_score=0.96, ended_at=1000.0)],
        config=_config(mode="on"),
    )

    executed, _ = execute_openings(
        records,
        config=_config(mode="on"),
        opener=opener,
        already={},
        now=1000.0 + ACT_WITHIN_SECONDS + 1,
    )

    assert opener.calls == []
    assert executed[0].decision == "denied_stale"
    assert "ушёл из кадра" in (executed[0].reason or "")


@pytest.mark.parametrize(
    ("ended_at", "now"),
    [
        pytest.param(None, 1000.0, id="ещё в кадре"),
        pytest.param(1000.0, 1000.0 + ACT_WITHIN_SECONDS - 1, id="только что вышел"),
    ],
)
def test_a_pulse_still_goes_while_the_person_is_there(ended_at: float | None, now: float) -> None:
    """Пока человек у калитки, опоздание решения не повод его не пускать."""
    opener = _FakeOpener(PulseResult(True, "ок", 500))
    records, _ = _passes(
        [_event(sub_label="Алексей", name_score=0.96, ended_at=ended_at)],
        config=_config(mode="on"),
    )

    executed, _ = execute_openings(
        records, config=_config(mode="on"), opener=opener, already={}, now=now
    )

    assert executed[0].decision == "opened"
    assert opener.calls == [(2, 500)]


def _prior(**overrides: object) -> PriorPass:
    values: dict[str, object] = {
        "decision": "opened",
        "pulses": 1,
        "last_pulse_at": 1000.0,
        # Импульс, попавший в уже открытую створку, — тот единственный случай,
        # ради которого второй шанс и заведён.
        "pulse_gate_state": "open",
    }
    values.update(overrides)
    return PriorPass(**values)  # type: ignore[arg-type]


def _chance(**overrides: object) -> GateAccessConfig:
    rules: dict[str, object] = {"enabled": True}
    rules.update(overrides)
    return _config(mode="on", second_chance=rules)


def test_the_gate_opens_again_when_it_closed_and_the_person_is_still_there() -> None:
    """Ради этого случая ветка и заведена.

    12 августа событие Лёши шло 55 секунд: импульс ушёл на третьей, когда
    створка была ещё открыта от прошедшего перед ним человека, и пропал зря.
    Через две секунды створка закрылась, а Лёша остался перед ней — и второго
    импульса ему не полагалось.
    """
    opener = _FakeOpener(PulseResult(True, "Импульс 500 мс на канал 2", 500))
    records, _ = _passes([_event(sub_label="Алексей", name_score=0.96)], config=_chance())

    executed, _ = execute_openings(
        records,
        config=_chance(),
        opener=opener,
        already={records[0].frigate_event_id: _prior()},
        gate_closed_at=1005.0,
        now=1010.0,
    )

    assert opener.calls == [(2, 500)]
    assert executed[0].decision == "opened"
    assert executed[0].pulses == 2
    assert executed[0].last_pulse_at == 1010.0
    assert "ещё раз" in (executed[0].reason or "")


@pytest.mark.parametrize(
    ("prior", "config", "closed_at", "now", "why"),
    [
        pytest.param(
            _prior(), _config(mode="on"), 1005.0, 1010.0, "выключено", id="ветка выключена"
        ),
        pytest.param(
            _prior(pulses=2), _chance(), 1005.0, 1010.0, "потолок", id="импульсы исчерпаны"
        ),
        pytest.param(
            _prior(), _chance(), 1005.0, 1003.0, "рано", id="пауза между импульсами"
        ),
        pytest.param(_prior(), _chance(), None, 1010.0, "створка", id="створка не закрывалась"),
        pytest.param(
            _prior(), _chance(), 999.0, 1010.0, "до импульса", id="закрылась до импульса"
        ),
        pytest.param(
            _prior(last_pulse_at=None), _chance(), 1005.0, 1010.0, "когда", id="строка без времени"
        ),
        pytest.param(
            _prior(pulse_gate_state="closed"),
            _chance(),
            1005.0,
            1010.0,
            "импульс сработал",
            id="импульс попал в закрытую створку",
        ),
        pytest.param(
            _prior(pulse_gate_state="unknown"),
            _chance(),
            1005.0,
            1010.0,
            "неизвестно",
            id="состояние створки неизвестно",
        ),
    ],
)
def test_the_second_chance_has_its_limits(
    prior: PriorPass,
    config: GateAccessConfig,
    closed_at: float | None,
    now: float,
    why: str,
) -> None:
    """Повтор без закрывшейся створки — это то самое качание, а не второй шанс."""
    opener = _FakeOpener(PulseResult(True, "ок", 500))
    records, _ = _passes([_event(sub_label="Алексей", name_score=0.96)], config=config)

    executed, _ = execute_openings(
        records,
        config=config,
        opener=opener,
        already={records[0].frigate_event_id: prior},
        gate_closed_at=closed_at,
        now=now,
    )

    assert opener.calls == [], why
    assert executed[0].decision == "opened"


def test_the_repeat_is_counted_in_the_journal(runtime_path: Path) -> None:
    """Решение при повторе не меняется — оно и там и там «открыл».

    Значит, ветка «решение изменилось» второй импульс не запишет, и счётчику
    нужна своя. Без него потолок импульсов терялся бы при каждом перезапуске.
    """
    factory = _session(runtime_path)
    opened = _record(decision="opened", person_name="Алексей", pulses=1, last_pulse_at=1000.0)

    with factory() as session:
        record_passes(session, [opened])
        session.commit()
    with factory() as session:
        record_passes(
            session,
            [replace(opened, pulses=2, last_pulse_at=1010.0, reason="открыто ещё раз")],
        )
        session.commit()
    with factory() as session:
        row = session.query(GatePass).one()

    assert row.pulses == 2
    assert row.last_pulse_at == pytest.approx(1010.0)
    assert row.reason == "открыто ещё раз"


def test_the_gate_does_not_beep_after_someone_who_got_in() -> None:
    """12 августа калитка пикнула вслед прошедшему, и это была не мелочь.

        17:46:31  импульс — створка была закрыта
        17:46:36  створка открыта   ← человек прошёл
        17:46:51  створка закрыта
        17:46:51  импульс второй    ← лишний

    Отличие удачи от неудачи — состояние створки в момент импульса. Закрыта:
    защёлку отпустили для человека, он открыл и прошёл. Открыта: защёлка и так
    была отпущена, наш импульс ничего не изменил.
    """
    opener = _FakeOpener(PulseResult(True, "ок", 500))
    records, _ = _passes([_event(sub_label="Алексей", name_score=0.96)], config=_chance())

    executed, _ = execute_openings(
        records,
        config=_chance(),
        opener=opener,
        already={records[0].frigate_event_id: _prior(pulse_gate_state="closed")},
        gate_closed_at=1005.0,
        now=1010.0,
    )

    assert opener.calls == []
    assert executed[0].decision == "opened"


def test_the_state_of_the_gate_at_the_pulse_is_remembered() -> None:
    """Без этого второй шанс не отличит пропавший импульс от сработавшего."""
    opener = _FakeOpener(PulseResult(True, "ок", 500))
    records, _ = _passes([_event(sub_label="Алексей", name_score=0.96)], config=_chance())

    executed, _ = execute_openings(
        records,
        config=_chance(),
        opener=opener,
        already={},
        gate_state="open",
        now=1000.0,
    )

    assert executed[0].pulse_gate_state == "open"


def test_a_clock_behind_the_event_never_opens_the_gate() -> None:
    """После обрыва питания часы узла какое-то время идут позади.

    Событие «из будущего» выглядит вечно свежим, и старый проход получил бы
    импульс на первом же опросе после включения. Незнание времени — не повод
    трогать плату.
    """
    opener = _FakeOpener(PulseResult(True, "ок", 500))
    records, _ = _passes(
        [_event(sub_label="Алексей", name_score=0.96, ended_at=2000.0)],
        config=_config(mode="on"),
    )

    executed, _ = execute_openings(
        records, config=_config(mode="on"), opener=opener, already={}, now=1000.0
    )

    assert opener.calls == []
    assert executed[0].decision == "denied_stale"
    assert "часы узла позади" in (executed[0].reason or "")


def test_the_reason_catches_up_with_the_name(runtime_path: Path) -> None:
    """13 августа журнал выдал «Нина … Лицо не узнано» — одна строка о разном.

    Первый опрос видит человека без имени и отказывает с «Лицо не узнано».
    Через секунду приезжает имя, и отказ становится «Нина не в списке». Код
    решения в обоих случаях один — `denied_not_allowed`, — поэтому причина
    обновляться обязана отдельно от него.
    """
    factory = _session(runtime_path)
    nameless = _record(decision="denied_not_allowed", reason="Лицо не узнано")

    with factory() as session:
        record_passes(session, [nameless])
        session.commit()
    with factory() as session:
        refined = record_passes(
            session,
            [
                replace(
                    nameless,
                    person_name="Нина",
                    score=0.93,
                    reason="Нина не в списке разрешённых",
                )
            ],
        )
        session.commit()
    with factory() as session:
        row = session.query(GatePass).one()

    assert refined == (0, 1)
    assert row.person_name == "Нина"
    assert row.reason == "Нина не в списке разрешённых"


def test_a_row_with_a_pulse_behind_it_keeps_its_reason(runtime_path: Path) -> None:
    """За «открыл» стоит ушедший импульс, и переоценка его не переписывает."""
    factory = _session(runtime_path)
    opened = _record(
        decision="opened", person_name="Алексей", reason="Алексей: открыто по лицу. Импульс"
    )

    with factory() as session:
        record_passes(session, [opened])
        session.commit()
    with factory() as session:
        record_passes(
            session,
            [replace(opened, decision=DECISION_DRY_RUN, reason="открыл бы")],
        )
        session.commit()
    with factory() as session:
        row = session.query(GatePass).one()

    assert row.decision == "opened"
    assert "открыто по лицу" in (row.reason or "")


def test_a_reread_does_not_replace_the_pulse_report_with_a_guess(
    runtime_path: Path,
) -> None:
    """Регрессия фикса #218, поймана 13 августа в 20:12 — через три часа.

    Пересчёт перечитанного события несёт carried-«opened» с пересчитанной
    dry-причиной «открыл бы», и ветка обновления причин затирала ею отчёт
    «Импульс 500 мс на канал 1». Причина терминальной строки меняется только
    вместе с новым импульсом.
    """
    factory = _session(runtime_path)
    opened = _record(
        decision="opened",
        person_name="Леша",
        reason="Леша: открыто по лицу. Импульс 500 мс на канал 1",
        pulse_ms=500,
        modbus_ok=True,
        pulses=1,
        last_pulse_at=1000.0,
    )

    with factory() as session:
        record_passes(session, [opened])
        session.commit()
    with factory() as session:
        record_passes(
            session,
            [
                replace(
                    opened,
                    reason="Леша — открыл бы",
                    pulse_ms=None,
                    modbus_ok=None,
                )
            ],
        )
        session.commit()
    with factory() as session:
        row = session.query(GatePass).one()

    assert "Импульс 500 мс" in (row.reason or "")


# ── Правки 13 августа: настоящее «сейчас», пауза от импульса, повтор ──

# Дневная база: окно 07:00–23:00 в решении настоящее, и полночь 1970 года
# честно получала бы denied_window вместо проверяемого кода.
NOON = datetime(2026, 8, 13, 12, 0, tzinfo=MOSCOW).timestamp()


def test_the_pause_ends_in_the_real_world_not_in_the_events_world() -> None:
    """Проход 17:14:11 простоял в denied_cooldown до смерти в denied_stale.

    Решению передавалось начало события, и оно замерзало: «прошло 5 с из 10 с»
    навсегда. Теперь решение получает настоящее «сейчас», и пауза кончается.
    """
    config = _config(mode="on", cooldown_seconds=10)
    event = _event(id="b", started_at=NOON + 5, sub_label="Алексей", name_score=0.96)
    opened_at = LastOpening("a", datetime.fromtimestamp(NOON, UTC))

    early, _ = _passes(
        [event], config=config, last_opening=opened_at,
        now=datetime.fromtimestamp(NOON + 6, UTC),
    )
    later, _ = _passes(
        [event], config=config, last_opening=opened_at,
        now=datetime.fromtimestamp(NOON + 11, UTC),
    )

    assert early[0].decision == "denied_cooldown"
    # То же событие, тот же журнал — но мир успел прожить паузу.
    assert later[0].decision == DECISION_DRY_RUN


def test_a_later_opening_does_not_lock_an_earlier_event_forever() -> None:
    """Отрицательная пауза: событие Б началось раньше открытия для В.

    Раньше waited = начало_Б − открытие_В < 0 запирало Б намертво. С настоящим
    «сейчас» возраст паузы не бывает отрицательным просто потому, что «сейчас»
    позже любого прошлого открытия.
    """
    config = _config(mode="on", cooldown_seconds=10)
    event = _event(id="b", started_at=NOON, sub_label="Алексей", name_score=0.96)
    later_opening = LastOpening("c", datetime.fromtimestamp(NOON + 20, UTC))

    records, _ = _passes(
        [event], config=config, last_opening=later_opening,
        now=datetime.fromtimestamp(NOON + 31, UTC),
    )

    assert records[0].decision == DECISION_DRY_RUN


def test_a_stale_decision_does_not_arm_the_pause() -> None:
    """Пауза считается от щелчка реле, а не от намерения.

    Решение, умершее в denied_stale, раньше взводило паузу, и следующий человек
    ждал несуществующего открытия.
    """
    config = _config(mode="on")
    opener = _FakeOpener(PulseResult(True, "ок", 500))
    records, opening = _passes(
        [_event(sub_label="Алексей", name_score=0.96, ended_at=NOON)],
        config=config,
        now=datetime.fromtimestamp(NOON + ACT_WITHIN_SECONDS + 1, UTC),
    )

    executed, pulsed = execute_openings(
        records, config=config, opener=opener, already={},
        now=NOON + ACT_WITHIN_SECONDS + 1,
    )

    assert executed[0].decision == "denied_stale"
    # В режиме «on» намерение паузу не взводит…
    assert opening is None
    # …а импульса не было, значит и щелчка нет.
    assert pulsed is None


def test_a_successful_pulse_arms_the_pause() -> None:
    config = _config(mode="on")
    opener = _FakeOpener(PulseResult(True, "ок", 500))
    records, _ = _passes(
        [_event(sub_label="Алексей", name_score=0.96)],
        config=config,
        now=datetime.fromtimestamp(NOON, UTC),
    )

    _, pulsed = execute_openings(
        records, config=config, opener=opener, already={}, now=NOON
    )

    assert pulsed is not None
    assert pulsed.at == datetime.fromtimestamp(NOON, UTC)


def test_the_dry_mode_still_arms_the_pause_by_decision() -> None:
    """Наблюдение обязано воспроизводить поведение «on», а импульсов в нём нет."""
    records, opening = _passes(
        [_event(sub_label="Алексей", name_score=0.96)],
        config=_config(mode="dry"),
        now=datetime.fromtimestamp(NOON, UTC),
    )

    assert records[0].decision == DECISION_DRY_RUN
    assert opening is not None


def test_a_board_error_earns_a_retry_after_a_pause() -> None:
    """Отказ платы — не приговор: 17:04 он стоил Лёше четырнадцати секунд.

    Пауза перед повтором защищает плату от долбёжки так же, как защищал
    приговор, но человеку достаётся вторая попытка.
    """
    from src.gate.access import ERROR_RETRY_SECONDS

    config = _config(mode="on")
    opener = _FakeOpener(PulseResult(True, "ок", 500))
    records, _ = _passes(
        [_event(sub_label="Алексей", name_score=0.96)],
        config=config,
        now=datetime.fromtimestamp(NOON, UTC),
    )
    prior = PriorPass("error", 0, NOON)

    too_soon, _ = execute_openings(
        records, config=config, opener=opener,
        already={records[0].frigate_event_id: prior}, now=NOON + 5,
    )
    assert too_soon[0].decision == "error"
    assert opener.calls == []  # рано — плату не трогали

    after_pause, _ = execute_openings(
        records, config=config, opener=opener,
        already={records[0].frigate_event_id: prior},
        now=NOON + ERROR_RETRY_SECONDS + 1,
    )
    assert after_pause[0].decision == "opened"
    assert opener.calls == [(2, 500)]


def test_ongoing_events_are_fetched_without_an_age_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Фильтр `after` режет по началу события.

    Человек, простоявший у калитки дольше окна наблюдения, выпадал из опроса,
    хотя его событие живо: постоял шесть минут, решил войти — а служба его уже
    не видит.
    """
    from src.gate.notify import fetch_ongoing_events

    seen: list[str] = []

    def fake_urlopen(url, timeout=0):
        seen.append(url if isinstance(url, str) else url.full_url)
        return _Response(b"[]")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    fetch_ongoing_events("http://127.0.0.1:5000")

    assert "in_progress=1" in seen[0]
    assert "after=" not in seen[0]


def test_failed_attempts_throttle_through_the_journal(runtime_path: Path) -> None:
    """Находка адверсарного ревью: троттлинг повторов работал ровно один раз.

    Время неудачной попытки не доезжало до базы, каждый следующий опрос видел
    «пауза выдержана» — и плата долбилась раз в секунду, а Телеграм получал
    сообщение на каждый опрос. Здесь цикл прокручивается честно: execute →
    record_passes → PriorPass из базы → следующий опрос.
    """
    from src.gate.access import ERROR_RETRY_SECONDS

    factory = _session(runtime_path)
    config = _config(mode="on")
    opener = _FakeOpener(PulseResult(False, "Плата не ответила", 500))
    attempts = 0

    for tick in range(int(ERROR_RETRY_SECONDS * 3)):
        now = NOON + tick
        records, _ = _passes(
            [_event(sub_label="Алексей", name_score=0.96)],
            config=config,
            now=datetime.fromtimestamp(now, UTC),
        )
        with factory() as session:
            already = {
                row.frigate_event_id: PriorPass(
                    row.decision, row.pulses or 0, row.last_pulse_at, row.pulse_gate_state
                )
                for row in session.query(GatePass)
            }
            before = len(opener.calls)
            executed, _ = execute_openings(
                records, config=config, opener=opener, already=already, now=now
            )
            attempts += len(opener.calls) - before
            record_passes(session, executed)
            session.commit()

    # 30 секунд у неотвечающей платы: первая попытка и по одной на каждые
    # ERROR_RETRY_SECONDS — а не тридцать.
    assert attempts == 1 + int(30 / ERROR_RETRY_SECONDS - 1)


def test_a_failed_second_chance_does_not_demote_opened(runtime_path: Path) -> None:
    """Неудачный второй шанс не разжалует «opened» и не открывает обход лимитов.

    Находка ревью: запись «error» поверх успешного открытия и врала журналу, и
    пускала следующий импульс через секунду мимо gap_seconds и max_pulses.
    """
    factory = _session(runtime_path)
    config = _chance()
    opener = _FakeOpener(PulseResult(False, "Плата не ответила", 500))
    records, _ = _passes(
        [_event(sub_label="Алексей", name_score=0.96)],
        config=config,
        now=datetime.fromtimestamp(NOON + 20, UTC),
    )
    prior = _prior(last_pulse_at=NOON)

    executed, _ = execute_openings(
        records,
        config=config,
        opener=opener,
        already={records[0].frigate_event_id: prior},
        gate_closed_at=NOON + 5,
        now=NOON + 20,
    )
    with factory() as session:
        record_passes(
            session,
            [
                replace(
                    executed[0],
                    person_name="Алексей",
                )
            ],
        )
        session.commit()
    with factory() as session:
        row = session.query(GatePass).one()

    assert executed[0].decision == "opened"  # исход не разжалован
    assert row.decision == "opened"
    # Время попытки записано: следующая ждёт нового закрытия створки.
    assert row.last_pulse_at == pytest.approx(NOON + 20)


def test_one_poll_batch_gets_one_pulse(runtime_path: Path) -> None:
    """Два события в одной пачке — один импульс, второму пауза.

    Находка ревью: пауза внутри опроса исчезла вместе с переездом отметки на
    импульс, и реле щёлкало дважды подряд.
    """
    config = _config(mode="on", cooldown_seconds=10)
    opener = _FakeOpener(PulseResult(True, "ок", 500))
    records, _ = _passes(
        [
            _event(id="a", sub_label="Алексей", name_score=0.96),
            _event(id="b", started_at=NOON + 1, sub_label="Алексей", name_score=0.97),
        ],
        config=config,
        now=datetime.fromtimestamp(NOON + 2, UTC),
    )

    executed, _ = execute_openings(
        records, config=config, opener=opener, already={}, now=NOON + 2
    )

    assert [record.decision for record in executed] == ["opened", "denied_cooldown"]
    assert len(opener.calls) == 1
