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
    GatePassRecord,
    LastOpening,
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
        "person_name": None,
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


def test_a_car_is_not_a_pass() -> None:
    """Журнал про людей: машину лицом не открывают, а строка о ней — шум."""
    records, _ = _passes([_event(id="a", label="car"), _event(id="b", label="person")])

    assert [record.frigate_event_id for record in records] == ["b"]


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

    executed = execute_openings(
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

    executed = execute_openings(records, config=_config(), opener=opener, already={})

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
        already={records[0].frigate_event_id: "opened"},
    )

    assert opener.calls == [(2, 500)]


def test_a_failed_opening_is_not_retried_every_second() -> None:
    """Отказ платы не повод долбить её без остановки: у человека есть кнопка."""
    opener = _FakeOpener(PulseResult(False, "Плата не ответила", 500))
    records, _ = _passes(
        [_event(sub_label="Алексей", name_score=0.96)],
        config=_config(mode="on"),
    )

    executed = execute_openings(records, config=_config(mode="on"), opener=opener, already={})
    execute_openings(
        records,
        config=_config(mode="on"),
        opener=opener,
        already={records[0].frigate_event_id: "error"},
    )

    assert executed[0].decision == "error"
    assert executed[0].modbus_ok is False
    assert opener.calls == [(2, 500)]


def test_a_refused_decision_never_reaches_the_board() -> None:
    """Всё, кроме «открыл бы», до платы не доходит даже в режиме `on`."""
    opener = _FakeOpener(PulseResult(True, "ок", 500))
    records, _ = _passes([_event(sub_label=None)], config=_config(mode="on"))

    executed = execute_openings(records, config=_config(mode="on"), opener=opener, already={})

    assert executed[0].decision != "opened"
    assert opener.calls == []


def test_what_went_to_the_board_reaches_the_journal(runtime_path: Path) -> None:
    factory = _session(runtime_path)
    records, _ = _passes(
        [_event(sub_label="Алексей", name_score=0.96)],
        config=_config(mode="on"),
    )
    executed = execute_openings(
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
            already = {row.frigate_event_id: row.decision for row in session.query(GatePass)}
            record_passes(
                session,
                execute_openings(records, config=config, opener=opener, already=already),
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
            already = {row.frigate_event_id: row.decision for row in session.query(GatePass)}
            record_passes(
                session,
                execute_openings(records, config=config, opener=opener, already=already),
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
