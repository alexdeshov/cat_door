from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Self

import pytest

from src.common.database import GatePass, create_session_factory
from src.gate.access import (
    DECISION_MODE_OFF,
    GatePassRecord,
    pass_snapshot_path,
    passes_from_events,
    purge_old_passes,
    record_passes,
)
from src.gate.notify import GateEvent, fetch_events


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
    records = passes_from_events(
        [_event(id="a", camera="gate"), _event(id="b", camera="face")],
        cameras={"gate": "Калитка"},
    )

    assert [record.frigate_event_id for record in records] == ["a"]


def test_a_car_is_not_a_pass() -> None:
    """Журнал про людей: машину лицом не открывают, а строка о ней — шум."""
    records = passes_from_events(
        [_event(id="a", label="car"), _event(id="b", label="person")],
        cameras={"gate": "Калитка"},
    )

    assert [record.frigate_event_id for record in records] == ["b"]


def test_a_stranger_is_written_down_too() -> None:
    """«В 02:40 у калитки был кто-то неузнанный» — самая ценная строка журнала."""
    records = passes_from_events([_event(sub_label=None)], cameras={"gate": "Калитка"})

    assert records[0].person_name is None
    assert records[0].decision == DECISION_MODE_OFF


def test_the_journal_records_who_and_how_certainly() -> None:
    records = passes_from_events(
        [_event(sub_label="Алексей", name_score=0.94, zones=("двор", "у калитки"))],
        cameras={"gate": "Калитка"},
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
