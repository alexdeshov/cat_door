from __future__ import annotations

import json
from pathlib import Path
from typing import Self

from src.common.camera_registry import camera_registry_path
from src.common.notifications import NotificationGate
from src.common.settings import Settings
from src.common.telegram import TelegramClient, TelegramError
from src.gate.notify import (
    GATE_ROLE,
    GateEvent,
    fetch_events,
    forget_old,
    gate_caption,
    load_state,
    notify_about_events,
    save_state,
    state_path,
)


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


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "display_timezone": "Europe/Moscow",
        "gate_api_url": "http://127.0.0.1:5000",
        "gate_notify_cameras": [],
        "gate_notify_labels": ["person"],
        "gate_notify_faces_only": False,
        "gate_notify_cooldown_seconds": 120.0,
    }
    values.update(overrides)
    return Settings.model_construct(**values)  # type: ignore[arg-type]


def test_events_arrive_oldest_first(monkeypatch) -> None:
    """Frigate отдаёт свежие первыми, а рассказывать надо по порядку."""
    payload = json.dumps(
        [
            {"id": "b", "camera": "gate", "label": "person", "start_time": 200.0},
            {"id": "a", "camera": "gate", "label": "person", "start_time": 100.0},
        ]
    ).encode()
    seen: dict[str, str] = {}

    def fake_urlopen(url, timeout):
        seen["url"] = url
        return _Response(payload)

    monkeypatch.setattr("src.gate.notify.urllib.request.urlopen", fake_urlopen)

    events = fetch_events("http://127.0.0.1:5000/", after=50.0)

    assert [event.id for event in events] == ["a", "b"]
    assert "after=50.000" in seen["url"]


def test_unreachable_frigate_is_a_reason_to_wait_not_to_die(monkeypatch) -> None:
    import urllib.error

    def fake_urlopen(url, timeout):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr("src.gate.notify.urllib.request.urlopen", fake_urlopen)

    try:
        fetch_events("http://127.0.0.1:5000", after=0.0)
    except RuntimeError as exc:
        assert "не ответил" in str(exc)
    else:  # pragma: no cover - защита от молчаливого прохода
        raise AssertionError("ожидалась RuntimeError")


def test_only_interesting_labels_reach_the_phone(runtime_path: Path, monkeypatch) -> None:
    sent: list[str] = []
    monkeypatch.setattr("src.gate.notify.fetch_snapshot", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        "src.common.telegram.TelegramClient.send_message",
        lambda self, text: sent.append(text),
    )
    gate = NotificationGate(roles=frozenset({GATE_ROLE}), cooldown_seconds=0.0)

    delivered = notify_about_events(
        _settings(),
        TelegramClient("8999303876:test", "12345"),
        gate,
        [_event(label="person"), _event(id="second", label="car")],
        data_dir=runtime_path,
        now=10.0,
        wall_now=_event().started_at + 60,
        cameras={"gate": "Калитка", "gate_2": "Калитка 2"},
    )

    assert delivered == 1
    assert len(sent) == 1
    # Человек без имени — «неизвестный»: сообщение приходит именно потому,
    # что узнать его не удалось. Машина при этом молчит по фильтру меток.
    assert "неизвестный" in sent[0]


def test_one_guest_gives_one_message(runtime_path: Path, monkeypatch) -> None:
    """Человек у калитки виден обеими камерами сразу — это один гость."""
    sent: list[str] = []
    monkeypatch.setattr("src.gate.notify.fetch_snapshot", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        "src.common.telegram.TelegramClient.send_message",
        lambda self, text: sent.append(text),
    )
    gate = NotificationGate(roles=frozenset({GATE_ROLE}), cooldown_seconds=120.0)

    notify_about_events(
        _settings(),
        TelegramClient("8999303876:test", "12345"),
        gate,
        [_event(camera="gate"), _event(id="second", camera="gate_2")],
        data_dir=runtime_path,
        now=10.0,
        wall_now=_event().started_at + 60,
        cameras={"gate": "Калитка", "gate_2": "Калитка 2"},
    )

    assert len(sent) == 1


def test_snapshot_goes_with_the_message_and_is_cleaned_up(
    runtime_path: Path,
    monkeypatch,
) -> None:
    photos: list[str] = []

    def fake_snapshot(api_url, event, destination, **kwargs):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"\xff\xd8jpeg\xff\xd9")
        return True

    monkeypatch.setattr("src.gate.notify.fetch_snapshot", fake_snapshot)
    monkeypatch.setattr(
        "src.common.telegram.TelegramClient.send_photo",
        lambda self, photo, caption: photos.append(caption),
    )
    gate = NotificationGate(roles=frozenset({GATE_ROLE}), cooldown_seconds=0.0)

    notify_about_events(
        _settings(),
        TelegramClient("8999303876:test", "12345"),
        gate,
        [_event(sub_label="Алекс")],
        data_dir=runtime_path,
        now=10.0,
        wall_now=_event().started_at + 60,
        cameras={"gate": "Калитка", "gate_2": "Калитка 2"},
    )

    assert photos and "Алекс" in photos[0]
    assert not list((runtime_path / "runtime").glob("gate-notify-*.jpg"))


def test_a_silent_telegram_does_not_stop_the_batch(runtime_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("src.gate.notify.fetch_snapshot", lambda *args, **kwargs: False)

    def refuse(self, text):
        raise TelegramError("Телеграм недоступен")

    monkeypatch.setattr("src.common.telegram.TelegramClient.send_message", refuse)
    gate = NotificationGate(roles=frozenset({GATE_ROLE}), cooldown_seconds=0.0)

    delivered = notify_about_events(
        _settings(),
        TelegramClient("8999303876:test", "12345"),
        gate,
        [_event()],
        data_dir=runtime_path,
        now=10.0,
        wall_now=_event().started_at + 60,
        cameras={"gate": "Калитка", "gate_2": "Калитка 2"},
    )

    assert delivered == 0


def test_sent_events_are_remembered_across_restarts(runtime_path: Path) -> None:
    """Окно наблюдения перечитывает те же события каждый опрос."""
    path = state_path(runtime_path)

    assert load_state(path) == {}

    save_state(path, {"a": 1786171119.5})

    assert load_state(path) == {"a": 1786171119.5}


def test_old_events_leave_the_memory(runtime_path: Path) -> None:
    kept = forget_old({"старое": 100.0, "свежее": 500.0}, older_than=400.0)

    assert kept == {"свежее": 500.0}


def test_caption_shows_the_local_time_and_the_recognised_name() -> None:
    caption = gate_caption(_event(sub_label="Алекс", started_at=1786171119.0), "Europe/Moscow")

    assert caption.startswith("Калитка: Алекс · ")


def test_only_recognised_faces_reach_the_phone(runtime_path: Path, monkeypatch) -> None:
    """Владелец просил сообщать о людях с лицом из базы, а не о всяком движении."""
    sent: list[str] = []
    monkeypatch.setattr("src.gate.notify.fetch_snapshot", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        "src.common.telegram.TelegramClient.send_message",
        lambda self, text: sent.append(text),
    )
    gate = NotificationGate(roles=frozenset({GATE_ROLE}), cooldown_seconds=0.0)

    notify_about_events(
        _settings(gate_notify_faces_only=True),
        TelegramClient("8999303876:test", "12345"),
        gate,
        [_event(id="без имени"), _event(id="с именем", sub_label="Алекс")],
        data_dir=runtime_path,
        now=10.0,
        wall_now=_event().started_at + 60,
        cameras={"gate": "Калитка", "gate_2": "Калитка 2"},
    )

    assert len(sent) == 1
    assert "Алекс" in sent[0]


def test_a_nameless_event_is_not_thrown_away_forever(runtime_path: Path, monkeypatch) -> None:
    """Frigate приписывает имя не сразу — событие ждёт следующего опроса."""
    sent: list[str] = []
    monkeypatch.setattr("src.gate.notify.fetch_snapshot", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        "src.common.telegram.TelegramClient.send_message",
        lambda self, text: sent.append(text),
    )
    gate = NotificationGate(roles=frozenset({GATE_ROLE}), cooldown_seconds=0.0)
    settings = _settings(gate_notify_faces_only=True)
    client = TelegramClient("8999303876:test", "12345")
    already: dict[str, float] = {}

    # первый опрос: имени ещё нет
    notify_about_events(
        settings, client, gate, [_event(id="гость")],
        data_dir=runtime_path, now=10.0, wall_now=_event().started_at + 60, already_sent=already,
        cameras={"gate": "Калитка"},
    )
    # второй опрос: то же событие, но Frigate успел его опознать
    notify_about_events(
        settings, client, gate, [_event(id="гость", sub_label="Алекс")],
        data_dir=runtime_path, now=20.0, wall_now=_event().started_at + 60, already_sent=already,
        cameras={"gate": "Калитка"},
    )

    assert len(sent) == 1
    assert "Алекс" in sent[0]
    assert "гость" in already


def test_the_same_event_is_never_sent_twice(runtime_path: Path, monkeypatch) -> None:
    sent: list[str] = []
    monkeypatch.setattr("src.gate.notify.fetch_snapshot", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        "src.common.telegram.TelegramClient.send_message",
        lambda self, text: sent.append(text),
    )
    gate = NotificationGate(roles=frozenset({GATE_ROLE}), cooldown_seconds=0.0)
    settings = _settings()
    client = TelegramClient("8999303876:test", "12345")
    already: dict[str, float] = {}

    for _ in range(3):
        notify_about_events(
            settings, client, gate, [_event()],
            data_dir=runtime_path, now=10.0, wall_now=_event().started_at + 60, already_sent=already,
        cameras={"gate": "Калитка"},
        )

    assert len(sent) == 1


def test_a_stranger_gets_a_message_of_his_own(runtime_path: Path, monkeypatch) -> None:
    """О своих — по имени, о чужих — «неизвестный», и паузы у них разные."""
    sent: list[str] = []
    monkeypatch.setattr("src.gate.notify.fetch_snapshot", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        "src.common.telegram.TelegramClient.send_message",
        lambda self, text: sent.append(text),
    )
    known = NotificationGate(roles=frozenset({GATE_ROLE}), cooldown_seconds=0.0)
    unknown = NotificationGate(roles=frozenset({GATE_ROLE}), cooldown_seconds=0.0)

    notify_about_events(
        _settings(gate_notify_face_grace_seconds=20.0),
        TelegramClient("8999303876:test", "12345"),
        known,
        [_event(id="свой", sub_label="Алекс"), _event(id="чужой")],
        data_dir=runtime_path,
        now=10.0,
        wall_now=_event().started_at + 60,
        unknown_gate=unknown,
        cameras={"gate": "Калитка"},
    )

    assert len(sent) == 2
    assert "Алекс" in sent[0]
    assert "неизвестный" in sent[1]


def test_a_stranger_waits_out_the_grace_period(runtime_path: Path, monkeypatch) -> None:
    """Иначе один гость дал бы два сообщения: сперва «неизвестный», потом имя."""
    sent: list[str] = []
    monkeypatch.setattr("src.gate.notify.fetch_snapshot", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        "src.common.telegram.TelegramClient.send_message",
        lambda self, text: sent.append(text),
    )
    gate = NotificationGate(roles=frozenset({GATE_ROLE}), cooldown_seconds=0.0)
    settings = _settings(gate_notify_face_grace_seconds=20.0)
    client = TelegramClient("8999303876:test", "12345")
    already: dict[str, float] = {}
    guest = _event(id="гость")

    # через пять секунд имени ещё нет — молчим
    notify_about_events(
        settings, client, gate, [guest],
        data_dir=runtime_path, now=10.0,
        wall_now=guest.started_at + 5, already_sent=already, cameras={"gate": "Калитка"},
    )
    assert sent == []

    # через десять Frigate его узнал — уходит одно сообщение, с именем
    notify_about_events(
        settings, client, gate, [_event(id="гость", sub_label="Алекс")],
        data_dir=runtime_path, now=20.0,
        wall_now=guest.started_at + 10, already_sent=already, cameras={"gate": "Калитка"},
    )

    assert len(sent) == 1
    assert "Алекс" in sent[0]


def test_frigate_word_for_a_stranger_is_not_a_name() -> None:
    """Разные версии Frigate пишут «unknown» вместо пустого поля."""
    caption = gate_caption(_event(sub_label="unknown"), "Europe/Moscow")

    assert "неизвестный" in caption


def test_only_the_gate_circuit_reaches_the_phone(runtime_path: Path, monkeypatch) -> None:
    """Frigate ведёт общую ленту на оба контура узла.

    Без отбора в чат шли уведомления и от камер регистратора — да ещё и с
    подписью «Калитка», потому что источник считался очевидным.
    """
    sent: list[str] = []
    monkeypatch.setattr("src.gate.notify.fetch_snapshot", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        "src.common.telegram.TelegramClient.send_message",
        lambda self, text: sent.append(text),
    )
    gate = NotificationGate(roles=frozenset({GATE_ROLE}), cooldown_seconds=0.0)

    notify_about_events(
        _settings(),
        TelegramClient("8999303876:test", "12345"),
        gate,
        [
            _event(id="калитка", camera="test_cam", sub_label="Алекс"),
            _event(id="регистратор", camera="nvr_2", sub_label="Алекс"),
        ],
        data_dir=runtime_path,
        now=10.0,
        wall_now=_event().started_at + 60,
        cameras={"test_cam": "Калитка"},
    )

    assert len(sent) == 1
    assert sent[0].startswith("Калитка: Алекс · ")


def test_gate_cameras_come_from_the_inventory(runtime_path: Path) -> None:
    """Камеру могли перевести в другой контур минуту назад."""
    from src.common.camera_registry import CameraRecord, CameraRegistry, save_registry
    from src.gate.notify import gate_cameras

    save_registry(
        camera_registry_path(runtime_path),
        CameraRegistry(
            cameras=[
                CameraRecord(id="test-cam", title="Калитка", circuit="gate", host="10.0.0.1"),
                CameraRecord(id="nvr-2", title="Двор", circuit="nvr", host="10.0.0.2"),
                CameraRecord(id="off", title="Снята", circuit="gate", host="10.0.0.3", enabled=False),
            ]
        ),
    )

    assert gate_cameras(runtime_path) == {"test_cam": "Калитка"}
