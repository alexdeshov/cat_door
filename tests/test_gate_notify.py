from __future__ import annotations

import json
from pathlib import Path
from typing import Self

from src.common.notifications import NotificationGate
from src.common.settings import Settings
from src.common.telegram import TelegramClient, TelegramError
from src.gate.notify import (
    GATE_ROLE,
    GateEvent,
    fetch_events,
    gate_caption,
    load_last_seen,
    notify_about_events,
    save_last_seen,
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
    return Settings.model_construct(
        display_timezone="Europe/Moscow",
        gate_api_url="http://127.0.0.1:5000",
        gate_notify_cameras=[],
        gate_notify_labels=["person"],
        gate_notify_cooldown_seconds=120.0,
        **overrides,  # type: ignore[arg-type]
    )


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
    )

    assert delivered == 1
    assert len(sent) == 1
    assert "человек" in sent[0]


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
    )

    assert delivered == 0


def test_first_start_does_not_dump_the_archive(runtime_path: Path) -> None:
    """Файла отметки нет — значит служба поднялась впервые."""
    path = state_path(runtime_path)

    assert load_last_seen(path) is None

    save_last_seen(path, 1786171119.5)

    assert load_last_seen(path) == 1786171119.5


def test_caption_shows_the_local_time_and_the_recognised_name() -> None:
    caption = gate_caption(_event(sub_label="Алекс", started_at=1786171119.0), "Europe/Moscow")

    assert caption.startswith("Калитка: Алекс · gate · ")
