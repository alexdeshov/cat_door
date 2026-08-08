from __future__ import annotations

import io
import json
import urllib.error
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

import pytest

from src.common.notifications import NotificationGate, pass_caption
from src.common.telegram import TelegramClient, TelegramError

TOKEN = "8999303876:secret-part-of-the-token"


def _client(**overrides: object) -> TelegramClient:
    return TelegramClient(token=TOKEN, chat_id="12345", **overrides)  # type: ignore[arg-type]


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        return None


def test_message_goes_to_the_configured_chat(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["body"] = request.data
        return _Response({"ok": True, "result": {}})

    monkeypatch.setattr("src.common.telegram.urllib.request.urlopen", fake_urlopen)

    _client().send_message("Проход в 14:32")

    assert seen["url"] == f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    assert b"chat_id=12345" in seen["body"]  # type: ignore[operator]


def test_photo_is_sent_as_multipart(runtime_path: Path, monkeypatch) -> None:
    photo = runtime_path / "pass.jpg"
    photo.write_bytes(b"\xff\xd8jpeg-body\xff\xd9")
    seen: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["type"] = request.headers["Content-type"]
        seen["body"] = request.data
        return _Response({"ok": True, "result": {}})

    monkeypatch.setattr("src.common.telegram.urllib.request.urlopen", fake_urlopen)

    _client().send_photo(photo, "Черныш: проход в 14:32")

    assert str(seen["url"]).endswith("/sendPhoto")
    assert str(seen["type"]).startswith("multipart/form-data; boundary=")
    body = seen["body"]
    assert isinstance(body, bytes)
    assert b'name="photo"; filename="pass.jpg"' in body
    assert b"\xff\xd8jpeg-body\xff\xd9" in body
    assert "Черныш: проход в 14:32".encode() in body


def test_token_never_leaks_into_the_error(monkeypatch) -> None:
    """Токен лежит в адресе запроса, а адрес urllib кладёт в текст ошибки.

    Эта строка уезжает в журнал systemd, который читают люди и который никто
    не считает секретом.
    """

    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            401,
            "Unauthorized",
            {},  # type: ignore[arg-type]
            io.BytesIO(b'{"ok":false,"description":"Unauthorized"}'),
        )

    monkeypatch.setattr("src.common.telegram.urllib.request.urlopen", fake_urlopen)

    with pytest.raises(TelegramError) as failure:
        _client().send_message("проверка")

    assert TOKEN not in str(failure.value)
    assert "Unauthorized" in str(failure.value)


def test_unreachable_telegram_is_an_error_not_a_crash(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        raise urllib.error.URLError("Network is unreachable")

    monkeypatch.setattr("src.common.telegram.urllib.request.urlopen", fake_urlopen)

    with pytest.raises(TelegramError, match="недоступен"):
        _client().send_message("проверка")


def test_unconfigured_client_refuses_before_touching_the_network() -> None:
    with pytest.raises(TelegramError, match="не настроен"):
        TelegramClient(token="", chat_id="").send_message("проверка")


def test_gate_lets_the_door_through_and_silences_the_approach() -> None:
    """33 из 38 ложных срабатываний пришли с подхода — там пока молчим."""
    gate = NotificationGate(roles=frozenset({"face"}), cooldown_seconds=300)

    assert gate.blocked_reason("face", now=100.0) is None
    assert gate.blocked_reason("side", now=100.0) == "camera_not_notified:side"
    assert gate.blocked_reason(None, now=100.0) == "unknown_camera"


def test_gate_keeps_quiet_after_a_recent_message() -> None:
    gate = NotificationGate(roles=frozenset({"face"}), cooldown_seconds=300)
    gate.remember_sent(now=100.0)

    assert gate.blocked_reason("face", now=200.0) == "cooldown"
    assert gate.blocked_reason("face", now=401.0) is None


def test_gate_without_roles_is_the_off_switch() -> None:
    gate = NotificationGate(roles=frozenset(), cooldown_seconds=0)

    assert gate.blocked_reason("face", now=1.0) == "notifications_disabled"


def test_caption_reads_the_stored_time_as_utc() -> None:
    """sqlite отдаёт метку без зоны; принять её за местную — соврать на три часа."""
    caption = pass_caption(
        when=datetime(2026, 8, 8, 11, 32),  # noqa: DTZ001 — так метку отдаёт sqlite
        timezone="Europe/Moscow",
        role="face",
        confidence=0.98,
        identity=None,
    )

    assert caption == "Проход в 14:32 · дверца · уверенность 0.98"


def test_caption_puts_the_name_first_when_the_cat_is_recognised() -> None:
    caption = pass_caption(
        when=datetime(2026, 8, 8, 11, 32, tzinfo=UTC),
        timezone="Europe/Moscow",
        role="side",
        confidence=None,
        identity="Черныш",
    )

    assert caption == "Черныш: проход в 14:32 · подход"


def test_recorder_sends_a_photo_after_publishing_a_pass(runtime_path: Path, monkeypatch) -> None:
    """Уведомление идёт следом за записью, а не вместо неё."""
    from src.common.settings import Settings
    from src.recorder import app as recorder_app
    from src.recorder.detection_journal import DetectionJournal
    from src.recorder.events import AutomaticEventRequest

    recorder_app.reset_notification_gate()
    settings = Settings.model_construct(
        data_dir=runtime_path,
        database_path=runtime_path / "catdoor.sqlite3",
        ffmpeg_path="ffmpeg",
        ring_segment_seconds=0,
        display_timezone="Europe/Moscow",
        telegram_bot_token=TOKEN,
        telegram_chat_id="12345",
        notify_roles=["face"],
        notify_cooldown_seconds=300.0,
    )
    event = type("StoredEvent", (), {"id": 7, "created_at": datetime(2026, 8, 8, 11, 32),  # noqa: DTZ001
                                     "media_path": "events/00000007/face.mp4",
                                     "side_media_path": None})()
    sent: list[tuple[str, str]] = []
    def fake_extract(ffmpeg_path, video, destination, **kwargs):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"\xff\xd8jpeg\xff\xd9")
        return True

    monkeypatch.setattr("src.recorder.app.extract_frame", fake_extract)
    monkeypatch.setattr(
        "src.common.telegram.TelegramClient.send_photo",
        lambda self, photo, caption: sent.append(("photo", caption)),
    )
    journal = DetectionJournal(
        runtime_path / "runtime" / "detection.jsonl",
        max_bytes=100_000,
        backup_count=1,
    )

    request = AutomaticEventRequest(
        started_at=1.0,
        ended_at=9.0,
        initiating_camera="face",
        confidence_face=0.98,
        confidence_side=0.0,
    )
    recorder_app._notify_about_pass(settings, request, event, journal)  # type: ignore[arg-type]
    # Второй проход подряд молчит: пауза общая на процесс.
    recorder_app._notify_about_pass(settings, request, event, journal)  # type: ignore[arg-type]
    journal.close()

    assert sent == [("photo", "Проход в 14:32 · дверца · уверенность 0.98")]
    written = (runtime_path / "runtime" / "detection.jsonl").read_text(encoding="utf-8")
    assert "notification_sent" in written
    assert '"reason":"cooldown"' in written
    assert not list((runtime_path / "runtime").glob("notification-*.jpg"))


def test_recorder_survives_a_silent_telegram(runtime_path: Path, monkeypatch) -> None:
    """Проход уже записан, и падать из-за мессенджера ему незачем."""
    from src.common.settings import Settings
    from src.recorder import app as recorder_app
    from src.recorder.detection_journal import DetectionJournal
    from src.recorder.events import AutomaticEventRequest

    recorder_app.reset_notification_gate()
    settings = Settings.model_construct(
        data_dir=runtime_path,
        database_path=runtime_path / "catdoor.sqlite3",
        ffmpeg_path="ffmpeg",
        display_timezone="Europe/Moscow",
        telegram_bot_token=TOKEN,
        telegram_chat_id="12345",
        notify_roles=["face"],
        notify_cooldown_seconds=300.0,
    )
    event = type("StoredEvent", (), {"id": 8, "created_at": datetime(2026, 8, 8, 11, 32),  # noqa: DTZ001
                                     "media_path": None, "side_media_path": None})()

    def refuse(self, text):
        raise TelegramError("Телеграм недоступен: сеть")

    monkeypatch.setattr("src.common.telegram.TelegramClient.send_message", refuse)
    journal = DetectionJournal(
        runtime_path / "runtime" / "detection.jsonl",
        max_bytes=100_000,
        backup_count=1,
    )

    recorder_app._notify_about_pass(  # type: ignore[arg-type]
        settings,
        AutomaticEventRequest(
            started_at=1.0,
            ended_at=9.0,
            initiating_camera="face",
            confidence_face=0.98,
            confidence_side=0.0,
        ),
        event,
        journal,
    )
    journal.close()

    written = (runtime_path / "runtime" / "detection.jsonl").read_text(encoding="utf-8")
    assert "notification_failed" in written


def test_confusing_telegram_refusals_get_a_human_hint() -> None:
    """«Бот не может писать боту» формально точно и совершенно непонятно."""
    from src.common.telegram import hint_for

    assert "не идентификатор бота" in (
        hint_for("Телеграм отказал (403): Forbidden: the bot can't send messages to the bot") or ""
    )
    assert "нажали «Start»" in (hint_for("Bad Request: chat not found") or "")
    assert hint_for("Телеграм отказал: что-то новое") is None
