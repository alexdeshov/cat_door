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


def test_every_listed_recipient_gets_the_message(monkeypatch) -> None:
    """Семья — не группа: chat id через запятую, каждому лично."""
    bodies: list[bytes] = []

    def fake_urlopen(request, timeout):
        bodies.append(request.data)
        return _Response({"ok": True, "result": {}})

    monkeypatch.setattr("src.common.telegram.urllib.request.urlopen", fake_urlopen)

    TelegramClient(token=TOKEN, chat_id="12345, 462634240").send_message("Проход в 14:32")

    assert [b"chat_id=12345" in body for body in bodies] == [True, False]
    assert b"chat_id=462634240" in bodies[1]


def test_one_blocked_recipient_does_not_fail_the_delivery(monkeypatch) -> None:
    """Заблокировавший бота не должен ронять доставку остальным —
    иначе сторож повторял бы «узел перезагружался» до бесконечности."""

    def fake_urlopen(request, timeout):
        if b"chat_id=666" in request.data:
            raise urllib.error.HTTPError(
                request.full_url, 403, "Forbidden", {}, io.BytesIO(b"bot was blocked")
            )
        return _Response({"ok": True, "result": {}})

    monkeypatch.setattr("src.common.telegram.urllib.request.urlopen", fake_urlopen)

    TelegramClient(token=TOKEN, chat_id="666,12345").send_message("проверка")  # не падает

    with pytest.raises(TelegramError, match="403"):
        TelegramClient(token=TOKEN, chat_id="666").send_message("проверка")


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
        detector_confidence=0.98,
        identity=None,
    )

    # Число у безымянного прохода — детекторное, и подписано тем, чем оно и
    # является: «в кадре кот». Раньше оно называлось «уверенность» и читалось
    # как уверенность опознания — разбор прохода #688.
    assert caption == "Проход в 14:32 · дверца · кот 0.98"


def test_caption_puts_the_name_first_when_the_cat_is_recognised() -> None:
    caption = pass_caption(
        when=datetime(2026, 8, 8, 11, 32, tzinfo=UTC),
        timezone="Europe/Moscow",
        role="side",
        detector_confidence=None,
        identity="Черныш",
        identity_confidence=0.93,
        votes=56,
        frames_used=154,
    )

    assert caption == "Черныш: проход в 14:32 · подход · опознание 0.93 · 56 из 154 кадров"


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

    assert sent == [("photo", "Проход в 14:32 · дверца · кот 0.98")]
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


def test_notification_uses_the_frame_the_detector_was_surest_about(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """Первый кадр записи не годится: она начинается с предбуфера, кота там нет."""
    import numpy as np

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
    event = type(
        "StoredEvent",
        (),
        {
            "id": 9,
            "created_at": datetime(2026, 8, 8, 11, 32),  # noqa: DTZ001
            "media_path": "events/00000009/face.mp4",
            "side_media_path": None,
        },
    )()
    best_frame = np.full((540, 960, 3), 42, dtype=np.uint8)
    photos: list[Path] = []

    def refuse_extraction(*args, **kwargs):
        raise AssertionError("кадр из записи брать не должны — есть запомненный")

    monkeypatch.setattr("src.recorder.app.extract_frame", refuse_extraction)
    monkeypatch.setattr(
        "src.common.telegram.TelegramClient.send_photo",
        lambda self, photo, caption: photos.append(Path(photo)),
    )
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
        best_frame,
    )
    journal.close()

    assert len(photos) == 1
    assert not list((runtime_path / "runtime").glob("notification-*.jpg"))


def test_uncertain_pass_is_recorded_but_not_announced() -> None:
    """Порог уведомления выше порога записи намеренно.

    Писать стоит всё сомнительное — потом разберём по ленте, — а будить
    человека только тем, в чём детектор уверен.
    """
    gate = NotificationGate(
        roles=frozenset({"face"}),
        cooldown_seconds=0.0,
        min_confidence=0.8,
    )

    assert gate.blocked_reason("face", now=1.0, confidence=0.62) == "below_notify_threshold:0.62"
    assert gate.blocked_reason("face", now=1.0, confidence=0.81) is None
    # Без переданной уверенности порог не проверяется: калитка судит иначе.
    assert gate.blocked_reason("face", now=1.0) is None

def test_caption_takes_the_name_from_aggregated_decision(
    runtime_path: Path, monkeypatch
) -> None:
    """Решение агрегатора — «known:Имя», и классы отказа именем не являются.

    Сравнение с голым «known» не пропускало имя в сообщение никогда, а
    «known:__nocat__» подписал бы прохода служебным ярлыком.
    """
    import numpy as np

    from src.common.settings import Settings
    from src.recorder import app as recorder_app
    from src.recorder.detection_journal import DetectionJournal
    from src.recorder.events import AutomaticEventPrediction, AutomaticEventRequest

    recorder_app.reset_notification_gate()
    settings = Settings.model_construct(
        data_dir=runtime_path,
        database_path=runtime_path / "catdoor.sqlite3",
        ffmpeg_path="ffmpeg",
        display_timezone="Europe/Moscow",
        telegram_bot_token=TOKEN,
        telegram_chat_id="12345",
        telegram_api_base="https://api.telegram.org",
        notify_roles=["face"],
        notify_cooldown_seconds=0.0,
        notify_min_confidence=0.0,
    )
    event = type(
        "StoredEvent",
        (),
        {
            "id": 10,
            "created_at": datetime(2026, 8, 10, 16, 40),  # noqa: DTZ001
            "media_path": "events/00000010/face.mp4",
            "side_media_path": None,
        },
    )()
    captions: list[str] = []
    monkeypatch.setattr(
        "src.common.telegram.TelegramClient.send_photo",
        lambda self, photo, caption: captions.append(caption),
    )
    journal = DetectionJournal(
        runtime_path / "runtime" / "detection.jsonl",
        max_bytes=100_000,
        backup_count=1,
    )

    def prediction(view: str, decision: str, name: str | None):
        return AutomaticEventPrediction(
            model_version_id=1,
            view=view,
            decision=decision,
            predicted_name=name,
            confidence=0.9,
            frames_used=8,
            votes=6,
            max_detector_confidence=0.9,
            mean_inference_ms=1.0,
            raw_scores=(),
        )

    recorder_app._notify_about_pass(  # type: ignore[arg-type]
        settings,
        AutomaticEventRequest(
            started_at=1.0,
            ended_at=9.0,
            initiating_camera="face",
            confidence_face=0.98,
            confidence_side=0.0,
            predictions=(
                prediction("face", "known:__nocat__", "__nocat__"),
                prediction("side", "known:Черныш", "Черныш"),
            ),
        ),
        event,
        journal,
        np.full((540, 960, 3), 42, dtype=np.uint8),
    )
    journal.close()

    assert captions and captions[0].startswith("Черныш: проход")


def test_a_night_pass_that_reaches_the_door_is_notified(runtime_path: Path, monkeypatch) -> None:
    """Ночью коты заходят через террасу, и событие начинается с подхода.

    13–14 августа восемь ночных проходов подряд ушли в
    `camera_not_notified:side`, хотя кот дошёл до самой дверцы и был узнан, —
    владелец счёл уведомления сломанными. Роль для гейта — «докуда дошло», а
    не «кто начал»: дверца, увидевшая кота уверенно, делает проход дверным.
    """
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
    event = type(
        "StoredEvent",
        (),
        {
            "id": 545,
            "created_at": datetime(2026, 8, 14, 4, 43),  # noqa: DTZ001
            "media_path": "events/00000545/face.mp4",
            "side_media_path": None,
        },
    )()
    sent: list[str] = []

    def fake_extract(ffmpeg_path, video, destination, **kwargs):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"\xff\xd8jpeg\xff\xd9")
        return True

    monkeypatch.setattr("src.recorder.app.extract_frame", fake_extract)
    monkeypatch.setattr(
        "src.common.telegram.TelegramClient.send_photo",
        lambda self, photo, caption: sent.append(caption),
    )
    journal = DetectionJournal(
        runtime_path / "runtime" / "detection.jsonl", max_bytes=100_000, backup_count=1
    )

    # Начал подход, но дверца увидела кота уверенно — раньше это молчало.
    reached_door = AutomaticEventRequest(
        started_at=1.0,
        ended_at=9.0,
        initiating_camera="side",
        confidence_face=0.90,
        confidence_side=0.95,
    )
    recorder_app._notify_about_pass(settings, reached_door, event, journal)

    # Чисто террасная прогулка остаётся тихой.
    recorder_app.reset_notification_gate()
    terrace_only = AutomaticEventRequest(
        started_at=1.0,
        ended_at=9.0,
        initiating_camera="side",
        confidence_face=0.2,
        confidence_side=0.97,
    )
    recorder_app._notify_about_pass(settings, terrace_only, event, journal)
    journal.close()

    assert len(sent) == 1
    assert "дверца" in sent[0]
