"""Кнопка «Видео» под уведомлением: постановка, нажатие, отказ.

Проверяется весь путь ролика, кроме самой сети: кнопка появляется у обоих
отправителей, нажатие разбирается, ролик уезжает тому, кто нажал, а любой отказ
объясняется человеку словами. Молчащая кнопка неотличима от сломанной — и
именно это здесь и стерегут.
"""

from __future__ import annotations

import json
import urllib.error
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

import pytest

from src.common.clip_button import (
    GATE_KIND,
    PASS_KIND,
    clip_callback_data,
    gate_clip_buttons,
    parse_clip_callback,
    pass_clip_buttons,
)
from src.common.database import Event, apply_migrations, create_session_factory
from src.common.notifications import NotificationGate
from src.common.settings import Settings
from src.common.telegram import UPLOAD_LIMIT_BYTES, TelegramClient, TelegramError
from src.gate.notify import GATE_ROLE, GateEvent, notify_about_events
from src.telegram_bot.clips import (
    ClipUnavailable,
    clip_directory,
    fetch_gate_clip,
    forget_old_presses,
    handle_press,
    load_offset,
    offset_path,
    pass_clip_path,
    save_offset,
    sweep_abandoned_clips,
)

EVENT_ID = "1786171119.126838-b1jywn"


class _Response:
    """Ответ Frigate: тело плюс заголовки, как их отдаёт urlopen."""

    def __init__(self, body: bytes, declared: int | None = None) -> None:
        self._body = body
        self._offset = 0
        self.headers = {} if declared is None else {"Content-Length": str(declared)}

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            chunk, self._offset = self._body[self._offset :], len(self._body)
            return chunk
        chunk = self._body[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        return None


class _Client(TelegramClient):
    """Клиент, который вместо сети складывает отправленное в списки."""

    def __init__(self, chat_id: str = "12345") -> None:
        super().__init__("8999303876:test", chat_id)
        object.__setattr__(self, "videos", [])
        object.__setattr__(self, "texts", [])
        object.__setattr__(self, "answers", [])

    def send_video(self, video, caption, *, chat_id, reply_to=None) -> None:  # type: ignore[override]
        self.videos.append(  # type: ignore[attr-defined]
            {
                "path": Path(video),
                "size": Path(video).stat().st_size,
                "caption": caption,
                "chat_id": chat_id,
                "reply_to": reply_to,
            }
        )

    def send_text_to(self, chat_id, text) -> None:  # type: ignore[override]
        self.texts.append((chat_id, text))  # type: ignore[attr-defined]

    def answer_callback(self, callback_id, text="") -> None:  # type: ignore[override]
        self.answers.append(text)  # type: ignore[attr-defined]


def _settings(runtime_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "data_dir": runtime_path,
        "display_timezone": "Europe/Moscow",
        "gate_api_url": "http://127.0.0.1:5000",
        "gate_notify_cameras": [],
        "gate_notify_labels": ["person"],
        "gate_notify_faces_only": False,
        "gate_notify_cooldown_seconds": 0.0,
        "telegram_clip_button": True,
        "telegram_clip_timeout_seconds": 30.0,
    }
    values.update(overrides)
    return Settings.model_construct(**values)  # type: ignore[arg-type]


def _press(data: str, *, chat_id: str = "12345", message_id: int = 7) -> dict[str, Any]:
    return {
        "id": "cb-1",
        "data": data,
        "message": {"message_id": message_id, "chat": {"id": int(chat_id)}},
    }


# --- кнопка под сообщением ----------------------------------------------------


def test_the_button_carries_the_event_back() -> None:
    """Нагрузка кнопки — единственное, что возвращается нажатием.

    Собирает её один модуль, разбирает другой: разъехавшиеся форматы означали
    бы мёртвую кнопку в чате, то есть поломку, которую видно только на
    телефоне владельца.
    """
    ((_, data),) = gate_clip_buttons(EVENT_ID)

    request = parse_clip_callback(data)

    assert request is not None
    assert (request.kind, request.event_id) == (GATE_KIND, EVENT_ID)


def test_a_pass_gets_one_button_per_recorded_view() -> None:
    """Ракурсы названы по-человечески: в чат смотрит не тот, кто знает роли."""
    buttons = pass_clip_buttons("412", has_face=True, has_side=False)

    assert [text for text, _ in buttons] == ["🎬 Видео с дверцы"]
    assert parse_clip_callback(buttons[0][1]).view == "f"  # type: ignore[union-attr]


def test_a_payload_over_the_limit_gives_no_button() -> None:
    """Сообщение с длинной нагрузкой Телеграм отвергает целиком.

    Уведомление без кнопки — потеря удобства; уведомление, не ушедшее вовсе, —
    потеря сообщения о человеке у калитки. Поэтому кнопка молча не ставится.
    """
    assert clip_callback_data(GATE_KIND, "x" * 70) is None
    assert gate_clip_buttons("x" * 70) == ()


def test_the_gate_notification_arrives_with_a_button(runtime_path: Path, monkeypatch) -> None:
    sent: list[tuple[str, tuple]] = []
    monkeypatch.setattr("src.gate.notify.fetch_snapshot", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        "src.common.telegram.TelegramClient.send_message",
        lambda self, text, buttons=(): sent.append((text, buttons)),
    )
    event = GateEvent(
        id=EVENT_ID, camera="gate", label="person", sub_label="Алекс",
        started_at=1786171119.126838, has_snapshot=False,
    )

    notify_about_events(
        _settings(runtime_path),
        TelegramClient("8999303876:test", "12345"),
        NotificationGate(roles=frozenset({GATE_ROLE}), cooldown_seconds=0.0),
        [event],
        data_dir=runtime_path,
        now=10.0,
        wall_now=event.started_at + 60,
        cameras={"gate": "Калитка"},
    )

    (_, buttons), = sent
    assert parse_clip_callback(buttons[0][1]).event_id == EVENT_ID  # type: ignore[union-attr]


def test_the_button_disappears_when_the_bot_is_off(runtime_path: Path, monkeypatch) -> None:
    """Кнопка без службы, ловящей нажатия, крутила бы часики и молчала.

    Поэтому выключается она одним флагом сразу у всех отправителей.
    """
    sent: list[tuple[str, tuple]] = []
    monkeypatch.setattr("src.gate.notify.fetch_snapshot", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        "src.common.telegram.TelegramClient.send_message",
        lambda self, text, buttons=(): sent.append((text, buttons)),
    )
    event = GateEvent(
        id=EVENT_ID, camera="gate", label="person", sub_label="Алекс",
        started_at=1786171119.126838, has_snapshot=False,
    )

    notify_about_events(
        _settings(runtime_path, telegram_clip_button=False),
        TelegramClient("8999303876:test", "12345"),
        NotificationGate(roles=frozenset({GATE_ROLE}), cooldown_seconds=0.0),
        [event],
        data_dir=runtime_path,
        now=10.0,
        wall_now=event.started_at + 60,
        cameras={"gate": "Калитка"},
    )

    ((text, buttons),) = sent
    assert text.startswith("Калитка: Алекс")
    assert buttons == ()


# --- нажатие ------------------------------------------------------------------


def test_a_press_sends_the_clip_under_the_notification(runtime_path: Path, monkeypatch) -> None:
    """Ролик подвешивается к тому сообщению, под которым нажали.

    За день в чате набирается десяток проходов, и видео само по себе не
    отвечает на вопрос, чьё оно.
    """
    monkeypatch.setattr(
        "src.telegram_bot.clips.urllib.request.urlopen",
        lambda url, timeout: _Response(b"video-bytes"),
    )
    client = _Client()

    delivered = handle_press(
        _settings(runtime_path),
        client,
        None,
        _press(f"clip:{GATE_KIND}:{EVENT_ID}"),
        now=100.0,
        recent={},
    )

    assert delivered
    assert client.videos[0]["chat_id"] == "12345"  # type: ignore[attr-defined]
    assert client.videos[0]["reply_to"] == 7  # type: ignore[attr-defined]
    assert client.answers == ["Готовлю видео…"]  # type: ignore[attr-defined]


def test_the_downloaded_clip_is_not_left_behind(runtime_path: Path, monkeypatch) -> None:
    """Скачанное удаляется сразу: том данных не кладовая для чужих копий."""
    monkeypatch.setattr(
        "src.telegram_bot.clips.urllib.request.urlopen",
        lambda url, timeout: _Response(b"video-bytes"),
    )

    handle_press(
        _settings(runtime_path), _Client(), None,
        _press(f"clip:{GATE_KIND}:{EVENT_ID}"), now=100.0, recent={},
    )

    assert list(clip_directory(runtime_path).glob("*")) == []


def test_a_press_from_a_stranger_gets_nothing(runtime_path: Path) -> None:
    """Бота находят по имени, и чужой чат не обязан молчать сам по себе.

    Список получателей уведомлений и есть список тех, кому положено видео.
    """
    client = _Client(chat_id="12345")

    delivered = handle_press(
        _settings(runtime_path), client, None,
        _press(f"clip:{GATE_KIND}:{EVENT_ID}", chat_id="777"), now=100.0, recent={},
    )

    assert not delivered
    assert client.videos == []  # type: ignore[attr-defined]
    assert client.answers == ["Не для вас"]  # type: ignore[attr-defined]


def test_a_second_press_does_not_send_a_second_clip(runtime_path: Path, monkeypatch) -> None:
    """Ролик едет не мгновенно, и нетерпеливый жмёт ещё раз."""
    monkeypatch.setattr(
        "src.telegram_bot.clips.urllib.request.urlopen",
        lambda url, timeout: _Response(b"video-bytes"),
    )
    client = _Client()
    recent: dict[str, float] = {}
    settings = _settings(runtime_path)
    press = _press(f"clip:{GATE_KIND}:{EVENT_ID}")

    handle_press(settings, client, None, press, now=100.0, recent=recent)
    handle_press(settings, client, None, press, now=130.0, recent=recent)

    assert len(client.videos) == 1  # type: ignore[attr-defined]
    assert client.answers[-1] == "Уже готовлю"  # type: ignore[attr-defined]


def test_the_press_is_forgotten_after_the_window() -> None:
    assert forget_old_presses({"a": 10.0}, now=10.0 + 179.0) == {"a": 10.0}
    assert forget_old_presses({"a": 10.0}, now=10.0 + 181.0) == {}


def test_a_foreign_button_is_left_alone(runtime_path: Path) -> None:
    """Не наша нагрузка — не наше нажатие: часики гасит тот, кто её поставил."""
    client = _Client()

    assert not handle_press(
        _settings(runtime_path), client, None, _press("vote:yes"), now=100.0, recent={}
    )
    assert client.answers == []  # type: ignore[attr-defined]


# --- отказы, о которых человеку говорят вслух ---------------------------------


def test_a_clip_frigate_has_not_written_yet_asks_to_wait(runtime_path: Path, monkeypatch) -> None:
    """404 — это почти всегда «ещё не готов», а не «нет никогда».

    Человек всё равно нажмёт ещё раз, и лучше пусть знает зачем.
    """
    def refuse(url, timeout):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr("src.telegram_bot.clips.urllib.request.urlopen", refuse)
    client = _Client()

    delivered = handle_press(
        _settings(runtime_path), client, None,
        _press(f"clip:{GATE_KIND}:{EVENT_ID}"), now=100.0, recent={},
    )

    assert not delivered
    chat_id, text = client.texts[0]  # type: ignore[attr-defined]
    assert chat_id == "12345"
    assert "не готов" in text


def test_a_refusal_does_not_burn_the_press(runtime_path: Path, monkeypatch) -> None:
    """«Ещё не готов» — приглашение нажать снова, и память об этом мешала бы.

    Иначе повторное нажатие три минуты отвечало бы «уже готовлю», хотя никто
    ничего не готовит.
    """
    def refuse(url, timeout):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr("src.telegram_bot.clips.urllib.request.urlopen", refuse)
    recent: dict[str, float] = {}

    handle_press(
        _settings(runtime_path), _Client(), None,
        _press(f"clip:{GATE_KIND}:{EVENT_ID}"), now=100.0, recent=recent,
    )

    assert recent == {}


def test_an_oversized_clip_is_named_not_uploaded(runtime_path: Path, monkeypatch) -> None:
    """Телеграм принимает от бота 50 МБ — это предел Bot API, а не настройки.

    Припаркованная машина когда-то дала гигабайтный клип и повесила облачную
    очередь; здесь тот же клип не должен ни уехать, ни съесть память узла.
    """
    monkeypatch.setattr(
        "src.telegram_bot.clips.urllib.request.urlopen",
        lambda url, timeout: _Response(b"x", declared=UPLOAD_LIMIT_BYTES + 1),
    )
    client = _Client()

    handle_press(
        _settings(runtime_path), client, None,
        _press(f"clip:{GATE_KIND}:{EVENT_ID}"), now=100.0, recent={},
    )

    assert client.videos == []  # type: ignore[attr-defined]
    assert "МБ" in client.texts[0][1]  # type: ignore[attr-defined]


def test_a_clip_that_grows_past_the_limit_is_dropped(runtime_path: Path, monkeypatch) -> None:
    """Заголовка может не быть: Frigate отдаёт склейку потоком.

    Тогда предел ловится по ходу чтения — иначе гигабайт уехал бы в память
    узла, чтобы там же и умереть.
    """
    monkeypatch.setattr("src.telegram_bot.clips.UPLOAD_LIMIT_BYTES", 4)
    monkeypatch.setattr(
        "src.telegram_bot.clips.urllib.request.urlopen",
        lambda url, timeout: _Response(b"0123456789"),
    )
    destination = clip_directory(runtime_path) / "gate.mp4"

    with pytest.raises(ClipUnavailable):
        fetch_gate_clip("http://127.0.0.1:5000", EVENT_ID, destination, timeout=5.0)

    assert not destination.exists()


def test_telegram_refusing_the_video_is_explained(runtime_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "src.telegram_bot.clips.urllib.request.urlopen",
        lambda url, timeout: _Response(b"video-bytes"),
    )

    class _Refusing(_Client):
        def send_video(self, *args, **kwargs) -> None:  # type: ignore[override]
            raise TelegramError("Телеграм отказал (413)")

    client = _Refusing()

    handle_press(
        _settings(runtime_path), client, None,
        _press(f"clip:{GATE_KIND}:{EVENT_ID}"), now=100.0, recent={},
    )

    assert "413" in client.texts[0][1]  # type: ignore[attr-defined]
    assert list(clip_directory(runtime_path).glob("*")) == []


# --- ролики кошачьих проходов -------------------------------------------------


def _pass_event(runtime_path: Path, *, face: str | None, side: str | None) -> Any:
    database = runtime_path / "catdoor.sqlite3"
    apply_migrations(database)
    factory = create_session_factory(database)
    with factory() as session:
        session.add(
            Event(
                created_at=datetime(2026, 9, 14, 3, 20, tzinfo=UTC),
                status="ready",
                duration_seconds=6.0,
                media_path=face,
                side_media_path=side,
            )
        )
        session.commit()
    return factory


def test_a_cat_pass_clip_comes_from_the_data_volume(runtime_path: Path) -> None:
    """Пути в базе относительные: том данных переезжал между узлами."""
    (runtime_path / "events").mkdir()
    (runtime_path / "events" / "face.mp4").write_bytes(b"cat")
    factory = _pass_event(runtime_path, face="events/face.mp4", side=None)
    client = _Client()

    delivered = handle_press(
        _settings(runtime_path), client, factory,
        _press(f"clip:{PASS_KIND}:1:f"), now=100.0, recent={},
    )

    assert delivered
    assert "дверца" in client.videos[0]["caption"]  # type: ignore[attr-defined]


def test_a_view_that_was_never_recorded_says_so(runtime_path: Path) -> None:
    factory = _pass_event(runtime_path, face="events/face.mp4", side=None)

    with pytest.raises(ClipUnavailable, match="терраса"):
        pass_clip_path(factory, runtime_path, "1", "side")


def test_a_clip_eaten_by_retention_says_so(runtime_path: Path) -> None:
    """В чате строка живёт дольше, чем ролик на диске, и это норма."""
    factory = _pass_event(runtime_path, face="events/face.mp4", side=None)

    with pytest.raises(ClipUnavailable, match="окном хранения"):
        pass_clip_path(factory, runtime_path, "1", "face")


def test_a_pass_that_no_longer_exists_says_so(runtime_path: Path) -> None:
    factory = _pass_event(runtime_path, face="events/face.mp4", side=None)

    with pytest.raises(ClipUnavailable, match="не найден"):
        pass_clip_path(factory, runtime_path, "404", "face")


# --- состояние службы ---------------------------------------------------------


def test_the_offset_survives_a_restart(runtime_path: Path) -> None:
    """Потеря смещения стоит одного повторно уехавшего ролика, не больше."""
    path = offset_path(runtime_path)
    (runtime_path / "runtime").mkdir(parents=True, exist_ok=True)

    save_offset(path, 4711)

    assert load_offset(path) == 4711
    assert load_offset(runtime_path / "runtime" / "missing.json") == 0
    path.write_text("{битый", encoding="utf-8")
    assert load_offset(path) == 0


def test_the_offset_is_stored_before_the_press_is_handled() -> None:
    """Нажатие, от которого служба падает, иначе читалось бы после каждого
    перезапуска — и служба падала бы вечно."""
    source = Path("src/telegram_bot/clips.py").read_text(encoding="utf-8")
    body = source[source.index("for update in updates:") :]

    assert body.index("save_offset(path, offset)") < body.index("handle_press(")


def test_abandoned_downloads_are_swept_at_start(runtime_path: Path) -> None:
    """`finally` не исполняется при пропаже питания и при SIGKILL от сторожа.

    Каталог отведён под одну задачу, поэтому всё в нём при старте заведомо
    брошено.
    """
    directory = clip_directory(runtime_path)
    directory.mkdir(parents=True)
    (directory / "gate-old.mp4").write_bytes(b"leftover")

    sweep_abandoned_clips(runtime_path)

    assert list(directory.glob("*")) == []


def test_the_watched_units_include_the_bot() -> None:
    """Упавшая служба оставила бы кнопку молча крутить часики в чате."""
    from src.monitor.healthwatch import WATCHED_UNITS

    assert "catdoor-telegram-bot" in WATCHED_UNITS


def test_the_button_is_enabled_together_with_the_notifications() -> None:
    """Читать обновления бота имеет право ровно один процесс, и это он."""
    installer = Path("install/install.sh").read_text(encoding="utf-8")

    assert "systemctl enable catdoor-telegram-bot.service" in installer
    assert "-m src.telegram_bot.clips" in (
        Path("install/catdoor-telegram-bot.service").read_text(encoding="utf-8")
    )


def test_the_poll_asks_only_for_presses(monkeypatch) -> None:
    """Чужие сообщения боту заставляли бы службу вычитывать чужую переписку."""
    seen: dict[str, Any] = {}

    def fake_call(self, method, fields, file=None, timeout=None):
        seen.update({"method": method, "fields": fields, "timeout": timeout})
        return {"ok": True, "result": [{"update_id": 5}]}

    monkeypatch.setattr("src.common.telegram.TelegramClient._call", fake_call)

    updates = TelegramClient("8999303876:test", "12345").poll_updates(4, timeout_seconds=25.0)

    assert updates == [{"update_id": 5}]
    assert json.loads(seen["fields"]["allowed_updates"]) == ["callback_query"]
    # Сетевой таймаут даётся с запасом: штатное молчание Телеграма длиной в
    # опрос иначе выглядело бы обрывом связи.
    assert seen["timeout"] > 25.0
