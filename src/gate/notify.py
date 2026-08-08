"""Уведомления о событиях калитки: опрос Frigate и отправка в Телеграм.

Отдельная служба, а не поток внутри веб-службы или recorder, по двум причинам.
Контур калитки живёт своей жизнью, и опрос чужого API не должен ронять запись
кошачьих проходов. И вторая, приземлённая: перезапуск веб-службы обрывает
идущую офлайн-оценку, а уведомления хочется чинить и перезапускать когда
угодно.

Источник — HTTP API самого Frigate на локальном порту, без пароля: наружу этот
порт не смотрит. MQTT дал бы события мгновенно, но потребовал бы брокера ради
одной подписки; опрос раз в десять секунд для «кто-то у калитки» неотличим —
запись к этому моменту всё равно уже сделана.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.common.durable_write import write_text_durably
from src.common.notifications import NotificationGate
from src.common.sd_notify import Notifier
from src.common.settings import ConfigurationError, Settings, validate_runtime_paths
from src.common.telegram import TelegramClient, TelegramError

logger = logging.getLogger("catdoor.gate-notify")

# Сколько событий забирать за раз. Больше нужно только после долгого простоя, и
# тогда лишние всё равно отсеются по отметке времени.
EVENTS_PER_POLL = 20
# Первый запуск не должен вывалить в чат весь архив: берём только то, что
# случилось после старта службы.
LOOKBACK_SECONDS = 60.0
# Пауза считается на контур целиком: у `NotificationGate` она привязана к роли,
# и роль здесь одна на все камеры калитки.
GATE_ROLE = "gate"


@dataclass(frozen=True)
class GateEvent:
    id: str
    camera: str
    label: str
    sub_label: str | None
    started_at: float
    has_snapshot: bool


def state_path(data_dir: Path) -> Path:
    return data_dir / "runtime" / "gate-notify.json"


def load_last_seen(path: Path) -> float | None:
    """Отметка последнего разобранного события.

    `None` — файла нет: служба запускается впервые, и архив в чат не поедет.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = float(payload["last_event_at"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    return value


def save_last_seen(path: Path, value: float) -> None:
    write_text_durably(
        path,
        json.dumps({"last_event_at": value}, ensure_ascii=False),
        mode=0o640,
    )


def fetch_events(api_url: str, after: float, *, timeout: float = 10.0) -> list[GateEvent]:
    query = urllib.parse.urlencode({"after": f"{after:.3f}", "limit": EVENTS_PER_POLL})
    url = f"{api_url.rstrip('/')}/api/events?{query}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Frigate не ответил: {exc}") from exc
    if not isinstance(payload, list):
        # Ответ не того вида — это тоже «Frigate не ответил» с точки зрения
        # вызывающего: он умеет только подождать и спросить снова.
        raise RuntimeError("Frigate вернул не список событий")  # noqa: TRY004
    events = [
        GateEvent(
            id=str(item.get("id", "")),
            camera=str(item.get("camera", "")),
            label=str(item.get("label", "")),
            sub_label=(str(item["sub_label"]) if item.get("sub_label") else None),
            started_at=float(item.get("start_time", 0.0)),
            has_snapshot=bool(item.get("has_snapshot")),
        )
        for item in payload
        if isinstance(item, dict)
    ]
    # Frigate отдаёт свежие первыми, а рассказывать надо по порядку.
    return sorted(events, key=lambda event: event.started_at)


def fetch_snapshot(api_url: str, event: GateEvent, destination: Path, *, timeout: float = 15.0) -> bool:
    if not event.has_snapshot:
        return False
    url = f"{api_url.rstrip('/')}/api/events/{event.id}/snapshot.jpg?h=720"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.warning("Снимок события %s не забрался: %s", event.id, exc)
        return False
    if not body:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(body)
    return True


def gate_caption(event: GateEvent, timezone: str) -> str:
    """Подпись под снимком: кто, где и когда."""
    labels = {"person": "человек", "car": "машина", "cat": "кот", "dog": "собака"}
    who = event.sub_label or labels.get(event.label, event.label or "движение")
    moment = datetime.fromtimestamp(event.started_at, UTC).astimezone(ZoneInfo(timezone))
    return f"Калитка: {who} · {event.camera} · {moment.strftime('%H:%M')}"


def notify_about_events(
    settings: Settings,
    client: TelegramClient,
    gate: NotificationGate,
    events: list[GateEvent],
    *,
    data_dir: Path,
    now: float,
) -> int:
    """Разобрать пачку событий. Возвращает число отправленных сообщений."""
    labels = set(settings.gate_notify_labels)
    cameras = set(settings.gate_notify_cameras)
    sent = 0
    for event in events:
        if labels and event.label not in labels:
            continue
        # Пустой список камер означает «все»: у калитки их одна-две, и
        # перечислять их руками — лишний повод забыть про новую.
        if cameras and event.camera not in cameras:
            continue
        reason = gate.blocked_reason(GATE_ROLE, now)
        if reason is not None:
            logger.info("Событие %s пропущено: %s", event.id, reason)
            continue
        snapshot = data_dir / "runtime" / f"gate-notify-{event.id}.jpg"
        caption = gate_caption(event, settings.display_timezone)
        try:
            if fetch_snapshot(settings.gate_api_url, event, snapshot):
                client.send_photo(snapshot, caption)
            else:
                client.send_message(caption)
        except TelegramError as exc:
            logger.warning("Событие %s не отправилось: %s", event.id, exc)
            continue
        finally:
            snapshot.unlink(missing_ok=True)
        gate.remember_sent(now)
        sent += 1
    return sent


def run(*, once: bool = False) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    settings = Settings()
    validate_runtime_paths(settings)
    client = TelegramClient(settings.telegram_bot_token, settings.telegram_chat_id)
    if not client.configured:
        raise ConfigurationError(
            "Уведомления калитки включены, но бот не настроен: нет токена или адреса чата"
        )
    # Пауза общая на весь контур, а не на камеру: человек у калитки виден
    # обеими сразу, и два сообщения об одном госте — это шум.
    gate = NotificationGate(
        roles=frozenset({GATE_ROLE}),
        cooldown_seconds=settings.gate_notify_cooldown_seconds,
    )
    path = state_path(settings.data_dir)
    last_seen = load_last_seen(path)
    if last_seen is None:
        last_seen = time.time() - LOOKBACK_SECONDS
        save_last_seen(path, last_seen)
    notifier = Notifier.from_environment()
    notifier.ready("Уведомления калитки на связи")
    while True:
        try:
            events = fetch_events(settings.gate_api_url, last_seen)
        except RuntimeError as exc:
            # Frigate могут перезапускать; это не повод падать.
            logger.warning("%s", exc)
            events = []
        if events:
            notify_about_events(
                settings,
                client,
                gate,
                events,
                data_dir=settings.data_dir,
                now=time.monotonic(),
            )
            last_seen = max(event.started_at for event in events)
            save_last_seen(path, last_seen)
        notifier.watchdog()
        if once:
            return
        time.sleep(settings.gate_notify_poll_seconds)


def main() -> None:
    try:
        run()
    except ConfigurationError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
