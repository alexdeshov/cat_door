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
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.common.camera_registry import camera_registry_path, load_registry
from src.common.durable_write import write_text_durably
from src.common.frigate_config import frigate_camera_name
from src.common.notifications import NotificationGate, notification_snapshot_path
from src.common.sd_notify import Notifier
from src.common.settings import ConfigurationError, Settings, validate_runtime_paths
from src.common.telegram import TelegramClient, TelegramError, TelegramUnreachable

logger = logging.getLogger("catdoor.gate-notify")

# Сколько событий забирать за раз. Больше нужно только после долгого простоя, и
# тогда лишние всё равно отсеются по отметке времени.
EVENTS_PER_POLL = 20
# Окно наблюдения. События внутри него перечитываются каждый опрос — потому что
# имя распознанного лица Frigate приписывает событию не сразу, а через несколько
# секунд после появления человека. Уже отправленные помнятся по идентификатору,
# так что повторов это не даёт.
WATCH_WINDOW_SECONDS = 300.0
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


def gate_cameras(data_dir: Path) -> dict[str, str]:
    """Камеры контура калитки: имя во Frigate — заголовок из инвентаря.

    Frigate обслуживает оба контура узла разом, и его лента событий общая. Без
    этого отбора в чат шли уведомления и от камер регистратора — с подписью
    «Калитка», потому что источник считался очевидным. Инвентарь читается на
    каждом опросе: камеру могли перевести в другой контур минуту назад.
    """
    registry = load_registry(camera_registry_path(data_dir))
    if registry is None:
        return {}
    return {
        frigate_camera_name(camera.id): (camera.title or camera.id)
        for camera in registry.cameras
        if camera.circuit == "gate" and camera.enabled
    }


def state_path(data_dir: Path) -> Path:
    return data_dir / "runtime" / "gate-notify.json"


def load_state(path: Path) -> dict[str, float]:
    """Что уже отправлено: идентификатор события — время его начала.

    Пустой словарь означает и «файла нет» (первый запуск), и «ничего ещё не
    отправляли». Различать их не нужно: окно наблюдения всё равно ограничено
    минутами, и архив в чат не поедет ни в том, ни в другом случае.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        sent = payload["sent"]
        return {str(key): float(value) for key, value in sent.items()}
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return {}


def save_state(path: Path, sent: dict[str, float]) -> None:
    write_text_durably(
        path,
        json.dumps({"sent": sent}, ensure_ascii=False),
        mode=0o640,
    )


def forget_old(sent: dict[str, float], *, older_than: float) -> dict[str, float]:
    """Выкинуть из памяти события старше окна наблюдения.

    Иначе файл рос бы вечно, а смысла в нём нет: событие, вышедшее из окна, мы
    и так больше не запрашиваем.
    """
    return {key: value for key, value in sent.items() if value >= older_than}


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


def recognised_name(event: GateEvent) -> str | None:
    """Имя из базы лиц, если Frigate его поставил.

    Пустой `sub_label` и слово «unknown» значат одно и то же — лицо не узнано,
    — но приходят из разных версий Frigate, поэтому проверяются оба.
    """
    name = (event.sub_label or "").strip()
    if not name or name.lower() in {"unknown", "unrecognized"}:
        return None
    return name


def gate_caption(event: GateEvent, timezone: str, title: str = "Калитка") -> str:
    """Подпись под снимком: кто, где и когда."""
    labels = {"person": "человек", "car": "машина", "cat": "кот", "dog": "собака"}
    name = recognised_name(event)
    if name is not None:
        who = name
    elif event.label == "person":
        # «Человек» здесь сказало бы меньше, чем нужно: сообщение приходит
        # именно потому, что этого человека узнать не удалось.
        who = "неизвестный"
    else:
        who = labels.get(event.label, event.label or "движение")
    moment = datetime.fromtimestamp(event.started_at, UTC).astimezone(ZoneInfo(timezone))
    return f"{title}: {who} · {moment.strftime('%H:%M')}"


def notify_about_events(
    settings: Settings,
    client: TelegramClient,
    gate: NotificationGate,
    events: list[GateEvent],
    *,
    data_dir: Path,
    now: float,
    wall_now: float,
    unknown_gate: NotificationGate | None = None,
    already_sent: dict[str, float] | None = None,
    heartbeat: Callable[[], None] | None = None,
    cameras: dict[str, str] | None = None,
) -> int:
    """Разобрать пачку событий. Возвращает число отправленных сообщений."""
    labels = set(settings.gate_notify_labels)
    # Явный список камер в настройке — исключение на случай нестандартной
    # схемы; обычно источник берётся из инвентаря по контуру.
    allowed = (
        {name: name for name in settings.gate_notify_cameras}
        if settings.gate_notify_cameras
        else (cameras if cameras is not None else gate_cameras(data_dir))
    )
    seen = already_sent if already_sent is not None else {}
    beat = heartbeat or (lambda: None)
    if events and not allowed:
        # Молчание из-за нечитаемого инвентаря выглядело бы как «ничего не
        # происходит», а это разные вещи.
        logger.warning("Ни одной камеры контура калитки — сообщать не о чем")
    sent = 0
    for event in events:
        # Сторож кормится на каждом событии, а не после всей пачки: одна
        # отправка стоит до пятнадцати секунд, и четырёх хватало, чтобы
        # перевалить за минуту и быть убитым посреди работы.
        beat()
        if event.id in seen:
            continue
        if labels and event.label not in labels:
            continue
        # Frigate ведёт общую ленту на оба контура узла, и без этого отбора в
        # чат шли бы события камер регистратора.
        if event.camera not in allowed:
            continue
        name = recognised_name(event)
        if name is None:
            # Имя Frigate приписывает событию не сразу: человек сперва
            # появляется, и лишь потом распознаётся. Объявлять его неизвестным
            # немедленно значило бы прислать два сообщения про одного гостя —
            # сперва «неизвестный», потом имя. Поэтому выдержка.
            if settings.gate_notify_faces_only:
                continue
            if wall_now - event.started_at < settings.gate_notify_face_grace_seconds:
                continue
        # Пауза у своих и у чужих разная: знакомых дома ждут, а чужой у калитки
        # может оказаться прохожим, и частить о нём незачем.
        active_gate = gate if name is not None else (unknown_gate or gate)
        reason = active_gate.blocked_reason(GATE_ROLE, now)
        if reason is not None:
            # Пропущенное по паузе событие забывается, а не откладывается.
            # Иначе оно ждало бы в окне наблюдения и уходило в чат, как только
            # пауза истечёт, — уведомление о человеке, которого видели три
            # минуты назад. Пауза означает «не беспокоить», а не «потом».
            logger.info("Событие %s пропущено: %s", event.id, reason)
            seen[event.id] = event.started_at
            continue
        snapshot = notification_snapshot_path(data_dir, f"gate-{event.id}")
        caption = gate_caption(event, settings.display_timezone, allowed[event.camera])
        try:
            if fetch_snapshot(settings.gate_api_url, event, snapshot):
                client.send_photo(snapshot, caption)
            else:
                client.send_message(caption)
        except TelegramUnreachable as exc:
            # Разбор пачки прекращается целиком. Следующее событие уйдёт в тот
            # же обрыв и потратит те же пятнадцать секунд таймаута, а служба
            # обязана вернуться к сторожевому таймеру за минуту.
            #
            # 2026-08-09 этого не было, и вышла смертельная петля: Телеграм стал
            # недоступен, неотправленные события копились и пробовались заново
            # каждый цикл, цикл перевалил за минуту, systemd убивал службу по
            # сторожевому таймеру, она поднималась и повторяла всё сначала.
            logger.warning("Разбор прерван, Телеграм недоступен: %s", exc)
            snapshot.unlink(missing_ok=True)
            break
        except TelegramError as exc:
            # Отказ по конкретному сообщению — частное дело этого сообщения:
            # остальные могут уйти.
            logger.warning("Событие %s не отправилось: %s", event.id, exc)
            continue
        finally:
            snapshot.unlink(missing_ok=True)
        active_gate.remember_sent(now)
        seen[event.id] = event.started_at
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
    unknown_gate = NotificationGate(
        roles=frozenset({GATE_ROLE}),
        cooldown_seconds=settings.gate_notify_unknown_cooldown_seconds,
    )
    path = state_path(settings.data_dir)
    sent = load_state(path)
    started_at = time.time()
    notifier = Notifier.from_environment()
    notifier.ready("Уведомления калитки на связи")
    while True:
        # Архив в чат не поедет: раньше запуска службы не смотрим вовсе.
        window_start = max(started_at, time.time() - WATCH_WINDOW_SECONDS)
        try:
            events = fetch_events(settings.gate_api_url, window_start)
        except RuntimeError as exc:
            # Frigate могут перезапускать; это не повод падать.
            logger.warning("%s", exc)
            events = []
        if events:
            delivered = notify_about_events(
                settings,
                client,
                gate,
                events,
                data_dir=settings.data_dir,
                now=time.monotonic(),
                wall_now=time.time(),
                unknown_gate=unknown_gate,
                already_sent=sent,
                heartbeat=notifier.watchdog,
            )
            if delivered:
                sent = forget_old(sent, older_than=window_start)
                save_state(path, sent)
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
