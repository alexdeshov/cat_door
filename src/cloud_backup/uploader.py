"""Выгрузка клипов событий Frigate в S3-бакет — раз в минуту.

Назначение — страховка от выноса оборудования: к моменту, когда вор дошёл от
калитки до сервера, клип его подхода уже в облаке. Отсюда устройство:

- **опрос, а не подписка**: раз в минуту спрашиваем у Frigate завершённые
  события за окно оглядки и заливаем недостающие. Опрос идемпотентен и
  догоняет после любого простоя — облака, узла, самого выгрузчика; подписка
  на MQTT теряла бы всё, что случилось, пока служба лежала;
- **клип целиком, а не сегменты**: `GET /api/events/<id>/clip.mp4` отдаёт
  склеенный ролик события — смотреть его после кражи можно без плясок с
  10-секундными кусками;
- **журнал залитого** живёт в состоянии на диске: перезапуск не приводит ни к
  повторной заливке, ни к дырам. Ключи в бакете детерминированы
  (`events/дата/камера/id.mp4`), так что даже потерянное состояние даёт лишь
  перезапись того же объекта новой версией, а не дубль;
- деньги: входящий трафик у Яндекса бесплатен, DELETE не тарифицируется,
  чистит бакет lifecycle — службе удалять ничего не нужно (и нельзя: ключ
  только на запись).

Настройки перечитываются каждый цикл — включение из админки подхватывается
без перезапуска.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.common.cloud_backup_config import (
    CloudBackupConfig,
    cloud_backup_config_path,
    cloud_backup_state_path,
    load_cloud_backup_config,
)
from src.common.durable_write import write_text_durably
from src.common.sd_notify import Notifier
from src.common.settings import Settings
from src.common.telegram import TelegramClient, TelegramError

logger = logging.getLogger("catdoor.cloud-backup")

# Как часто спрашивать Frigate, если брокер молчит. Опрос — страховочная сеть:
# он догоняет всё, что случилось, пока служба или брокер лежали. Тридцать
# секунд — компромисс между «узнать быстро» и «не дёргать зря».
POLL_SECONDS = 30.0
# Сколько идентификаторов помнить. Событий ~400 в сутки, окно оглядки сутки —
# четырёх тысяч хватает с десятикратным запасом, а файл состояния не растёт.
REMEMBER_IDS = 4000
# Сколько копия может молчать, прежде чем это станет поводом разбудить
# человека. Час — примерно два десятка событий обычного дня: достаточно, чтобы
# переждать перезагрузку роутера, и мало, чтобы узнать о беде тем же вечером.
ALERT_AFTER_SECONDS = 3600.0
# Как часто напоминать, пока не починено. Тревога раз в минуту превратилась бы
# в шум, который перестают читать, — и настоящую беду в нём не заметят.
ALERT_REPEAT_SECONDS = 6 * 3600.0
# Потолок длительности клипа. Frigate склеивает событие в один mp4, и трёхчасовая
# машина у калитки даёт около гигабайта: узел собирает такой клип дольше любого
# разумного таймаута, а канал на 300 КБ/с отдавал бы его час. 14 августа одно
# такое событие встало первым в очереди и остановило выгрузку целиком — при
# живом канале и работоспособной службе.
#
# Такие события не пропадают молча: снимок уходит всегда, а сам ролик остаётся
# на узле и в ночной локальной копии. Для страховки от выноса важен факт и
# кадр, а не три часа записи стоящей машины.
MAX_CLIP_SECONDS = 600.0


def _fetch_json(url: str, timeout: float = 15.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", "replace"))


def _fetch_bytes(url: str, timeout: float = 120.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def eligible_events(
    events: list[dict[str, Any]],
    config: CloudBackupConfig,
    uploaded: set[str],
) -> list[dict[str, Any]]:
    """Что заливать: завершённые, с записью, нужной камеры, ещё не залитые.

    Незавершённые события пропускаются без пометки — их клип ещё растёт, и
    заливать половину значило бы никогда не залить целое: пометка «сделано»
    закрыла бы дорогу повтору.
    """
    selected = []
    # Новые вперёд. Восходящий канал узла — 1.8–3.6 Мбит/с (замерено
    # 14 августа), и суточный хвост едет часами; всё это время в облаке должны
    # оказываться прежде всего последние минуты — ради них копия и заводилась.
    # Хронологический порядок был нужен, пока один ядовитый клип мог застопорить
    # очередь; теперь каждый отказ обрабатывается отдельно, и очередь не встаёт.
    events = sorted(events, key=lambda item: float(item.get("start_time") or 0), reverse=True)
    for event in events:
        event_id = str(event.get("id") or "")
        if not event_id or event_id in uploaded:
            continue
        if not event.get("end_time"):
            continue
        if not event.get("has_clip", True):
            # Записи нет — заливать нечего; помечаем, чтобы не спрашивать вечно.
            selected.append({**event, "_skip": True})
            continue
        camera = str(event.get("camera") or "")
        if config.cameras and camera not in config.cameras:
            continue
        duration = float(event.get("end_time") or 0) - float(event.get("start_time") or 0)
        if duration > MAX_CLIP_SECONDS:
            # Ролик неподъёмный — но снимок этого события всё равно нужен.
            selected.append({**event, "_snapshot_only": True})
            continue
        selected.append(event)
    return selected


# Сколько весит секунда видео, пока своих замеров нет: 5 Мбит/с основного
# потока. Первые же залитые клипы заменят это число фактом.
FALLBACK_BYTES_PER_SECOND = 640_000


def video_seconds(events: list[dict[str, Any]]) -> float:
    """Длительность очереди в секундах видео — из неё считается вес.

    События, от которых уедет только снимок, в вес не входят: иначе оценка
    «осталось 11.7 ГБ» состояла бы в основном из ролика, который никто не
    собирается заливать.
    """
    total = 0.0
    for event in events:
        if event.get("_snapshot_only"):
            continue
        start = float(event.get("start_time") or 0)
        end = float(event.get("end_time") or start)
        total += max(0.0, end - start)
    return total


def human_size(size_bytes: float) -> str:
    """Гигабайты, когда их больше одного, иначе мегабайты: «осталось 0.04 ГБ»
    человеку не говорит ничего."""
    if size_bytes >= 1e9:
        return f"{size_bytes / 1e9:.1f} ГБ"
    return f"{size_bytes / 1e6:.0f} МБ"


def _object_key(config: CloudBackupConfig, event: dict[str, Any], suffix: str) -> str:
    day = datetime.fromtimestamp(float(event["start_time"]), UTC).strftime("%Y-%m-%d")
    camera = str(event.get("camera") or "unknown")
    return f"{config.prefix}/{day}/{camera}/{event['id']}.{suffix}"


def alert_decision(
    state: dict[str, Any],
    *,
    now: float,
    after: float = ALERT_AFTER_SECONDS,
    repeat: float = ALERT_REPEAT_SECONDS,
) -> str | None:
    """О чём сказать человеку: `alarm`, `recovered` или молчание.

    Чистая функция: решение о тревоге проверяется таблицей случаев, а не
    ожиданием у телефона. Отсчёт идёт от последней удачной выгрузки, а если её
    не было ни разу — от момента, когда копию включили: «ни разу не заработало»
    — такая же беда, как «перестало».
    """
    reference = float(state.get("last_success_at") or state.get("watching_since") or 0.0)
    if not reference:
        return None
    silent_for = now - reference
    alerted_at = float(state.get("alerted_at") or 0.0)
    if silent_for < after:
        # Всё в порядке. Если перед этим тревожили — сказать, что отпустило.
        return "recovered" if alerted_at else None
    if alerted_at and now - alerted_at < repeat:
        return None
    return "alarm"


class EventTrigger:
    """Будильник от брокера: событие закончилось — не ждать конца опроса.

    Замер 14 августа: событие человека живёт 8 секунд (медиана), а опрос
    занимал минуту — то есть почти вся задержка между «человек ушёл» и
    «клип в облаке» приходилась именно на ожидание следующего круга.
    Подписка на `frigate/events` будит цикл сразу.

    Подписка **дополняет** опрос, а не заменяет: сообщение, пришедшее в
    мёртвую службу, потеряно навсегда, а опрос переживает и перезапуск, и
    падение брокера. Поэтому брокера нет — просто ждём по таймеру.
    """

    def __init__(self, host: str, port: int = 1883) -> None:
        self.host = host
        self.port = port
        self._wake = threading.Event()
        self._client: Any = None

    def start(self) -> bool:
        if not self.host:
            return False
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            logger.info("paho-mqtt нет: выгрузка пойдёт по таймеру")
            return False
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="catdoor-cloud-backup")
        client.on_connect = lambda c, *_: c.subscribe("frigate/events")
        client.on_message = self._on_message
        try:
            client.connect(self.host, self.port, 60)
        except OSError as exc:
            logger.info("Брокер %s:%s недоступен: пойду по таймеру (%s)", self.host, self.port, exc)
            return False
        client.loop_start()
        self._client = client
        logger.info("Слушаю окончания событий: frigate/events")
        return True

    def _on_message(self, _client: Any, _userdata: Any, message: Any) -> None:
        try:
            payload = json.loads(getattr(message, "payload", b"").decode("utf-8", "replace"))
        except ValueError:
            return
        # Только конец события: у начатого клип ещё растёт, и будить цикл на
        # него значит гонять его вхолостую.
        if isinstance(payload, dict) and payload.get("type") == "end":
            self._wake.set()

    def wait(self, timeout: float) -> None:
        self._wake.wait(timeout)
        self._wake.clear()

    def stop(self) -> None:
        client = self._client
        if client is not None:
            client.loop_stop()
            client.disconnect()
            self._client = None


def _remember(uploaded: set[str], order: list[str], event_id: str) -> None:
    """Отметить залитым, сохранив порядок: по нему обрезается журнал."""
    if event_id not in uploaded:
        uploaded.add(event_id)
        order.append(event_id)


def _load_state(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now(UTC).isoformat()
    write_text_durably(path, json.dumps(state, ensure_ascii=False), mode=0o600)


def _alert_sender(settings: Settings) -> Callable[[str], None] | None:
    """Отправитель тревог. `None` — бот не настроен, тогда только журнал."""
    client = TelegramClient(
        settings.telegram_bot_token,
        settings.telegram_chat_id,
        api_base=settings.telegram_api_base,
    )
    if not client.configured:
        logger.warning("Бот не настроен: о простое копии будет знать только журнал")
        return None

    def send(text: str) -> None:
        try:
            client.send_message(text)
        except (TelegramError, OSError) as exc:
            # Тревога, которая роняет службу, хуже отсутствующей.
            logger.warning("Тревога не ушла в Телеграм: %s", exc)

    return send


def _notify_about_silence(
    state: dict[str, Any], send: Callable[[str], None] | None, *, now: float
) -> None:
    decision = alert_decision(state, now=now)
    if decision is None:
        return
    if decision == "recovered":
        state.pop("alerted_at", None)
        logger.info("Копия снова работает")
        if send is not None:
            send("Облачная копия снова работает — очередь пошла")
        return
    reference = float(state.get("last_success_at") or state.get("watching_since") or now)
    hours = max(0.0, (now - reference) / 3600)
    state["alerted_at"] = now
    logger.warning("Копия молчит %.0f ч", hours)
    if send is not None:
        # Про телеграм сказано честно: при мёртвом канале это сообщение само
        # не уйдёт и придёт с возвратом связи — тревога не заменяет проверки
        # страницы «Состояние», а дополняет её.
        send(
            f"Облачная копия не выгружает {hours:.0f} ч.\n"
            f"Последняя причина: {state.get('last_error') or state.get('message') or '—'}\n"
            "Подробности — страница «Состояние» узла."
        )


def run(settings: Settings, *, once: bool = False) -> None:
    notifier = Notifier.from_environment()
    notifier.ready("Облачная копия событий на связи")
    config_path = cloud_backup_config_path(settings.data_dir)
    state_path = cloud_backup_state_path(settings.data_dir)
    config = CloudBackupConfig()
    trigger = EventTrigger(settings.mqtt_host, settings.mqtt_port)
    trigger.start()
    alert = _alert_sender(settings)
    client = None
    client_key: tuple[str, str, str, str] | None = None
    while True:
        notifier.watchdog()
        config = load_cloud_backup_config(config_path, config)
        state = _load_state(state_path)
        if not config.enabled or not config.ready:
            # Выключенная копия не тревожит: молчание по решению владельца —
            # не беда. Отсчёт начнётся заново, когда её включат.
            state.pop("watching_since", None)
            state.pop("alerted_at", None)
            state["status"] = "disabled" if not config.enabled else "not_configured"
            state["message"] = (
                "Выключено в админке" if not config.enabled
                else "Не заданы бакет или ключи"
            )
            _save_state(state_path, state)
            if once:
                return
            trigger.wait(POLL_SECONDS)
            continue
        state.setdefault("watching_since", time.time())
        try:
            import boto3  # type: ignore[import-not-found]  # ставится на узле, в dev-среде не нужен

            wanted = (config.endpoint, config.region, config.access_key_id, config.secret_key)
            if client is None or client_key != wanted:
                # Умолчания botocore (минута на соединение, несколько попыток)
                # растягивали «минутный» цикл на часы при мёртвом облаке — и всё
                # это время сторож исправно кормился. Настройка необязательна:
                # без botocore клиент создаётся с умолчаниями, а не падает.
                extra: dict[str, Any] = {}
                try:
                    from botocore.config import (  # type: ignore[import-not-found]
                        Config as BotoConfig,
                    )

                    extra["config"] = BotoConfig(
                        connect_timeout=15,
                        read_timeout=120,
                        retries={"max_attempts": 2},
                    )
                except ImportError:
                    pass
                client = boto3.client(
                    "s3",
                    **extra,
                    endpoint_url=config.endpoint,
                    region_name=config.region,
                    aws_access_key_id=config.access_key_id,
                    aws_secret_access_key=config.secret_key,
                )
                client_key = wanted
            after = time.time() - config.lookback_hours * 3600
            events = _fetch_json(
                f"{settings.gate_api_url.rstrip('/')}/api/events?"
                f"after={after:.0f}&limit=500"
            )
            if not isinstance(events, list):
                raise RuntimeError("Frigate вернул не список событий")  # noqa: TRY004 — это отказ API, не тип
            # Журнал — список, а не множество: обрезка по REMEMBER_IDS должна
            # выбрасывать самые старые отметки, а обход set() отдаёт их в
            # произвольном порядке и жертвует случайными (находка ревью 14.08).
            uploaded_order: list[str] = list(state.get("uploaded") or [])
            uploaded = set(uploaded_order)

            attempts: dict[str, int] = dict(state.get("attempts") or {})
            sent_bytes = 0
            sent_events = 0
            failed = 0
            batch = eligible_events(events, config, uploaded)
            # Вес очереди считается по длительности клипов и наблюдаемому
            # весу секунды видео: у Frigate нет способа спросить размер клипа
            # заранее, а «осталось N файлов» ничего не говорит о времени —
            # события бывают и по три секунды, и по три минуты.
            rate = float(state.get("bytes_per_video_second") or FALLBACK_BYTES_PER_SECOND)
            queue_seconds = video_seconds(batch)
            done_seconds = 0.0
            if len(batch) > 20:
                # Длинный хвост — сразу показать фронт работ, не дожидаясь
                # первого промежуточного сохранения.
                state["status"] = "running"
                state["progress"] = {
                    "done": 0,
                    "total": len(batch),
                    "remaining_bytes": int(queue_seconds * rate),
                }
                state["message"] = (
                    f"Догоняю: 0 из {len(batch)}, "
                    f"осталось ~{human_size(queue_seconds * rate)}"
                )
                _save_state(state_path, state)
            for position, event in enumerate(batch):
                event_id = str(event["id"])
                if event.get("_skip"):
                    _remember(uploaded, uploaded_order, event_id)
                    continue
                if event.get("_snapshot_only"):
                    # Клип длиннее потолка: берём только снимок и идём дальше,
                    # иначе одно такое событие держит очередь мёртвой хваткой.
                    try:
                        snapshot = _fetch_bytes(
                            f"{settings.gate_api_url.rstrip('/')}"
                            f"/api/events/{event_id}/snapshot.jpg"
                        )
                        client.put_object(
                            Bucket=config.bucket,
                            Key=_object_key(config, event, "jpg"),
                            Body=snapshot,
                            ContentType="image/jpeg",
                        )
                        sent_bytes += len(snapshot)
                        _remember(uploaded, uploaded_order, event_id)
                        state["last_success_at"] = time.time()
                        logger.info(
                            "Событие %s длиннее %.0f мин: залит только снимок",
                            event_id, MAX_CLIP_SECONDS / 60,
                        )
                    except Exception as exc:  # noqa: BLE001
                        attempts[event_id] = attempts.get(event_id, 0) + 1
                        failed += 1
                        logger.info("Снимок %s не ушёл: %s", event_id, str(exc)[:120])
                    notifier.watchdog()
                    continue
                # Одно ядовитое событие не должно ронять цикл: Frigate может
                # собирать клип часового события дольше любого таймаута.
                # 14 августа такой клип оборвал догоняющую выгрузку, а
                # застрявший сторож перезапустил службу целиком.
                try:
                    clip = _fetch_bytes(
                        f"{settings.gate_api_url.rstrip('/')}/api/events/{event_id}/clip.mp4"
                    )
                    client.put_object(
                        Bucket=config.bucket,
                        Key=_object_key(config, event, "mp4"),
                        Body=clip,
                        ContentType="video/mp4",
                    )
                except Exception as exc:  # noqa: BLE001 — пропуск, не крушение
                    attempts[event_id] = attempts.get(event_id, 0) + 1
                    failed += 1
                    # Списание — только в конце цикла и только если канал в этом
                    # же цикле доказал, что жив. Иначе пять минут без интернета
                    # (пять циклов) списывали бы ВСЮ очередь как «залитую», и
                    # вернувшаяся связь уже не догнала бы ни одного клипа —
                    # включая тот, ради которого копия и заводилась. Найдено
                    # адверсарным ревью 14 августа.
                    logger.info(
                        "Событие %s отложено (попытка %d): %s",
                        event_id, attempts[event_id], str(exc)[:120],
                    )
                    notifier.watchdog()
                    continue
                attempts.pop(event_id, None)
                sent_bytes += len(clip)
                done_seconds += max(
                    0.0,
                    float(event.get("end_time") or 0) - float(event.get("start_time") or 0),
                )
                if done_seconds > 0:
                    # Свой замер точнее умолчания: битрейт зависит от камеры и
                    # сцены, а не от паспортных цифр.
                    rate = sent_bytes / done_seconds
                    state["bytes_per_video_second"] = round(rate)
                if sent_events % 20 == 0:
                    # Догоняющий цикл идёт десятки минут, и страница всё это
                    # время показывала бы последнее прошлое состояние — 14
                    # августа это дважды прочиталось как «сломано», хотя шла
                    # успешная выгрузка на 3 МБ/с. Прогресс пишется по ходу.
                    state["status"] = "running"
                    remaining = max(0.0, (queue_seconds - done_seconds) * rate)
                    state["progress"] = {
                        "done": sent_events + 1,
                        "total": len(batch),
                        "remaining_bytes": int(remaining),
                    }
                    state["message"] = (
                        f"Догоняю: {sent_events + 1} из {len(batch)}, "
                        f"залито {human_size(sent_bytes)}, "
                        f"осталось ~{human_size(remaining)}"
                    )
                    # Журнал залитого — в каждое промежуточное сохранение.
                    # Оба первых догоняющих цикла умерли до конца, и час
                    # выгрузки не оставил ни одной отметки: 460 клипов поехали
                    # по второму разу, а при версионировании с блокировкой
                    # каждая перезаливка — лишняя запертая версия и деньги.
                    state["uploaded"] = uploaded_order[-REMEMBER_IDS:]
                    state["attempts"] = attempts
                    _save_state(state_path, state)
                try:
                    snapshot = _fetch_bytes(
                        f"{settings.gate_api_url.rstrip('/')}/api/events/{event_id}/snapshot.jpg"
                    )
                    client.put_object(
                        Bucket=config.bucket,
                        Key=_object_key(config, event, "jpg"),
                        Body=snapshot,
                        ContentType="image/jpeg",
                    )
                    sent_bytes += len(snapshot)
                except Exception:  # noqa: BLE001 — снимок второстепенен
                    # Ловить надо всё: ошибки botocore не наследуют OSError, и
                    # отказ на снимке вылетал из цикла мимо отметки о клипе —
                    # тот же клип заливался заново каждую минуту, а очередь
                    # стояла (находка ревью 14.08).
                    logger.info("Событие %s: снимок не ушёл", event_id)
                _remember(uploaded, uploaded_order, event_id)
                # Отметка успеха ставится на каждый залитый объект, а не в
                # конце цикла. Догоняющий цикл идёт часами и может не дожить до
                # конца — перезапуск службы, ядовитый клип, обрыв, — и тогда
                # «последнего успеха» не появлялось вовсе: тревога о молчании
                # ударила бы через час по работающей во всю силу выгрузке.
                state["last_success_at"] = time.time()
                state.pop("last_error", None)
                sent_events += 1
                notifier.watchdog()
            if sent_events:
                # Канал жив — значит, застрявшие клипы застряли сами по себе.
                for event_id, count in list(attempts.items()):
                    if count >= 5:
                        _remember(uploaded, uploaded_order, event_id)
                        attempts.pop(event_id, None)
                        logger.warning(
                            "Событие %s не отдалось за %d попыток при живом канале",
                            event_id, count,
                        )
            state["uploaded"] = uploaded_order[-REMEMBER_IDS:]
            state["attempts"] = attempts
            today = datetime.now(UTC).strftime("%Y-%m-%d")
            if state.get("day") != today:
                state["day"] = today
                state["day_events"] = 0
                state["day_bytes"] = 0
            state["day_events"] = int(state.get("day_events") or 0) + sent_events
            state["day_bytes"] = int(state.get("day_bytes") or 0) + sent_bytes
            state.pop("progress", None)
            if failed and not sent_events:
                # Ни одного доставленного при непустой очереди — это не успех.
                # Зелёная строка со свежим «последний успех» врала бы неделю,
                # пока в облаке пусто (находка ревью 14.08).
                state["status"] = "error"
                state["message"] = (
                    f"Не удалось залить ни одного события: ждут {failed}"
                )
            else:
                state["status"] = "ok"
                tail = f", отложено {failed}" if failed else ""
                state["message"] = (
                    f"За сутки: {state['day_events']} событий, "
                    f"{state['day_bytes'] / 1e6:.0f} МБ{tail}"
                )
                state["last_success_at"] = time.time()
                state.pop("last_error", None)
            if sent_events:
                logger.info(
                    "Залито %d событий, %.1f МБ", sent_events, sent_bytes / 1e6
                )
        except Exception as exc:  # noqa: BLE001 — облако и Frigate падают, служба нет
            state["status"] = "error"
            state["last_error"] = str(exc)[:300]
            state["message"] = f"Ошибка выгрузки: {str(exc)[:120]}"
            logger.warning("Цикл выгрузки не удался: %s", exc)
        _notify_about_silence(state, alert, now=time.time())
        _save_state(state_path, state)
        if once:
            return
        trigger.wait(POLL_SECONDS)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    run(Settings())


if __name__ == "__main__":
    main()
