"""Выгрузка клипов событий Frigate в S3-бакет.

Назначение — страховка от выноса оборудования: к моменту, когда вор дошёл от
калитки до сервера, клип его подхода уже в облаке. Отсюда устройство:

- **опрос, а брокер сверх него**: раз в полминуты спрашиваем у Frigate
  завершённые события за окно оглядки и заливаем недостающие. Опрос
  идемпотентен и догоняет после любого простоя — облака, узла, самого
  выгрузчика; подписка на MQTT теряла бы всё, что случилось, пока служба
  лежала, поэтому она только будит цикл раньше срока;
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
from collections.abc import Callable, Iterator
from contextlib import contextmanager
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
# Сколько дать Frigate на сборку клипа, прежде чем его просить. Будильник
# брокера приходит в ту же секунду, в которую событие закончилось, а клипа в
# этот момент ещё нет: 14 августа КАЖДОЕ новое событие получало
# `HTTP 400 Bad Request` на первой попытке — время отказа совпадало с
# `end_time` до секунды. Выигрыша это не давало (событие всё равно уезжало
# следующим кругом), зато копило попытки, а пять попыток при живом канале
# списывают событие как «залитое», не залив его.
CLIP_READY_SECONDS = 10.0
# Сколько дать Frigate на выставление `has_clip`. Флаг в выдаче `/api/events`
# появляется не мгновенно: событие, увиденное с `has_clip=False` сразу после
# конца, раньше помечалось «без записи» НАВСЕГДА — и клип, появившийся секундой
# позже, не уезжал уже никогда (находка ревью 15.08). Две минуты — с большим
# запасом к любой сборке.
HAS_CLIP_GRACE_SECONDS = 120.0
# Бэкофф повторов вместо списания. Списание при однонаправленном ключе
# необратимо (чтения нет — проверить, залито ли, нельзя), а оба его сигнала
# оказались ложными: попытки копились в мёртвых циклах, «живость» канала
# доказывалась одним мелким снимком, и рваная связь хоронила всю очередь
# (находка ревью 15.08). Теперь событие не хоронится никогда: неудача лишь
# отодвигает следующую попытку — 1, 2, 4… минут, потолок час, — а из памяти
# событие выходит только с окном оглядки. Цена ядовитого события — один
# таймаут в час, а не потерянный навсегда клип.
RETRY_BASE_SECONDS = 60.0
RETRY_CEILING_SECONDS = 3600.0


def retry_delay(count: int) -> float:
    """Через сколько секунд событию положена следующая попытка."""
    return min(RETRY_CEILING_SECONDS, RETRY_BASE_SECONDS * 2 ** max(0, count - 1))


def _attempt_entry(raw: Any) -> dict[str, float]:
    """Запись о попытках; старый формат (голое число) поднимается на ходу."""
    if isinstance(raw, dict):
        return {"count": int(raw.get("count") or 0), "last_at": float(raw.get("last_at") or 0.0)}
    try:
        return {"count": int(raw), "last_at": 0.0}
    except (TypeError, ValueError):
        return {"count": 0, "last_at": 0.0}


def deferred_ids(attempts: dict[str, Any], now: float) -> set[str]:
    """Кому ещё рано на повтор. `last_at` в будущем (часы прыгнули назад)
    не откладывает попытку — иначе перевод часов заморозил бы очередь."""
    waiting = set()
    for event_id, raw in attempts.items():
        entry = _attempt_entry(raw)
        since = now - entry["last_at"]
        if 0 <= since < retry_delay(int(entry["count"])):
            waiting.add(event_id)
    return waiting


# Сколько нить-кормилец живёт максимум. Таймауты передач по-операционные:
# полуживое TCP-соединение, отдающее байт раз в сто секунд, держит
# `_fetch_bytes`/`put_object` вечно — и нить, пингующая без предела, делала бы
# такое зависание невидимым НАВСЕГДА: сторож сыт, тревога живёт в том же
# заблокированном потоке (находка ревью 15.08). Честная передача самого
# крупного клипа (~150 МБ на 300 КБ/с в обе стороны) укладывается в
# двадцать минут; после часа нить замолкает, и systemd перезапускает
# застрявшую службу штатно.
HEARTBEAT_LIFETIME_SECONDS = 3600.0


@contextmanager
def _heartbeat(notifier: Notifier) -> Iterator[None]:
    """Кормить сторожа во время передачи, а не только между событиями.

    Клип в десять минут — это около 150 МБ, а восходящий канал узла отдаёт
    300 КБ/с: один `put_object` идёт восемь минут и больше. 14 августа сторож
    (тогда 300 с) трижды убил выгрузчик посреди заливки, и каждый SIGABRT стоил
    потерянных отметок и повторной заливки уже залитого окна — оплаченных
    лишних версий в бакете.

    Нить пингует от имени того же процесса, поэтому `NotifyAccess=main` не
    против: проверка идёт по pid. Без сторожа (`watchdog_interval_seconds` —
    `None`) нить не заводится вовсе. Жизнь нити ограничена
    `HEARTBEAT_LIFETIME_SECONDS`: бесконечный пинг превращал бы зависшую
    навсегда передачу в невидимую.
    """
    interval = notifier.watchdog_interval_seconds
    if interval is None:
        yield
        return
    done = threading.Event()

    def beat() -> None:
        beats = 0
        while not done.wait(interval) and beats * interval < HEARTBEAT_LIFETIME_SECONDS:
            notifier.watchdog()
            beats += 1

    thread = threading.Thread(target=beat, name="cloud-backup-watchdog", daemon=True)
    thread.start()
    try:
        yield
    finally:
        done.set()
        thread.join(timeout=1.0)


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
    *,
    now: float | None = None,
    deferred: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Что заливать: завершённые, с записью, нужной камеры, ещё не залитые.

    Незавершённые события пропускаются без пометки — их клип ещё растёт, и
    заливать половину значило бы никогда не залить целое: пометка «сделано»
    закрыла бы дорогу повтору. По той же причине пропускаются только что
    закончившиеся (`CLIP_READY_SECONDS`), свежие с ещё не выставленным
    `has_clip` (`HAS_CLIP_GRACE_SECONDS`) и отдыхающие после неудачи
    (`deferred`) — все они вернутся следующими кругами.

    Кривое событие в выдаче (нечисловые времена, без start_time) пропускается
    с жалобой, но не роняет отбор: одна мусорная строка Frigate раньше
    стопорила ВСЮ очередь на всё окно оглядки (находка ревью 15.08).
    """
    now = time.time() if now is None else now
    deferred = deferred or set()
    parsed = []
    for event in events:
        try:
            start = float(event["start_time"])
            end = float(event["end_time"]) if event.get("end_time") else None
        except (KeyError, TypeError, ValueError):
            logger.warning("Событие %s с кривыми временами пропущено", event.get("id"))
            continue
        parsed.append((start, end, event))
    # Новые вперёд. Восходящий канал узла — 1.8–3.6 Мбит/с (замерено
    # 14 августа), и суточный хвост едет часами; всё это время в облаке должны
    # оказываться прежде всего последние минуты — ради них копия и заводилась.
    parsed.sort(key=lambda item: item[0], reverse=True)
    selected = []
    seen: set[str] = set()
    for start, end, event in parsed:
        event_id = str(event.get("id") or "")
        if not event_id or event_id in uploaded or event_id in seen:
            # Дубликат id в выдаче — иначе клип уехал бы дважды за цикл.
            continue
        seen.add(event_id)
        if end is None:
            continue
        if 0 <= now - end < CLIP_READY_SECONDS:
            # Клип ещё собирается. Без пометки: возьмём следующим кругом.
            # Отрицательный возраст (часы прыгнули назад) не задерживает:
            # ждать, пока часы «догонят» будущее, можно вечно.
            continue
        if event_id in deferred:
            # Отдыхает после неудачи — бэкофф скажет, когда снова пора.
            continue
        if not event.get("has_clip", True):
            if now - end < HAS_CLIP_GRACE_SECONDS:
                # Флаг мог ещё не выставиться — рано хоронить.
                continue
            # Записи нет — заливать нечего; помечаем, чтобы не спрашивать вечно.
            selected.append({**event, "_skip": True})
            continue
        camera = str(event.get("camera") or "")
        if config.cameras and camera not in config.cameras:
            continue
        if end - start > MAX_CLIP_SECONDS:
            # Ролик неподъёмный — но снимок этого события всё равно нужен.
            selected.append({**event, "_snapshot_only": True})
            continue
        selected.append(event)
    return selected


# Сколько весит секунда видео, пока своих замеров нет: 5 Мбит/с основного
# потока. Первые же залитые клипы заменят это число фактом.
FALLBACK_BYTES_PER_SECOND = 640_000


# Сколько событий в час считать потолком при запросе к Frigate. Замер
# 14 августа: 465 событий за сутки, то есть двадцать в час; двухсот хватает с
# десятикратным запасом. Число нужно потому, что предел выборки был жёстким
# (500) при настраиваемом окне: стоило поставить оглядку в неделю, и Frigate,
# отдающий новые первыми, возвращал бы только свежие пятьсот — остальное окно
# не заливалось бы никогда, а панель показывала бы «очередь пуста».
EVENTS_PER_HOUR_CAP = 200
EVENTS_LIMIT_CEILING = 20_000


def events_limit(lookback_hours: int) -> int:
    """Предел выборки, растущий вместе с окном оглядки."""
    return min(EVENTS_LIMIT_CEILING, max(500, int(lookback_hours) * EVENTS_PER_HOUR_CAP))


def video_seconds(events: list[dict[str, Any]]) -> float:
    """Длительность очереди в секундах видео — из неё считается вес.

    События, от которых уедет только снимок, в вес не входят: иначе оценка
    «осталось 11.7 ГБ» состояла бы в основном из ролика, который никто не
    собирается заливать.
    """
    total = 0.0
    for event in events:
        if event.get("_snapshot_only") or event.get("_skip"):
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
    — такая же беда, как «перестало». Нечисловой мусор в состоянии (ручная
    правка, кривой restore) читается как ноль, а не роняет службу в вечный
    crash-loop до `_save_state`.
    """
    def _stamp(key: str) -> float:
        try:
            return float(state.get(key) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    reference = _stamp("last_success_at") or _stamp("watching_since")
    if not reference:
        return None
    # Часы прыгнули назад (обрыв питания): отметка из «будущего» не должна
    # ни глушить тревогу на часы вперёд, ни давать ложное «recovered».
    reference = min(reference, now)
    silent_for = now - reference
    alerted_at = min(_stamp("alerted_at"), now)
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
            # connect_async, а не connect: после обрыва питания служба и брокер
            # стартуют наперегонки, и одиночный неудачный connect хоронил
            # подписку до следующего рестарта службы — будильник был мёртв
            # после каждой холодной загрузки (находка ревью 15.08). Нить paho
            # сама добивается соединения и переподключается после обрывов;
            # on_connect пересабскрайбит на каждом CONNACK.
            client.connect_async(self.host, self.port, 60)
            client.loop_start()
        except (OSError, ValueError) as exc:
            logger.info("Брокер %s:%s недоступен: пойду по таймеру (%s)", self.host, self.port, exc)
            return False
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

    def wait(self, timeout: float) -> bool:
        """Дождаться повода для круга. `True` — разбудил брокер, а не таймер.

        Разница важна: после сообщения брокера клипа у Frigate ещё нет, и цикл
        обязан выждать (`CLIP_READY_SECONDS`), иначе спросит заведомо рано.
        """
        woken = self._wake.wait(timeout)
        self._wake.clear()
        return woken

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


def _load_state(path: Path) -> dict[str, Any] | None:
    """Состояние с диска; `None` — файл есть, но не читается.

    Разница принципиальна: раньше любой сбойный `read` (транзиентный EIO,
    гонка прав) выглядел как «чистый старт», цикл работал с пустым журналом и
    в конце **перезаписывал** файл без отметок — вся тщательность
    `durable_write` обнулялась одной неудачной попыткой чтения, и всё окно
    ехало по второму разу (находка ревью 15.08). Нечитаемый файл — повод
    пропустить цикл, а не стереть журнал.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        return None


def _save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now(UTC).isoformat()
    write_text_durably(path, json.dumps(state, ensure_ascii=False), mode=0o600)


def _save_state_quietly(path: Path, state: dict[str, Any]) -> None:
    """Сохранить, не роняя службу: полный диск — не повод для crash-loop.

    `ENOSPC` на общем томе превращал перезапись состояния в вечный цикл
    «залить клип → не суметь отметить → умереть → перезалить тот же клип» —
    платная петля при уже мёртвом диске (находка ревью 15.08). Несохранённый
    прогресс дешевле: отметки допишутся, когда диск оживёт.
    """
    try:
        _save_state(path, state)
    except OSError as exc:
        logger.warning("Состояние не сохранилось (%s) — продолжаю без отметки", exc)


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


# Как часто мерить занятое в бакете. Перечисление — единственное, что ключ
# умеет кроме записи; раз в час на ~14 тысячах объектов (30 дней хранения) —
# это полтора десятка страниц листинга, незаметно и для канала, и для счёта.
MEASURE_EVERY_SECONDS = 3600.0
# На сколько возраст старейшего объекта может превышать срок хранения, прежде
# чем это станет поводом сказать вслух. Lifecycle Яндекса убирает раз в сутки
# и не в момент истечения; трое суток люфта отделяют «уборка запаздывает» от
# «уборка не настроена».
RETENTION_SLACK_DAYS = 3


def _measure_cloud_usage(client: Any, config: CloudBackupConfig, state: dict[str, Any]) -> None:
    """Сколько лежит в бакете — и работает ли уборка.

    Служба удалять не умеет по построению (украденный ключ не должен стирать
    архив), уборку делает lifecycle бакета — но его настраивает владелец
    руками, и забытая настройка означала бы бесконечный рост и счёт. Поэтому
    выгрузчик раз в час перечисляет свой префикс и честно говорит, если
    старейший объект живёт дольше срока хранения: значит, lifecycle не работает.
    """
    cloud = dict(state.get("cloud") or {})
    measured_at = float(cloud.get("measured_at") or 0.0)
    now = time.time()
    if 0 <= now - measured_at < MEASURE_EVERY_SECONDS:
        return
    try:
        total_bytes = 0
        objects = 0
        oldest: Any = None
        token: str | None = None
        while True:
            kwargs = {"Bucket": config.bucket, "Prefix": f"{config.prefix}/", "MaxKeys": 1000}
            if token:
                kwargs["ContinuationToken"] = token
            page = client.list_objects_v2(**kwargs)
            for item in page.get("Contents") or []:
                total_bytes += int(item.get("Size") or 0)
                objects += 1
                stamp = item.get("LastModified")
                if stamp is not None and (oldest is None or stamp < oldest):
                    oldest = stamp
            if not page.get("IsTruncated"):
                break
            token = page.get("NextContinuationToken")
        cloud = {
            "measured_at": now,
            "bytes": total_bytes,
            "objects": objects,
            "oldest_at": oldest.isoformat() if oldest is not None else None,
        }
        if oldest is not None and config.retention_days:
            age_days = (now - oldest.timestamp()) / 86400
            cloud["retention_working"] = age_days <= config.retention_days + RETENTION_SLACK_DAYS
            if not cloud["retention_working"]:
                logger.warning(
                    "Старейший объект живёт %.0f дн. при сроке %d: lifecycle не убирает",
                    age_days, config.retention_days,
                )
        state["cloud"] = cloud
    except Exception as exc:  # noqa: BLE001 — замер вторичен, выгрузку не ронять
        logger.info("Замер бакета не удался: %s", str(exc)[:120])


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
        loaded = _load_state(state_path)
        if loaded is None:
            # Файл есть, но не читается. Работать «с чистого листа» нельзя:
            # в конце цикла чистый лист перезаписал бы журнал залитого.
            logger.warning("Состояние не читается — пропускаю цикл, журнал не трогаю")
            if once:
                return
            trigger.wait(POLL_SECONDS)
            continue
        state = loaded
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
            _save_state_quietly(state_path, state)
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
            limit = events_limit(config.lookback_hours)
            events = _fetch_json(
                f"{settings.gate_api_url.rstrip('/')}/api/events?"
                f"after={after:.0f}&limit={limit}"
            )
            if not isinstance(events, list):
                raise RuntimeError("Frigate вернул не список событий")  # noqa: TRY004 — это отказ API, не тип
            if len(events) >= limit:
                # Frigate отдаёт новые первыми, поэтому упёршийся предел
                # означает молча отрезанный хвост окна: «очередь пуста» тогда
                # врало бы про самые старые события, ради которых окно и
                # расширяли. Молчать об урезанном охвате нельзя.
                logger.warning(
                    "Frigate вернул ровно предел (%d): хвост окна в %d ч не виден",
                    limit, config.lookback_hours,
                )
            # Журнал — список, а не множество: обрезка должна выбрасывать
            # самые старые отметки, а обход set() отдаёт их в произвольном
            # порядке и жертвует случайными (находка ревью 14.08). Потолок —
            # не меньше двух пределов выборки: журнал в 4000 при выборке в
            # 20 000 ронял отметки о свежем и перезаливал излишек по кругу
            # (находка ревью 15.08).
            remember_cap = max(REMEMBER_IDS, 2 * limit)
            uploaded_order: list[str] = list(state.get("uploaded") or [])
            uploaded = set(uploaded_order)

            now = time.time()
            attempts: dict[str, dict[str, float]] = {
                str(key): _attempt_entry(value)
                for key, value in dict(state.get("attempts") or {}).items()
            }
            waiting = deferred_ids(attempts, now)
            sent_bytes = 0
            clip_bytes = 0
            sent_events = 0
            failed = 0
            batch = eligible_events(events, config, uploaded, now=now, deferred=waiting)
            if config.cameras and events and not batch and not waiting:
                cameras_seen = sorted({str(e.get("camera") or "?") for e in events})
                if not set(cameras_seen) & set(config.cameras):
                    # Фильтр не совпал ни с одной камерой окна — прецедент
                    # `test-cam` против `test_cam`. Раньше это выглядело как
                    # «Очередь пуста» со свежим успехом, и тревога молчала
                    # неделями при пустом бакете (находка ревью 15.08).
                    raise RuntimeError(
                        f"Фильтр камер {config.cameras} не совпал ни с одной "
                        f"из окна: {cameras_seen}"
                    )
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
                _save_state_quietly(state_path, state)
            for position, event in enumerate(batch):
                event_id = str(event["id"])
                if event.get("_skip"):
                    _remember(uploaded, uploaded_order, event_id)
                    attempts.pop(event_id, None)
                    continue
                if event.get("_snapshot_only"):
                    # Клип длиннее потолка: берём только снимок и идём дальше,
                    # иначе одно такое событие держит очередь мёртвой хваткой.
                    try:
                        with _heartbeat(notifier):
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
                        attempts.pop(event_id, None)
                        state["last_success_at"] = time.time()
                        # Доставка есть доставка: снимок-only событие раньше не
                        # считалось, и цикл из одних снимков писал «не удалось
                        # залить ни одного» при живом канале.
                        sent_events += 1
                        state["uploaded"] = uploaded_order[-remember_cap:]
                        state["attempts"] = attempts
                        _save_state_quietly(state_path, state)
                        logger.info(
                            "Событие %s длиннее %.0f мин: залит только снимок",
                            event_id, MAX_CLIP_SECONDS / 60,
                        )
                    except Exception as exc:  # noqa: BLE001
                        entry = attempts.get(event_id) or _attempt_entry(None)
                        entry = {"count": entry["count"] + 1, "last_at": time.time()}
                        attempts[event_id] = entry
                        failed += 1
                        logger.info("Снимок %s не ушёл: %s", event_id, str(exc)[:120])
                    notifier.watchdog()
                    continue
                # Одно ядовитое событие не должно ронять цикл: Frigate может
                # собирать клип часового события дольше любого таймаута.
                # 14 августа такой клип оборвал догоняющую выгрузку, а
                # застрявший сторож перезапустил службу целиком.
                try:
                    # Сторож кормится изнутри: клип у потолка длительности
                    # уезжает восемь минут и дольше, а между событиями пинга
                    # он бы не дождался — 14 августа именно так systemd убил
                    # выгрузчик посреди работающей заливки.
                    with _heartbeat(notifier):
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
                    entry = attempts.get(event_id) or _attempt_entry(None)
                    entry = {"count": entry["count"] + 1, "last_at": time.time()}
                    attempts[event_id] = entry
                    failed += 1
                    # Неудача не хоронит событие никогда — только отодвигает
                    # следующую попытку (см. retry_delay). Списание убрано:
                    # при однонаправленном ключе оно необратимо, а оба его
                    # сигнала — счётчик попыток и «живость» канала — на рваной
                    # связи оказались ложными (находка ревью 15.08).
                    logger.info(
                        "Событие %s отложено (попытка %.0f, следующая через %.0f с): %s",
                        event_id, entry["count"], retry_delay(int(entry["count"])),
                        str(exc)[:120],
                    )
                    notifier.watchdog()
                    continue
                attempts.pop(event_id, None)
                sent_bytes += len(clip)
                clip_bytes += len(clip)
                done_seconds += max(
                    0.0,
                    float(event.get("end_time") or 0) - float(event.get("start_time") or 0),
                )
                if done_seconds > 0:
                    # Свой замер точнее умолчания: битрейт зависит от камеры и
                    # сцены, а не от паспортных цифр. Снимки в замер не входят —
                    # они завышали вес секунды видео.
                    rate = clip_bytes / done_seconds
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
                try:
                    with _heartbeat(notifier):
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
                # Отметка о заливке пишется на диск сразу же, а не раз в
                # двадцать клипов. Цикл убивают: 14 августа сторож трижды
                # прислал SIGABRT посреди догона, и каждая смерть стоила до
                # двадцати перезалитых клипов — при версионировании с
                # блокировкой это оплаченные запертые версии. Одна durable
                # запись на событие при четырёх сотнях событий в сутки не
                # стоит ничего.
                state["uploaded"] = uploaded_order[-remember_cap:]
                state["attempts"] = attempts
                _save_state_quietly(state_path, state)
                notifier.watchdog()
            # Память о попытках живёт вместе с окном оглядки: событие, ушедшее
            # из окна, Frigate больше не отдаст — его запись только копила бы
            # файл состояния навсегда (находка ревью 15.08).
            window_ids = {str(e.get("id") or "") for e in events}
            attempts = {key: value for key, value in attempts.items() if key in window_ids}
            state["uploaded"] = uploaded_order[-remember_cap:]
            state["attempts"] = attempts
            today = datetime.now(UTC).strftime("%Y-%m-%d")
            if state.get("day") != today:
                state["day"] = today
                state["day_events"] = 0
                state["day_bytes"] = 0
            state["day_events"] = int(state.get("day_events") or 0) + sent_events
            state["day_bytes"] = int(state.get("day_bytes") or 0) + sent_bytes
            state.pop("progress", None)
            # Отложенных считаем вместе: и упавших в этом цикле, и отдыхающих
            # по бэкоффу — для панели это одна очередь «ещё не в облаке».
            state["pending"] = len(attempts)
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
                day_line = (
                    f"за сутки {state['day_events']} событий, "
                    f"{state['day_bytes'] / 1e6:.0f} МБ"
                )
                if sent_events:
                    tail = f", отложено {len(attempts)}" if attempts else ""
                    state["message"] = (
                        f"За сутки: {state['day_events']} событий, "
                        f"{state['day_bytes'] / 1e6:.0f} МБ{tail}"
                    )
                elif attempts:
                    state["message"] = (
                        f"Жду повтора: отложено {len(attempts)}, {day_line}"
                    )
                else:
                    # Заливать было нечего — и это надо сказать прямо. Прежняя
                    # строка «За сутки: N событий, X МБ» одинаково выглядела и
                    # при пустой очереди, и при вставшей: 14 августа ровно из-за
                    # этого ноль на графике бакета прочитался как поломка, хотя
                    # догон просто закончился.
                    state["message"] = f"Очередь пуста: всё залито, {day_line}"
                if len(events) >= limit:
                    # Урезанный охват не должен прятаться в journalctl:
                    # «очередь пуста» при отрезанном хвосте — враньё.
                    state["message"] += f" · охват урезан пределом {limit}"
                if sent_events or not attempts:
                    # Отметка здоровья — только когда цикл что-то доставил или
                    # честно убедился, что доставлять нечего. Цикл, где вся
                    # очередь отдыхает по бэкоффу, ничего не доказал — свежая
                    # отметка глушила бы тревогу при неделями стоящих клипах.
                    state["last_success_at"] = time.time()
                    state.pop("last_error", None)
            if sent_events:
                logger.info(
                    "Залито %d событий, %.1f МБ", sent_events, sent_bytes / 1e6
                )
            _measure_cloud_usage(client, config, state)
        except Exception as exc:  # noqa: BLE001 — облако и Frigate падают, служба нет
            state["status"] = "error"
            state["last_error"] = str(exc)[:300]
            state["message"] = f"Ошибка выгрузки: {str(exc)[:120]}"
            logger.warning("Цикл выгрузки не удался: %s", exc)
        _notify_about_silence(state, alert, now=time.time())
        _save_state_quietly(state_path, state)
        if once:
            return
        if trigger.wait(POLL_SECONDS):
            # Разбудил брокер: событие закончилось только что, и клипа у
            # Frigate ещё нет. Выждать дешевле, чем спросить рано — ранний
            # вопрос стоил гарантированного отказа на каждом событии.
            time.sleep(CLIP_READY_SECONDS)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    run(Settings())


if __name__ == "__main__":
    main()
