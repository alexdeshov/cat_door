"""Состояние створки калитки от классификатора Frigate — через MQTT.

Обученная владельцем модель смотрит на калитку камерой регистратора и на каждое
изменение публикует `Открыта` или `Закрыта`. Это замена геркону, от которого
отказались: тот сказал бы то же самое, но проводом.

**Состояние не запоминается брокером.** Проверено 2026-08-12: свежая подписка не
получает ничего, пока створка не шевельнётся. Поэтому после запуска службы
состояние неизвестно — и это надо говорить прямо, а не подставлять «закрыта»:
«не знаю» и «закрыта» ведут к разным решениям.

**Известному состоянию доверяем по цепочке, а не по секундомеру.** До
2026-08-14 состояние протухало за пять минут — и большую часть суток (ночь,
затишья) створка была «неизвестна», а второй шанс молчал. Решение владельца:
последнее состояние верно, пока непрерывна цепочка доставки — узел не
перезагружался (память процесса пуста после рестарта сама по себе), связь с
брокером не рвалась (disconnect сбрасывает), и Frigate не перезапускался (его
завещание `frigate/available: offline` — retained, ловится подпиской). Порвалась
любая нить — «не знаю» до первого свежего срабатывания классификатора. Событие
о створке, случившееся в дыру доставки, потеряно навсегда, и никакой срок
годности этого не чинил; целая же цепочка гарантирует, что событий не было —
классификатор публикует каждое движение створки.

По HTTP этого нет вовсе: `/api/classification/<модель>` отвечает 405, маршрут
существует только для управления обучением.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger("catdoor.gate-state")

OPEN = "open"
CLOSED = "closed"
UNKNOWN = "unknown"

# Что публикует классификатор. Названия задал владелец при обучении, и они
# по-русски — сверяем без учёта регистра и лишних пробелов.
_LABELS = {"открыта": OPEN, "закрыта": CLOSED}
# Завещание Frigate: retained-флаг его живости. `offline` означает дыру в
# доставке — движение створки за время простоя потеряно навсегда.
AVAILABILITY_TOPIC = "frigate/available"


@dataclass(frozen=True)
class GateState:
    """Что известно про створку и насколько этому можно верить."""

    value: str
    at: float | None = None
    # Когда створка в последний раз закрылась. Отсюда считается окно
    # возвращения: человек вышел, закрыл за собой и хочет обратно.
    closed_at: float | None = None

    @property
    def known(self) -> bool:
        return self.value in {OPEN, CLOSED}

    def closed_within(self, seconds: float, *, now: float) -> bool:
        """Закрывалась ли створка только что.

        Именно закрывалась, а не «сейчас закрыта»: открытая створка не требует
        открытия, а закрытая полдня назад ничего не говорит о том, кто стоит
        перед ней.
        """
        if self.closed_at is None:
            return False
        return 0 <= now - self.closed_at <= seconds


class GateStateWatcher:
    """Держит последнее состояние створки, слушая брокер в фоне.

    Подписка может отвалиться, брокер — перезапуститься. Тогда состояние
    становится неизвестным, а не остаётся прежним: молча продолжать утверждать
    «закрыта», потеряв связь, — ровно тот способ соврать, которого мы избегаем
    везде в этом проекте.
    """

    def __init__(
        self,
        host: str,
        port: int = 1883,
        *,
        topic: str = "frigate/+/classification/+",
        availability_topic: str = AVAILABILITY_TOPIC,
        clock: Callable[[], float] = time.time,
        client_id: str = "catdoor-gate-access",
    ) -> None:
        self.host = host
        self.port = port
        self.topic = topic
        self.availability_topic = availability_topic
        self._clock = clock
        self._client_id = client_id
        self._lock = threading.Lock()
        self._state = GateState(UNKNOWN)
        self._connected = False
        self._client: object | None = None

    def start(self) -> bool:
        """Подключиться и слушать в фоне. `False` — брокер не настроен или не найден."""
        if not self.host:
            logger.info("Брокер не настроен: состояние створки останется неизвестным")
            return False
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            logger.warning("paho-mqtt не установлен: состояние створки недоступно")
            return False
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=self._client_id)
        client.on_connect = self._on_connect
        client.on_message = self._on_message
        client.on_disconnect = self._on_disconnect
        try:
            # Асинхронно, а не `connect()`: после перезагрузки узла служба
            # поднимается раньше брокера, и синхронная попытка падала одним
            # «Connection refused» — навсегда. 2026-08-21 в 05:40 узел
            # перезагрузило обновление ядра, и до ручного перезапуска в 08:56
            # служба прожила три часа без створки: правило стороны разоружено,
            # второй шанс слеп. Фоновому циклу paho первая неудача не приговор —
            # он добивается соединения сам, с нарастающей паузой.
            client.connect_async(self.host, self.port, 60)
        except (OSError, ValueError) as exc:
            logger.warning("Брокер %s:%s не принят: %s", self.host, self.port, exc)
            return False
        client.loop_start()
        self._client = client
        return True

    def stop(self) -> None:
        client = self._client
        if client is not None:
            client.loop_stop()  # type: ignore[attr-defined]
            client.disconnect()  # type: ignore[attr-defined]
            self._client = None

    def _on_connect(self, client: object, *_: object) -> None:
        client.subscribe(self.topic)  # type: ignore[attr-defined]
        client.subscribe(self.availability_topic)  # type: ignore[attr-defined]
        with self._lock:
            self._connected = True
        logger.info("Слушаю состояние створки: %s", self.topic)

    def _on_disconnect(self, *_: object) -> None:
        # Потеряв связь, перестаём утверждать что бы то ни было. Отметка о
        # закрытии тоже уходит: за время без связи створку могли открыть.
        with self._lock:
            self._state = GateState(UNKNOWN)
            self._connected = False
        logger.warning("Связь с брокером потеряна: состояние створки неизвестно")

    def _on_message(self, _client: object, _userdata: object, message: object) -> None:
        payload = getattr(message, "payload", b"")
        text = payload.decode("utf-8", "replace").strip().lower()
        if getattr(message, "topic", "") == self.availability_topic:
            if text == "offline":
                # Frigate умер — движение створки за его простой потеряно.
                # Забываем всё, включая отметку закрытия: она кормит окно
                # возвращения и второй шанс, и врать им нельзя.
                with self._lock:
                    self._state = GateState(UNKNOWN)
                logger.warning("Frigate offline: состояние створки неизвестно")
            return
        value = _LABELS.get(text)
        if value is None:
            logger.warning("Классификатор прислал незнакомое состояние: %r", text[:40])
            return
        now = self._clock()
        with self._lock:
            closed_at = now if value == CLOSED else self._state.closed_at
            self._state = GateState(value, now, closed_at)
        logger.info("Створка: %s", text)

    @property
    def connected(self) -> bool:
        """Жива ли подписка на брокер. Для файла здоровья службы:
        2026-08-21 служба три часа прожила без створки, и снаружи это не было
        видно ничем — правило стороны стояло разоружённым."""
        with self._lock:
            return self._connected

    def snapshot(self) -> GateState:
        """Текущее знание о створке.

        Секундомера здесь больше нет: состояние верно, пока целы все нити
        доставки, а за их обрывами следят обработчики — disconnect брокера,
        завещание Frigate, пустая память после собственного рестарта. Возраст
        состояния остаётся в `at` для журнала, но решением не управляет.
        """
        with self._lock:
            state = self._state
        if not state.known or state.at is None:
            return GateState(UNKNOWN)
        return state
