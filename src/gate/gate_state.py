"""Состояние створки калитки от классификатора Frigate — через MQTT.

Обученная владельцем модель смотрит на калитку камерой регистратора и на каждое
изменение публикует `Открыта` или `Закрыта`. Это замена геркону, от которого
отказались: тот сказал бы то же самое, но проводом.

**Состояние не запоминается брокером.** Проверено 2026-08-12: свежая подписка не
получает ничего, пока створка не шевельнётся. Отсюда два следствия, и оба важнее,
чем кажется.

Во-первых, после запуска службы состояние неизвестно — и это надо говорить
прямо, а не подставлять «закрыта». «Не знаю» и «закрыта» ведут к разным
решениям: на первом нельзя основывать вывод «калитка не открылась, значит замок
заклинил».

Во-вторых, известное состояние стареет. Створку могли открыть, пока подписка
лежала, и тогда наше «закрыта» — воспоминание, а не факт. Поэтому у состояния
есть срок годности.

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
# Сколько живёт последнее известное состояние. Створка меняется редко, но
# «закрыта пять минут назад» и «закрыта сейчас» — разные утверждения, и
# основывать на первом вывод о заклинившем замке нельзя.
FRESH_FOR_SECONDS = 300.0


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
        clock: Callable[[], float] = time.time,
        client_id: str = "catdoor-gate-access",
    ) -> None:
        self.host = host
        self.port = port
        self.topic = topic
        self._clock = clock
        self._client_id = client_id
        self._lock = threading.Lock()
        self._state = GateState(UNKNOWN)
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
            client.connect(self.host, self.port, 60)
        except OSError as exc:
            logger.warning("Брокер %s:%s недоступен: %s", self.host, self.port, exc)
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
        logger.info("Слушаю состояние створки: %s", self.topic)

    def _on_disconnect(self, *_: object) -> None:
        # Потеряв связь, перестаём утверждать что бы то ни было. Отметка о
        # закрытии тоже уходит: за время без связи створку могли открыть.
        with self._lock:
            self._state = GateState(UNKNOWN)
        logger.warning("Связь с брокером потеряна: состояние створки неизвестно")

    def _on_message(self, _client: object, _userdata: object, message: object) -> None:
        payload = getattr(message, "payload", b"")
        text = payload.decode("utf-8", "replace").strip().lower()
        value = _LABELS.get(text)
        if value is None:
            logger.warning("Классификатор прислал незнакомое состояние: %r", text[:40])
            return
        now = self._clock()
        with self._lock:
            closed_at = now if value == CLOSED else self._state.closed_at
            self._state = GateState(value, now, closed_at)
        logger.info("Створка: %s", text)

    def snapshot(self) -> GateState:
        """Текущее знание о створке. Устаревшее превращается в «не знаю»."""
        with self._lock:
            state = self._state
        if not state.known or state.at is None:
            return GateState(UNKNOWN)
        if self._clock() - state.at > FRESH_FOR_SECONDS:
            return GateState(UNKNOWN, closed_at=state.closed_at)
        return state
