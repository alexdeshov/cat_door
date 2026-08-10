"""Положение кошачьей дверцы и состояние замка — по Modbus с реле-контроллера.

Контроллер Waveshare Modbus POE ETH Relay (B) живёт в изолированной камерной
сети и говорит на Modbus RTU поверх TCP (заводской прозрачный режим, порт
4196). Геркон на дверце заведён на цифровой вход платы, замок — на канал реле.

**Опрос непрерывный, а не по запросу страницы.** Положение дверцы — это не
мгновенное значение входа, а его устойчивость: геркон замкнут непрерывно
несколько секунд — дверца закрыта; разомкнут или дребезжит — открыта (кот
проходит, створка качается). Мгновенный опрос по HTTP-запросу дал бы «закрыта»
ровно в момент, когда створка пролетает мимо магнита.

Поток только читает: команд включения реле здесь нет и быть не должно —
дверь по правилам проекта не управляется.
"""

from __future__ import annotations

import socket
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

# Позиции дверцы, какими их видит страница. «unknown» — честное состояние
# первых секунд после старта и любых перебоев связи: врать «закрыта», не
# набрав окна наблюдений, хуже, чем признаться, что данных ещё нет.
POSITION_CLOSED = "closed"
POSITION_OPEN = "open"
POSITION_UNKNOWN = "unknown"


def modbus_crc16(payload: bytes) -> bytes:
    """CRC16/Modbus, младший байт вперёд — как на проводе."""
    crc = 0xFFFF
    for byte in payload:
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return bytes((crc & 0xFF, crc >> 8))


def read_discrete_inputs_frame(first: int, count: int) -> bytes:
    """Команда 0x02 «читать входы». Адрес устройства у платы фиксирован: 0x01."""
    body = bytes((0x01, 0x02)) + first.to_bytes(2, "big") + count.to_bytes(2, "big")
    return body + modbus_crc16(body)


def read_coils_frame(first: int, count: int) -> bytes:
    """Команда 0x01 «читать реле»."""
    body = bytes((0x01, 0x01)) + first.to_bytes(2, "big") + count.to_bytes(2, "big")
    return body + modbus_crc16(body)


def parse_bits_reply(reply: bytes, expected_function: int) -> int:
    """Достать байт битов из ответа 0x01/0x02.

    Проверяются адрес, функция и CRC: контроллер делит порт с чем угодно, и
    молча принять чужой или битый ответ значило бы показать «дверца закрыта»
    по мусору из сети.
    """
    if len(reply) < 6:
        raise ValueError(f"ответ короче минимального: {reply.hex()}")
    if reply[0] != 0x01 or reply[1] != expected_function:
        raise ValueError(f"чужой ответ: {reply.hex()}")
    length = reply[2]
    frame = reply[: 3 + length + 2]
    if len(frame) != 3 + length + 2:
        raise ValueError(f"ответ оборван: {reply.hex()}")
    if modbus_crc16(frame[:-2]) != frame[-2:]:
        raise ValueError(f"CRC не сходится: {reply.hex()}")
    return frame[3]


@dataclass
class DoorSnapshot:
    """То, что уезжает в API. Всё вычислено на момент снятия."""

    available: bool
    position: str
    position_stable_seconds: float | None
    reed_closed: bool | None
    lock_relay_on: bool | None
    error: str | None = None


@dataclass
class DoorMonitor:
    """Фоновый опрос контроллера и вывод положения дверцы из окна наблюдений."""

    host: str
    port: int = 4196
    reed_input: int = 1
    lock_relay: int = 1
    # Порог устойчивости из требования владельца: «геркон соединён 2–3 секунды —
    # закрыта». Берём середину; правится настройкой, а не правкой кода.
    stable_seconds: float = 2.5
    poll_interval: float = 0.5
    timeout: float = 3.0
    clock: Callable[[], float] = time.monotonic
    _samples: deque[tuple[float, bool]] = field(default_factory=lambda: deque(maxlen=64))
    _lock_relay_on: bool | None = field(default=None)
    _last_success: float | None = field(default=None)
    _last_error: str | None = field(default=None)
    _sock: socket.socket | None = field(default=None, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)
    _state_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="catdoor-door-monitor"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.timeout + 1)
        self._close()

    # -- сеть -------------------------------------------------------------

    def _close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _exchange(self, frame: bytes, expected_function: int) -> int:
        # Соединение живёт между опросами: плата держит TCP-сессию, а два
        # рукопожатия в секунду на ровном месте — лишний шум в камерной сети.
        if self._sock is None:
            self._sock = socket.create_connection(
                (self.host, self.port), timeout=self.timeout
            )
            self._sock.settimeout(self.timeout)
        self._sock.sendall(frame)
        reply = self._sock.recv(64)
        if not reply:
            raise OSError("контроллер закрыл соединение")
        return parse_bits_reply(reply, expected_function)

    def poll_once(self) -> None:
        """Один опрос: вход геркона и состояние реле замка."""
        try:
            inputs = self._exchange(
                read_discrete_inputs_frame(self.reed_input - 1, 1), 0x02
            )
            coils = self._exchange(read_coils_frame(self.lock_relay - 1, 1), 0x01)
        except (OSError, ValueError) as error:
            self._close()
            with self._state_lock:
                self._last_error = str(error)
            return
        with self._state_lock:
            self._samples.append((self.clock(), bool(inputs & 1)))
            self._lock_relay_on = bool(coils & 1)
            self._last_success = self.clock()
            self._last_error = None

    def _run(self) -> None:
        while not self._stop.wait(self.poll_interval):
            self.poll_once()

    # -- вывод состояния ---------------------------------------------------

    def snapshot(self) -> DoorSnapshot:
        now = self.clock()
        with self._state_lock:
            samples = list(self._samples)
            lock_on = self._lock_relay_on
            last_success = self._last_success
            error = self._last_error
        # Недоступность — это «давно не было удачного опроса», а не «последний
        # опрос неудачен»: одиночный сетевой чих не должен мигать панелью.
        if last_success is None or now - last_success > max(5.0, self.poll_interval * 6):
            return DoorSnapshot(
                available=False,
                position=POSITION_UNKNOWN,
                position_stable_seconds=None,
                reed_closed=None,
                lock_relay_on=None,
                error=error,
            )
        reed_closed = samples[-1][1] if samples else None
        position, stable_for = self._position(samples, now)
        return DoorSnapshot(
            available=True,
            position=position,
            position_stable_seconds=stable_for,
            reed_closed=reed_closed,
            lock_relay_on=lock_on,
            error=None,
        )

    def _position(
        self, samples: list[tuple[float, bool]], now: float
    ) -> tuple[str, float | None]:
        """Положение по требованию владельца.

        Замкнут непрерывно ≥ stable_seconds — закрыта. Разомкнут — открыта
        сразу, без выдержки: разомкнувшийся геркон однозначен, створка ушла от
        магнита. Замкнут, но ещё не выдержал окно (дребезг, кот в проёме) —
        тоже «открыта»: часто меняющийся сигнал и есть открытая дверца.
        """
        if not samples:
            return POSITION_UNKNOWN, None
        if not samples[-1][1]:
            changed_at = self._last_change(samples)
            return POSITION_OPEN, now - changed_at
        changed_at = self._last_change(samples)
        stable_for = now - changed_at
        if stable_for >= self.stable_seconds:
            return POSITION_CLOSED, stable_for
        return POSITION_OPEN, stable_for

    @staticmethod
    def _last_change(samples: list[tuple[float, bool]]) -> float:
        current = samples[-1][1]
        changed_at = samples[0][0]
        for timestamp, value in reversed(samples):
            if value != current:
                break
            changed_at = timestamp
        return changed_at
