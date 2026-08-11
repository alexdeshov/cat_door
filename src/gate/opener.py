"""Импульс на реле калитки. Единственное место в проекте, которое пишет на плату.

Разрешено письменно владельцем 2026-08-11 и только в этих границах
(`AGENTS.md`): замыкание входа «кнопка выхода» контроллера Novicam SX120KW на
заданном канале, и ничего больше. Замком распоряжается сам SX120KW — в цепь
замка система не вмешивается, поэтому отказ узла означает «не открывается по
лицу», а не «калитка не работает».

**Импульс задаётся одной командой.** У платы есть регистр «мигнуть»:
`0x0200 + канал`, значение — время в сотнях миллисекунд. Одна запись замыкает
контакт на заданный срок и **сама его размыкает**.

Пара команд «включить» и через полсекунды «выключить» этим свойством не
обладает: если процесс умрёт, сеть моргнёт или узел уйдёт в перезагрузку между
ними, калитка останется разомкнутой навсегда и молча. Ради этого свойства
регистр «мигнуть» здесь и используется, хотя команда «включить» проще.

Соединение живёт ровно на время команды. Постоянный сокет держит монитор
дверцы внутри веб-службы, и второе постоянное соединение к тому же порту
шлюза — лишний повод для отказа; короткое переживает и перезапуск монитора.
"""

from __future__ import annotations

import logging
import socket
from dataclasses import dataclass

from src.web.door_monitor import modbus_crc16

logger = logging.getLogger("catdoor.gate-opener")

# Адрес платы на шине фиксирован заводскими настройками.
DEVICE_ADDRESS = 0x01
WRITE_REGISTER = 0x06
# База регистров «мигнуть». Канал 1 — это 0x0200, канал 2 — 0x0201.
FLASH_REGISTER_BASE = 0x0200
# Единица значения регистра — сотня миллисекунд.
PULSE_UNIT_MS = 100
# Потолок импульса. Полсекунды хватает контроллеру с запасом, а долгое
# замыкание кнопки выхода некоторые контроллеры считают удержанием.
MAX_PULSE_MS = 5000


def flash_register(channel: int) -> int:
    """Регистр «мигнуть» для канала. Каналы человеку — с единицы, плате — с нуля."""
    if not 1 <= channel <= 8:
        raise ValueError(f"канал реле вне диапазона платы: {channel}")
    return FLASH_REGISTER_BASE + channel - 1


def write_register_frame(register: int, value: int) -> bytes:
    """Команда 0x06 «записать один регистр»."""
    body = (
        bytes((DEVICE_ADDRESS, WRITE_REGISTER))
        + register.to_bytes(2, "big")
        + value.to_bytes(2, "big")
    )
    return body + modbus_crc16(body)


def parse_write_reply(reply: bytes, register: int, value: int) -> None:
    """Проверить эхо платы. Молчаливое «наверное, дошло» здесь недопустимо.

    Плата отвечает на 0x06 копией запроса. Несовпадение означает, что команду
    исполнил кто-то другой или не исполнил никто, — а в журнал в этот момент
    собирается строка «открыл».
    """
    expected = write_register_frame(register, value)
    if len(reply) < len(expected):
        raise ValueError(f"ответ короче эха: {reply.hex()}")
    frame = reply[: len(expected)]
    if modbus_crc16(frame[:-2]) != frame[-2:]:
        raise ValueError(f"CRC не сходится: {reply.hex()}")
    if frame != expected:
        raise ValueError(f"плата ответила не тем: {reply.hex()}")


@dataclass(frozen=True)
class PulseResult:
    ok: bool
    detail: str
    pulse_ms: int


@dataclass
class RelayOpener:
    """Отправитель импульса. Ничего не хранит между вызовами."""

    host: str
    port: int = 4196
    timeout: float = 3.0

    def pulse(self, channel: int, milliseconds: int) -> PulseResult:
        """Замкнуть контакт на заданное время. Исключений не бросает.

        Ошибка — это строка для журнала и для человека у калитки, а не
        падение службы: она обязана пережить и выключенную плату, и обрыв.
        """
        if not self.host:
            return PulseResult(False, "Адрес платы не задан", 0)
        duration = max(PULSE_UNIT_MS, min(int(milliseconds), MAX_PULSE_MS))
        units = duration // PULSE_UNIT_MS
        try:
            register = flash_register(channel)
        except ValueError as exc:
            return PulseResult(False, str(exc), 0)
        try:
            with socket.create_connection((self.host, self.port), timeout=self.timeout) as link:
                link.settimeout(self.timeout)
                link.sendall(write_register_frame(register, units))
                parse_write_reply(link.recv(64), register, units)
        except (OSError, TimeoutError) as exc:
            logger.warning("Импульс на канал %d не ушёл: %s", channel, exc)
            return PulseResult(False, f"Плата не ответила: {exc}", duration)
        except ValueError as exc:
            logger.warning("Плата ответила неожиданно: %s", exc)
            return PulseResult(False, f"Плата ответила неожиданно: {exc}", duration)
        logger.info("Импульс %d мс на канал %d ушёл", duration, channel)
        return PulseResult(True, f"Импульс {duration} мс на канал {channel}", duration)
