"""Проверки единственного места, которому позволено писать на плату."""

from __future__ import annotations

import socket
import threading

import pytest

from src.gate.opener import (
    MAX_PULSE_MS,
    RelayOpener,
    flash_address,
    flash_frame,
    parse_flash_reply,
)
from src.web.door_monitor import modbus_crc16


def _fake_board(sent: list[bytes]) -> tuple[int, threading.Thread, socket.socket]:
    """Плата, отвечающая эхом. Возвращает порт, поток и сокет для закрытия."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)

    def serve() -> None:
        link, _ = server.accept()
        with link:
            request = link.recv(64)
            sent.append(request)
            link.sendall(request)

    thread = threading.Thread(target=serve)
    thread.start()
    return server.getsockname()[1], thread, server


def test_the_channel_numbering_starts_at_one_for_people() -> None:
    """На плате каналы считаются с нуля, у человека — с единицы."""
    assert flash_address(1) == 0x0200
    assert flash_address(2) == 0x0201
    with pytest.raises(ValueError, match="диапазона"):
        flash_address(0)
    with pytest.raises(ValueError, match="диапазона"):
        flash_address(9)


def test_the_frame_matches_the_board() -> None:
    """Команда 0x05 «записать одну катушку», адрес платы 0x01, CRC в конце.

    Не 0x06: регистров хранения у платы нет вовсе — измерено на живом железе
    2026-08-11, чтение функцией 0x03 отвечает отказом по любому адресу.
    Проверенный на плате кадр «мигнуть» на полсекунды выглядит ровно так.
    """
    frame = flash_frame(0x0200, 5)

    assert frame[:6] == bytes((0x01, 0x05, 0x02, 0x00, 0x00, 0x05))
    assert modbus_crc16(frame[:-2]) == frame[-2:]
    # Этот кадр плата приняла эхом на живой проверке 2026-08-11: канал 8,
    # одна десятая секунды. Сверка с ним ловит и порядок байтов, и CRC.
    assert flash_frame(0x0207, 1).hex() == "010502070001bc73"


def test_a_refusal_is_translated_into_words() -> None:
    """`01 86 02` — это «недопустимый адрес», и человеку надо сказать именно это.

    Ровно этот кадр плата прислала 2026-08-11 на первую попытку открыть калитку
    функцией 0x06. Сообщение «ответ короче эха» было формально верным и
    совершенно бесполезным: причину пришлось искать руками.
    """
    with pytest.raises(ValueError, match="недопустимый адрес"):
        parse_flash_reply(bytes.fromhex("018602c3a1"), 0x0200, 5)
    with pytest.raises(ValueError, match="функция не поддерживается"):
        parse_flash_reply(bytes.fromhex("01850100"), 0x0200, 5)


def test_a_reply_that_is_not_the_echo_is_refused() -> None:
    """Плата отвечает на 0x06 копией запроса, и это единственное подтверждение.

    В этот момент в журнал собирается строка «открыл» — принять чужой или битый
    ответ значило бы записать открытие, которого не было.
    """
    good = flash_frame(0x0200, 5)
    parse_flash_reply(good, 0x0200, 5)

    with pytest.raises(ValueError, match="короче"):
        parse_flash_reply(good[:4], 0x0200, 5)
    with pytest.raises(ValueError, match="CRC"):
        parse_flash_reply(good[:-2] + b"\x00\x00", 0x0200, 5)
    with pytest.raises(ValueError, match="не тем"):
        parse_flash_reply(flash_frame(0x0201, 5), 0x0200, 5)


def test_a_pulse_is_one_command_that_ends_itself() -> None:
    """Пара «включить» и «выключить» оставила бы калитку разомкнутой навсегда.

    Если процесс умрёт между двумя командами, размыкать будет некому. Регистр
    «мигнуть» замыкает контакт на заданный срок и размыкает его сам, поэтому
    команда ровно одна.
    """
    sent: list[bytes] = []
    port, thread, server = _fake_board(sent)

    result = RelayOpener(host="127.0.0.1", port=port).pulse(1, 500)
    thread.join(timeout=5)
    server.close()

    assert result.ok
    assert len(sent) == 1
    assert sent[0] == flash_frame(0x0200, 5)


def test_an_unreachable_board_is_reported_not_raised() -> None:
    """Отказ — строка для журнала и для человека у калитки, а не падение службы."""
    result = RelayOpener(host="127.0.0.1", port=1, timeout=0.2).pulse(1, 500)

    assert result.ok is False
    assert "не ответила" in result.detail


def test_a_missing_address_says_so() -> None:
    assert RelayOpener(host="").pulse(1, 500).detail == "Адрес платы не задан"


def test_the_pulse_is_bounded() -> None:
    """Долгое замыкание кнопки выхода контроллеры считают удержанием.

    Опечатка в поле «импульс, мс» не должна превращаться в минуту замкнутого
    контакта: значение прижимается к потолку, а не отвергается.
    """
    sent: list[bytes] = []
    port, thread, server = _fake_board(sent)

    result = RelayOpener(host="127.0.0.1", port=port).pulse(1, 600_000)
    thread.join(timeout=5)
    server.close()

    assert result.ok
    assert result.pulse_ms == MAX_PULSE_MS
    assert sent[0] == flash_frame(0x0200, MAX_PULSE_MS // 100)
