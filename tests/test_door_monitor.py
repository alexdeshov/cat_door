"""Датчик дверцы: кадры Modbus и вывод положения из окна наблюдений."""

from __future__ import annotations

import pytest

from src.web.door_monitor import (
    POSITION_CLOSED,
    POSITION_OPEN,
    POSITION_UNKNOWN,
    DoorMonitor,
    parse_bits_reply,
    read_coils_frame,
    read_discrete_inputs_frame,
)


def test_frames_match_the_vendor_documentation() -> None:
    """Кадры сверены с официальной вики Waveshare и живой платой 2026-08-10."""
    assert read_discrete_inputs_frame(0, 8) == bytes.fromhex("01020000000879CC")
    assert read_coils_frame(0, 8) == bytes.fromhex("0101000000083DCC")


def test_replies_from_the_real_board_parse() -> None:
    """Ответы записаны с настоящего контроллера при проверке геркона."""
    assert parse_bits_reply(bytes.fromhex("010201016048"), 0x02) == 0x01
    assert parse_bits_reply(bytes.fromhex("01020100a188"), 0x02) == 0x00
    assert parse_bits_reply(bytes.fromhex("010101005188"), 0x01) == 0x00


def test_corrupt_replies_are_rejected() -> None:
    with pytest.raises(ValueError):
        parse_bits_reply(bytes.fromhex("010201016049"), 0x02)  # битый CRC
    with pytest.raises(ValueError):
        parse_bits_reply(bytes.fromhex("010201016048"), 0x01)  # чужая функция
    with pytest.raises(ValueError):
        parse_bits_reply(b"\x01\x02", 0x02)  # обрывок


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def monitor_with(samples: list[tuple[float, bool]], clock: FakeClock) -> DoorMonitor:
    monitor = DoorMonitor(host="test", clock=clock)
    for stamp, closed in samples:
        monitor._samples.append((stamp, closed))
    monitor._last_success = clock.now
    monitor._lock_relay_on = False
    return monitor


def test_reed_closed_long_enough_means_door_closed() -> None:
    clock = FakeClock()
    monitor = monitor_with([(996.0, True), (998.0, True), (999.5, True)], clock)
    shot = monitor.snapshot()
    assert shot.position == POSITION_CLOSED
    assert shot.reed_closed is True
    assert shot.available is True


def test_reed_open_means_door_open_immediately() -> None:
    clock = FakeClock()
    monitor = monitor_with([(996.0, True), (999.8, False)], clock)
    assert monitor.snapshot().position == POSITION_OPEN


def test_chattering_reed_means_door_open_even_when_currently_closed() -> None:
    """Дребезг — кот в проёме: замкнуто прямо сейчас ещё не значит закрыто."""
    clock = FakeClock()
    monitor = monitor_with(
        [(998.0, True), (998.5, False), (999.0, True), (999.5, True)], clock
    )
    shot = monitor.snapshot()
    assert shot.reed_closed is True
    assert shot.position == POSITION_OPEN


def test_stability_threshold_uses_setting() -> None:
    clock = FakeClock()
    monitor = monitor_with([(999.0, True)], clock)
    monitor.stable_seconds = 0.5
    assert monitor.snapshot().position == POSITION_CLOSED


def test_stale_polls_mean_unavailable_not_stuck_state() -> None:
    """Обрыв связи не должен вечно показывать последнее положение."""
    clock = FakeClock()
    monitor = monitor_with([(999.0, True)], clock)
    clock.now = 1030.0
    shot = monitor.snapshot()
    assert shot.available is False
    assert shot.position == POSITION_UNKNOWN
    assert shot.reed_closed is None


def test_no_samples_yet_means_unknown() -> None:
    clock = FakeClock()
    monitor = DoorMonitor(host="test", clock=clock)
    shot = monitor.snapshot()
    assert shot.available is False
    assert shot.position == POSITION_UNKNOWN


def test_poll_failure_keeps_last_good_window() -> None:
    """Одиночный сетевой чих не роняет доступность: окно ещё свежее."""
    clock = FakeClock()
    monitor = monitor_with([(997.0, True), (999.0, True)], clock)
    monitor._last_error = "connection reset"
    shot = monitor.snapshot()
    assert shot.available is True
    assert shot.error is None
