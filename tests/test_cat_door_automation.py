"""Проверки исполнителя автоматики дверцы.

Решение проверяется таблицей случаев в `test_cat_door_decision.py`; здесь —
всё, что вокруг него: задержка, одно назначение за раз и просьба на диске.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.common.cat_door_config import CatDoorConfig
from src.common.gate_open_request import open_request_path
from src.recorder.cat_door_automation import CatDoorAutomation


class _Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _config(**overrides: object) -> CatDoorConfig:
    values: dict[str, object] = {
        "mode": "on",
        "allowed_cats": ["Черныш"],
        "min_confidence": 0.9,
        "open_delay_seconds": 5.0,
        "cooldown_seconds": 30,
    }
    values.update(overrides)
    return CatDoorConfig.model_validate(values)


def _seen(
    automation: CatDoorAutomation,
    config: CatDoorConfig,
    *,
    times: int | None = None,
    **overrides: object,
) -> None:
    """Кот перед камерой. По умолчанию — целой серией, как того требует правило.

    Один кадр защёлку больше не отпускает: см. `confirm_frames`.
    """
    values: dict[str, object] = {"name": "Черныш", "confidence": 0.95, "role": "face"}
    values.update(overrides)
    for _ in range(config.confirm_frames if times is None else times):
        automation.observe(config=config, **values)  # type: ignore[arg-type]


def test_the_latch_waits_for_the_cat_to_walk_up(tmp_path: Path) -> None:
    """Кот, узнанный на подходе, доходит до дверцы не мгновенно.

    Открыть в момент опознания значит израсходовать окно удержания вхолостую и
    запереть дверцу перед носом.
    """
    clock = _Clock()
    automation = CatDoorAutomation(tmp_path, clock=clock)
    config = _config(open_delay_seconds=5.0)

    _seen(automation, config)
    early = automation.tick(config)
    clock.now += 5.0
    on_time = automation.tick(config)

    assert not early
    assert on_time
    request = json.loads(open_request_path(tmp_path).read_text(encoding="utf-8"))
    assert request["target"] == "cat_door"
    assert request["source"] == "auto"
    assert request["actor"] == "Черныш"


def test_the_delay_does_not_block_the_loop(tmp_path: Path) -> None:
    """Ожидание `sleep`-ом остановило бы разбор кадров обеих камер."""
    clock = _Clock()
    automation = CatDoorAutomation(tmp_path, clock=clock)
    config = _config()

    _seen(automation, config)

    # Ни одна из этих проверок не должна ничего ждать: время двигаем мы сами.
    assert [automation.tick(config) for _ in range(3)] == [False, False, False]
    assert automation.pending


def test_one_appearance_gives_one_opening(tmp_path: Path) -> None:
    """Кот в кадре десять раз подряд — это один кот, а не десять удержаний.

    Защёлка держится пять секунд, и очередь таких удержаний греет катушку.
    """
    clock = _Clock()
    automation = CatDoorAutomation(tmp_path, clock=clock)
    config = _config()

    for _ in range(10):
        _seen(automation, config)
        clock.now += 0.2
    clock.now += 10
    opened = [automation.tick(config) for _ in range(5)]

    assert opened.count(True) == 1


def test_the_pause_applies_to_the_next_appearance(tmp_path: Path) -> None:
    """Пауза считается от поданного открытия, а не от опознания."""
    clock = _Clock()
    automation = CatDoorAutomation(tmp_path, clock=clock)
    config = _config(cooldown_seconds=30, open_delay_seconds=0.0)

    _seen(automation, config)
    automation.tick(config)
    clock.now += 10
    _seen(automation, config)
    within_pause = automation.pending
    clock.now += 25
    _seen(automation, config)
    after_pause = automation.pending

    assert not within_pause
    assert after_pause


def test_switching_off_while_waiting_cancels_the_opening(tmp_path: Path) -> None:
    """Человек выключает автоматику именно чтобы дверца не открывалась.

    Пять секунд задержки — достаточный срок, чтобы успеть передумать.
    """
    clock = _Clock()
    automation = CatDoorAutomation(tmp_path, clock=clock)

    _seen(automation, _config())
    clock.now += 10
    fired = automation.tick(_config(mode="off"))

    assert not fired
    assert not open_request_path(tmp_path).exists()
    assert not automation.pending


def test_a_dry_run_never_touches_the_latch(tmp_path: Path) -> None:
    """Режим наблюдения обязан писать в журнал и молчать на плате."""
    clock = _Clock()
    automation = CatDoorAutomation(tmp_path, clock=clock)
    config = _config(mode="dry")

    _seen(automation, config)
    clock.now += 10

    assert not automation.pending
    assert not automation.tick(config)
    assert not open_request_path(tmp_path).exists()


def test_the_terrace_camera_schedules_nothing(tmp_path: Path) -> None:
    """Условие владельца: открывать, только когда кот у самой дверцы."""
    automation = CatDoorAutomation(tmp_path, clock=_Clock())

    _seen(automation, _config(), role="side")

    assert not automation.pending


def test_one_lucky_frame_does_not_open_the_latch(tmp_path: Path) -> None:
    """Разбор 17 августа: чужой кот дал 12 кадров с близостью ≥ 0.90.

    По правилу «хватит одного кадра» дверца открылась бы ему. Серия требует,
    чтобы кот выглядел своим не мгновение, а несколько кадров подряд.
    """
    automation = CatDoorAutomation(tmp_path, clock=_Clock())
    config = _config(confirm_frames=5)

    _seen(automation, config, times=4)
    short = automation.pending
    _seen(automation, config, times=1)

    assert not short
    assert automation.pending


def test_a_gap_breaks_the_series(tmp_path: Path) -> None:
    """Опознание идёт только на кадрах с котом.

    Без проверки разрыва «пять кадров подряд» растянулись бы на кота, ушедшего
    и вернувшегося через минуту, — то есть на пять разных появлений.
    """
    clock = _Clock()
    automation = CatDoorAutomation(tmp_path, clock=clock)
    config = _config(confirm_frames=5)

    _seen(automation, config, times=4)
    clock.now += 60
    _seen(automation, config, times=1)

    assert not automation.pending


def test_the_terrace_frames_do_not_break_the_series(tmp_path: Path) -> None:
    """Кадры обеих камер идут вперемешку в одном потоке.

    Если бы кадр террасы обнулял счётчик, серия у дверцы не набралась бы
    никогда: камеры разбираются по очереди.
    """
    automation = CatDoorAutomation(tmp_path, clock=_Clock())
    config = _config(confirm_frames=3)

    for _ in range(3):
        _seen(automation, config, times=1)
        _seen(automation, config, times=1, role="side")

    assert automation.pending


def test_another_cat_starts_the_series_over(tmp_path: Path) -> None:
    """Серия — про одного кота, а не про то, что кто-то стоял у дверцы."""
    automation = CatDoorAutomation(tmp_path, clock=_Clock())
    config = _config(confirm_frames=3, allowed_cats=["Черныш", "Яся"])

    _seen(automation, config, times=2)
    _seen(automation, config, times=2, name="Яся")

    assert not automation.pending
