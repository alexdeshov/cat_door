"""Открытие дверцы по опознанному коту — со стороны рекордера.

Решение живёт здесь, а не в службе калитки, по одной причине: рекордер —
единственный, кто видит кота у дверцы **сейчас**. Опознание прохода попадает в
базу, когда проход уже закончился; для журнала это нормально, для защёлки
поздно.

Плату при этом рекордер не трогает. Он кладёт просьбу файлом, а исполняет её
`catdoor-gate-access` — та же дорога, которой ходит кнопка в админке. Двум
процессам в один TCP-порт шлюза лезть нельзя, и заводить второго писателя ради
удобства мы не станем.

**Задержка не блокирует цикл.** Кот, узнанный на подходе, доходит до дверцы
секунд за пять; ждать эти секунды `sleep`-ом означало бы остановить разбор
кадров обеих камер. Поэтому открытие назначается на будущее, а цикл продолжает
работать и просто заглядывает сюда на каждом кадре.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from src.common.cat_door_config import CatDoorConfig
from src.common.cat_door_decision import (
    DECISION_COOLDOWN,
    DECISION_DRY_RUN,
    DECISION_MODE_OFF,
    DECISION_WRONG_CAMERA,
    CatDoorDecision,
    decide,
)
from src.common.gate_open_request import (
    TARGET_CAT_DOOR,
    OpenRequest,
    boot_id,
    open_request_path,
    save_open_request,
)

logger = logging.getLogger("catdoor.cat-door")

# Об этих отказах молчим: они случаются на каждом втором кадре и залили бы лог,
# ничего не сообщив. Остальные — редкие и объясняют человеку, почему не открылось.
QUIET_DECISIONS = frozenset({DECISION_MODE_OFF, DECISION_WRONG_CAMERA, DECISION_COOLDOWN})


class CatDoorAutomation:
    """Назначает и подаёт открытие дверцы. Одно назначенное открытие за раз.

    Второе назначение поверх первого означало бы, что кот, попавший в кадр
    десять раз подряд, получит десять импульсов подряд, — а держится защёлка
    пять секунд, и очередь таких удержаний греет катушку.
    """

    def __init__(
        self,
        data_dir: Path,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._request_path = open_request_path(data_dir)
        self._clock = clock
        self._open_at: float | None = None
        self._pending_name: str | None = None
        self._last_opened_at: float | None = None
        self._last_reason: str | None = None

    @property
    def pending(self) -> bool:
        return self._open_at is not None

    def observe(
        self,
        *,
        name: str | None,
        confidence: float | None,
        role: str,
        config: CatDoorConfig,
    ) -> CatDoorDecision:
        """Посмотреть на кадр и, если пора, назначить открытие."""
        now = self._clock()
        decision = decide(
            name=name,
            confidence=confidence,
            role=role,
            config=config,
            now=now,
            last_opened_at=self._last_opened_at,
        )
        if decision.code not in QUIET_DECISIONS and decision.reason != self._last_reason:
            # Одна и та же причина подряд — это тот же кот в тех же кадрах.
            # Печатаем смену причины, а не каждый кадр.
            logger.info("Дверца: %s", decision.reason)
            self._last_reason = decision.reason
        if decision.code == DECISION_DRY_RUN:
            return decision
        if not decision.opens or self.pending:
            return decision
        self._open_at = now + config.open_delay_seconds
        self._pending_name = name
        logger.info(
            "Дверца: открою через %.0f с — %s",
            config.open_delay_seconds,
            name,
        )
        return decision

    def tick(self, config: CatDoorConfig) -> bool:
        """Подать назначенное открытие, если время пришло. `True` — подали.

        Зовётся на каждом кадре и почти всегда ничего не делает: это дешевле и
        честнее, чем отдельный поток с таймером, который пришлось бы гасить при
        каждой смене настроек.
        """
        if self._open_at is None:
            return False
        now = self._clock()
        if now < self._open_at:
            return False
        # Настройки могли выключить автоматику за те секунды, что мы ждали.
        # Открывать после этого нельзя: человек нажал «выключить» именно чтобы
        # дверца не открывалась.
        if config.mode != "on":
            logger.info("Дверца: автоматика выключена, пока я ждал — не открываю")
            self._forget()
            return False
        request = OpenRequest(
            id=uuid4().hex,
            requested_at=now,
            target=TARGET_CAT_DOOR,
            source="auto",
            boot_id=boot_id(),
            actor=self._pending_name or "",
        )
        try:
            save_open_request(self._request_path, request)
        except OSError as exc:
            logger.warning("Просьба открыть дверцу не записана: %s", exc)
            self._forget()
            return False
        logger.info("Дверца: прошу открыть — %s", self._pending_name)
        self._last_opened_at = now
        self._forget()
        return True

    def _forget(self) -> None:
        self._open_at = None
        self._pending_name = None
