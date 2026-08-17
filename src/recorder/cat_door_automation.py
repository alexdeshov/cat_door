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
    DECISION_CONFIRMING,
    DECISION_COOLDOWN,
    DECISION_DRY_RUN,
    DECISION_MODE_OFF,
    DECISION_OPEN,
    DECISION_WRONG_CAMERA,
    DOOR_ROLE,
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
QUIET_DECISIONS = frozenset(
    {DECISION_MODE_OFF, DECISION_WRONG_CAMERA, DECISION_COOLDOWN, DECISION_CONFIRMING}
)

# Разрыв, после которого серия считается прерванной. Кадры с опознанием идут
# только пока кот в кадре: без этой проверки кот, ушедший и вернувшийся через
# минуту, продолжил бы копить ту же серию.
STREAK_GAP_SECONDS = 3.0


class CatDoorAutomation:
    """Назначает и подаёт открытие дверцы. Одно назначенное открытие за раз.

    Второе назначение поверх первого означало бы, что кот, попавший в кадр
    десять раз подряд, получит десять импульсов подряд, — а держится защёлка
    пять секунд, и очередь таких удержаний греет катушку.

    **Открывает серия, а не кадр.** Решение по одному кадру принимает чистая
    `decide`, но защёлка ждёт, пока подряд идущие кадры камеры дверцы назовут
    одного и того же кота `confirm_frames` раз. Один кадр — это самый удачный
    кроп из сотни: в разборе 17 августа чужой кот набрал 12 таких кадров и
    получил бы дверцу.
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
        self._streak = 0
        self._streak_name: str | None = None
        self._streak_at: float | None = None

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
        # Серию копят только кадры камеры дверцы: кадры террасы идут в том же
        # потоке вперемешку, и обнулять счётчик о них значило бы, что серия не
        # наберётся никогда.
        confirmed = (
            self._count(decision, name, now, config) if role == DOOR_ROLE else False
        )
        if decision.code in {DECISION_OPEN, DECISION_DRY_RUN} and not confirmed:
            # Кот подходит по всем условиям, но серия ещё копится. Это не
            # отказ: сухой режим обязан отчитываться тем же решением, что и
            # рабочий, иначе «открыл бы» в журнале означает не то, что будет.
            decision = CatDoorDecision(
                DECISION_CONFIRMING,
                f"{name}: {self._streak} кадр(ов) из {config.confirm_frames}",
            )
        if decision.code not in QUIET_DECISIONS and decision.reason != self._last_reason:
            # Одна и та же причина подряд — это тот же кот в тех же кадрах.
            # Печатаем смену причины, а не каждый кадр.
            logger.info("Дверца: %s", decision.reason)
            self._last_reason = decision.reason
        if not decision.opens or self.pending:
            return decision
        self._open_at = now + config.open_delay_seconds
        self._pending_name = name
        logger.info(
            "Дверца: открою через %.0f с — %s (%d кадра подряд)",
            config.open_delay_seconds,
            name,
            config.confirm_frames,
        )
        return decision

    def _count(
        self,
        decision: CatDoorDecision,
        name: str | None,
        now: float,
        config: CatDoorConfig,
    ) -> bool:
        """Продлить серию этим кадром. `True` — серия набрана.

        Разрыв во времени рвёт серию: опознание запускается только на кадрах с
        котом, поэтому пять кадров «подряд» могли бы растянуться на кота,
        ушедшего и вернувшегося через минуту.
        """
        would_open = decision.code in {DECISION_OPEN, DECISION_DRY_RUN}
        stale = (
            self._streak_at is not None
            and not 0 <= now - self._streak_at <= STREAK_GAP_SECONDS
        )
        if not would_open or name != self._streak_name or stale:
            self._streak = 0
            self._streak_name = name if would_open else None
        if not would_open:
            self._streak_at = None
            return False
        self._streak += 1
        self._streak_at = now
        return self._streak >= config.confirm_frames

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
        # Следующее открытие набирает серию заново: иначе кот, стоящий у
        # дверцы, получал бы импульс каждый кадр после паузы.
        self._streak = 0
        self._streak_name = None
        self._streak_at = None
