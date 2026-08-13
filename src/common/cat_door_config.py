"""Настройки защёлки кошачьей дверцы: файл, который пишет веб и читает служба.

Та же схема, что у калитки, и по той же причине: веб механикой не
распоряжается и прав на неё не получает.

Режима три. `off` — ни одной команды на плату; `dry` — решение принимается и
пишется, защёлка не трогается; `on` — дверца открывается. Умолчание `off`:
свежая установка не открывает дверцу никому, пока человек не выбрал, кому
именно.

Отличие от калитки одно, но существенное: там сухой контакт замыкает кнопку
чужого контроллера, а здесь реле подаёт питание на катушку. Катушка греется,
поэтому время удержания ограничено, а команда выбрана такая, которая снимает
питание сама.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from src.common.durable_write import write_text_durably

logger = logging.getLogger("catdoor.cat-door-config")

# Потолок удержания. Защёлки на 12 В обычно кратковременного включения, и
# «подержать подольше, чтобы точно прошёл» — верный способ спалить катушку.
MAX_HOLD_MS = 15_000


def cat_door_config_path(data_dir: Path) -> Path:
    return data_dir / "runtime" / "cat-door.json"


class CatDoorConfig(BaseModel):
    mode: Literal["off", "dry", "on"] = "off"
    # Реле 1 занято калиткой, вход DI1 — герконом дверцы.
    relay: int = Field(default=2, ge=1, le=8)
    # Сколько держать защёлку открытой. Не импульс в полсекунды, как у калитки:
    # коту надо успеть протиснуться. Верхняя граница — про нагрев катушки.
    hold_ms: int = Field(default=5000, ge=200, le=MAX_HOLD_MS)
    # Кого пускать. Пусто — никого: список имён, а не «любого узнанного».
    allowed_cats: list[str] = Field(default_factory=list)
    # Порог уверенности опознания. Отдельный от порога записи: писать стоит всё
    # сомнительное, а открывать — только уверенное.
    #
    # С 13 августа он, пауза и задержка **работают**: решение принимает
    # `src/common/cat_door_decision.py` по живому кадру камеры дверцы. Ручная
    # кнопка в админке по-прежнему их не спрашивает — на то она и ручная.
    min_confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    cooldown_seconds: int = Field(default=30, ge=0, le=3600)
    # Сколько ждать от опознания до подачи питания. Кот, узнанный на подходе,
    # доходит до дверцы не мгновенно — по наблюдению владельца около пяти
    # секунд. Открыть сразу значит израсходовать окно удержания вхолостую и
    # запереть дверцу перед носом; ждать — значит раскрыть её к его приходу.
    #
    # Число это сейчас взято с чужих слов и подлежит замеру: к нему добавится
    # собственная задержка распознавания, которую видно будет только на живых
    # проходах.
    open_delay_seconds: float = Field(default=5.0, ge=0.0, le=30.0)

    @field_validator("mode", mode="before")
    @classmethod
    def normalise_mode(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    def allows(self, name: str | None) -> bool:
        return bool(name) and name in self.allowed_cats


def load_cat_door_config(path: Path, defaults: CatDoorConfig) -> CatDoorConfig:
    """Настройки с диска; при негодном файле — прежние значения и громкая жалоба."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return defaults
    except (OSError, ValueError) as exc:
        logger.warning("Настройки дверцы не прочитаны (%s), беру прежние", exc)
        return defaults
    try:
        return CatDoorConfig.model_validate(payload)
    except ValueError as exc:
        logger.warning(
            "Настройки дверцы из %s не приняты, работаю на прежних: %s",
            path.name,
            str(exc).replace("\n", " ")[:300],
        )
        return defaults


def save_cat_door_config(path: Path, config: CatDoorConfig) -> None:
    write_text_durably(path, json.dumps(config.model_dump(), ensure_ascii=False, indent=2))
