"""Настройки автооткрытия калитки: файл, который пишет веб и читает служба.

Та же схема, что у настроек детекции и у переключателя контуров: веб кладёт
файл, служба перечитывает его на ходу. Веб механикой не распоряжается и прав на
неё не получает.

Режимов в этой версии два. `off` — ни одной команды на плату ни при каких
условиях, `dry` — решение принимается и пишется в журнал, реле не трогается.
Значение `on` **не принимается**: правило владельца в `AGENTS.md` запрещает
управлять реальной дверью до отдельного письменного разрешения, и снимать его
молча нельзя.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from src.common.durable_write import write_text_durably

logger = logging.getLogger("catdoor.gate-access-config")

MAC_PATTERN = re.compile(r"^[0-9a-f]{2}(:[0-9a-f]{2}){5}$")
TIME_PATTERN = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


def gate_access_config_path(data_dir: Path) -> Path:
    return data_dir / "runtime" / "gate-access.json"


def mask_mac(mac: str) -> str:
    """MAC для показа в админке: видны две последние группы.

    Адреса телефонов — персональные данные членов семьи. Отличить свою запись
    от чужой по хвосту можно, а снять адрес с чужого экрана — нет.
    """
    if not mac:
        return ""
    parts = mac.split(":")
    if len(parts) != 6:
        return "··"
    return ":".join(["··"] * 4 + parts[4:])


class GateSpec(BaseModel):
    """Одна калитка: камера, канал реле и зона, в которой человек считается «у калитки».

    Камера может быть пустой: канал реле нужен и до того, как настроено
    автооткрытие, — кнопкой в админке калитку открывают руками. Пустая камера
    не совпадёт ни с одним событием, и автоматика останется молчать.
    """

    camera: str = ""
    # Канал платы, к которому подключена кнопка выхода контроллера калитки. На
    # объекте это 1: план резервировал единицу за кошачьей дверцей, но механики
    # у неё нет, и канал ей подберут отдельно.
    relay: int = Field(default=1, ge=1, le=8)
    # Пусто — вся картинка. Зона нужна, чтобы узнанное лицо на террасе не
    # открывало калитку с другой стороны участка.
    zone: str = ""


class Person(BaseModel):
    """Человек из библиотеки лиц Frigate и, возможно, его телефон.

    Адрес и MAC — персональные данные: живут только в файле на узле, в
    репозиторий не попадают и в коде их нет.
    """

    name: str
    phone_ip: str = ""
    phone_mac: str = ""

    @field_validator("phone_mac", mode="before")
    @classmethod
    def normalise_mac(cls, value: object) -> object:
        if not isinstance(value, str) or not value.strip():
            return ""
        mac = value.strip().lower().replace("-", ":")
        if not MAC_PATTERN.match(mac):
            raise ValueError("MAC-адрес пишется шестью группами вида aa:bb:cc:dd:ee:ff")
        return mac


class GateAccessConfig(BaseModel):
    mode: Literal["off", "dry"] = "off"
    gates: list[GateSpec] = Field(default_factory=list)
    # Белый список имён, а не «любой узнанный»: распознавание лица — удобство,
    # а не фактор безопасности, и фотография на экране телефона его обманывает.
    allowed_names: list[str] = Field(default_factory=list)
    require_second_factor: bool = False
    people: list[Person] = Field(default_factory=list)
    # Выше порога самого Frigate (0.9) намеренно: имя он ставит и тогда, когда
    # уверен «на троечку», а открывать по такому имени незачем.
    min_score: float = Field(default=0.93, ge=0.0, le=1.0)
    pulse_ms: int = Field(default=500, ge=100, le=5000)
    # Обратной связи от геркона нет, поэтому пауза — единственная защита от
    # очереди импульсов на одного человека, которого камера узнаёт снова и снова.
    cooldown_seconds: int = Field(default=20, ge=0, le=3600)
    # Ночью распознавание почти не работает: модель обучена на цветных кадрах,
    # а ИК-подсветка даёт серое. Ложное срабатывание ночью — худший случай.
    open_from: str = "07:00"
    open_to: str = "23:00"

    @field_validator("mode", mode="before")
    @classmethod
    def refuse_the_open_mode(cls, value: object) -> object:
        if isinstance(value, str) and value.strip().lower() == "on":
            raise ValueError(
                "Режим «on» не принимается: управление замком требует отдельного "
                "письменного разрешения владельца, и кода исполнения в системе ещё нет"
            )
        return value

    @field_validator("open_from", "open_to")
    @classmethod
    def validate_time(cls, value: str) -> str:
        if not TIME_PATTERN.match(value):
            raise ValueError("Время задаётся как ЧЧ:ММ, например 07:00")
        return value

    @model_validator(mode="after")
    def validate_people(self) -> GateAccessConfig:
        names = [person.name for person in self.people]
        if len(names) != len(set(names)):
            raise ValueError("Один человек — одна запись: имена в списке повторяются")
        if len({gate.camera for gate in self.gates}) != len(self.gates):
            raise ValueError("Одна камера — одна калитка: камеры в списке повторяются")
        return self

    def gate_for(self, camera: str) -> GateSpec | None:
        for gate in self.gates:
            if gate.camera == camera:
                return gate
        return None

    def person(self, name: str) -> Person | None:
        for person in self.people:
            if person.name == name:
                return person
        return None


def load_gate_access_config(path: Path, defaults: GateAccessConfig) -> GateAccessConfig:
    """Настройки с диска; при негодном файле — прежние значения и громкая жалоба.

    Молчаливый откат здесь дороже, чем в детекции: человек, переключивший режим
    в админке, уверен, что переключил его, — а служба продолжала бы работать по
    старому. Разница между «настройка записана» и «настройка применена» не
    видна ничем, кроме этой строки в журнале.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        # Файла нет — это не беда, а первый запуск: режим `off`.
        return defaults
    except (OSError, ValueError) as exc:
        logger.warning("Настройки калитки не прочитаны (%s), беру прежние", exc)
        return defaults
    try:
        return GateAccessConfig.model_validate(payload)
    except ValueError as exc:
        logger.warning(
            "Настройки калитки из %s не приняты, работаю на прежних: %s",
            path.name,
            str(exc).replace("\n", " ")[:300],
        )
        return defaults


def save_gate_access_config(path: Path, config: GateAccessConfig) -> None:
    """Записать так, чтобы настройки не потеряла пропажа питания.

    Служба перечитывает файл на ходу, поэтому подмена обязана быть атомарной, а
    обрыв через секунду после «Сохранить» не должен оставить файл нулевой
    длины. Права — только владельцу: внутри персональные данные семьи.
    """
    write_text_durably(
        path,
        json.dumps(config.model_dump(), ensure_ascii=False, indent=2),
        mode=0o600,
    )
