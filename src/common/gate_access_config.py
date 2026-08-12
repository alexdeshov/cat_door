"""Настройки автооткрытия калитки: файл, который пишет веб и читает служба.

Та же схема, что у настроек детекции и у переключателя контуров: веб кладёт
файл, служба перечитывает его на ходу. Веб механикой не распоряжается и прав на
неё не получает.

Режимов три. `off` — ни одной команды на плату ни при каких условиях; `dry` —
решение принимается и пишется в журнал, реле не трогается; `on` — импульс
уходит.

Умолчание — `off`, и это не формальность: свежая установка не должна открывать
калитку никому, пока человек не выбрал, кому именно.

Кнопка «Открыть калитку» в админке работает во всех трёх режимах: они про
автоматику, а не про человека, который нажал сам.
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


class MovementRules(BaseModel):
    """Правила направления: кого пускать по тому, куда он идёт.

    Числа измерены 12 августа на живых проходах, разбор — в
    `docs/gate-direction-experiments.md`. Меняются они не от вкуса, а когда
    изменится геометрия: подвинули камеру — пороги перемеривают по журналу, там
    отношение площадей пишется по каждому проходу.

    Выключаются целиком, и это умолчание: свежая установка ведёт себя так же,
    как до появления правил.
    """

    enabled: bool = False
    # Во сколько раз выросла рамка, чтобы считать «идёт к калитке».
    approaching_ratio: float = Field(default=2.0, gt=1.0, le=20.0)
    # И во сколько упала, чтобы считать «уходит».
    leaving_ratio: float = Field(default=0.5, gt=0.0, lt=1.0)
    # Пускать ли того, кто стоит у калитки и никуда не идёт, — но только сразу
    # после того, как створка закрылась. Это возвращение: вышел, закрыл за
    # собой, развернулся.
    open_when_still: bool = True
    return_window_seconds: int = Field(default=30, ge=0, le=600)

    @model_validator(mode="after")
    def validate_ratios(self) -> MovementRules:
        if self.leaving_ratio >= self.approaching_ratio:
            raise ValueError("порог ухода должен быть меньше порога прихода")
        return self


class SecondChanceRules(BaseModel):
    """Второй импульс тому же человеку в том же событии.

    Одно событие — один импульс: защита от качания, из-за которой 11 августа
    калитка щёлкала каждые две секунды. У неё есть цена, и 12 августа она
    предъявилась. Событие Лёши шло 55 секунд; импульс ушёл на третьей, когда
    створка была ещё открыта от прошедшего перед ним человека, и пропал зря.
    Через две секунды створка закрылась, а Лёша остался перед ней — и второго
    импульса ему не полагалось, потому что Frigate, пока человек не вышел из
    кадра, считает это всё тем же событием.

    Отсюда правило: **створка закрылась, а свой всё ещё здесь — открыть ещё
    раз.** Не «повторять, пока не сработает»: повтор без закрывшейся створки
    вернул бы то самое качание.

    Выключены по умолчанию, как и правила движения: свежая установка ведёт
    себя так же, как до появления этой ветки.
    """

    enabled: bool = False
    # Сколько всего импульсов позволено одному событию, включая первый.
    max_pulses: int = Field(default=2, ge=1, le=5)
    # Сколько ждать между импульсами. Створка после импульса успевает
    # открыться и закрыться примерно за пять секунд — замерено 12 августа по
    # классификатору: импульс 17:18:49, открыта 17:18:55, закрыта 17:18:58.
    gap_seconds: int = Field(default=8, ge=1, le=120)
    # Требовать ли, чтобы створка успела закрыться между импульсами. Это и есть
    # само правило; выключать его стоит, только если классификатор состояния
    # врёт, и тогда защитой остаётся один лишь `gap_seconds`.
    require_closed: bool = True


class GateAccessConfig(BaseModel):
    mode: Literal["off", "dry", "on"] = "off"
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
    movement: MovementRules = Field(default_factory=MovementRules)
    second_chance: SecondChanceRules = Field(default_factory=SecondChanceRules)
    # Что было до применения пресета. Хранится здесь же, чтобы откат не зависел
    # ни от чего, кроме этого файла: «вернуть как было» должно работать и после
    # перезагрузки узла, и через неделю.
    replaced_by_preset: dict[str, object] | None = None
    # То же самое для пресета второго шанса. Отдельным полем, а не общим:
    # пресеты накатываются и откатываются порознь, и общая ячейка означала бы,
    # что откат одного втихую роняет другой.
    replaced_by_chance_preset: dict[str, object] | None = None

    @field_validator("mode", mode="before")
    @classmethod
    def normalise_mode(cls, value: object) -> object:
        """Файл правят и руками, а «ON» с пробелом — тот же режим, что «on»."""
        return value.strip().lower() if isinstance(value, str) else value

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


# Пресет «правила движения» — то, что измерено 12 августа 2026. Накатывается и
# откатывается одним нажатием: числа подобраны разбором живых проходов, и
# перебирать их руками человеку незачем.
MOVEMENT_PRESET_NAME = "Правила движения, 12 августа"
MOVEMENT_PRESET = MovementRules(
    enabled=True,
    approaching_ratio=2.0,
    leaving_ratio=0.5,
    open_when_still=True,
    return_window_seconds=30,
)


def apply_movement_preset(config: GateAccessConfig) -> GateAccessConfig:
    """Включить пресет, запомнив прежние правила для отката."""
    return config.model_copy(
        update={
            "movement": MOVEMENT_PRESET.model_copy(),
            "replaced_by_preset": config.movement.model_dump(),
        }
    )


def rollback_movement_preset(config: GateAccessConfig) -> GateAccessConfig:
    """Вернуть правила, которые были до пресета.

    Если возвращать нечего — выключаем правила совсем: это заведомо то
    поведение, которое было до их появления.
    """
    previous = config.replaced_by_preset
    restored = (
        MovementRules.model_validate(previous)
        if isinstance(previous, dict)
        else MovementRules()
    )
    return config.model_copy(update={"movement": restored, "replaced_by_preset": None})


# Пресет «второй шанс» — ответ на разбор 12 августа: импульс, ушедший в уже
# открытую створку, пропадал, и человек перед закрывшейся калиткой оставался ни
# с чем до тех пор, пока не выйдет из кадра.
SECOND_CHANCE_PRESET_NAME = "Второй шанс, 12 августа"
SECOND_CHANCE_PRESET = SecondChanceRules(
    enabled=True,
    max_pulses=2,
    gap_seconds=8,
    require_closed=True,
)


def apply_second_chance_preset(config: GateAccessConfig) -> GateAccessConfig:
    """Включить второй шанс, запомнив прежние правила для отката."""
    return config.model_copy(
        update={
            "second_chance": SECOND_CHANCE_PRESET.model_copy(),
            "replaced_by_chance_preset": config.second_chance.model_dump(),
        }
    )


def rollback_second_chance_preset(config: GateAccessConfig) -> GateAccessConfig:
    """Вернуть то, что было до пресета; если возвращать нечего — выключить."""
    previous = config.replaced_by_chance_preset
    restored = (
        SecondChanceRules.model_validate(previous)
        if isinstance(previous, dict)
        else SecondChanceRules()
    )
    return config.model_copy(
        update={"second_chance": restored, "replaced_by_chance_preset": None}
    )


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
