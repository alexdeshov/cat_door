"""Какие контуры узла работают прямо сейчас.

На узле два контура: коты у дверцы (этот репозиторий) и люди у калитки
(`gate_access`, Frigate). Процессор один, встроенная графика на текущем железе
сломана драйвером, поэтому считает всё он. Одновременная работа не «замедляет
немного», а деградирует оба контура незаметно: кадры стареют, проходы
пропускаются, и узнаётся это из пустого журнала через неделю.

Отсюда признак `exclusive`. Он не архитектурное украшение, а запись факта про
конкретное железо, и снимается одной строкой после переезда — интерфейс и
модель состояния переделывать не придётся.

Режим контура котов живёт отдельно, в `capture_mode`: там сказано, *как* он
работает, здесь — *разрешено* ли ему работать вообще. Разделение нужно, чтобы
выключение с уровня узла не стирало выбранный режим наблюдения или записи.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from src.common.durable_write import write_text_durably

CircuitName = Literal["cats", "gate", "nvr"]
GateRecording = Literal["events", "continuous"]

CIRCUIT_TITLES: dict[CircuitName, str] = {
    "cats": "Дверца",
    "gate": "Калитка 1",
    "nvr": "Регистратор",
}

# Контуров три, а движка два. Калитка и регистратор — это один и тот же
# Frigate с разными камерами в конфигурации, поэтому вместе они не стоят почти
# ничего сверх одного из них. Ограничение узла — про движки, а не про контуры:
# запрещать «калитку вместе с регистратором» значило бы запрещать процессу
# работать вместе с самим собой.
FRIGATE_CIRCUITS: tuple[CircuitName, ...] = ("gate", "nvr")


class CircuitConflict(ValueError):
    """Запрошены оба контура там, где узел тянет только один."""


class NodeCircuits(BaseModel):
    cats: bool = True
    gate: bool = False
    nvr: bool = False
    exclusive: bool = True
    # Круглосуточная запись заполняет диск предсказуемо и быстро, а для сбора
    # библиотеки лиц достаточно клипов вокруг событий. Значение попадает в
    # конфигурацию Frigate при развёртывании контура.
    gate_recording: GateRecording = "events"

    @property
    def frigate_wanted(self) -> bool:
        """Нужен ли Frigate вообще — хоть одному из своих контуров."""
        return any(self.is_active(name) for name in FRIGATE_CIRCUITS)

    @property
    def has_conflict(self) -> bool:
        return self.exclusive and self.cats and self.frigate_wanted

    def resolved(self) -> NodeCircuits:
        """Обезвредить конфликт в правленом руками файле.

        Отвергать файл целиком нельзя: тогда опечатка останавливала бы запись
        проходов. Побеждает кошачий контур — он работал до появления второго и
        не требует внешней службы.
        """
        if not self.has_conflict:
            return self
        return self.model_copy(update=dict.fromkeys(FRIGATE_CIRCUITS, False))

    def is_active(self, circuit: CircuitName) -> bool:
        return bool(getattr(self, circuit))

    def with_only(self, circuit: CircuitName) -> NodeCircuits:
        """Отдать узел одному контуру, остановив остальные."""
        return self.model_copy(update={name: name == circuit for name in CIRCUIT_TITLES})

    def with_circuit(self, circuit: CircuitName, enabled: bool) -> NodeCircuits:
        """Включить или выключить контур, посторонившись перед чужим движком.

        Включение контура при действующем ограничении гасит контуры другого
        движка, но не соседей по своему: регистратор рядом с калиткой — та же
        самая работа Frigate, и выключать её незачем.
        """
        updated = self.model_copy(update={circuit: enabled})
        if not enabled or not updated.has_conflict:
            return updated
        if circuit == "cats":
            return updated.model_copy(update=dict.fromkeys(FRIGATE_CIRCUITS, False))
        return updated.model_copy(update={"cats": False})

    @property
    def active_names(self) -> tuple[CircuitName, ...]:
        return tuple(name for name in CIRCUIT_TITLES if self.is_active(name))


def ensure_allowed(circuits: NodeCircuits) -> None:
    """Проверка вынесена из валидатора модели намеренно.

    Валидатор pydantic заворачивает своё исключение в `ValidationError`, и
    вызывающий код не смог бы отличить «узел не тянет оба контура» от опечатки
    в типе поля — а это разные ответы: осмысленный отказ и ошибка запроса.
    """
    if circuits.has_conflict:
        raise CircuitConflict(
            "Узел не тянет кошачий контур вместе с Frigate: включите что-то "
            "одно или снимите ограничение на вкладке узла"
        )


def circuits_path(data_dir: Path) -> Path:
    return data_dir / "runtime" / "circuits.json"


def load_circuits(path: Path, defaults: NodeCircuits | None = None) -> NodeCircuits:
    """Прочитать состояние узла. Нечитаемый файл не меняет поведения.

    Умолчание — кошачий контур разрешён: это то, что работало до появления
    второго контура, и неожиданная остановка записи из-за испорченного файла
    была бы худшим исходом, чем игнорирование файла.
    """
    fallback = defaults or NodeCircuits()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback
    try:
        return NodeCircuits.model_validate(payload).resolved()
    except ValueError:
        return fallback


def save_circuits(path: Path, circuits: NodeCircuits) -> None:
    write_text_durably(path, json.dumps(circuits.model_dump(), ensure_ascii=False, indent=2))
