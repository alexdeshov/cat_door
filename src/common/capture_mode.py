"""Режим автоматического сбора, переключаемый на ходу.

Раньше это были две переменные окружения — «работает ли детектор» и «пишутся
ли подтверждённые проходы». Обе читались один раз при старте, поэтому включить
сбор означало зайти по SSH, поправить файл в `/etc` и перезапустить службу.

Здесь они сведены в одно значение с тремя положениями. Два флага порождали
четыре сочетания, одно из которых бессмысленно: выключенный детектор, которому
велено писать. Три положения описывают ровно то, что бывает на самом деле:

- `off` — детектор не запускается;
- `observe` — детектор работает и всё показывает, но на диск ничего не пишет;
- `record` — подтверждённые проходы сохраняются.

Файл лежит рядом с настройками детекции и читается тем же циклом recorder, так
что переключение применяется в течение секунды — но только когда автомат
событий в ожидании: обрывать запись начатого прохода на полуслове незачем.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from src.common.durable_write import write_text_durably
from src.common.settings import Settings

CaptureState = Literal["off", "observe", "record"]


class CaptureMode(BaseModel):
    enabled: bool = False
    record_events: bool = False

    @classmethod
    def from_settings(cls, settings: Settings) -> CaptureMode:
        """Начальное значение из окружения — до первого переключения в админке."""
        return cls(
            enabled=settings.automatic_capture_enabled,
            record_events=not settings.automatic_capture_dry_run,
        )

    @classmethod
    def from_state(cls, state: CaptureState) -> CaptureMode:
        return cls(enabled=state != "off", record_events=state == "record")

    @property
    def state(self) -> CaptureState:
        if not self.enabled:
            return "off"
        return "record" if self.record_events else "observe"

    @property
    def dry_run(self) -> bool:
        """Прежнее имя того же признака: им подписан статус и журнал."""
        return not self.record_events


def capture_mode_path(data_dir: Path) -> Path:
    return data_dir / "runtime" / "capture-mode.json"


def load_capture_mode(path: Path, defaults: CaptureMode) -> CaptureMode:
    """Прочитать режим. Нечитаемый файл — это ещё не повод менять поведение.

    Возврат к значениям окружения безопаснее выдумывания: администратор, который
    ничего не переключал, получит то, что задано при установке.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return defaults
    try:
        return CaptureMode.model_validate(payload)
    except ValueError:
        return defaults


def save_capture_mode(path: Path, mode: CaptureMode) -> None:
    write_text_durably(path, json.dumps(mode.model_dump(), ensure_ascii=False, indent=2))
