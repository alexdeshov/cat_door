"""Просьба открыть калитку и ответ на неё — двумя файлами.

Кнопка живёт в админке, а плата — за службой `catdoor-gate-access`, и это не
случайность: у веб-процесса нет и не должно быть прав на механику. Ровно та же
схема, что у переключателя контуров, где веб пишет файл, а `catdoor-gate-sync`
исполняет.

Просьба одноразовая. У неё есть идентификатор и время, и служба исполняет её
только один раз: файл, оставшийся на диске, не должен открыть калитку при
следующем запуске службы — это худшее, что мог бы сделать перезапуск. По той же
причине просьба стареет: нажатие, до которого служба не добралась за десяток
секунд, уже не нужно человеку у калитки, а сработало бы оно, когда он ушёл.

**Возраст по часам — недостаточная защита от перезагрузки.** Проверено
2026-08-11: если узел поднялся после обрыва питания с отставшими часами (RTC
сбился, chrony ещё не подтянул время), просьба из прошлой жизни выглядит
свежей — и калитка открывается сама, ночью, без человека. Поэтому в просьбе
записан ещё и идентификатор загрузки ядра: он меняется при каждой перезагрузке
и от часов не зависит вовсе.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel, Field

from src.common.durable_write import write_text_durably

logger = logging.getLogger("catdoor.gate-open-request")

# Сколько живёт нажатие. Служба смотрит файл раз в секунду, так что десяти
# хватает с запасом на неспешный опрос и на перезапуск в этот самый момент.
REQUEST_TTL_SECONDS = 10.0


BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")


def boot_id() -> str:
    """Идентификатор текущей загрузки ядра. Пусто — система его не даёт.

    Меняется при каждой перезагрузке и не зависит от часов, поэтому отличает
    «служба не успела» от «это было до обрыва питания». На системах без
    `/proc` возвращается пустая строка, и проверка становится незначащей —
    рабочий узел это Ubuntu, а суита не должна зависеть от ядра.
    """
    try:
        return BOOT_ID_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


class OpenRequest(BaseModel):
    id: str
    requested_at: float
    source: str = "admin"
    # Загрузка, в которой нажали кнопку. Пусто — система не даёт его знать.
    boot_id: str = ""
    # Кто нажал — на будущее, когда в админке появятся учётные записи.
    actor: str = ""


class OpenResult(BaseModel):
    id: str
    ok: bool
    detail: str
    finished_at: float
    pulse_ms: int = Field(default=0, ge=0)


def open_request_path(data_dir: Path) -> Path:
    return data_dir / "runtime" / "gate-open-request.json"


def open_result_path(data_dir: Path) -> Path:
    return data_dir / "runtime" / "gate-open-result.json"


def save_open_request(path: Path, request: OpenRequest) -> None:
    write_text_durably(path, request.model_dump_json(), mode=0o640)


def load_open_request(path: Path) -> OpenRequest | None:
    """Просьба с диска. `None` — файла нет или он негоден.

    Негодный файл здесь означает «не открывать»: разобрать мусор как команду на
    механику нельзя ни при каких обстоятельствах.
    """
    try:
        return OpenRequest.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        logger.warning("Просьба открыть калитку не прочитана (%s) — игнорирую", exc)
        return None


def save_open_result(path: Path, result: OpenResult) -> None:
    write_text_durably(path, result.model_dump_json(), mode=0o640)


def load_open_result(path: Path) -> OpenResult | None:
    try:
        return OpenResult.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return None
