"""Запись файла, переживающая внезапное отключение питания.

`replace()` даёт атомарность на уровне имени: читатель увидит либо старое
содержимое, либо новое, но не половину. Долговечности он не даёт совсем.
Ядро держит и данные временного файла, и саму запись каталога в кэше; если
питание пропадёт в ближайшие секунды, ext4 при монтировании восстановит журнал
метаданных, и переименование останется — а данных, на которые оно указывает, в
файле не будет. На месте настроек камер это выглядит как файл нулевой длины,
то есть потеря адресов и паролей.

Поэтому порядок здесь строгий: содержимое на диск, затем подмена имени, затем
запись каталога на диск. Так после любого обрыва остаётся либо целый старый
файл, либо целый новый.
"""

from __future__ import annotations

import os
from contextlib import suppress
from pathlib import Path
from uuid import uuid4


def _fsync_directory(directory: Path) -> None:
    """Сбросить саму запись каталога.

    Без этого переименование может не пережить обрыв, даже когда содержимое
    файла уже на диске. Не все файловые системы позволяют открыть каталог для
    такой операции, поэтому отказ здесь не считается ошибкой записи.
    """
    with suppress(OSError):
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def publish_file_durably(temporary: Path, destination: Path) -> None:
    """Обнародовать уже записанный файл так, чтобы он пережил обрыв питания.

    Для видео события `replace()` без fsync означал ровно тот же дефект, что и
    у конфигов: переименование переживает обрыв раньше данных, и в ленте
    остаётся «готовое» событие с пустым файлом — как правило, последний проход
    перед отключением, самый интересный.
    """
    descriptor = os.open(temporary, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    temporary.replace(destination)
    _fsync_directory(destination.parent)


def write_text_durably(path: Path, text: str, *, mode: int = 0o600) -> None:
    """Записать файл целиком, с гарантией, что он переживёт потерю питания.

    Права выставляются в момент создания, а не после записи: между `open` и
    `chmod` существовало бы окно, в котором пароли камер лежат с правами по
    умолчанию. По умолчанию файл виден только владельцу — оба процесса работают
    под одним служебным пользователем, и делать эти файлы шире незачем.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(text.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
