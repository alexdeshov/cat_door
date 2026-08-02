import os
import stat
from pathlib import Path

import pytest

from src.common.durable_write import write_text_durably


def test_content_and_permissions_are_set_together(runtime_path: Path) -> None:
    path = runtime_path / "nested" / "secrets.json"

    write_text_durably(path, '{"пароль": "есть"}')

    assert path.read_text(encoding="utf-8") == '{"пароль": "есть"}'
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_previous_content_survives_a_failed_write(runtime_path: Path, monkeypatch) -> None:
    """Обрыв на середине не должен оставлять ни половины, ни пустого файла."""
    path = runtime_path / "cameras.json"
    write_text_durably(path, "старое")

    def explode(descriptor: int) -> None:
        raise OSError("диск отвалился")

    monkeypatch.setattr("src.common.durable_write.os.fsync", explode)
    with pytest.raises(OSError):
        write_text_durably(path, "новое")

    assert path.read_text(encoding="utf-8") == "старое"
    assert list(runtime_path.glob(".cameras.json.*.tmp")) == []


def test_data_reaches_the_disk_before_the_rename(runtime_path: Path, monkeypatch) -> None:
    """Порядок и есть смысл модуля: сначала содержимое, потом имя, потом каталог.

    Переименование, переживiшee обрыв питания раньше данных, оставит запись
    каталога, указывающую на пустоту, — ровно тот файл нулевой длины, ради
    которого всё это написано.
    """
    order: list[str] = []
    real_fsync = os.fsync
    real_replace = Path.replace

    def record_fsync(descriptor: int) -> None:
        order.append("fsync")
        real_fsync(descriptor)

    def record_replace(self: Path, target) -> Path:  # type: ignore[no-untyped-def]
        order.append("replace")
        return real_replace(self, target)

    monkeypatch.setattr("src.common.durable_write.os.fsync", record_fsync)
    monkeypatch.setattr(Path, "replace", record_replace)

    write_text_durably(runtime_path / "settings.json", "{}")

    assert order == ["fsync", "replace", "fsync"]


def test_directory_fsync_failure_is_not_fatal(runtime_path: Path, monkeypatch) -> None:
    """Не всякая файловая система позволяет открыть каталог для сброса."""
    real_open = os.open

    def refuse_directories(path, flags, *rest):  # type: ignore[no-untyped-def]
        if Path(path).is_dir():
            raise OSError("каталог так не открыть")
        return real_open(path, flags, *rest)

    monkeypatch.setattr("src.common.durable_write.os.open", refuse_directories)

    write_text_durably(runtime_path / "settings.json", "{}")

    assert (runtime_path / "settings.json").read_text(encoding="utf-8") == "{}"
