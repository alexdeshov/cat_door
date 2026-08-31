from __future__ import annotations

import re
from pathlib import Path

from src.common.settings import Settings

ENV_EXAMPLE = Path(__file__).resolve().parent.parent / ".env.example"
ASSIGNMENT = re.compile(r"^(CATDOOR_[A-Z0-9_]+)=(.*)$", re.MULTILINE)
PATH_KEYS = {
    "CATDOOR_DATA_DIR",
    "CATDOOR_DATABASE_PATH",
    "CATDOOR_DETECTOR_MODEL_PATH",
}


def _assignments() -> dict[str, str]:
    return {
        match.group(1): match.group(2).strip()
        for match in ASSIGNMENT.finditer(ENV_EXAMPLE.read_text(encoding="utf-8"))
    }


def test_example_holds_no_windows_paths() -> None:
    """A backslash is a normal filename character on POSIX, not a separator.

    Path("models\\cat-detector.onnx") is a single name there, so with_suffix()
    would look for a manifest that cannot exist and the detector would refuse
    to start.
    """
    for key, value in _assignments().items():
        assert "\\" not in value, f"{key} contains a backslash"
        assert not re.match(r"^[A-Za-z]:", value), f"{key} starts with a drive letter"


def test_path_settings_are_absolute() -> None:
    """Services run under systemd, where a relative path resolves against /."""
    assignments = _assignments()
    for key in PATH_KEYS:
        assert key in assignments, f"{key} is missing from .env.example"
        assert assignments[key].startswith("/"), f"{key} is not absolute"


def _shell_keys() -> set[str]:
    """Keys the backup shell script reads. It has its own settings — retention
    windows, archive paths — and they never reach `Settings`.

    Read out of the script rather than listed here on purpose: a hand-kept list
    would happily accept a typo, which is the one thing this file exists to
    catch.
    """
    scripts = Path(__file__).resolve().parent.parent / "scripts"
    text = (scripts / "backup_to_archive.sh").read_text(encoding="utf-8")
    return set(re.findall(r"\$\{(CATDOOR_[A-Z0-9_]+)", text))


def test_every_documented_key_is_read_by_something() -> None:
    """An unknown key is silently dropped by extra="ignore", so it would lie.

    Two readers, not one: most keys land in `Settings`, but the backup script
    is a shell script with its own. Until 31.08 its unit had no
    `EnvironmentFile` at all, so none of them were read anywhere — documented
    and dead at the same time.
    """
    known = {f"CATDOOR_{name.upper()}" for name in Settings.model_fields} | _shell_keys()

    unknown = [key for key in _assignments() if key not in known]

    assert unknown == []


def test_the_backup_unit_can_actually_receive_its_keys() -> None:
    """Documenting a key for a service that reads no environment file is a lie
    of the same kind: it looks configurable and is not."""
    unit = Path(__file__).resolve().parent.parent / "install" / "catdoor-backup.service"

    assert "EnvironmentFile" in unit.read_text(encoding="utf-8")
