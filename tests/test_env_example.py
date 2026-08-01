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


def test_every_documented_key_exists_in_settings() -> None:
    """An unknown key is silently dropped by extra="ignore", so it would lie."""
    fields = set(Settings.model_fields)
    unknown = [
        key
        for key in _assignments()
        if key.removeprefix("CATDOOR_").lower() not in fields
    ]
    assert unknown == []
