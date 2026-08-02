from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.common.settings import (
    ConfigurationError,
    Settings,
    validate_runtime_paths,
)


def _settings(**overrides: object) -> Settings:
    return Settings.model_construct(**overrides)  # type: ignore[arg-type]


def test_relative_data_dir_is_rejected() -> None:
    """Under systemd a relative path resolves against /, so it can never work."""
    settings = _settings(
        data_dir=Path("data"),
        database_path=None,
        detector_model_path=Path("/var/lib/catdoor/models/cat-detector.onnx"),
    )

    with pytest.raises(ConfigurationError, match="абсолютным"):
        validate_runtime_paths(settings)


def test_backslash_in_model_path_is_rejected(tmp_path: Path) -> None:
    settings = _settings(
        data_dir=tmp_path,
        database_path=None,
        detector_model_path=Path("models\\cat-detector.onnx"),
    )

    with pytest.raises(ConfigurationError, match="обратный слэш"):
        validate_runtime_paths(settings)


def test_valid_configuration_creates_directories(tmp_path: Path) -> None:
    data_dir = tmp_path / "catdoor"
    settings = _settings(
        data_dir=data_dir,
        database_path=tmp_path / "db" / "catdoor.sqlite3",
        detector_model_path=data_dir / "models" / "cat-detector.onnx",
    )

    validate_runtime_paths(settings)

    assert data_dir.is_dir()
    assert (tmp_path / "db").is_dir()
    assert list(data_dir.glob(".catdoor-write-check.*")) == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_unwritable_data_dir_is_rejected(tmp_path: Path) -> None:
    data_dir = tmp_path / "readonly"
    data_dir.mkdir(mode=0o500)
    settings = _settings(
        data_dir=data_dir,
        database_path=None,
        detector_model_path=data_dir / "models" / "cat-detector.onnx",
    )

    try:
        with pytest.raises(ConfigurationError, match="недоступен для записи"):
            validate_runtime_paths(settings)
    finally:
        data_dir.chmod(0o700)


def test_environment_cameras_are_only_a_seed() -> None:
    """Разбор списка остаётся; всё остальное про камеры живёт в инвентаре.

    Проверки поиска по id, ролей и подмены адреса переехали в
    `tests/test_camera_registry.py` вместе с самим поведением.
    """
    settings = Settings(
        camera_face_rtsp="rtsp://face",
        cameras=[{"id": "yard", "rtsp": "rtsp://yard", "title": "Двор"}],
    )

    assert [camera.id for camera in settings.cameras] == ["yard"]
    assert settings.cameras[0].title == "Двор"


def test_camera_id_must_survive_a_url_path_segment() -> None:
    with pytest.raises(ValidationError):
        Settings(cameras=[{"id": "../etc", "rtsp": "rtsp://x"}])
