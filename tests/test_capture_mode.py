from pathlib import Path

from src.common.capture_mode import (
    CaptureMode,
    capture_mode_path,
    load_capture_mode,
    save_capture_mode,
)
from src.common.settings import Settings


def test_three_positions_cover_every_sensible_combination() -> None:
    """Четвёртое сочетание двух флагов бессмысленно и потому недостижимо."""
    assert CaptureMode.from_state("off").state == "off"
    assert CaptureMode.from_state("off").enabled is False
    assert CaptureMode.from_state("observe") == CaptureMode(enabled=True, record_events=False)
    assert CaptureMode.from_state("record") == CaptureMode(enabled=True, record_events=True)
    # Выключенный детектор, которому велено писать, всё равно считается
    # выключенным: писать нечего, кадры никто не смотрит.
    assert CaptureMode(enabled=False, record_events=True).state == "off"


def test_dry_run_is_the_old_name_of_the_same_flag() -> None:
    assert CaptureMode.from_state("observe").dry_run is True
    assert CaptureMode.from_state("record").dry_run is False


def test_environment_provides_the_starting_position(runtime_path: Path) -> None:
    settings = Settings(
        data_dir=runtime_path,
        automatic_capture_enabled=True,
        automatic_capture_dry_run=False,
    )

    assert CaptureMode.from_settings(settings).state == "record"
    # Файла ещё нет — остаётся то, что задано при установке.
    assert load_capture_mode(
        capture_mode_path(runtime_path),
        CaptureMode.from_settings(settings),
    ).state == "record"


def test_saved_mode_wins_over_the_environment(runtime_path: Path) -> None:
    settings = Settings(data_dir=runtime_path, automatic_capture_enabled=False)
    path = capture_mode_path(runtime_path)

    save_capture_mode(path, CaptureMode.from_state("record"))

    assert load_capture_mode(path, CaptureMode.from_settings(settings)).state == "record"


def test_broken_file_falls_back_to_the_environment(runtime_path: Path) -> None:
    """Переключатель не должен менять поведение из-за испорченного файла."""
    settings = Settings(data_dir=runtime_path, automatic_capture_enabled=True)
    path = capture_mode_path(runtime_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ это не json", encoding="utf-8")

    assert load_capture_mode(path, CaptureMode.from_settings(settings)).enabled is True
