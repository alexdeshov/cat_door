"""Сжатие ролика под предел Телеграма: сколько давить, чем и когда не браться.

Проверяется арифметика бюджета и поведение вокруг ffmpeg, но не сам ffmpeg:
запускать кодировщик в тестах значило бы мерить не наш код, а скорость машины,
на которой их гоняют.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.telegram_bot.shrink import (
    AUDIO_KBPS,
    MIN_TOTAL_KBPS,
    QUALITY_LADDER,
    Plan,
    ShrinkFailed,
    encode_command,
    encode_timeout,
    ffprobe_path,
    first_plan,
    probe_duration,
    shrink_to_fit,
    width_for,
)

LIMIT = 50 * 1024 * 1024


def _budget(limit: int = LIMIT) -> int:
    from src.telegram_bot.shrink import BUDGET_SHARE

    return int(limit * BUDGET_SHARE)


class _Encoder:
    """Подставной ffmpeg: пишет файл заданного размера и помнит команды."""

    def __init__(self, sizes: list[int]) -> None:
        self.sizes = sizes
        self.commands: list[list[str]] = []

    def __call__(self, command, **kwargs):
        self.commands.append(list(command))
        size = self.sizes[min(len(self.commands), len(self.sizes)) - 1]
        destination = Path(command[-1])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"x" * size)
        return SimpleNamespace(returncode=0, stderr="")


def _bitrate_of(command: list[str]) -> int:
    return int(command[command.index("-b:v") + 1].removesuffix("k"))


def _width_of(command: list[str]) -> int:
    filter_argument = command[command.index("-vf") + 1]
    return int(filter_argument.split("min(iw,")[1].split(")")[0])


# --- сколько битрейта давать ---------------------------------------------------


def test_the_budget_is_split_by_length_not_taken_from_a_table() -> None:
    """Пятьдесят мегабайт на полминуты и на пять минут — разные картинки.

    Постоянное «сожми в 2 Мбит/с» промахивалось бы в обе стороны: короткое
    событие вышло бы хуже, чем могло, а длинное не влезло бы вовсе.
    """
    short = first_plan(30.0, _budget(), LIMIT)
    long = first_plan(300.0, _budget(), LIMIT)

    assert short.video_kbps > long.video_kbps * 9
    assert short.crf == 0 and long.crf == 0


def test_a_thin_budget_shrinks_the_frame_too() -> None:
    """1080p на 600 кбит/с разваливается в квадраты, 854 точки — нет."""
    assert width_for(3000) == 1920
    assert width_for(1500) == 1280
    assert width_for(600) == 854
    assert width_for(MIN_TOTAL_KBPS) == 640


def test_half_an_hour_is_refused_in_words() -> None:
    """Под границей читаемости видео перестаёт быть видео.

    Прислать полчаса серой мути хуже, чем сказать, где посмотреть запись.
    """
    with pytest.raises(ShrinkFailed, match="мин"):
        first_plan(3600.0, _budget(), LIMIT)


def test_without_a_measured_length_quality_leads() -> None:
    """Молчащий ffprobe не повод остаться без видео вовсе."""
    plan = first_plan(None, _budget(), LIMIT)

    assert (plan.width, plan.crf) == QUALITY_LADDER[0]
    assert plan.video_kbps == 0


# --- попытки -------------------------------------------------------------------


def test_a_miss_is_corrected_by_the_size_that_came_out(runtime_path: Path, monkeypatch) -> None:
    """Битрейт ffmpeg держит в среднем, и промах вверх случается.

    Вторая попытка считается по факту — во сколько раз файл оказался больше
    предела, — а не вслепую: слепой повтор стоил бы третьей перекодировки.
    """
    encoder = _Encoder([int(LIMIT * 1.5), int(LIMIT * 0.8)])
    monkeypatch.setattr("src.telegram_bot.shrink.subprocess.run", encoder)

    result = shrink_to_fit(
        "ffmpeg",
        runtime_path / "source.mp4",
        runtime_path / "fit.mp4",
        limit_bytes=LIMIT,
        duration_seconds=120.0,
    )

    assert result.stat().st_size <= LIMIT
    first, second = (_bitrate_of(command) for command in encoder.commands)
    assert second < first / 1.4
    assert _width_of(encoder.commands[1]) < _width_of(encoder.commands[0])


def test_a_clip_that_never_fits_is_explained(runtime_path: Path, monkeypatch) -> None:
    """Отказ обязан быть словами и без мусора на томе данных."""
    encoder = _Encoder([LIMIT * 4])
    monkeypatch.setattr("src.telegram_bot.shrink.subprocess.run", encoder)
    destination = runtime_path / "fit.mp4"

    with pytest.raises(ShrinkFailed):
        shrink_to_fit(
            "ffmpeg",
            runtime_path / "source.mp4",
            destination,
            limit_bytes=LIMIT,
            duration_seconds=60.0,
        )

    assert not destination.exists()


def test_ffmpeg_refusing_is_explained(runtime_path: Path, monkeypatch) -> None:
    def refuse(command, **kwargs):
        return SimpleNamespace(returncode=1, stderr="Invalid data found")

    monkeypatch.setattr("src.telegram_bot.shrink.subprocess.run", refuse)

    with pytest.raises(ShrinkFailed, match="админке"):
        shrink_to_fit(
            "ffmpeg",
            runtime_path / "source.mp4",
            runtime_path / "fit.mp4",
            limit_bytes=LIMIT,
            duration_seconds=60.0,
        )


def test_a_missing_ffmpeg_is_explained(runtime_path: Path, monkeypatch) -> None:
    """Без кодировщика кнопка обязана сказать это, а не молчать."""
    def explode(command, **kwargs):
        raise OSError("No such file or directory: 'ffmpeg'")

    monkeypatch.setattr("src.telegram_bot.shrink.subprocess.run", explode)

    with pytest.raises(ShrinkFailed, match="ffmpeg"):
        shrink_to_fit(
            "ffmpeg",
            runtime_path / "source.mp4",
            runtime_path / "fit.mp4",
            limit_bytes=LIMIT,
            duration_seconds=60.0,
        )


# --- сама команда --------------------------------------------------------------


def test_the_command_keeps_the_clip_playable_on_a_phone() -> None:
    """Ролик обязан открываться на том единственном экране, где его смотрят."""
    command = encode_command("ffmpeg", Path("a.mp4"), Path("b.mp4"), Plan(width=1280, video_kbps=900))

    assert "+faststart" in command
    assert command[command.index("-pix_fmt") + 1] == "yuv420p"
    # Кадр только уменьшается: растягивать его значило бы платить битрейтом за
    # придуманные пиксели.
    assert "min(iw,1280)" in command[command.index("-vf") + 1]
    # Звук у записи прохода отсутствует вовсе, у склейки Frigate он есть —
    # вопросительный знак оставляет команду рабочей в обоих случаях.
    assert "0:a:0?" in command
    assert command[command.index("-b:a") + 1] == f"{AUDIO_KBPS}k"


def test_constant_quality_is_used_when_the_length_is_unknown() -> None:
    command = encode_command("ffmpeg", Path("a.mp4"), Path("b.mp4"), Plan(width=854, crf=34))

    assert "-b:v" not in command
    assert command[command.index("-crf") + 1] == "34"


# --- длина ролика --------------------------------------------------------------


def test_the_length_comes_from_the_neighbouring_ffprobe(monkeypatch) -> None:
    """Настройка на узле одна: разводить ffmpeg и ffprobe по разным сборкам
    значило бы завести ещё один способ сломать узел молча."""
    seen: dict[str, list[str]] = {}

    def fake_run(command, **kwargs):
        seen["command"] = list(command)
        return SimpleNamespace(
            returncode=0, stdout=json.dumps({"format": {"duration": "12.5"}}), stderr=""
        )

    monkeypatch.setattr("src.telegram_bot.shrink.subprocess.run", fake_run)

    assert probe_duration("/opt/bin/ffmpeg", Path("clip.mp4")) == 12.5
    assert seen["command"][0] == "/opt/bin/ffprobe"
    assert ffprobe_path("ffmpeg") == "ffprobe"


def test_an_unmeasurable_clip_does_not_raise(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.telegram_bot.shrink.subprocess.run",
        lambda command, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="broken"),
    )

    assert probe_duration("ffmpeg", Path("clip.mp4")) is None


def test_the_encode_waits_longer_for_a_longer_clip() -> None:
    """Предел не про качество, а про то, чтобы зависший ffmpeg не держал службу."""
    assert encode_timeout(10.0) == 120.0
    assert encode_timeout(300.0) == 1800.0
    assert encode_timeout(None) > 0
