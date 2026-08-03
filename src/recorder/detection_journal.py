from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

# События, которые пишутся на каждый разобранный кадр. Их тысячи в час, и в
# общем файле они вытесняли всё остальное меньше чем за два часа: обрыв камеры,
# случившийся ночью, к утру было уже не найти, хотя ради таких находок журнал и
# заведён. Поэтому они живут отдельным файлом со своей ротацией — покадровая
# история остаётся такой же короткой, а летопись происшествий живёт неделями.
FRAME_EVENTS = frozenset({"observation", "identity_skipped"})


def frame_journal_path(path: Path) -> Path:
    return path.with_name("observations.jsonl")


class DetectionJournal:
    def __init__(self, path: Path, *, max_bytes: int, backup_count: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._logger = self._open(path, max_bytes, backup_count, "detection")
        self._frames = self._open(frame_journal_path(path), max_bytes, backup_count, "frames")

    def _open(self, path: Path, max_bytes: int, backup_count: int, name: str) -> logging.Logger:
        logger = logging.getLogger(f"catdoor.{name}.{id(self)}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = RotatingFileHandler(
            path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        return logger

    def write(self, event: str, **values: Any) -> None:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            **values,
        }
        logger = self._frames if event in FRAME_EVENTS else self._logger
        logger.info(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    def close(self) -> None:
        for logger in (self._logger, self._frames):
            for handler in tuple(logger.handlers):
                handler.close()
                logger.removeHandler(handler)
