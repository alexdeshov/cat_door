from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


class DetectionJournal:
    def __init__(self, path: Path, *, max_bytes: int, backup_count: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(f"catdoor.detection.{id(self)}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        handler = RotatingFileHandler(
            path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(handler)

    def write(self, event: str, **values: Any) -> None:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            **values,
        }
        self._logger.info(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    def close(self) -> None:
        for handler in tuple(self._logger.handlers):
            handler.close()
            self._logger.removeHandler(handler)
