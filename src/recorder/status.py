from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


class RecentFrameCounters:
    def __init__(
        self,
        roles: Iterable[str] = ("face", "side"),
        window_seconds: float = 60.0,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self.window_seconds = window_seconds
        self._processed = {role: deque[float]() for role in roles}
        self._skipped_no_motion = {role: deque[float]() for role in roles}

    def record_processed(self, role: str, timestamp: float) -> None:
        self._processed[role].append(timestamp)

    def record_skipped_no_motion(self, role: str, timestamp: float) -> None:
        self._skipped_no_motion[role].append(timestamp)

    def snapshot(self, timestamp: float) -> dict[str, dict[str, int]]:
        cutoff = timestamp - self.window_seconds
        for counters in (self._processed, self._skipped_no_motion):
            for values in counters.values():
                while values and values[0] < cutoff:
                    values.popleft()
        return {
            "frames_processed": {
                role: len(values)
                for role, values in self._processed.items()
            },
            "frames_skipped_no_motion": {
                role: len(values)
                for role, values in self._skipped_no_motion.items()
            },
        }


def write_recorder_status(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": datetime.now(UTC).isoformat(), **values}
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        for attempt in range(20):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.025)
    finally:
        temporary.unlink(missing_ok=True)


def read_recorder_status(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None
