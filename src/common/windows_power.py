from __future__ import annotations

import ctypes
import sys
from types import TracebackType
from typing import Self

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def _set_thread_execution_state(flags: int) -> int:
    return int(ctypes.windll.kernel32.SetThreadExecutionState(flags))


class SleepInhibitor:
    """Keep Windows awake for the lifetime of the current server thread."""

    def __init__(self) -> None:
        self._active = False

    def __enter__(self) -> Self:
        if sys.platform != "win32":
            return self
        if _set_thread_execution_state(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) == 0:
            raise OSError("Windows не удалось запретить переход в спящий режим")
        self._active = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._active:
            _set_thread_execution_state(ES_CONTINUOUS)
            self._active = False
