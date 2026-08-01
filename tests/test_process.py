from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from src.common import process


def test_spawn_detached_uses_posix_session_and_no_windows_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_popen(command: Sequence[str], **kwargs: Any) -> object:
        captured["command"] = list(command)
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(process.subprocess, "Popen", fake_popen)

    process.spawn_detached(["python", "-V"], cwd=Path("/tmp"))

    assert captured["command"] == ["python", "-V"]
    assert captured["start_new_session"] is True
    assert captured["close_fds"] is True
    assert captured["stdin"] is subprocess.DEVNULL
    assert "creationflags" not in captured


def test_spawn_detached_copies_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        process.subprocess,
        "Popen",
        lambda command, **kwargs: captured.update(kwargs) or object(),
    )
    environment = {"CATDOOR_TEST_MODE": "true"}

    process.spawn_detached(["python", "-V"], env=environment)

    assert captured["env"] == environment
    assert captured["env"] is not environment


def test_spawn_detached_child_leaves_parent_session() -> None:
    """The child must survive a SIGHUP to the parent, so it needs its own session."""
    script = "import os, sys; sys.stdout.write(str(os.getsid(0)))"
    with tempfile.TemporaryFile() as output:
        child = process.spawn_detached([sys.executable, "-c", script], stdout=output)
        assert child.wait(timeout=30) == 0
        output.seek(0)
        child_session = int(output.read())

    assert child_session != os.getsid(0)
