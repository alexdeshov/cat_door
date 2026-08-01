from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO


def spawn_detached(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    stdout: IO[bytes] | int = subprocess.DEVNULL,
    stderr: IO[bytes] | int = subprocess.DEVNULL,
) -> subprocess.Popen[bytes]:
    """Start a background process that survives the caller's session.

    `start_new_session=True` puts the child in its own session and process group,
    so a SIGHUP or SIGINT delivered to the parent (closing an SSH session, Ctrl+C
    in a terminal) does not reach it.
    """
    return subprocess.Popen(
        list(command),
        cwd=cwd,
        env=dict(env) if env is not None else None,
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
        close_fds=True,
    )
