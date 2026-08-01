from __future__ import annotations

import os
import sys
from pathlib import Path

from src.common.process import spawn_detached
from src.common.settings import Settings


def start_web() -> int:
    settings = Settings()
    log_dir = settings.data_dir / "runtime"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (
        (log_dir / "web.stdout.log").open("ab") as stdout,
        (log_dir / "web.stderr.log").open("ab") as stderr,
    ):
        process = spawn_detached(
            [sys.executable, "-m", "src.web.app"],
            cwd=Path(__file__).resolve().parent.parent,
            env=dict(os.environ),
            stdout=stdout,
            stderr=stderr,
        )
    return process.pid


def main() -> None:
    print(start_web())


if __name__ == "__main__":
    main()
