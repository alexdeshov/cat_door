from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from src.common.settings import Settings


def main() -> None:
    settings = Settings()
    log_dir = settings.data_dir / "runtime"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (
        (log_dir / "web.stdout.log").open("ab") as stdout,
        (log_dir / "web.stderr.log").open("ab") as stderr,
    ):
        process = subprocess.Popen(
            [sys.executable, "-m", "src.web.app"],
            cwd=Path(__file__).resolve().parent.parent,
            env=dict(os.environ),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
            close_fds=True,
        )
    print(process.pid)


if __name__ == "__main__":
    main()
