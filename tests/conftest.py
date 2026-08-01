from pathlib import Path

import pytest


@pytest.fixture
def runtime_path(tmp_path: Path) -> Path:
    """Per-test scratch directory that pytest removes on its own.

    This used to create data/test-runtime/<uuid> relative to the working
    directory and never clean it up, which required running pytest from the
    repository root and left the tree littered.
    """
    return tmp_path
