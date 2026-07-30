from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture
def runtime_path() -> Path:
    path = Path("data") / "test-runtime" / uuid4().hex
    path.mkdir(parents=True)
    return path
