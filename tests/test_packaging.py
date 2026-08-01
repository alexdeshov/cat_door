"""Guards for packaging metadata that broke the first Linux deployment.

Three separate defects surfaced only when the project was installed on the
Ubuntu runtime host: the suite could not be collected through the ``pytest``
console script, ``.[dev]`` did not install everything the tests import, and
setuptools mistook the flat layout for a src-layout.
"""

import importlib
import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def test_pytest_adds_repository_root_to_path(pyproject: dict) -> None:
    """Tests import ``src``/``scripts``/``training`` from the repository root.

    Without this the suite only collects under ``python -m pytest``, which
    happens to prepend the working directory. The ``pytest`` console script
    does not, and every module fails on import.
    """
    options = pyproject["tool"]["pytest"]["ini_options"]
    assert "." in options["pythonpath"]


@pytest.mark.parametrize("package", ["numpy", "pillow"])
def test_dev_extra_covers_test_imports(pyproject: dict, package: str) -> None:
    """``.[dev]`` must be enough to run the suite on a clean machine.

    numpy and Pillow used to live only in the training and inference extras,
    so the suite passed only in a virtualenv that also carried the training
    dependencies.
    """
    dev = pyproject["project"]["optional-dependencies"]["dev"]
    assert any(requirement.startswith(package) for requirement in dev)


def test_setuptools_keeps_flat_layout(pyproject: dict) -> None:
    """``src`` is a regular package, not the root of a src-layout.

    Left to auto-discovery, setuptools sees the ``src/`` directory, puts it on
    ``sys.path`` and exports ``common``/``recorder``/``web`` instead of
    ``src.common`` and friends, which no import in this repository uses.
    """
    include = pyproject["tool"]["setuptools"]["packages"]["find"]["include"]
    assert set(include) >= {"src*", "scripts*", "training*"}


def test_console_script_targets_are_importable(pyproject: dict) -> None:
    """Entry points must resolve, otherwise systemd units cannot call them."""
    for entry_point in pyproject["project"]["scripts"].values():
        module_name, _, attribute = entry_point.partition(":")
        module = importlib.import_module(module_name)
        assert callable(getattr(module, attribute))
