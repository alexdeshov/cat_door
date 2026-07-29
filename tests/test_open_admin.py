from __future__ import annotations

import pytest

from scripts import open_admin


def test_find_chrome_uses_program_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program_files = r"C:\Program Files"
    monkeypatch.setenv("PROGRAMFILES", program_files)
    monkeypatch.delenv("PROGRAMFILES(X86)", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setattr(open_admin.Path, "is_file", lambda _: True)

    assert open_admin.find_chrome() == (
        open_admin.Path(program_files) / "Google" / "Chrome" / "Application" / "chrome.exe"
    )


def test_wait_for_admin_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    class Response:
        def __enter__(self) -> object:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def fake_urlopen(url: str, timeout: int) -> Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError
        return Response()

    monkeypatch.setattr(open_admin, "urlopen", fake_urlopen)
    monkeypatch.setattr(open_admin.time, "sleep", lambda _: None)

    assert open_admin.wait_for_admin(attempts=2)
    assert calls == 2
