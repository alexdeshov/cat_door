"""Журналы служб в админке: источники, чтение, отказ, сверка с юнитами."""

from __future__ import annotations

import subprocess
from pathlib import Path

from src.web import logs
from src.web.logs import SOURCES, LogSource, clamp_lines, read_file_tail, read_unit_log

INSTALL = Path(__file__).resolve().parents[1] / "install"


def test_every_unit_has_a_log_source() -> None:
    """Юнит без источника здесь — служба, чей журнал снова добывать по ssh.

    Список сверяется с каталогом install/, а не ведётся руками: рукописный
    список — ровно то, из-за чего сторож сутки не видел кошачью копию.
    """
    units = {p.stem for p in INSTALL.glob("*.service")}
    covered = {s.unit for s in SOURCES if s.unit}

    assert units - covered == set()


def test_source_keys_are_unique_and_url_safe() -> None:
    keys = [s.key for s in SOURCES]
    assert len(keys) == len(set(keys))
    assert all(k.replace("-", "").isalnum() for k in keys)


def test_lines_are_clamped() -> None:
    assert clamp_lines(None) == logs.DEFAULT_LINES
    assert clamp_lines(0) == 1
    assert clamp_lines(10**9) == logs.MAX_LINES


def test_empty_journal_is_not_an_error(monkeypatch) -> None:
    """Пустой журнал и недоступный журнал — разные ответы."""
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "-- No entries --\n", ""))

    text, error = read_unit_log("catdoor-recorder", 50)

    assert text == "" and error is None


def test_forbidden_journal_names_the_fix(monkeypatch) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "No journal files were found.\n"))

    text, error = read_unit_log("catdoor-recorder", 50)

    assert text == ""
    assert error is not None and "systemd-journal" in error and "install.sh" in error


def test_journal_lines_are_passed_through(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "2026-08-31T05:00:00+0300 acer python[1]: ok\n", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    text, error = read_unit_log("catdoor-backup", 7)

    assert error is None and "ok" in text
    assert calls[0][:5] == ["journalctl", "-u", "catdoor-backup", "-n", "7"]


def test_file_tail_keeps_the_last_lines(tmp_path: Path) -> None:
    path = tmp_path / "detection.jsonl"
    path.write_text("".join(f'{{"n": {i}}}\n' for i in range(50)), encoding="utf-8")

    text, error = read_file_tail(path, 3)

    assert error is None
    assert text.splitlines() == ['{"n": 47}', '{"n": 48}', '{"n": 49}']


def test_missing_file_says_so(tmp_path: Path) -> None:
    text, error = read_file_tail(tmp_path / "нет.jsonl", 3)

    assert text == "" and error is not None and "ещё нет" in error


def test_the_page_and_api_are_served(runtime_path: Path, monkeypatch) -> None:
    from tests.test_admin import make_client

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "строка журнала\n", ""))
    client = make_client(runtime_path, monkeypatch)

    page = client.get("/logs?source=healthwatch")
    body = client.get("/api/logs/healthwatch?lines=5").json()
    missing = client.get("/api/logs/нет-такого")

    assert page.status_code == 200 and "Журналы служб" in page.text
    assert '<option value="healthwatch" selected>' in page.text
    assert body["text"] == "строка журнала\n" and body["error"] is None
    assert missing.status_code == 404


def test_detection_decisions_are_a_source_too(runtime_path: Path, monkeypatch) -> None:
    """detection.jsonl раньше доставался только через владельца."""
    from tests.test_admin import make_client

    client = make_client(runtime_path, monkeypatch)
    (runtime_path / "runtime").mkdir(parents=True, exist_ok=True)
    (runtime_path / "runtime" / "detection.jsonl").write_text('{"kind": "candidate"}\n', encoding="utf-8")

    body = client.get("/api/logs/detection").json()

    assert body["text"].strip() == '{"kind": "candidate"}'
    assert isinstance(SOURCES[-1], LogSource)
