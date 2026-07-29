from __future__ import annotations

from src.common import windows_power


def test_sleep_inhibitor_prevents_sleep_and_restores_default(monkeypatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(windows_power.sys, "platform", "win32")
    monkeypatch.setattr(
        windows_power,
        "_set_thread_execution_state",
        lambda flags: calls.append(flags) or 1,
    )

    with windows_power.SleepInhibitor():
        assert calls == [windows_power.ES_CONTINUOUS | windows_power.ES_SYSTEM_REQUIRED]

    assert calls == [
        windows_power.ES_CONTINUOUS | windows_power.ES_SYSTEM_REQUIRED,
        windows_power.ES_CONTINUOUS,
    ]


def test_sleep_inhibitor_is_noop_outside_windows(monkeypatch) -> None:
    monkeypatch.setattr(windows_power.sys, "platform", "darwin")
    monkeypatch.setattr(
        windows_power,
        "_set_thread_execution_state",
        lambda flags: (_ for _ in ()).throw(AssertionError(flags)),
    )

    with windows_power.SleepInhibitor():
        pass
