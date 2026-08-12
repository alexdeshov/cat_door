"""Состояние створки: что известно, чему верить и когда сказать «не знаю»."""

from __future__ import annotations

from types import SimpleNamespace

from src.gate.gate_state import CLOSED, OPEN, UNKNOWN, GateStateWatcher


def _watcher(now: list[float]) -> GateStateWatcher:
    return GateStateWatcher("127.0.0.1", clock=lambda: now[0])


def _send(watcher: GateStateWatcher, text: str) -> None:
    watcher._on_message(None, None, SimpleNamespace(payload=text.encode("utf-8")))


def test_nothing_is_known_before_the_gate_moves() -> None:
    """Брокер состояние не запоминает: свежая подписка не получает ничего.

    Проверено на живом узле 2026-08-12. Подставлять «закрыта» нельзя: на «не
    знаю» нельзя основывать вывод «калитка не открылась, значит замок заклинил».
    """
    assert _watcher([100.0]).snapshot().value == UNKNOWN


def test_the_classifier_speaks_russian() -> None:
    """Названия состояний задал владелец при обучении, и они по-русски."""
    now = [100.0]
    watcher = _watcher(now)

    _send(watcher, "Открыта")
    assert watcher.snapshot().value == OPEN

    _send(watcher, "  закрыта  ")
    assert watcher.snapshot().value == CLOSED


def test_an_unfamiliar_state_is_ignored_not_guessed() -> None:
    now = [100.0]
    watcher = _watcher(now)
    _send(watcher, "Открыта")

    _send(watcher, "непонятно")

    assert watcher.snapshot().value == OPEN


def test_knowledge_goes_stale() -> None:
    """«Закрыта пять минут назад» и «закрыта сейчас» — разные утверждения.

    Створку могли открыть, пока подписка лежала.
    """
    now = [100.0]
    watcher = _watcher(now)
    _send(watcher, "Закрыта")

    now[0] = 100.0 + 299
    assert watcher.snapshot().value == CLOSED
    now[0] = 100.0 + 301
    assert watcher.snapshot().value == UNKNOWN


def test_losing_the_broker_means_losing_the_knowledge() -> None:
    """Потеряв связь, перестаём утверждать что бы то ни было.

    Молча продолжать говорить «закрыта» — ровно тот способ соврать, которого
    этот проект избегает везде.
    """
    now = [100.0]
    watcher = _watcher(now)
    _send(watcher, "Открыта")

    watcher._on_disconnect()

    assert watcher.snapshot().value == UNKNOWN


def test_a_missing_broker_is_not_a_crash() -> None:
    """Служба обязана работать и без брокера: журнал проходов важнее."""
    assert GateStateWatcher("").start() is False
