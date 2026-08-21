"""Состояние створки: что известно, чему верить и когда сказать «не знаю»."""

from __future__ import annotations

from types import SimpleNamespace

from src.gate.gate_state import CLOSED, OPEN, UNKNOWN, GateStateWatcher


def _watcher(now: list[float]) -> GateStateWatcher:
    return GateStateWatcher("127.0.0.1", clock=lambda: now[0])


def _send(watcher: GateStateWatcher, text: str, topic: str = "frigate/cam/classification/gate") -> None:
    watcher._on_message(
        None, None, SimpleNamespace(payload=text.encode("utf-8"), topic=topic)
    )


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


def test_knowledge_survives_quiet_hours_while_the_chain_holds() -> None:
    """Решение владельца 2026-08-14: доверие по цепочке, а не по секундомеру.

    Пятиминутный срок годности делал створку «неизвестной» большую часть суток
    (ночь, затишья), и второй шанс молчал именно тогда, когда импульсы чаще
    пропадают. Пока связь с брокером цела и Frigate жив, событий о створке не
    было — классификатор публикует каждое её движение, и молчание означает
    «не шевелилась», а не «неизвестно».
    """
    now = [100.0]
    watcher = _watcher(now)
    _send(watcher, "Закрыта")

    now[0] = 100.0 + 8 * 3600  # восемь часов тишины
    assert watcher.snapshot().value == CLOSED


def test_frigate_going_offline_breaks_the_trust_chain() -> None:
    """Завещание Frigate — обрыв нити: движение створки за простой потеряно.

    Забывается и отметка закрытия — она кормит окно возвращения и второй шанс.
    """
    now = [100.0]
    watcher = _watcher(now)
    _send(watcher, "Закрыта")

    _send(watcher, "offline", topic="frigate/available")

    state = watcher.snapshot()
    assert state.value == UNKNOWN
    assert state.closed_at is None
    # «online» сам по себе доверия не возвращает: нужно свежее срабатывание.
    _send(watcher, "online", topic="frigate/available")
    assert watcher.snapshot().value == UNKNOWN
    _send(watcher, "Открыта")
    assert watcher.snapshot().value == "open"


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


def test_the_return_window_starts_when_the_gate_closes() -> None:
    """Именно закрылась, а не «сейчас закрыта».

    Открытая створка открытия не требует, а закрытая полдня назад ничего не
    говорит о том, кто стоит перед ней.
    """
    now = [100.0]
    watcher = _watcher(now)

    _send(watcher, "Открыта")
    assert watcher.snapshot().closed_within(30, now=now[0]) is False

    now[0] = 105.0
    _send(watcher, "Закрыта")
    now[0] = 130.0
    assert watcher.snapshot().closed_within(30, now=now[0]) is True
    now[0] = 140.0
    assert watcher.snapshot().closed_within(30, now=now[0]) is False


def test_losing_the_broker_forgets_the_closing_too() -> None:
    """За время без связи створку могли открыть и закрыть не один раз."""
    now = [100.0]
    watcher = _watcher(now)
    _send(watcher, "Закрыта")

    watcher._on_disconnect()

    assert watcher.snapshot().closed_within(30, now=now[0]) is False


def test_a_broker_that_is_not_up_yet_is_awaited_not_abandoned(monkeypatch) -> None:
    """Служба стартует раньше брокера — подключение добивается фоном.

    2026-08-21, 05:40: обновление ядра перезагрузило узел, служба поднялась
    раньше mosquitto, синхронный connect упал одним «Connection refused» — и
    три часа, до ручного перезапуска, створка не слушалась вовсе: правило
    стороны стояло разоружённым. Поэтому подключение асинхронное: первая
    неудача — не приговор, paho добивается соединения сам.
    """
    import sys
    from types import SimpleNamespace

    calls: list[str] = []

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.on_connect = None
            self.on_message = None
            self.on_disconnect = None

        def connect_async(self, *args: object) -> None:
            # Настоящий connect_async не ходит в сеть и не падает от того,
            # что брокер ещё не поднялся.
            calls.append("connect_async")

        def connect(self, *args: object) -> None:
            raise AssertionError("синхронный connect вернул бы вечную слепоту")

        def loop_start(self) -> None:
            calls.append("loop_start")

    fake = SimpleNamespace(
        Client=FakeClient,
        CallbackAPIVersion=SimpleNamespace(VERSION2=2),
    )
    monkeypatch.setitem(sys.modules, "paho", SimpleNamespace(mqtt=SimpleNamespace(client=fake)))
    monkeypatch.setitem(sys.modules, "paho.mqtt", SimpleNamespace(client=fake))
    monkeypatch.setitem(sys.modules, "paho.mqtt.client", fake)

    assert GateStateWatcher("127.0.0.1").start() is True
    assert calls == ["connect_async", "loop_start"]
