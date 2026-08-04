import pytest

from src.recorder.automatic import (
    AutomaticEventAction,
    AutomaticEventConfig,
    AutomaticEventMachine,
    AutomaticEventState,
)


def make_machine(**overrides: object) -> AutomaticEventMachine:
    values = {
        "confidence_threshold": 0.6,
        "confirmation_frames": 2,
        "window_frames": 3,
        "candidate_timeout_seconds": 2.0,
        "absence_seconds": 2.0,
        "tail_seconds": 3.0,
        "cooldown_seconds": 2.0,
        **overrides,
    }
    return AutomaticEventMachine(AutomaticEventConfig(**values))


def test_single_detection_does_not_start_event() -> None:
    machine = make_machine()

    first = machine.observe(0.0, {"face": 0.9, "side": 0.0})
    second = machine.observe(2.1, {"face": 0.0, "side": 0.0})

    assert first.state is AutomaticEventState.CANDIDATE
    assert first.action is None
    assert second.state is AutomaticEventState.IDLE
    assert second.action is None


def test_detections_from_either_camera_confirm_one_event() -> None:
    machine = make_machine()

    machine.observe(0.0, {"face": 0.8, "side": 0.0})
    update = machine.observe(0.5, {"face": 0.0, "side": 0.7})

    assert update.state is AutomaticEventState.ACTIVE
    assert update.action is AutomaticEventAction.START
    assert update.initiating_camera == "side"


def test_single_frame_confirmation_can_start_immediately() -> None:
    machine = make_machine(confirmation_frames=1)

    update = machine.observe(0.0, {"face": 0.8})

    assert update.state is AutomaticEventState.ACTIVE
    assert update.action is AutomaticEventAction.START


def test_event_uses_absence_tail_and_cooldown() -> None:
    machine = make_machine()
    machine.observe(0.0, {"face": 0.8})
    machine.observe(0.5, {"face": 0.8})

    assert machine.observe(2.4, {"face": 0.0}).state is AutomaticEventState.ACTIVE
    assert machine.observe(2.5, {"face": 0.0}).state is AutomaticEventState.TAIL
    stop = machine.observe(5.5, {"face": 0.0})
    assert stop.state is AutomaticEventState.COOLDOWN
    assert stop.action is AutomaticEventAction.STOP

    ignored = machine.observe(6.0, {"face": 0.9})
    assert ignored.state is AutomaticEventState.COOLDOWN
    assert ignored.action is None
    assert machine.observe(7.5, {"face": 0.0}).state is AutomaticEventState.IDLE


def test_detection_during_tail_continues_same_event() -> None:
    machine = make_machine()
    machine.observe(0.0, {"face": 0.8})
    machine.observe(0.5, {"face": 0.8})
    machine.observe(2.5, {"face": 0.0})

    resumed = machine.observe(3.0, {"side": 0.9})

    assert resumed.state is AutomaticEventState.ACTIVE
    assert resumed.action is None


def test_rejects_invalid_configuration_and_time_order() -> None:
    with pytest.raises(ValueError):
        AutomaticEventConfig(confirmation_frames=4, window_frames=3)

    machine = make_machine()
    machine.observe(2.0, {"face": 0.0})
    with pytest.raises(ValueError):
        machine.observe(1.0, {"face": 0.0})


def test_a_cat_asleep_at_the_door_does_not_record_forever() -> None:
    """Главная защита от переполнения диска.

    Детектор видит спящего кота так же уверенно, как идущего, поэтому без
    потолка событие оставалось бы ACTIVE, пока кот не уйдёт, — одна детекция
    записала бы диск досуха.
    """
    machine = make_machine(confirmation_frames=1, max_event_seconds=60.0)
    machine.observe(0.0, {"face": 0.9})

    # Кот в кадре непрерывно, но событие закрывается по потолку.
    last = None
    for tenth in range(1, 621):
        last = machine.observe(tenth / 10.0, {"face": 0.9})
        if last.action is AutomaticEventAction.STOP:
            break

    assert last is not None
    assert last.action is AutomaticEventAction.STOP
    assert last.state is AutomaticEventState.COOLDOWN
    # Стоп наступил ровно на потолке, а не позже.
    assert machine._last_timestamp == pytest.approx(60.0, abs=0.15)


def test_the_cap_counts_from_the_first_start_across_tail() -> None:
    """Кот, вышедший и вернувшийся, не обнуляет счётчик — клип один."""
    machine = make_machine(confirmation_frames=1, max_event_seconds=10.0, absence_seconds=2.0)
    machine.observe(0.0, {"face": 0.9})
    # Ушёл в tail и вернулся в active — это то же событие.
    machine.observe(3.0, {"face": 0.0})
    assert machine.observe(4.0, {"face": 0.9}).state is AutomaticEventState.ACTIVE

    stop = machine.observe(10.0, {"face": 0.9})

    assert stop.action is AutomaticEventAction.STOP


def test_a_new_event_starts_after_the_cap_if_the_cat_stays() -> None:
    """После принудительного стопа спящий кот даёт новое событие, не вечное."""
    machine = make_machine(
        confirmation_frames=1, max_event_seconds=5.0, cooldown_seconds=2.0
    )
    machine.observe(0.0, {"face": 0.9})
    stop = machine.observe(5.0, {"face": 0.9})
    assert stop.action is AutomaticEventAction.STOP

    # Пауза прошла, кот всё ещё там — начинается следующий клип.
    restart = machine.observe(7.0, {"face": 0.9})
    assert restart.action is AutomaticEventAction.START


def test_zero_cap_means_no_limit() -> None:
    """Ноль отключает потолок — событие живёт, пока кот в кадре."""
    machine = make_machine(confirmation_frames=1, max_event_seconds=0.0)
    machine.observe(0.0, {"face": 0.9})

    for tenth in range(1, 1200):
        update = machine.observe(tenth / 10.0, {"face": 0.9})
        assert update.state is AutomaticEventState.ACTIVE
