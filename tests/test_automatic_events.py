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
