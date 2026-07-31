from __future__ import annotations

import numpy as np

from src.recorder.motion import MotionGate


def test_static_scene_has_no_motion_after_warmup() -> None:
    gate = MotionGate()
    frame = np.full((320, 320, 3), 80, dtype=np.uint8)

    warmup = [gate.has_motion(frame) for _ in range(10)]

    assert warmup == [True] * 10
    assert gate.has_motion(frame) is False


def test_moving_block_covering_five_percent_is_motion() -> None:
    gate = MotionGate()
    background = np.zeros((416, 416, 3), dtype=np.uint8)
    for _ in range(11):
        gate.has_motion(background)
    changed = background.copy()
    changed[100:196, 100:196] = 255

    assert gate.has_motion(changed) is True


def test_reset_restores_warmup_after_stream_reopens() -> None:
    gate = MotionGate()
    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    for _ in range(11):
        gate.has_motion(frame)

    gate.reset()

    assert gate.has_motion(frame) is True
