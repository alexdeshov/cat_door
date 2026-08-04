from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


class AutomaticEventState(StrEnum):
    IDLE = "idle"
    CANDIDATE = "candidate"
    ACTIVE = "active"
    TAIL = "tail"
    COOLDOWN = "cooldown"


class AutomaticEventAction(StrEnum):
    START = "start"
    STOP = "stop"


@dataclass(frozen=True)
class AutomaticEventConfig:
    confidence_threshold: float = 0.55
    confirmation_frames: int = 3
    window_frames: int = 5
    candidate_timeout_seconds: float = 3.0
    absence_seconds: float = 3.0
    tail_seconds: float = 10.0
    cooldown_seconds: float = 5.0
    # Потолок длительности одного события. Без него кот, заснувший у дверцы,
    # держал бы событие активным сколько угодно — детектор видит спящего так же
    # уверенно, как идущего, — и одна детекция записала бы диск досуха. По
    # достижении потолка событие закрывается принудительно; если кот всё ещё
    # там, после паузы начнётся новое, но каждый клип ограничен. Ноль — без
    # лимита.
    max_event_seconds: float = 60.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if self.confirmation_frames < 1:
            raise ValueError("confirmation_frames must be positive")
        if self.window_frames < self.confirmation_frames:
            raise ValueError("window_frames must be at least confirmation_frames")
        if self.candidate_timeout_seconds <= 0:
            raise ValueError("candidate_timeout_seconds must be positive")
        if self.absence_seconds <= 0:
            raise ValueError("absence_seconds must be positive")
        if self.tail_seconds < 0 or self.cooldown_seconds < 0:
            raise ValueError("tail_seconds and cooldown_seconds cannot be negative")
        if self.max_event_seconds < 0:
            raise ValueError("max_event_seconds cannot be negative")


@dataclass(frozen=True)
class AutomaticEventUpdate:
    state: AutomaticEventState
    action: AutomaticEventAction | None
    detected: bool
    initiating_camera: str | None
    max_confidence: float


class AutomaticEventMachine:
    """Turns sparse cat detections from either camera into stable event boundaries."""

    def __init__(self, config: AutomaticEventConfig) -> None:
        self.config = config
        self.state = AutomaticEventState.IDLE
        self._window: deque[bool] = deque(maxlen=config.window_frames)
        self._candidate_started_at: float | None = None
        self._last_detection_at: float | None = None
        self._active_started_at: float | None = None
        self._tail_started_at: float | None = None
        self._cooldown_started_at: float | None = None
        self._last_timestamp: float | None = None

    def observe(
        self,
        timestamp: float,
        confidences: Mapping[str, float | None],
    ) -> AutomaticEventUpdate:
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("timestamps must be monotonic")
        self._last_timestamp = timestamp

        camera, confidence = max(
            ((role, value or 0.0) for role, value in confidences.items()),
            key=lambda item: item[1],
            default=(None, 0.0),
        )
        detected = confidence >= self.config.confidence_threshold
        initiating_camera = camera if detected else None
        action: AutomaticEventAction | None = None

        if self.state is AutomaticEventState.COOLDOWN:
            assert self._cooldown_started_at is not None
            if timestamp - self._cooldown_started_at >= self.config.cooldown_seconds:
                self._reset_to_idle()
            else:
                return self._update(action, detected, initiating_camera, confidence)

        # Потолок длительности перебивает обычную логику: событие закрывается,
        # даже если кот по-прежнему в кадре. Иначе спящий у дверцы кот держал бы
        # запись бесконечно. Проверка до состояний, чтобы ни ACTIVE, ни TAIL не
        # успели продлить событие в этом же кадре.
        if self.state in (AutomaticEventState.ACTIVE, AutomaticEventState.TAIL) and (
            self._event_ran_out_of_time(timestamp)
        ):
            self.state = AutomaticEventState.COOLDOWN
            self._cooldown_started_at = timestamp
            self._active_started_at = None
            self._tail_started_at = None
            return self._update(
                AutomaticEventAction.STOP, detected, initiating_camera, confidence
            )

        if self.state is AutomaticEventState.IDLE:
            if detected:
                self.state = AutomaticEventState.CANDIDATE
                self._candidate_started_at = timestamp
                self._last_detection_at = timestamp
                self._window.append(True)
                if self.config.confirmation_frames == 1:
                    self.state = AutomaticEventState.ACTIVE
                    self._active_started_at = timestamp
                    action = AutomaticEventAction.START

        elif self.state is AutomaticEventState.CANDIDATE:
            self._window.append(detected)
            if detected:
                self._last_detection_at = timestamp
            if sum(self._window) >= self.config.confirmation_frames:
                self.state = AutomaticEventState.ACTIVE
                self._last_detection_at = timestamp
                self._active_started_at = timestamp
                action = AutomaticEventAction.START
            elif (
                self._candidate_started_at is not None
                and timestamp - self._candidate_started_at
                >= self.config.candidate_timeout_seconds
            ):
                self._reset_to_idle()

        elif self.state is AutomaticEventState.ACTIVE:
            if detected:
                self._last_detection_at = timestamp
            elif (
                self._last_detection_at is not None
                and timestamp - self._last_detection_at >= self.config.absence_seconds
            ):
                self.state = AutomaticEventState.TAIL
                self._tail_started_at = timestamp

        elif self.state is AutomaticEventState.TAIL:
            if detected:
                self.state = AutomaticEventState.ACTIVE
                self._last_detection_at = timestamp
                self._tail_started_at = None
            elif (
                self._tail_started_at is not None
                and timestamp - self._tail_started_at >= self.config.tail_seconds
            ):
                self.state = AutomaticEventState.COOLDOWN
                self._cooldown_started_at = timestamp
                action = AutomaticEventAction.STOP

        return self._update(action, detected, initiating_camera, confidence)

    def _event_ran_out_of_time(self, timestamp: float) -> bool:
        return (
            self.config.max_event_seconds > 0
            and self._active_started_at is not None
            and timestamp - self._active_started_at >= self.config.max_event_seconds
        )

    def _reset_to_idle(self) -> None:
        self.state = AutomaticEventState.IDLE
        self._window.clear()
        self._candidate_started_at = None
        self._last_detection_at = None
        self._active_started_at = None
        self._tail_started_at = None
        self._cooldown_started_at = None

    def _update(
        self,
        action: AutomaticEventAction | None,
        detected: bool,
        initiating_camera: str | None,
        confidence: float,
    ) -> AutomaticEventUpdate:
        return AutomaticEventUpdate(
            state=self.state,
            action=action,
            detected=detected,
            initiating_camera=initiating_camera,
            max_confidence=confidence,
        )
