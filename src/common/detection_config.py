from __future__ import annotations

import json
import time
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from src.common.settings import Settings
from src.recorder.automatic import AutomaticEventConfig


class DetectorRoi(BaseModel):
    x: float = Field(ge=0.0, lt=1.0)
    y: float = Field(ge=0.0, lt=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_bounds(self) -> DetectorRoi:
        if self.x + self.width > 1.0 or self.y + self.height > 1.0:
            raise ValueError("ROI must fit inside normalized frame bounds")
        return self


class CameraDetectionConfig(BaseModel):
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    box_min_confidence: float = Field(ge=0.0, le=1.0)
    confirmation_frames: int = Field(ge=1, le=20)
    window_frames: int = Field(ge=1, le=30)
    sample_fps: float = Field(default=3.0, gt=0.0, le=10.0)
    roi: DetectorRoi | None = None
    motion_gating_enabled: bool = False
    motion_delta: float = Field(default=12.0, ge=0.0, le=255.0)
    motion_min_area_fraction: float = Field(default=0.005, gt=0.0, le=1.0)
    motion_alpha: float = Field(default=0.1, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_camera(self) -> CameraDetectionConfig:
        if self.window_frames < self.confirmation_frames:
            raise ValueError("window_frames must be at least confirmation_frames")
        if self.box_min_confidence > self.confidence_threshold:
            raise ValueError("box_min_confidence must not exceed confidence_threshold")
        return self


class DetectionConfig(BaseModel):
    face: CameraDetectionConfig
    side: CameraDetectionConfig
    candidate_timeout_seconds: float = Field(gt=0.0, le=30.0)
    absence_seconds: float = Field(gt=0.0, le=60.0)
    tail_seconds: float = Field(ge=0.0, le=60.0)
    cooldown_seconds: float = Field(ge=0.0, le=300.0)

    @classmethod
    def from_settings(cls, settings: Settings) -> DetectionConfig:
        side = CameraDetectionConfig(
            confidence_threshold=settings.detection_confidence_threshold,
            box_min_confidence=settings.detection_box_min_confidence,
            confirmation_frames=settings.detection_confirmation_frames,
            window_frames=settings.detection_window_frames,
            sample_fps=settings.detector_sample_fps,
            motion_gating_enabled=settings.motion_gating_enabled,
            motion_delta=settings.motion_delta,
            motion_min_area_fraction=settings.motion_min_area_fraction,
            motion_alpha=settings.motion_alpha,
        )
        return cls(
            face=CameraDetectionConfig(
                confidence_threshold=0.30,
                box_min_confidence=0.15,
                confirmation_frames=2,
                window_frames=5,
                sample_fps=settings.detector_face_sample_fps,
                motion_gating_enabled=settings.motion_gating_enabled,
                motion_delta=settings.motion_delta,
                motion_min_area_fraction=settings.motion_min_area_fraction,
                motion_alpha=settings.motion_alpha,
            ),
            side=side,
            candidate_timeout_seconds=settings.detection_candidate_timeout_seconds,
            absence_seconds=settings.detection_absence_seconds,
            tail_seconds=settings.detection_tail_seconds,
            cooldown_seconds=settings.detection_cooldown_seconds,
        )

    def event_config(self) -> AutomaticEventConfig:
        return AutomaticEventConfig(
            confidence_threshold=0.5,
            confirmation_frames=1,
            window_frames=1,
            candidate_timeout_seconds=self.candidate_timeout_seconds,
            absence_seconds=self.absence_seconds,
            tail_seconds=self.tail_seconds,
            cooldown_seconds=self.cooldown_seconds,
        )


def detection_config_path(data_dir: Path) -> Path:
    return data_dir / "runtime" / "detection-settings.json"


def load_detection_config(path: Path, defaults: DetectionConfig) -> DetectionConfig:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return defaults
    if "face" not in payload or "side" not in payload:
        camera = {
            key: payload[key]
            for key in (
                "confidence_threshold",
                "box_min_confidence",
                "confirmation_frames",
                "window_frames",
            )
            if key in payload
        }
        payload = {
            "face": {
                **defaults.face.model_dump(),
                **camera,
                "confidence_threshold": min(float(camera.get("confidence_threshold", 0.30)), 0.30),
                "box_min_confidence": min(float(camera.get("box_min_confidence", 0.15)), 0.15),
            },
            "side": {**defaults.side.model_dump(), **camera},
            **{
                key: payload.get(key, getattr(defaults, key))
                for key in (
                    "candidate_timeout_seconds",
                    "absence_seconds",
                    "tail_seconds",
                    "cooldown_seconds",
                )
            },
        }
    elif isinstance(payload["face"], dict) and isinstance(payload["side"], dict):
        payload = {
            **payload,
            "face": {**defaults.face.model_dump(), **payload["face"]},
            "side": {**defaults.side.model_dump(), **payload["side"]},
        }
    try:
        return DetectionConfig.model_validate(payload)
    except ValueError:
        return defaults


def save_detection_config(path: Path, config: DetectionConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(config.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        for attempt in range(10):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.01)
    finally:
        temporary.unlink(missing_ok=True)
