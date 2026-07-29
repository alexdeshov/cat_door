from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CATDOOR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    camera_face_rtsp: str = ""
    camera_side_rtsp: str = ""
    data_dir: Path = Path("data")
    database_path: Path | None = None
    ffmpeg_path: str = "ffmpeg"
    preview_fps: int = Field(default=5, ge=1, le=10)
    capture_duration_seconds: int = Field(default=8, ge=3, le=30)
    automatic_capture_enabled: bool = False
    automatic_capture_dry_run: bool = True
    detector_model_path: Path = Path("models/cat-detector.onnx")
    detector_device: str = "CPU"
    detector_sample_fps: float = Field(default=3.0, gt=0.0, le=10.0)
    ring_segment_seconds: int = Field(default=2, ge=1, le=10)
    ring_retention_seconds: int = Field(default=180, ge=30, le=1800)
    automatic_prebuffer_seconds: int = Field(default=12, ge=0, le=60)
    detection_confidence_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    detection_confirmation_frames: int = Field(default=3, ge=1, le=20)
    detection_window_frames: int = Field(default=5, ge=1, le=30)
    detection_candidate_timeout_seconds: float = Field(default=3.0, gt=0.0, le=30.0)
    detection_absence_seconds: float = Field(default=3.0, gt=0.0, le=60.0)
    detection_tail_seconds: float = Field(default=10.0, ge=0.0, le=60.0)
    detection_cooldown_seconds: float = Field(default=5.0, ge=0.0, le=300.0)
    local_training_epochs: int = Field(default=2, ge=1, le=100)
    web_host: str = "127.0.0.1"
    web_port: int = Field(default=8080, ge=1, le=65535)

    @property
    def resolved_database_path(self) -> Path:
        return self.database_path or self.data_dir / "catdoor.sqlite3"
