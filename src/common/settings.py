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
    identity_face_model_path: Path | None = None
    identity_side_model_path: Path | None = None
    identity_pretrained_weights_path: Path | None = None
    identity_confidence_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    detector_face_sample_fps: float = Field(default=8.0, gt=0.0, le=10.0)
    detector_sample_fps: float = Field(default=3.0, gt=0.0, le=10.0)
    ring_segment_seconds: int = Field(default=2, ge=1, le=10)
    ring_retention_seconds: int = Field(default=180, ge=30, le=1800)
    automatic_prebuffer_seconds: int = Field(default=12, ge=0, le=60)
    detection_confidence_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    detection_box_min_confidence: float = Field(default=0.20, ge=0.0, le=1.0)
    detection_confirmation_frames: int = Field(default=2, ge=1, le=20)
    detection_window_frames: int = Field(default=5, ge=1, le=30)
    detection_candidate_timeout_seconds: float = Field(default=3.0, gt=0.0, le=30.0)
    detection_absence_seconds: float = Field(default=3.0, gt=0.0, le=60.0)
    detection_tail_seconds: float = Field(default=10.0, ge=0.0, le=60.0)
    detection_cooldown_seconds: float = Field(default=5.0, ge=0.0, le=300.0)
    detection_log_max_bytes: int = Field(default=10_000_000, ge=100_000, le=100_000_000)
    detection_log_backup_count: int = Field(default=5, ge=1, le=20)
    local_training_epochs: int = Field(default=2, ge=1, le=100)
    web_host: str = "127.0.0.1"
    web_port: int = Field(default=8080, ge=1, le=65535)
    watchdog_recorder_stale_seconds: int = Field(default=180, ge=30, le=1800)
    watchdog_restart_cooldown_seconds: int = Field(default=600, ge=60, le=3600)
    watchdog_max_restart_attempts: int = Field(default=3, ge=1, le=10)
    watchdog_failure_backoff_seconds: int = Field(default=1800, ge=300, le=21600)
    watchdog_web_timeout_seconds: float = Field(default=2.0, gt=0.0, le=10.0)

    @property
    def resolved_database_path(self) -> Path:
        return self.database_path or self.data_dir / "catdoor.sqlite3"
