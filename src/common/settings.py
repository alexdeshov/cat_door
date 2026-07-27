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
    local_training_epochs: int = Field(default=2, ge=1, le=100)
    web_host: str = "127.0.0.1"
    web_port: int = Field(default=8080, ge=1, le=65535)

    @property
    def resolved_database_path(self) -> Path:
        return self.database_path or self.data_dir / "catdoor.sqlite3"
