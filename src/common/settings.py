import re
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigurationError(RuntimeError):
    """The configured paths cannot be used, so starting up would be pointless."""


# Camera ids end up inside stream URLs, so they stay to characters that survive
# a path segment without escaping.
_CAMERA_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

# The two roles the detector and the event schema are built around. Extra
# cameras are preview-only until that part of the model is generalised.
LEGACY_CAMERA_IDS = ("face", "side")


class CameraConfig(BaseModel):
    """One camera available for live preview.

    Separate from the `face`/`side` settings on purpose: those two carry a role
    the detector, the event tables and the identity models all depend on, while
    these are extra views used to choose mounting positions.
    """

    id: str
    rtsp: str
    rotate_180: bool = False
    title: str = ""

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        if not _CAMERA_ID.match(value):
            raise ValueError(
                "идентификатор камеры может содержать только латиницу в нижнем "
                f"регистре, цифры, дефис и подчёркивание, задано: {value!r}"
            )
        return value

    @property
    def display_title(self) -> str:
        return self.title or self.id


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CATDOOR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    camera_face_rtsp: str = ""
    camera_side_rtsp: str = ""
    camera_face_rotate_180: bool = True
    camera_side_rotate_180: bool = True
    cameras: list[CameraConfig] = Field(default_factory=list)
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

    @property
    def resolved_database_path(self) -> Path:
        return self.database_path or self.data_dir / "catdoor.sqlite3"

    def _known_cameras(self) -> dict[str, CameraConfig]:
        """Roles first, then the list, which may override a role by id.

        The two roles keep their own settings rather than being duplicated into
        the list, so an existing installation needs no configuration change.
        """
        by_id = {
            "face": CameraConfig(
                id="face",
                rtsp=self.camera_face_rtsp,
                rotate_180=self.camera_face_rotate_180,
                title="Камера идентификации",
            ),
            "side": CameraConfig(
                id="side",
                rtsp=self.camera_side_rtsp,
                rotate_180=self.camera_side_rotate_180,
                title="Камера подхода",
            ),
        }
        for camera in self.cameras:
            by_id[camera.id] = camera
        return by_id

    def preview_cameras(self) -> list[CameraConfig]:
        """Cameras worth putting on a page: the configured ones."""
        return [camera for camera in self._known_cameras().values() if camera.rtsp]

    def preview_camera(self, camera_id: str) -> CameraConfig | None:
        """Look up by id, including roles left unconfigured.

        A known role without an address has to stay distinguishable from a
        camera nobody ever declared: the first is a configuration gap, the
        second is a bad URL.
        """
        return self._known_cameras().get(camera_id)


def _require_writable_directory(directory: Path) -> None:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfigurationError(f"Не удалось создать каталог {directory}") from exc
    probe = directory / f".catdoor-write-check.{uuid4().hex}"
    try:
        probe.touch()
    except OSError as exc:
        raise ConfigurationError(f"Каталог недоступен для записи: {directory}") from exc
    finally:
        probe.unlink(missing_ok=True)


def validate_runtime_paths(settings: Settings) -> None:
    """Fail fast when the configured paths cannot work.

    The recorder's main loop swallows every exception and retries forever, so an
    unusable data directory would otherwise turn into a silent spin instead of a
    visible failure. Under systemd a relative path also resolves against /, which
    makes the default value unusable for a service.
    """
    if not settings.data_dir.is_absolute():
        raise ConfigurationError(
            f"CATDOOR_DATA_DIR должен быть абсолютным путём, задано: {settings.data_dir}"
        )
    if "\\" in str(settings.detector_model_path):
        raise ConfigurationError(
            "CATDOOR_DETECTOR_MODEL_PATH содержит обратный слэш — "
            f"на Linux это часть имени файла, а не разделитель: {settings.detector_model_path}"
        )
    _require_writable_directory(settings.data_dir)
    _require_writable_directory(settings.resolved_database_path.parent)
