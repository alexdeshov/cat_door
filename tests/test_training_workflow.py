import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.common.database import Event, create_session_factory
from src.common.detection_config import CameraDetectionConfig, DetectionConfig
from src.inference.cat_detector import CatDetection, CatDetectionResult
from training.dataset import load_manifest
from training.workflow import StatusHeartbeat, build_dataset


class FixedDetector:
    def __init__(self, detections: tuple[CatDetection, ...]) -> None:
        self.detections = detections
        self.frames: list[np.ndarray] = []

    def detect(self, frame: np.ndarray) -> CatDetectionResult:
        self.frames.append(frame)
        return CatDetectionResult(
            confidence=max(
                (detection.confidence for detection in self.detections),
                default=0.0,
            ),
            detections=self.detections,
        )


def detection_config() -> DetectionConfig:
    return DetectionConfig(
        face=CameraDetectionConfig(
            confidence_threshold=0.30,
            box_min_confidence=0.15,
            confirmation_frames=2,
            window_frames=5,
            sample_fps=8.0,
            roi={"x": 0.25, "y": 0.25, "width": 0.5, "height": 0.5},
        ),
        side=CameraDetectionConfig(
            confidence_threshold=0.55,
            box_min_confidence=0.20,
            confirmation_frames=2,
            window_frames=5,
            sample_fps=3.0,
        ),
        candidate_timeout_seconds=3,
        absence_seconds=3,
        tail_seconds=10,
        cooldown_seconds=5,
    )


def test_status_heartbeat_preserves_progress_and_refreshes_timestamp(
    runtime_path: Path,
) -> None:
    status_path = runtime_path / "status.json"
    status_path.write_text(
        json.dumps(
            {
                "updated_at": "2000-01-01T00:00:00+00:00",
                "state": "running",
                "completed": 3,
            }
        ),
        encoding="utf-8",
    )

    StatusHeartbeat(status_path).refresh()

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["updated_at"] != "2000-01-01T00:00:00+00:00"
    assert payload["state"] == "running"
    assert payload["completed"] == 3


def test_build_dataset_uses_selected_events_only(runtime_path: Path, monkeypatch) -> None:
    database_path = runtime_path / "catdoor.sqlite3"
    event_dir = runtime_path / "events" / "00000001"
    event_dir.mkdir(parents=True)
    (event_dir / "face.mp4").write_bytes(b"video")
    (event_dir / "side.mp4").write_bytes(b"video")
    factory = create_session_factory(database_path)
    with factory() as session:
        session.add(
            Event(
                created_at=datetime.now(UTC),
                status="ready",
                duration_seconds=5,
                media_path="events/00000001/face.mp4",
                side_media_path="events/00000001/side.mp4",
                label="known_cat",
                cat_name="Барсик",
                training_target="local",
            )
        )
        session.add(
            Event(
                created_at=datetime.now(UTC),
                status="ready",
                duration_seconds=5,
                media_path="events/00000002/face.mp4",
                label="known_cat",
                cat_name="Не выбран",
                training_target="macbook",
            )
        )
        session.commit()

    rotations: list[bool] = []

    def fake_extract(
        ffmpeg_path: str,
        source: Path,
        destination_pattern: Path,
        *,
        rotate_180: bool,
    ) -> list[Path]:
        rotations.append(rotate_180)
        destination = destination_pattern.parent / "frame-01.jpg"
        destination.parent.mkdir(parents=True)
        Image.fromarray(
            np.full((360, 640, 3), 100, dtype=np.uint8),
            mode="RGB",
        ).save(destination)
        return [destination]

    monkeypatch.setattr("training.workflow.extract_frames", fake_extract)
    destination = runtime_path / "training" / "dataset"
    detector = FixedDetector(
        (CatDetection(box=(80, 115, 240, 205), confidence=0.80),)
    )

    manifest, report = build_dataset(
        database_path,
        runtime_path,
        "ffmpeg",
        "local",
        destination,
        detector_model_path=Path("unused.onnx"),
        detector_device="CPU",
        detector=detector,
        default_detection_config=detection_config(),
    )

    items = load_manifest(manifest)
    assert report["events"] == 1
    assert report["events_per_label"] == {"Барсик": 1}
    assert report["ready_for_production"] is False
    assert report["warnings"] == [
        "Для классификатора нужны минимум два известных кота",
        "Барсик: независимых проходов — 1; рекомендуется не менее 100",
        "Мало примеров чужих котов для проверки ложных распознаваний",
    ]
    assert [item.label for item in items] == ["Барсик", "Барсик"]
    assert rotations == [True, True]
    assert report["detector_crops"] == {
        "included": 2,
        "excluded_no_confident_box": 0,
        "by_view": {
            "face": {"included": 1, "excluded_no_confident_box": 0},
            "side": {"included": 1, "excluded_no_confident_box": 0},
        },
    }
    assert all(frame.shape == (320, 320, 3) for frame in detector.frames)
    with Image.open(destination / items[0].image_path) as crop:
        assert crop.size == (224, 126)


def test_build_dataset_reports_and_excludes_frames_without_cat_box(
    runtime_path: Path,
    monkeypatch,
) -> None:
    database_path = runtime_path / "catdoor.sqlite3"
    event_dir = runtime_path / "events" / "00000001"
    event_dir.mkdir(parents=True)
    (event_dir / "face.mp4").write_bytes(b"video")
    factory = create_session_factory(database_path)
    with factory() as session:
        session.add(
            Event(
                created_at=datetime.now(UTC),
                status="ready",
                duration_seconds=5,
                media_path="events/00000001/face.mp4",
                label="known_cat",
                cat_name="Барсик",
                training_target="local",
            )
        )
        session.commit()

    def fake_extract(
        ffmpeg_path: str,
        source: Path,
        destination_pattern: Path,
        *,
        rotate_180: bool,
    ) -> list[Path]:
        destination = destination_pattern.parent / "frame-01.jpg"
        destination.parent.mkdir(parents=True)
        Image.fromarray(
            np.full((360, 640, 3), 100, dtype=np.uint8),
            mode="RGB",
        ).save(destination)
        return [destination]

    monkeypatch.setattr("training.workflow.extract_frames", fake_extract)
    destination = runtime_path / "training" / "dataset"

    with pytest.raises(ValueError, match="confident cat crops"):
        build_dataset(
            database_path,
            runtime_path,
            "ffmpeg",
            "local",
            destination,
            detector_model_path=Path("unused.onnx"),
            detector_device="CPU",
            detector=FixedDetector(()),
            default_detection_config=detection_config(),
        )

    report = json.loads(
        (destination / "dataset-report.json").read_text(encoding="utf-8")
    )
    assert report["detector_crops"]["included"] == 0
    assert report["detector_crops"]["excluded_no_confident_box"] == 1
    assert report["ready_for_production"] is False
    assert not list((destination / "images").rglob("*.jpg"))
