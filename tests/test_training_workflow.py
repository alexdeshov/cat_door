import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from PIL import Image

from src.common.camera_registry import (
    CameraRecord,
    CameraRegistry,
    camera_registry_path,
    save_registry,
)
from src.common.database import Event, create_session_factory
from src.common.detection_config import CameraDetectionConfig, DetectionConfig
from src.inference.cat_detector import CatDetection, CatDetectionResult
from training.dataset import load_manifest
from training.workflow import (
    StatusHeartbeat,
    build_dataset,
    extract_frames,
    thin_frames,
)


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
        "Мало проходов «кота не видно» — классификатору не на чем учиться отказу",
    ]
    assert [item.label for item in items] == ["Барсик", "Барсик"]
    # Инвентаря в этом наборе нет, значит и разворачивать нечего: клип записан
    # как есть, а переворот живёт в карточке камеры.
    assert rotations == [False, False]
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


def test_only_the_last_three_exports_are_kept(runtime_path, monkeypatch) -> None:
    """Каждый экспорт — гигабайты видео; раньше они копились вечно."""
    from training import workflow

    exports = runtime_path / "exports"
    exports.mkdir(parents=True)
    for index in range(5):
        (exports / f"macbook-2025010{index}-000000-aaaa.zip").write_bytes(b"x")

    monkeypatch.setattr(workflow, "build_dataset", lambda *args, **kwargs: None)
    def fake_archive(base, fmt, root):
        archive = Path(str(base) + ".zip")
        archive.write_bytes(b"x")
        return str(archive)

    monkeypatch.setattr(workflow.shutil, "make_archive", fake_archive)
    monkeypatch.setattr(workflow.shutil, "rmtree", lambda path: None)

    workflow.create_macbook_export(
        runtime_path / "db.sqlite3", runtime_path, "ffmpeg",
        runtime_path / "model.onnx", "CPU", None,
    )

    remaining = sorted(p.name for p in exports.glob("macbook-*.zip"))
    assert len(remaining) == 3
    # Остаются самые свежие: два последних старых плюс только что созданный.
    assert remaining[0] == "macbook-20250103-000000-aaaa.zip"
    assert remaining[1] == "macbook-20250104-000000-aaaa.zip"
    assert remaining[2].startswith("macbook-2026")


def test_frames_follow_the_camera_rotation_from_the_inventory(
    runtime_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Разворачивать кадры перед детектором нужно ровно тем камерам, что стоят вверх ногами.

    Клип пишется без фильтров ориентации, поэтому кадр из файла повторяет
    установку камеры. Детектор к повороту не инвариантен: константный разворот
    сделал 5 августа набор из 354 кадров, в котором уцелело 4 кропа.
    """
    database_path = runtime_path / "catdoor.db"
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
        session.commit()
    for view in ("face", "side"):
        media = runtime_path / "events" / "00000001" / f"{view}.mp4"
        media.parent.mkdir(parents=True, exist_ok=True)
        media.write_bytes(b"")

    save_registry(
        camera_registry_path(runtime_path),
        CameraRegistry(
            cameras=[
                CameraRecord(
                    id="upside-down",
                    host="10.0.0.2",
                    stream_path="/s",
                    role="face",
                    rotate_180=True,
                ),
                CameraRecord(
                    id="upright",
                    host="10.0.0.3",
                    stream_path="/s",
                    role="side",
                    rotate_180=False,
                ),
            ]
        ),
    )

    rotations: dict[str, bool] = {}

    def fake_extract(
        ffmpeg_path: str,
        source: Path,
        destination_pattern: Path,
        *,
        rotate_180: bool,
    ) -> list[Path]:
        rotations[destination_pattern.parent.name] = rotate_180
        destination = destination_pattern.parent / "frame-01.jpg"
        destination.parent.mkdir(parents=True)
        Image.fromarray(
            np.full((360, 640, 3), 100, dtype=np.uint8),
            mode="RGB",
        ).save(destination)
        return [destination]

    monkeypatch.setattr("training.workflow.extract_frames", fake_extract)

    build_dataset(
        database_path,
        runtime_path,
        "ffmpeg",
        "local",
        runtime_path / "training" / "dataset",
        detector_model_path=Path("unused.onnx"),
        detector_device="CPU",
        detector=FixedDetector((CatDetection(box=(80, 115, 240, 205), confidence=0.80),)),
        default_detection_config=detection_config(),
    )

    assert rotations == {"face": True, "side": False}


def test_frames_are_taken_from_the_whole_recording(tmp_path: Path) -> None:
    """Кадры берутся по всей записи, а не с её начала.

    Событие начинается с преролла: кот появляется позже и виден лишь малую
    часть записи. Прежние `-frames:v 3` отдавали набору ровно те секунды, где
    его гарантированно нет.
    """
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        pattern = Path(command[-1])
        for index in range(1, 4):
            pattern.parent.joinpath(f"frame-{index:04d}.jpg").write_bytes(b"jpeg")
        return subprocess.CompletedProcess(command, 0, "", "")

    with mock.patch("training.workflow.subprocess.run", fake_run):
        frames = extract_frames("ffmpeg", tmp_path / "clip.mp4", tmp_path / "frame-%04d.jpg", rotate_180=False)

    assert "-frames:v" not in calls[0]
    assert len(frames) == 3


def test_long_recordings_are_thinned_evenly() -> None:
    """Прореживание идёт по всей длине, иначе конец записи пропал бы целиком."""
    frames = [Path(f"frame-{index:04d}.jpg") for index in range(100)]

    kept = thin_frames(list(frames), 10)

    assert len(kept) == 10
    assert kept[0] == frames[0]
    # Последний оставленный кадр приходит из конца записи, а не из первой трети.
    assert kept[-1] == frames[90]


def test_extraction_asks_the_media_engine_first(tmp_path: Path, monkeypatch) -> None:
    """Декодирование — главный расход оценки, а не инференс."""
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        (tmp_path / "frame-01.jpg").write_bytes(b"jpeg")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("training.workflow.subprocess.run", fake_run)

    frames = extract_frames(
        "ffmpeg",
        tmp_path / "clip.mp4",
        tmp_path / "frame-%02d.jpg",
        rotate_180=False,
        vaapi_device=Path("/dev/dri/renderD128"),
    )

    assert len(commands) == 1
    command = commands[0]
    assert command.index("-hwaccel") < command.index("-i")
    assert command[command.index("-vf") + 1] == "fps=1,hwdownload,format=nv12"
    assert len(frames) == 1


def test_extraction_falls_back_to_the_processor(tmp_path: Path, monkeypatch, caplog) -> None:
    """Узел без видеоядра обязан считать так же, просто медленнее."""
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if "-hwaccel" in command:
            return SimpleNamespace(returncode=1, stderr="No VA display found")
        (tmp_path / "frame-01.jpg").write_bytes(b"jpeg")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("training.workflow.subprocess.run", fake_run)

    with caplog.at_level("WARNING", logger="catdoor.training"):
        frames = extract_frames(
            "ffmpeg",
            tmp_path / "clip.mp4",
            tmp_path / "frame-%02d.jpg",
            rotate_180=True,
            vaapi_device=Path("/dev/dri/renderD128"),
        )

    assert len(commands) == 2
    assert "-hwaccel" not in commands[1]
    assert commands[1][commands[1].index("-vf") + 1] == "fps=1,transpose=2,transpose=2"
    assert len(frames) == 1
    assert "No VA display found" in caplog.text
