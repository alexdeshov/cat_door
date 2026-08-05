import json
import shutil
import subprocess
import threading
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import numpy as np
from PIL import Image
from sqlalchemy import select

from src.common.camera_registry import CameraRegistry, camera_registry_path, load_registry
from src.common.database import Event, create_session_factory
from src.common.detection_config import (
    DetectionConfig,
    DetectorRoi,
    detection_config_path,
    load_detection_config,
)
from src.common.settings import Settings
from src.inference.cat_detector import CatDetectionResult, OpenVinoCatDetector
from src.inference.preprocessing import (
    DETECTOR_INPUT_SIZE,
    crop_normalized_roi,
    crop_strongest_cat,
    prepare_detector_frame,
)
from training.dataset import DatasetItem, assign_splits, validate_manifest, write_manifest

USABLE_LABELS = {"known_cat", "foreign_cat"}
_STATUS_LOCK = threading.RLock()


def view_rotation(registry: CameraRegistry | None, view: str) -> bool:
    """Нужно ли разворачивать кадры этого ракурса перед детектором.

    Клип пишется ровно так, как его отдала камера: `-c:v copy`, без единого
    фильтра ориентации. Переворот живёт в карточке камеры и отыгрывается на
    показе, поэтому здесь его приходится повторять — детектор не инвариантен к
    повороту, и перевёрнутый кот для него не кот.

    Проверено 5 августа на живых записях: те же восемь кадров дают уверенность
    0.67–0.99, если ориентация верна, и не проходят порог камеры подхода вовсе,
    если кадр развернуть. Прежде поворот стоял здесь константой `True` — набор
    из 354 кадров дал 4 кропа, и обучать было не на чем.

    Роль берётся из текущего инвентаря: какая камера снимала событие полгода
    назад, знать неоткуда, а перестановка камер местами меняет роль, но не
    ориентацию сцены.
    """
    if registry is None:
        return False
    camera = registry.by_role(view)
    return bool(camera is not None and camera.rotate_180)


class CatDetector(Protocol):
    def detect(self, rgb_frame: np.ndarray) -> CatDetectionResult: ...


def extract_frames(
    ffmpeg_path: str,
    source: Path,
    destination_pattern: Path,
    *,
    rotate_180: bool,
) -> list[Path]:
    destination_pattern.parent.mkdir(parents=True, exist_ok=True)
    filters = "fps=1"
    if rotate_180:
        filters += ",transpose=2,transpose=2"
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vf",
        filters,
        "-frames:v",
        "3",
        "-q:v",
        "3",
        "-y",
        str(destination_pattern),
    ]
    result = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
        check=False,
    )
    frames = sorted(destination_pattern.parent.glob(destination_pattern.name.replace("%02d", "*")))
    if result.returncode != 0 or not frames:
        raise RuntimeError(f"ffmpeg could not extract frames from {source.name}")
    return frames


def crop_extracted_frames(
    frames: Sequence[Path],
    detector: CatDetector,
    *,
    confidence_threshold: float,
    roi: DetectorRoi | None,
) -> tuple[list[Path], int]:
    accepted: list[Path] = []
    rejected = 0
    detector_input_size = int(
        getattr(detector, "input_size", DETECTOR_INPUT_SIZE)
    )
    detector_letterbox_value = int(getattr(detector, "letterbox_value", 0))
    detector_resize_mode = str(getattr(detector, "resize_mode", "letterbox"))
    detector_letterbox_alignment = str(
        getattr(detector, "letterbox_alignment", "center")
    )
    for frame_path in frames:
        with Image.open(frame_path) as image:
            frame = np.asarray(image.convert("RGB"), dtype=np.uint8)
        detector_frame = crop_normalized_roi(frame, roi)
        detection = detector.detect(
            prepare_detector_frame(
                detector_frame,
                detector_input_size,
                detector_letterbox_value,
                detector_resize_mode,
                detector_letterbox_alignment,
            )
        )
        crop = crop_strongest_cat(
            frame,
            detection.detections,
            min_confidence=confidence_threshold,
            detector_frame=detector_frame,
            roi=roi,
            input_size=detector_input_size,
            resize_mode=detector_resize_mode,
            letterbox_alignment=detector_letterbox_alignment,
        )
        if crop is None:
            frame_path.unlink(missing_ok=True)
            rejected += 1
            continue
        Image.fromarray(crop, mode="RGB").save(frame_path, format="JPEG", quality=95)
        accepted.append(frame_path)
    return accepted, rejected


def build_dataset(
    database_path: Path,
    data_dir: Path,
    ffmpeg_path: str,
    target: str,
    destination: Path,
    *,
    detector_model_path: Path,
    detector_device: str,
    detector: CatDetector | None = None,
    default_detection_config: DetectionConfig | None = None,
) -> tuple[Path, dict[str, object]]:
    if target not in {"local", "macbook"}:
        raise ValueError("unsupported training target")
    destination.mkdir(parents=True, exist_ok=False)
    images_root = destination / "images"
    factory = create_session_factory(database_path)
    with factory() as session:
        events = session.scalars(
            select(Event)
            .where(Event.training_target == target)
            .where(Event.label.in_(USABLE_LABELS))
            .order_by(Event.created_at)
        ).all()

    if not events:
        raise ValueError(f"no events selected for {target} training")

    defaults = default_detection_config or DetectionConfig.from_settings(
        Settings(data_dir=data_dir)
    )
    runtime_config = load_detection_config(
        detection_config_path(data_dir),
        defaults,
    )
    active_detector = detector or OpenVinoCatDetector(
        detector_model_path,
        detector_device,
    )
    registry = load_registry(camera_registry_path(data_dir))
    items: list[DatasetItem] = []
    crop_by_view: dict[str, dict[str, int]] = {
        "face": {"included": 0, "excluded_no_confident_box": 0},
        "side": {"included": 0, "excluded_no_confident_box": 0},
    }
    crops_included = 0
    crops_rejected = 0
    root = data_dir.resolve()
    for event in events:
        label = event.cat_name if event.label == "known_cat" else "__foreign__"
        if not label:
            continue
        for view, relative_media in (
            ("face", event.media_path),
            ("side", event.side_media_path),
        ):
            if not relative_media:
                continue
            source = (root / relative_media).resolve()
            if root not in source.parents or not source.is_file():
                raise ValueError(f"unsafe or missing media for event {event.id}, view {view}")
            event_dir = images_root / f"{event.id:08d}" / view
            frames = extract_frames(
                ffmpeg_path,
                source,
                event_dir / "frame-%02d.jpg",
                rotate_180=view_rotation(registry, view),
            )
            camera_config = getattr(runtime_config, view)
            cropped_frames, rejected = crop_extracted_frames(
                frames,
                active_detector,
                confidence_threshold=camera_config.confidence_threshold,
                roi=camera_config.roi,
            )
            view_report = crop_by_view[view]
            view_report["included"] += len(cropped_frames)
            view_report["excluded_no_confident_box"] += rejected
            crops_included += len(cropped_frames)
            crops_rejected += rejected
            for frame in cropped_frames:
                items.append(
                    DatasetItem(
                        event_id=str(event.id),
                        captured_at=event.created_at.isoformat(),
                        view=view,
                        image_path=frame.relative_to(destination).as_posix(),
                        label=label,
                    )
                )

    crop_report: dict[str, object] = {
        "included": crops_included,
        "excluded_no_confident_box": crops_rejected,
        "by_view": crop_by_view,
    }
    if not items:
        empty_report = {
            "items": 0,
            "events": 0,
            "labels": {},
            "views": {},
            "splits": {},
            "detector_crops": crop_report,
            "warnings": ["Детектор не нашёл уверенную рамку кошки ни на одном кадре"],
            "ready_for_production": False,
        }
        (destination / "dataset-report.json").write_text(
            json.dumps(empty_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise ValueError("selected events do not contain confident cat crops")
    items = assign_splits(items)
    manifest = destination / "dataset.jsonl"
    write_manifest(items, manifest)
    report = validate_manifest(items, destination)
    events_per_label: dict[str, set[str]] = defaultdict(set)
    for item in items:
        events_per_label[item.label].add(item.event_id)
    event_counts = {
        label: len(event_ids) for label, event_ids in sorted(events_per_label.items())
    }
    known_counts = {
        label: count for label, count in event_counts.items() if label != "__foreign__"
    }
    warnings: list[str] = []
    if len(known_counts) < 2:
        warnings.append("Для классификатора нужны минимум два известных кота")
    for label, count in known_counts.items():
        if count < 100:
            warnings.append(
                f"{label}: независимых проходов — {count}; рекомендуется не менее 100"
            )
    if event_counts.get("__foreign__", 0) < 30:
        warnings.append("Мало примеров чужих котов для проверки ложных распознаваний")
    report.update(
        {
            "events_per_label": event_counts,
            "detector_crops": crop_report,
            "warnings": warnings,
            "ready_for_production": not warnings,
        }
    )
    (destination / "dataset-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest, report


def create_macbook_export(
    database_path: Path,
    data_dir: Path,
    ffmpeg_path: str,
    detector_model_path: Path,
    detector_device: str,
    default_detection_config: DetectionConfig,
) -> Path:
    exports_root = data_dir / "exports"
    exports_root.mkdir(parents=True, exist_ok=True)
    name = f"macbook-{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
    work_dir = exports_root / name
    build_dataset(
        database_path,
        data_dir,
        ffmpeg_path,
        "macbook",
        work_dir,
        detector_model_path=detector_model_path,
        detector_device=detector_device,
        default_detection_config=default_detection_config,
    )
    archive = Path(shutil.make_archive(str(exports_root / name), "zip", work_dir))
    shutil.rmtree(work_dir)
    # Каждый экспорт — гигабайты размеченных видео, и раньше они копились
    # вечно: ни предела, ни индикации. Архив нужен один раз — донести датасет
    # до MacBook; пара предыдущих остаётся на случай «скачал и потерял».
    stale = sorted(exports_root.glob("macbook-*.zip"))[:-3]
    for old_archive in stale:
        old_archive.unlink(missing_ok=True)
    return archive


def _write_status_unlocked(path: Path, values: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": datetime.now(UTC).isoformat(), **values}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_status(path: Path, **values: object) -> None:
    with _STATUS_LOCK:
        _write_status_unlocked(path, values)


class StatusHeartbeat:
    def __init__(self, path: Path, *, interval_seconds: float = 30.0) -> None:
        self._path = path
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"catdoor-status-heartbeat-{path.parent.name}",
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self.refresh()

    def refresh(self) -> None:
        with _STATUS_LOCK:
            try:
                payload = json.loads(self._path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return
            if not isinstance(payload, dict):
                return
            values = {
                str(key): value
                for key, value in payload.items()
                if key != "updated_at"
            }
            _write_status_unlocked(self._path, values)
