import json
import logging
import math
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
from src.common.metrics import FOREIGN_LABEL, NOCAT_LABEL, REJECT_LABELS
from src.common.settings import Settings
from src.inference.cat_detector import CatDetectionResult, OpenVinoCatDetector
from src.inference.preprocessing import (
    DETECTOR_INPUT_SIZE,
    crop_normalized_roi,
    crop_strongest_cat,
    prepare_detector_frame,
)
from training.dataset import DatasetItem, assign_splits, validate_manifest, write_manifest

# Проходы, из которых режется обучающий материал. «Кота не видно» здесь не
# ошибка: это ложные срабатывания детектора, размеченные рукой, и они учат
# классификатор отказываться. «Несколько котов» остаётся снаружи — какое имя
# ставить кадру с двумя котами, неизвестно.
USABLE_LABELS = {"known_cat", "foreign_cat", "not_visible"}
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


logger = logging.getLogger("catdoor.training")

# Потолок кадров с одной записи. Событие длиной в пять минут иначе одно
# перевесило бы десяток обычных проходов, а соседние секунды всё равно почти
# одинаковы. Лишнее прореживается равномерно по всей записи, а не отрезается с
# конца.
MAX_FRAMES_PER_VIEW = 60


def _extraction_command(
    ffmpeg_path: str,
    source: Path,
    destination_pattern: Path,
    *,
    rotate_180: bool,
    vaapi_device: Path | None,
) -> list[str]:
    acceleration: list[str] = []
    filters = "fps=1"
    if vaapi_device is not None:
        acceleration = [
            "-hwaccel",
            "vaapi",
            "-hwaccel_device",
            str(vaapi_device),
            "-hwaccel_output_format",
            "vaapi",
        ]
        # Кадры прореживаются до скачивания: раз в секунду видеоядро отдаёт
        # один кадр вместо двадцати пяти, и в обычную память едет ровно то,
        # что нужно.
        filters += ",hwdownload,format=nv12"
    if rotate_180:
        filters += ",transpose=2,transpose=2"
    return [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        *acceleration,
        "-i",
        str(source),
        "-vf",
        filters,
        "-q:v",
        "3",
        "-y",
        str(destination_pattern),
    ]


def _run_extraction(command: list[str], destination_pattern: Path) -> tuple[bool, list[Path], str]:
    result = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        # Пять минут записи в 4 Мп декодируются заметно дольше трёх кадров.
        timeout=600,
        check=False,
    )
    pattern = destination_pattern.name
    for placeholder in ("%02d", "%03d", "%04d"):
        pattern = pattern.replace(placeholder, "*")
    frames = sorted(destination_pattern.parent.glob(pattern))
    return result.returncode == 0 and bool(frames), frames, (result.stderr or "").strip()


def extract_frames(
    ffmpeg_path: str,
    source: Path,
    destination_pattern: Path,
    *,
    rotate_180: bool,
    vaapi_device: Path | None = None,
) -> list[Path]:
    """Кадры записи, раз в секунду, по всей её длине.

    Раньше здесь стояло `-frames:v 3`, то есть в набор шли первые три секунды
    каждого прохода. Это худшие три секунды из возможных: событие начинается с
    преролла, кот появляется в кадре позже и виден в среднем десятую часть
    записи. Из 61 размеченного прохода набор получал 354 кадра, и кот был лишь
    на единицах из них.

    `vaapi_device` отдаёт декодирование медиадвижку. Здесь это главный расход:
    замер на настоящем проходе (58 секунд, 2688×1520) — 2.9 секунды по часам и
    **20 секунд процессорного времени** на одну камеру, тогда как инференс по
    полученным кадрам стоит полторы. Отказ видеоядра не должен ломать оценку,
    поэтому команда пересобирается программной и запускается заново.
    """
    destination_pattern.parent.mkdir(parents=True, exist_ok=True)
    if vaapi_device is not None:
        succeeded, frames, complaint = _run_extraction(
            _extraction_command(
                ffmpeg_path,
                source,
                destination_pattern,
                rotate_180=rotate_180,
                vaapi_device=vaapi_device,
            ),
            destination_pattern,
        )
        if succeeded:
            return thin_frames(frames, MAX_FRAMES_PER_VIEW)
        for leftover in frames:
            leftover.unlink(missing_ok=True)
        logger.warning(
            "Видеоядро не вырезало кадры из %s, перехожу на процессор: %s",
            source.name,
            complaint or "ffmpeg ничего не сообщил",
        )
    succeeded, frames, complaint = _run_extraction(
        _extraction_command(
            ffmpeg_path,
            source,
            destination_pattern,
            rotate_180=rotate_180,
            vaapi_device=None,
        ),
        destination_pattern,
    )
    if not succeeded:
        raise RuntimeError(f"ffmpeg could not extract frames from {source.name}")
    return thin_frames(frames, MAX_FRAMES_PER_VIEW)


def _ffprobe_path(ffmpeg_path: str) -> str:
    """Рядом с ffmpeg лежит ffprobe: это одна поставка, не две."""
    path = Path(ffmpeg_path)
    if "ffmpeg" in path.name:
        return str(path.with_name(path.name.replace("ffmpeg", "ffprobe", 1)))
    return "ffprobe"


def _video_geometry(ffmpeg_path: str, source: Path) -> tuple[int, int, float]:
    """Размер кадра и длительность записи.

    Без них сырой поток не разобрать: в нём нет ни заголовков, ни границ кадров
    — только байты один за другим, и знать, где кончается кадр, можно лишь
    заранее.
    """
    command = [
        _ffprobe_path(ffmpeg_path),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(source),
    ]
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        payload = json.loads(result.stdout or "{}")
        stream = payload["streams"][0]
        width, height = int(stream["width"]), int(stream["height"])
        duration = float(payload["format"]["duration"])
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, KeyError, ValueError, IndexError) as exc:
        raise RuntimeError(f"ffprobe could not describe {source.name}") from exc
    if width <= 0 or height <= 0 or duration <= 0:
        raise RuntimeError(f"ffprobe returned nonsense for {source.name}")
    return width, height, duration


def _kept_indices(total: int, limit: int) -> set[int]:
    """Те же кадры, что оставил бы `thin_frames`, но по номерам, а не по файлам."""
    if total <= limit:
        return set(range(total))
    step = total / limit
    return {int(index * step) for index in range(limit)}


def _decode_command(
    ffmpeg_path: str,
    source: Path,
    *,
    rotate_180: bool,
    vaapi_device: Path | None,
) -> list[str]:
    acceleration: list[str] = []
    filters = "fps=1"
    if vaapi_device is not None:
        acceleration = [
            "-hwaccel",
            "vaapi",
            "-hwaccel_device",
            str(vaapi_device),
            "-hwaccel_output_format",
            "vaapi",
        ]
        filters += ",hwdownload,format=nv12"
    if rotate_180:
        filters += ",transpose=2,transpose=2"
    return [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        *acceleration,
        "-i",
        str(source),
        "-vf",
        filters,
        "-an",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]


def _read_raw_frames(
    command: list[str],
    *,
    width: int,
    height: int,
    keep: set[int],
) -> list[np.ndarray]:
    frame_bytes = width * height * 3
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    frames: list[np.ndarray] = []
    try:
        assert process.stdout is not None
        index = 0
        while True:
            chunks = bytearray()
            while len(chunks) < frame_bytes:
                chunk = process.stdout.read(frame_bytes - len(chunks))
                if not chunk:
                    break
                chunks.extend(chunk)
            if len(chunks) != frame_bytes:
                return frames
            # Ненужные кадры не копируются в память вовсе: запись в пять минут
            # иначе держала бы в оперативке под четыре гигабайта.
            if index in keep:
                frames.append(
                    np.frombuffer(bytes(chunks), dtype=np.uint8).reshape((height, width, 3))
                )
            index += 1
    finally:
        if process.stdout:
            process.stdout.close()
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def decode_sampled_frames(
    ffmpeg_path: str,
    source: Path,
    *,
    rotate_180: bool,
    vaapi_device: Path | None = None,
    limit: int = MAX_FRAMES_PER_VIEW,
) -> list[np.ndarray]:
    """Те же кадры раз в секунду, но сразу в память, без круга через JPEG.

    Прежний путь кодировал каждый кадр в JPEG на диск, а следом распаковывал
    обратно через PIL — четыре мегапикселя туда и столько же назад на каждый
    кадр, полторы сотни кадров на событие. По замеру это около пятой части
    времени оценки, и она уходила целиком в переупаковку.

    Кадры выбираются тем же правилом, что и раньше: раз в секунду, не больше
    `limit`, равномерно по всей записи.
    """
    width, height, duration = _video_geometry(ffmpeg_path, source)
    # Фильтр `fps=1` отдаёт кадры в моменты 0, 1, 2 … пока они попадают в
    # запись, то есть ровно `ceil(duration)` штук.
    expected = max(1, math.ceil(duration))
    keep = _kept_indices(expected, limit)
    if vaapi_device is not None:
        frames = _read_raw_frames(
            _decode_command(
                ffmpeg_path,
                source,
                rotate_180=rotate_180,
                vaapi_device=vaapi_device,
            ),
            width=width,
            height=height,
            keep=keep,
        )
        if frames:
            return frames
        logger.warning(
            "Видеоядро не отдало кадров из %s, перехожу на процессор",
            source.name,
        )
    frames = _read_raw_frames(
        _decode_command(ffmpeg_path, source, rotate_180=rotate_180, vaapi_device=None),
        width=width,
        height=height,
        keep=keep,
    )
    if not frames:
        raise RuntimeError(f"ffmpeg could not decode frames from {source.name}")
    return frames


def thin_frames(frames: list[Path], limit: int) -> list[Path]:
    """Оставить не больше `limit` кадров, равномерно по всей записи."""
    if len(frames) <= limit:
        return frames
    step = len(frames) / limit
    keep = {frames[int(index * step)] for index in range(limit)}
    for frame in frames:
        if frame not in keep:
            frame.unlink(missing_ok=True)
    return [frame for frame in frames if frame in keep]


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
        if event.label == "known_cat":
            label = event.cat_name
        elif event.label == "not_visible":
            label = NOCAT_LABEL
        else:
            label = FOREIGN_LABEL
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
                # Четыре разряда, чтобы сортировка по имени оставалась
                # хронологической и после сотого кадра.
                event_dir / "frame-%04d.jpg",
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
        label: count
        for label, count in event_counts.items()
        if label not in REJECT_LABELS
    }
    warnings: list[str] = []
    if len(known_counts) < 2:
        warnings.append("Для классификатора нужны минимум два известных кота")
    for label, count in known_counts.items():
        if count < 100:
            warnings.append(
                f"{label}: независимых проходов — {count}; рекомендуется не менее 100"
            )
    if event_counts.get(FOREIGN_LABEL, 0) < 30:
        warnings.append("Мало примеров чужих котов для проверки ложных распознаваний")
    if event_counts.get(NOCAT_LABEL, 0) < 20:
        warnings.append(
            "Мало проходов «кота не видно» — классификатору не на чем учиться отказу"
        )
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
