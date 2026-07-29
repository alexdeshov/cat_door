import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select

from src.common.database import Event, create_session_factory
from training.dataset import DatasetItem, assign_splits, validate_manifest, write_manifest

USABLE_LABELS = {"known_cat", "foreign_cat"}


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


def build_dataset(
    database_path: Path,
    data_dir: Path,
    ffmpeg_path: str,
    target: str,
    destination: Path,
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

    items: list[DatasetItem] = []
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
                rotate_180=True,
            )
            for frame in frames:
                items.append(
                    DatasetItem(
                        event_id=str(event.id),
                        captured_at=event.created_at.isoformat(),
                        view=view,
                        image_path=frame.relative_to(destination).as_posix(),
                        label=label,
                    )
                )

    if not items:
        raise ValueError("selected events do not contain usable media")
    items = assign_splits(items)
    manifest = destination / "dataset.jsonl"
    write_manifest(items, manifest)
    report = validate_manifest(items, destination)
    (destination / "dataset-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest, report


def create_macbook_export(
    database_path: Path,
    data_dir: Path,
    ffmpeg_path: str,
) -> Path:
    exports_root = data_dir / "exports"
    exports_root.mkdir(parents=True, exist_ok=True)
    name = f"macbook-{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
    work_dir = exports_root / name
    build_dataset(database_path, data_dir, ffmpeg_path, "macbook", work_dir)
    archive = Path(shutil.make_archive(str(exports_root / name), "zip", work_dir))
    shutil.rmtree(work_dir)
    return archive


def write_status(path: Path, **values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": datetime.now(UTC).isoformat(), **values}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
