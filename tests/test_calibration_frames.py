import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from scripts.collect_calibration_frames import collect, is_night, read_provenance
from src.common.database import Event, create_session_factory

MANIFEST = {
    "architecture": "yolo11n",
    "input": {
        "name": "images",
        "shape": [1, 3, 16, 16],
        "channel_order": "rgb",
        "letterbox_value": 114,
        "resize_mode": "letterbox",
        "letterbox_alignment": "center",
    },
    "outputs": ["output0"],
    "cat_class_id": 15,
    "sha256": "",
}


@pytest.fixture
def calibration_world(runtime_path: Path, monkeypatch) -> tuple[Path, Path, Path]:
    """База с проходами, записи-заглушки и модель с манифестом."""
    model = runtime_path / "detector.onnx"
    model.write_bytes(b"onnx")
    model.with_suffix(".json").write_text(
        json.dumps({**MANIFEST, "sha256": hashlib.sha256(b"onnx").hexdigest()}),
        encoding="utf-8",
    )
    data_dir = runtime_path / "data"
    media = data_dir / "events"
    media.mkdir(parents=True)
    database = runtime_path / "catdoor.sqlite3"
    factory = create_session_factory(database)
    # Восемь проходов: половина с котом, половина без.
    with factory() as session:
        for index in range(8):
            for role in ("face", "side"):
                (media / f"{index}-{role}.mp4").write_bytes(b"video")
            session.add(
                Event(
                    created_at=datetime(2026, 8, 9, 3 if index % 2 else 13, tzinfo=UTC),
                    duration_seconds=8,
                    media_path=f"events/{index}-face.mp4",
                    side_media_path=f"events/{index}-side.mp4",
                    label="known_cat" if index % 2 else "not_visible",
                    training_target="excluded",
                )
            )
        session.commit()

    def fake_load_frames(ffmpeg_path, source, destination, vaapi_device=None):
        seed = abs(hash(source.name)) % 1000
        generator = np.random.default_rng(seed)
        return [
            generator.integers(0, 256, size=(24, 32, 3), dtype=np.uint8) for _ in range(10)
        ]

    monkeypatch.setattr(
        "scripts.collect_calibration_frames.load_frames",
        fake_load_frames,
    )
    return database, data_dir, model


def run_collect(world: tuple[Path, Path, Path], output: Path, **overrides: object) -> Path:
    database, data_dir, model = world
    arguments: dict[str, object] = {
        "frames": 16,
        "per_event": 2,
        "min_negative_fraction": 0.3,
        "seed": 7,
    }
    arguments.update(overrides)
    return collect(database, data_dir, "ffmpeg", model, output, **arguments)


def test_collected_frames_match_the_model_input(
    runtime_path: Path,
    calibration_world: tuple[Path, Path, Path],
) -> None:
    output = run_collect(calibration_world, runtime_path / "calib.npz")

    with np.load(output, allow_pickle=False) as archive:
        frames = archive["frames"]
        provenance = read_provenance(archive)

    assert frames.dtype == np.uint8
    assert frames.shape == (16, 16, 16, 3)
    assert provenance["input_size"] == 16
    assert provenance["letterbox_value"] == 114
    assert provenance["resize_mode"] == "letterbox"
    assert provenance["channel_order"] == "rgb"
    assert provenance["seed"] == 7
    assert provenance["events"]


def test_collection_is_deterministic_under_a_seed(
    runtime_path: Path,
    calibration_world: tuple[Path, Path, Path],
) -> None:
    """Набор, который нельзя собрать заново, нельзя и обвинить в результате."""
    first = run_collect(calibration_world, runtime_path / "one.npz")
    second = run_collect(calibration_world, runtime_path / "two.npz")

    with np.load(first, allow_pickle=False) as left, np.load(second, allow_pickle=False) as right:
        assert np.array_equal(left["frames"], right["frames"])
        assert read_provenance(left)["frame_list"] == read_provenance(right)["frame_list"]


def test_no_pass_contributes_more_than_the_cap(
    runtime_path: Path,
    calibration_world: tuple[Path, Path, Path],
) -> None:
    """Соседние кадры одного прохода дают один диапазон активаций на всех."""
    output = run_collect(calibration_world, runtime_path / "calib.npz", per_event=2)

    with np.load(output, allow_pickle=False) as archive:
        frame_list = read_provenance(archive)["frame_list"]

    per_pair: dict[tuple[int, str], int] = {}
    for item in frame_list:
        key = (item["event_id"], item["role"])
        per_pair[key] = per_pair.get(key, 0) + 1
    assert max(per_pair.values()) <= 2


def test_a_set_without_background_is_refused_loudly(
    runtime_path: Path,
    calibration_world: tuple[Path, Path, Path],
) -> None:
    """Диапазоны, снятые только по кадрам с котом, обрежут фон.

    Проявится это новыми ложными срабатываниями — тем самым, с чем проект
    боролся; и увидеть это можно будет только через сутки живой работы.
    """
    with pytest.raises(ValueError, match="negatives are"):
        run_collect(
            calibration_world,
            runtime_path / "calib.npz",
            labels=["known_cat"],
        )


def test_collection_refuses_to_overwrite(
    runtime_path: Path,
    calibration_world: tuple[Path, Path, Path],
) -> None:
    output = runtime_path / "calib.npz"
    output.write_bytes(b"already here")

    with pytest.raises(FileExistsError):
        run_collect(calibration_world, output)


def test_media_outside_the_data_directory_is_skipped(
    runtime_path: Path,
    calibration_world: tuple[Path, Path, Path],
) -> None:
    """Та же проверка, что в офлайн-оценке: путь из базы наружу не выпускает."""
    database, _, _ = calibration_world
    factory = create_session_factory(database)
    with factory() as session:
        for event in session.query(Event).all():
            event.media_path = "../../etc/passwd"
            event.side_media_path = None
        session.commit()

    with pytest.raises(ValueError, match="no calibration frames"):
        run_collect(calibration_world, runtime_path / "calib.npz")


def test_night_boundary_matches_the_evaluation_report() -> None:
    """Рубеж ночи берётся тот же, что у `EvaluatedDetectorEvent.is_night`."""
    assert is_night(datetime(2026, 8, 9, 20, tzinfo=UTC))
    assert is_night(datetime(2026, 8, 9, 3, tzinfo=UTC))
    assert not is_night(datetime(2026, 8, 9, 7, tzinfo=UTC))
    assert not is_night(datetime(2026, 8, 9, 19, tzinfo=UTC))
