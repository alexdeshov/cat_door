"""Калибровочные кадры для квантования — из собственных проходов.

Квантование подбирает диапазоны активаций по тем кадрам, которые ему показали.
Показать надо ровно ту сцену, на которой модель работает: террасу этого дома, её
тени, её ночную подсветку. На чужом наборе диапазоны снимаются с чужих чисел, и
ошибка не проявляется ничем, кроме качества.

Запускается **на узле**, где лежат записи и база. Результат — один `.npz` с уже
подготовленными кадрами, который увозится на мак к `scripts/quantize_detector.py`.
Разрез между скриптами проходит здесь потому, что записи читает только служебный
пользователь, а NNCF с onnxruntime нечего делать на боевом узле.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from random import Random
from typing import Any

import numpy as np

from src.common.database import create_session_factory
from src.common.detection_config import (
    DetectionConfig,
    detection_config_path,
    load_detection_config,
)
from src.common.settings import Settings
from src.inference.preprocessing import crop_normalized_roi, prepare_detector_frame
from training.evaluation_job import event_query, load_frames

# Метки, которые считаются негативами. Кадры без кота — самый ценный материал
# калибровки: диапазоны, снятые только по кадрам с котом, обрежут фон, и это
# вылезет новыми ложными срабатываниями — ровно тем, с чем проект боролся.
NEGATIVE_LABELS = frozenset({"not_visible", "no_cat", "false_trigger"})


def is_night(moment: datetime) -> bool:
    """Тот же рубеж, что у `EvaluatedDetectorEvent.is_night`."""
    return moment.hour >= 20 or moment.hour < 7


class ManifestPreprocessing:
    """Параметры подготовки кадра — из манифеста, без поднятия OpenVINO.

    Поднимать модель ради пяти чисел незачем: на узле её всё равно занимает
    живой recorder, а видеоядро с одной очередью не любит второго клиента.
    """

    def __init__(self, model_path: Path) -> None:
        manifest = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
        self.model_name = model_path.name
        self.model_sha256 = str(manifest["sha256"])
        source = manifest["input"]
        self.input_size = int(source["shape"][2])
        self.letterbox_value = int(source.get("letterbox_value", 0))
        self.resize_mode = str(source.get("resize_mode", "letterbox"))
        self.letterbox_alignment = str(source.get("letterbox_alignment", "center"))
        self.channel_order = str(source.get("channel_order", "rgb"))

    def prepare(self, frame: np.ndarray, roi: Any) -> np.ndarray:
        return prepare_detector_frame(
            crop_normalized_roi(frame, roi),
            self.input_size,
            self.letterbox_value,
            self.resize_mode,
            self.letterbox_alignment,
        )

    def provenance(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "model_sha256": self.model_sha256,
            "input_size": self.input_size,
            "letterbox_value": self.letterbox_value,
            "resize_mode": self.resize_mode,
            "letterbox_alignment": self.letterbox_alignment,
            "channel_order": self.channel_order,
        }


def _sample_indices(total: int, wanted: int, random: Random) -> list[int]:
    """Кадры внутри одного прохода — почти дубликаты друг друга.

    Поэтому берём не первые подряд, а равномерно по всей записи: вход в кадр,
    середина и уход дают три разные картинки, а три соседних — одну.
    """
    if total <= wanted:
        return list(range(total))
    step = total / wanted
    return sorted({min(total - 1, int(index * step)) for index in range(wanted)})


def collect(
    database_path: Path,
    data_dir: Path,
    ffmpeg_path: str,
    model_path: Path,
    output: Path,
    *,
    frames: int,
    per_event: int,
    min_negative_fraction: float,
    seed: int,
    labels: list[str] | None = None,
    roles: list[str] | None = None,
    vaapi_device: Path | None = None,
) -> Path:
    sidecar = output.with_suffix(".json")
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite calibration set {output}")
    preprocessing = ManifestPreprocessing(model_path)
    runtime_config = load_detection_config(
        detection_config_path(data_dir),
        DetectionConfig.from_settings(Settings(data_dir=data_dir)),
    )
    factory = create_session_factory(database_path)
    with factory() as session:
        events = list(
            session.scalars(event_query({"labels": labels} if labels else {})).all()
        )
    if not events:
        raise ValueError("calibration filter did not select any events")
    random = Random(seed)
    random.shuffle(events)
    wanted_roles = roles or ["face", "side"]
    root = data_dir.resolve()
    temporary_root = data_dir / "runtime" / "calibration"

    collected: list[np.ndarray] = []
    provenance_frames: list[dict[str, Any]] = []
    negatives = 0
    try:
        for event in events:
            if len(collected) >= frames:
                break
            for role in wanted_roles:
                relative = event.media_path if role == "face" else event.side_media_path
                if not relative:
                    continue
                source = (root / relative).resolve()
                if root not in source.parents or not source.is_file():
                    continue
                decoded = load_frames(
                    ffmpeg_path,
                    source,
                    temporary_root / f"{event.id:08d}" / role,
                    vaapi_device,
                )
                if not decoded:
                    continue
                roi = getattr(runtime_config, role).roi
                negative = event.label in NEGATIVE_LABELS
                for index in _sample_indices(len(decoded), per_event, random):
                    if len(collected) >= frames:
                        break
                    collected.append(preprocessing.prepare(decoded[index], roi))
                    negatives += negative
                    provenance_frames.append(
                        {
                            "event_id": event.id,
                            "role": role,
                            "frame_index": index,
                            "label": event.label,
                            "training_target": event.training_target,
                            "is_night": is_night(event.created_at),
                            "negative": negative,
                        }
                    )
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root, ignore_errors=True)

    if not collected:
        raise ValueError("no calibration frames could be decoded")
    fraction = negatives / len(collected)
    if fraction < min_negative_fraction:
        # Отказ громкий нарочно. Молча собранный набор без фона даёт модель,
        # которая прекрасно проходит замер времени и начинает видеть котов в
        # тенях; понять это можно будет только через сутки живой работы.
        raise ValueError(
            f"negatives are {fraction:.0%} of {len(collected)} frames, "
            f"need at least {min_negative_fraction:.0%}: расширьте фильтр меток"
        )

    stack = np.stack(collected).astype(np.uint8, copy=False)
    provenance = {
        **preprocessing.provenance(),
        "roi": {
            role: (
                getattr(runtime_config, role).roi.model_dump()
                if getattr(runtime_config, role).roi is not None
                else None
            )
            for role in wanted_roles
        },
        "seed": seed,
        "per_event": per_event,
        "frames": len(collected),
        "negative_fraction": round(fraction, 4),
        "night_frames": sum(1 for item in provenance_frames if item["is_night"]),
        "labels": dict(Counter(item["label"] for item in provenance_frames)),
        "roles": dict(Counter(item["role"] for item in provenance_frames)),
        "events": sorted({item["event_id"] for item in provenance_frames}),
        "frame_list": provenance_frames,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        frames=stack,
        provenance=np.frombuffer(
            json.dumps(provenance, ensure_ascii=False).encode("utf-8"),
            dtype=np.uint8,
        ),
    )
    sidecar.write_text(
        json.dumps(
            {
                "file": output.name,
                "sha256": _sha256(output),
                **{key: value for key, value in provenance.items() if key != "frame_list"},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_provenance(archive: Any) -> dict[str, Any]:
    """Провенанс из архива обратно в словарь."""
    return json.loads(bytes(archive["provenance"]).decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect own frames for post-training quantization calibration"
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--frames", type=int, default=256)
    parser.add_argument(
        "--per-event",
        type=int,
        default=4,
        help="cap per pass so the set spans many scenes instead of a few videos",
    )
    parser.add_argument(
        "--min-negative-fraction",
        type=float,
        default=0.3,
        help="frames without a cat; calibrating only on cats clips the background",
    )
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--label", action="append", dest="labels")
    parser.add_argument("--role", action="append", dest="roles", choices=("face", "side"))
    parser.add_argument("--vaapi-device", type=Path)
    arguments = parser.parse_args()
    print(
        collect(
            arguments.database,
            arguments.data_dir,
            arguments.ffmpeg,
            arguments.model,
            arguments.output,
            frames=arguments.frames,
            per_event=arguments.per_event,
            min_negative_fraction=arguments.min_negative_fraction,
            seed=arguments.seed,
            labels=arguments.labels,
            roles=arguments.roles,
            vaapi_device=arguments.vaapi_device,
        )
    )


if __name__ == "__main__":
    main()
