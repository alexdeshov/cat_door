from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.common.database import ModelVersion
from src.common.metrics import REJECT_LABELS

MODEL_KINDS = {"detector", "identity_face", "identity_side"}


def model_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_metadata(model_path: Path) -> dict[str, Any]:
    metadata_path = model_path.with_suffix(".json")
    try:
        value = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid model metadata: {metadata_path.name}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"invalid model metadata: {metadata_path.name}")
    return value


def identity_labels(model_paths: Iterable[Path | None]) -> set[str] | None:
    """Кого установленные модели вообще умеют называть.

    Нужно метрике: кот, которого модель не обучалась узнавать, для неё чужой,
    и считать его своим — значит прятать пропущенных чужих в
    `misidentified_known`. У эмбеддингов имена берутся из эталонов, у
    классификатора — из `labels`; классы отказа именами не являются.

    `None` — ни одной модели прочитать не удалось. Это не то же самое, что
    «модель не знает никого»: во втором случае метрика посчитала бы чужими
    всех до единого и отчиталась бы прекрасными числами ни о чём.
    """
    names: set[str] = set()
    found = False
    for path in model_paths:
        if path is None:
            continue
        try:
            metadata = load_model_metadata(path)
        except (ValueError, TypeError):
            continue
        found = True
        prototypes = metadata.get("prototypes")
        source = prototypes if isinstance(prototypes, dict) else metadata.get("labels")
        if isinstance(source, dict):
            source = list(source)
        if isinstance(source, list):
            names.update(
                str(name) for name in source if str(name) not in REJECT_LABELS
            )
    return names if found else None


def register_model_version(
    session: Session,
    *,
    kind: str,
    model_path: Path,
) -> ModelVersion:
    if kind not in MODEL_KINDS:
        raise ValueError(f"unsupported model kind: {kind}")
    if not model_path.is_file():
        raise ValueError(f"model is missing: {model_path.name}")
    sha256 = model_sha256(model_path)
    existing = session.scalar(select(ModelVersion).where(ModelVersion.sha256 == sha256))
    if existing is not None:
        if existing.kind != kind:
            raise ValueError(
                f"model hash is already registered as {existing.kind}: {model_path.name}"
            )
        return existing
    metadata = load_model_metadata(model_path)
    version = ModelVersion(
        kind=kind,
        file_name=model_path.name,
        sha256=sha256,
        metadata_json=json.dumps(metadata, ensure_ascii=False, sort_keys=True),
    )
    session.add(version)
    session.flush()
    return version
