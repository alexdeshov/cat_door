from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.common.database import ModelVersion

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
