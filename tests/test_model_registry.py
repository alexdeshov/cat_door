import json
from pathlib import Path

import pytest

from src.common.database import ModelVersion, create_session_factory
from src.common.model_registry import identity_labels, register_model_version


def _write_model(root: Path, name: str, payload: bytes = b"model") -> Path:
    model_path = root / name
    model_path.write_bytes(payload)
    model_path.with_suffix(".json").write_text(
        json.dumps({"labels": ["Барсик"], "image_size": 160}),
        encoding="utf-8",
    )
    return model_path


def test_register_model_version_is_idempotent_by_sha256(runtime_path: Path) -> None:
    model_path = _write_model(runtime_path, "face.onnx")
    factory = create_session_factory(runtime_path / "catdoor.sqlite3")

    with factory() as session:
        first = register_model_version(
            session,
            kind="identity_face",
            model_path=model_path,
        )
        session.commit()
        first_id = first.id
    with factory() as session:
        second = register_model_version(
            session,
            kind="identity_face",
            model_path=model_path,
        )
        session.commit()
        assert second.id == first_id
        assert session.query(ModelVersion).count() == 1


def test_same_model_bytes_cannot_be_registered_as_another_kind(
    runtime_path: Path,
) -> None:
    first_path = _write_model(runtime_path, "face.onnx")
    second_path = _write_model(runtime_path, "side.onnx")
    factory = create_session_factory(runtime_path / "catdoor.sqlite3")

    with factory() as session:
        register_model_version(
            session,
            kind="identity_face",
            model_path=first_path,
        )
        session.commit()
    with factory() as session, pytest.raises(ValueError, match="already registered"):
        register_model_version(
            session,
            kind="identity_side",
            model_path=second_path,
        )


def test_identity_labels_names_only_the_cats_the_model_can_produce(
    runtime_path: Path,
) -> None:
    """Метрике нужно знать, кого модель умеет называть.

    У эмбеддингов это эталоны, а не список классов обучения: класс «чужой»
    там есть, но именем не является и эталона не имеет.
    """
    model_path = runtime_path / "face.onnx"
    model_path.write_bytes(b"model")
    model_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "kind": "embedding",
                "prototypes": {"Яся": [0.1], "Черныш": [0.2], "__foreign__": [0.3]},
            }
        ),
        encoding="utf-8",
    )

    assert identity_labels([model_path]) == {"Яся", "Черныш"}


def test_identity_labels_says_nothing_when_it_knows_nothing(runtime_path: Path) -> None:
    """Пустое множество означало бы «модель не знает никого».

    Метрика посчитала бы тогда чужими всех до единого и отчиталась бы
    прекрасными числами ни о чём.
    """
    assert identity_labels([runtime_path / "missing.onnx", None]) is None
