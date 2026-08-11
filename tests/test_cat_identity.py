import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.inference.cat_identity import OpenVinoCatIdentity


class FakeCompiled:
    def output(self, _index: int) -> str:
        return "logits"

    def __call__(self, _inputs: object) -> dict[str, np.ndarray]:
        return {"logits": np.asarray([[0.0, 4.0]], dtype=np.float32)}


def test_identity_returns_name_only_above_threshold(
    runtime_path: Path,
    monkeypatch,
) -> None:
    model = runtime_path / "face-identity.onnx"
    model.write_bytes(b"synthetic")
    model.with_suffix(".json").write_text(
        json.dumps({"labels": ["Барсик", "Мурка"], "image_size": 16}),
        encoding="utf-8",
    )
    core = SimpleNamespace(
        read_model=lambda _path: object(),
        compile_model=lambda _model, _device: FakeCompiled(),
    )
    monkeypatch.setitem(sys.modules, "openvino", SimpleNamespace(Core=lambda: core))
    identity = OpenVinoCatIdentity(model)
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    known = identity.predict(frame, 0.8)
    unknown = identity.predict(frame, 0.99)

    assert known.name == "Мурка"
    assert known.confidence > 0.9
    assert unknown.name is None


class FakeEmbedding:
    def __init__(self, vector: list[float]) -> None:
        self.vector = vector

    def output(self, _index: int) -> str:
        return "embedding"

    def __call__(self, _inputs: object) -> dict[str, np.ndarray]:
        return {"embedding": np.asarray([self.vector], dtype=np.float32)}


def _embedding_model(
    runtime_path: Path,
    monkeypatch,
    vector: list[float],
    **metadata: object,
) -> OpenVinoCatIdentity:
    model = runtime_path / "face-identity.onnx"
    model.write_bytes(b"synthetic")
    payload: dict[str, object] = {
        "kind": "embedding",
        "image_size": 16,
        "prototypes": {"Черныш": [1.0, 0.0, 0.0], "Яся": [0.0, 1.0, 0.0]},
        "threshold": 0.55,
    }
    payload.update(metadata)
    model.with_suffix(".json").write_text(json.dumps(payload), encoding="utf-8")
    core = SimpleNamespace(
        read_model=lambda _path: object(),
        compile_model=lambda _model, _device: FakeEmbedding(vector),
    )
    monkeypatch.setitem(sys.modules, "openvino", SimpleNamespace(Core=lambda: core))
    return OpenVinoCatIdentity(model)


def test_the_nearest_prototype_names_the_cat(runtime_path: Path, monkeypatch) -> None:
    identity = _embedding_model(runtime_path, monkeypatch, [0.95, 0.31, 0.0])
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    prediction = identity.predict(frame, 0.99)

    # Порог узла (0.99) намеренно завышен: у эмбеддингов граница едет с
    # моделью, потому что она свойство пространства, а не настройка узла.
    assert prediction.name == "Черныш"
    assert prediction.confidence > 0.9


def test_a_stranger_is_far_from_every_prototype(runtime_path: Path, monkeypatch) -> None:
    """Незнакомец получается порогом, а не отдельным классом «чужой».

    Ради этого свойства эмбеддинги и заводились: класс «чужой» существовал ровно
    постольку, поскольку чужих успели показать.
    """
    identity = _embedding_model(runtime_path, monkeypatch, [0.0, 0.0, 1.0])

    prediction = identity.predict(np.zeros((20, 20, 3), dtype=np.uint8), 0.5)

    assert prediction.name is None
    assert prediction.candidate_name in {"Черныш", "Яся"}


def test_two_equally_close_cats_are_not_a_decision(runtime_path: Path, monkeypatch) -> None:
    """Два кота, похожие в сумерках, дают близкие расстояния до обоих эталонов.

    Выбирать между ними по третьему знаку — значит выдавать монетку за решение.
    """
    identity = _embedding_model(
        runtime_path, monkeypatch, [0.7072, 0.7070, 0.0], threshold=0.5
    )

    prediction = identity.predict(np.zeros((20, 20, 3), dtype=np.uint8), 0.5)

    assert prediction.name is None


def test_an_embedding_model_without_prototypes_is_refused(
    runtime_path: Path,
    monkeypatch,
) -> None:
    """Модель эмбеддингов без эталонов не умеет ничего и молчать об этом нельзя."""
    from src.inference.cat_detector import DetectorError

    model = runtime_path / "face-identity.onnx"
    model.write_bytes(b"synthetic")
    model.with_suffix(".json").write_text(
        json.dumps({"kind": "embedding", "image_size": 16, "prototypes": {}}),
        encoding="utf-8",
    )
    monkeypatch.setitem(
        sys.modules,
        "openvino",
        SimpleNamespace(Core=lambda: SimpleNamespace()),
    )

    with pytest.raises(DetectorError, match="prototypes"):
        OpenVinoCatIdentity(model)


def test_the_classifier_still_works(runtime_path: Path, monkeypatch) -> None:
    """Прежние модели читаются как раньше: на узле уже стоит такая."""
    model = runtime_path / "face-identity.onnx"
    model.write_bytes(b"synthetic")
    model.with_suffix(".json").write_text(
        json.dumps({"labels": ["Барсик", "Мурка"], "image_size": 16}),
        encoding="utf-8",
    )
    core = SimpleNamespace(
        read_model=lambda _path: object(),
        compile_model=lambda _model, _device: FakeCompiled(),
    )
    monkeypatch.setitem(sys.modules, "openvino", SimpleNamespace(Core=lambda: core))
    identity = OpenVinoCatIdentity(model)

    prediction = identity.predict(np.zeros((20, 20, 3), dtype=np.uint8), 0.8)

    assert identity.kind == "logits"
    assert prediction.name == "Мурка"
