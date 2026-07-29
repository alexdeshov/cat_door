import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

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
