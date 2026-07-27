from pathlib import Path

import pytest

from training.export_onnx import export_checkpoint


def test_export_checkpoint_creates_valid_onnx_with_dynamic_batch(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    onnx = pytest.importorskip("onnx")
    models = pytest.importorskip("torchvision.models")

    labels = ["known-cat-a", "known-cat-b"]
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(labels))
    checkpoint = tmp_path / "face.pt"
    torch.save(
        {
            "architecture": "mobilenet_v3_small",
            "view": "face",
            "labels": labels,
            "image_size": 224,
            "state_dict": model.state_dict(),
            "validation_accuracy": 0.0,
        },
        checkpoint,
    )

    destination = tmp_path / "face_model.onnx"
    model_info = export_checkpoint(checkpoint, destination)

    exported = onnx.load(destination)
    onnx.checker.check_model(exported)
    batch_dimension = exported.graph.input[0].type.tensor_type.shape.dim[0]
    assert batch_dimension.dim_param == "batch"
    assert model_info["onnx_file"] == destination.name
    assert model_info["labels"] == labels
