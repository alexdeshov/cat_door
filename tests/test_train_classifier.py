import sys
from pathlib import Path

import pytest

from training.train_classifier import (
    PretrainedWeightsError,
    build_pretrained_model,
    build_transforms,
)


def test_training_augmentations_preserve_view_semantics() -> None:
    pytest.importorskip("torchvision")

    training_transform, validation_transform = build_transforms(160)
    training_steps = [type(step).__name__ for step in training_transform.transforms]
    validation_steps = [type(step).__name__ for step in validation_transform.transforms]

    assert "RandomHorizontalFlip" not in training_steps
    assert "RandomResizedCrop" in training_steps
    assert "RandomRotation" in training_steps
    assert "ColorJitter" in training_steps
    assert validation_steps == ["Resize", "ToTensor", "Normalize"]


def test_explicit_local_imagenet_weights_are_loaded(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    models = pytest.importorskip("torchvision.models")
    weights_path = tmp_path / "mobilenet-v3-small.pth"
    source = models.mobilenet_v3_small(weights=None)
    torch.save(source.state_dict(), weights_path)

    model = build_pretrained_model(2, weights_path)

    assert model.classifier[3].out_features == 2
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_missing_cached_weights_has_actionable_offline_error(monkeypatch) -> None:
    models = pytest.importorskip("torchvision.models")

    def unavailable(*args, **kwargs):
        raise OSError("network unavailable")

    monkeypatch.setattr(models, "mobilenet_v3_small", unavailable)

    with pytest.raises(
        PretrainedWeightsError,
        match="network access.*--weights-path",
    ):
        build_pretrained_model(2, None)


def test_missing_explicit_weights_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(PretrainedWeightsError, match="does not exist"):
        build_pretrained_model(2, tmp_path / "missing.pth")


def test_weights_path_is_checked_before_torch_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typo in the path must be named as such even without torch installed.

    The runtime host carries no training dependencies, so the check has to run
    before the import rather than after it.
    """
    monkeypatch.setitem(sys.modules, "torch", None)

    with pytest.raises(PretrainedWeightsError, match="does not exist"):
        build_pretrained_model(2, tmp_path / "missing.pth")
