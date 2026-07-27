import argparse
import json
from pathlib import Path
from typing import Any

from training.dataset import DatasetItem, load_manifest


def train(
    manifest_path: Path,
    dataset_root: Path,
    view: str,
    output: Path,
    epochs: int,
    image_size: int,
) -> None:
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms

    items = [item for item in load_manifest(manifest_path) if item.view == view]
    labels = sorted({item.label for item in items if item.label != "__foreign__"})
    label_to_index = {label: index for index, label in enumerate(labels)}
    if len(labels) < 2:
        raise ValueError("training requires at least two known cat labels")

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    class ManifestDataset(Dataset[Any]):
        def __init__(self, rows: list[DatasetItem], split: str) -> None:
            self.rows = [
                row for row in rows if row.split == split and row.label in label_to_index
            ]

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> tuple[Any, int]:
            row = self.rows[index]
            with Image.open(dataset_root / row.image_path) as image:
                tensor = transform(image.convert("RGB"))
            return tensor, label_to_index[row.label]

    train_data = ManifestDataset(items, "train")
    validation_data = ManifestDataset(items, "validation")
    if not train_data or not validation_data:
        raise ValueError("train and validation splits must both be non-empty")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # No automatic weight download: training must remain usable in an offline environment.
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(labels))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_function = torch.nn.CrossEntropyLoss()
    loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
    for _ in range(epochs):
        model.train()
        for images, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(images.to(device)), targets.to(device))
            loss.backward()
            optimizer.step()
    model.eval()
    correct = 0
    validation_loader = DataLoader(validation_data, batch_size=32, num_workers=0)
    with torch.no_grad():
        for images, targets in validation_loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            correct += int((predictions == targets).sum())
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": "mobilenet_v3_small",
            "view": view,
            "labels": labels,
            "image_size": image_size,
            "state_dict": model.cpu().state_dict(),
            "validation_accuracy": correct / len(validation_data),
        },
        output,
    )
    print(json.dumps({"device": str(device), "checkpoint": str(output)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CatDoor image classifier on Mac/MPS")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--view", choices=("face", "side"), required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    train(
        args.manifest,
        args.dataset_root,
        args.view,
        args.output,
        args.epochs,
        args.image_size,
    )


if __name__ == "__main__":
    main()
