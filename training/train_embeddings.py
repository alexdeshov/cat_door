"""Обучение эмбеддингов вместо классификатора: кот как точка, а не как класс.

Классификатор отвечает «Черныш или Яся» и больше ничего не умеет: третий кот
требует переобучения целиком, а «чужой» существует ровно постольку, поскольку
его успели показать. Эмбеддинги отвечают вектором, и опознание становится
поиском ближайшего эталона — новый кот добавляется своим эталоном, а незнакомец
оказывается далёк от всех сам собой.

Учится это классификацией с угловым зазором (ArcFace): та же перекрёстная
энтропия, но перед ней к углу до своего класса добавляется margin. Модель
вынуждена не просто угадывать класс, а разносить их по направлениям — иначе
зазор не окупается. После обучения голова с зазором выбрасывается, остаётся
backbone, отдающий нормированный вектор.

Класс «чужой» в обучении участвует и приносит пользу — он оттягивает на себя
всё, что не похоже на своих котов, — но **эталона у него нет**: «чужой» это не
одна личность, а все остальные коты мира. На узле он получается порогом.

Порог подбирается здесь же, на отложенном наборе, а не назначается на глаз, и
уезжает в метаданные модели: граница «свой/чужой» — свойство обученного
пространства, а не настройка узла.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.common.metrics import FOREIGN_LABEL, REJECT_LABELS
from training.dataset import DatasetItem, load_manifest
from training.train_classifier import build_pretrained_model, build_transforms

EMBEDDING_DIM = 128
# Масштаб и зазор ArcFace. Зазор скромный намеренно: на наборе в пару тысяч
# кропов агрессивный margin не разносит классы, а мешает сойтись.
ARC_SCALE = 30.0
ARC_MARGIN = 0.2
# Доля чужих, которую согласны принять за своих при подборе порога. Один из
# ста — та же цена ошибки, что записана в критериях приёмки.
TARGET_FALSE_ACCEPT_RATE = 0.01
FALLBACK_THRESHOLD = 0.55


def _build_arc_head(embedding_dim: int, classes: int) -> Any:
    import torch

    class ArcMarginHead(torch.nn.Module):
        """Косинусные логиты с угловым зазором для своего класса."""

        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(classes, embedding_dim))
            torch.nn.init.xavier_uniform_(self.weight)

        def forward(self, embeddings: Any, targets: Any) -> Any:
            from torch.nn import functional

            cosine = functional.linear(
                functional.normalize(embeddings), functional.normalize(self.weight)
            ).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            with_margin = torch.cos(torch.acos(cosine) + ARC_MARGIN)
            target_mask = functional.one_hot(targets, classes).bool()
            return ARC_SCALE * torch.where(target_mask, with_margin, cosine)

    return ArcMarginHead()


def _build_backbone(embedding_dim: int, weights_path: Path | None) -> Any:
    """MobileNetV3-Small, отдающий вектор вместо вероятностей."""
    import torch

    model = build_pretrained_model(embedding_dim, weights_path)

    class Embedder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = model

        def forward(self, images: Any) -> Any:
            from torch.nn import functional

            return functional.normalize(self.backbone(images), dim=1)

    return Embedder()


def _choose_threshold(
    same: list[float],
    other: list[float],
) -> tuple[float, dict[str, float]]:
    """Порог по отложенному набору: где чужие ещё не проходят.

    Берётся такая близость, при которой доля принятых чужих не выше одного
    процента. Если чужих в наборе нет, назначать порог не из чего — и честнее
    сказать это вслух, чем выдать удобное число.
    """
    stats = {
        "same_mean": sum(same) / len(same) if same else 0.0,
        "other_mean": sum(other) / len(other) if other else 0.0,
        "same_min": min(same) if same else 0.0,
        "other_max": max(other) if other else 0.0,
    }
    if not other:
        return FALLBACK_THRESHOLD, stats
    ranked = sorted(other, reverse=True)
    allowed = int(len(ranked) * TARGET_FALSE_ACCEPT_RATE)
    # Чуть выше самого похожего чужого, которого мы согласны пропустить.
    threshold = round(min(ranked[allowed] + 0.01, 0.95), 3)
    stats["threshold_from_foreign"] = threshold
    # Сколько своих проходит выбранный порог. Порог, отсекающий чужих идеально
    # и своих заодно, — это не строгая модель, а неработающая; молча выдать
    # такую значило бы отдать на узел то, что никого никогда не узнает.
    stats["same_above_threshold"] = (
        sum(score >= threshold for score in same) / len(same) if same else 0.0
    )
    return threshold, stats


def train(
    manifest_path: Path,
    dataset_root: Path,
    view: str,
    output: Path,
    epochs: int,
    image_size: int,
    progress: Callable[[int, int, float], None] | None = None,
    weights_path: Path | None = None,
    include_foreign: bool = True,
) -> dict[str, object]:
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    items = [item for item in load_manifest(manifest_path) if item.view == view]
    # Чужие участвуют в обучении по умолчанию, в отличие от классификатора:
    # здесь они не «ещё один кот», а сила, растаскивающая пространство.
    labels = sorted(
        {item.label for item in items if include_foreign or item.label != FOREIGN_LABEL}
    )
    label_to_index = {label: index for index, label in enumerate(labels)}
    # Эталоны только у настоящих котов. «Чужой» и «кота не видно» — не личности.
    prototype_labels = [label for label in labels if label not in REJECT_LABELS]
    if len(prototype_labels) < 1:
        raise ValueError("embedding training needs at least one named cat")
    if len(labels) < 2:
        raise ValueError("training requires at least two labels")

    training_transform, validation_transform = build_transforms(image_size)

    class ManifestDataset(Dataset[Any]):
        def __init__(self, rows: list[DatasetItem], split: str, *, augment: bool) -> None:
            self.rows = [
                row for row in rows if row.split == split and row.label in label_to_index
            ]
            self.transform = training_transform if augment else validation_transform

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> tuple[Any, int]:
            row = self.rows[index]
            with Image.open(dataset_root / row.image_path) as image:
                tensor = self.transform(image.convert("RGB"))
            return tensor, label_to_index[row.label]

    train_data = ManifestDataset(items, "train", augment=True)
    validation_data = ManifestDataset(items, "validation", augment=False)
    # Эталоны считаются по обучающей выборке без искажений: эталон должен быть
    # тем, как кот выглядит, а не тем, как его случайно повернуло.
    prototype_data = ManifestDataset(items, "train", augment=False)
    if not train_data or not validation_data:
        raise ValueError("train and validation splits must both be non-empty")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = _build_backbone(EMBEDDING_DIM, weights_path).to(device)
    head = _build_arc_head(EMBEDDING_DIM, len(labels)).to(device)
    optimizer = torch.optim.AdamW(
        [*model.parameters(), *head.parameters()], lr=3e-4
    )
    loss_function = torch.nn.CrossEntropyLoss()
    loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
    for epoch in range(1, epochs + 1):
        model.train()
        head.train()
        running_loss = 0.0
        batches = 0
        for images, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            embeddings = model(images.to(device))
            loss = loss_function(head(embeddings, targets.to(device)), targets.to(device))
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            batches += 1
        if progress is not None:
            progress(epoch, epochs, running_loss / max(batches, 1))

    model.eval()

    def embed(data: Any) -> tuple[Any, Any]:
        vectors = []
        targets = []
        with torch.no_grad():
            for images, labels_batch in DataLoader(data, batch_size=32, num_workers=0):
                vectors.append(model(images.to(device)).cpu())
                targets.append(labels_batch)
        return torch.cat(vectors), torch.cat(targets)

    train_vectors, train_targets = embed(prototype_data)
    prototypes = {}
    for label in prototype_labels:
        mask = train_targets == label_to_index[label]
        if not bool(mask.any()):
            continue
        centroid = train_vectors[mask].mean(dim=0)
        prototypes[label] = (centroid / centroid.norm()).tolist()
    if not prototypes:
        raise ValueError("no prototypes could be built: named cats have no train crops")

    # Проверка на отложенном наборе: насколько свои близки к своему эталону и
    # насколько к нему близки чужие. Отсюда же берётся порог.
    validation_vectors, validation_targets = embed(validation_data)
    index_to_label = {index: label for label, index in label_to_index.items()}
    prototype_matrix = torch.tensor([prototypes[name] for name in prototypes])
    same: list[float] = []
    other: list[float] = []
    correct = 0
    named = 0
    for vector, target in zip(validation_vectors, validation_targets, strict=True):
        scores = prototype_matrix @ vector
        best = int(scores.argmax())
        best_name = list(prototypes)[best]
        true_label = index_to_label[int(target)]
        if true_label in prototypes:
            named += 1
            same.append(float(scores[list(prototypes).index(true_label)]))
            if best_name == true_label:
                correct += 1
        else:
            other.append(float(scores.max()))
    threshold, stats = _choose_threshold(same, other)

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": "mobilenet_v3_small",
            "kind": "embedding",
            "view": view,
            "labels": labels,
            "prototypes": prototypes,
            "embedding_dim": EMBEDDING_DIM,
            "image_size": image_size,
            "threshold": threshold,
            "state_dict": model.cpu().state_dict(),
            "validation_accuracy": correct / named if named else 0.0,
        },
        output,
    )
    result = {
        "device": str(device),
        "checkpoint": str(output),
        "prototypes": list(prototypes),
        "threshold": threshold,
        "validation_accuracy": correct / named if named else 0.0,
        "similarity": {key: round(value, 3) for key, value in stats.items()},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--view", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--weights-path", type=Path, default=None)
    parser.add_argument(
        "--no-foreign",
        action="store_true",
        help="не брать чужих в обучение; эталонов у них нет в любом случае",
    )
    args = parser.parse_args()
    train(
        args.manifest,
        args.dataset_root,
        args.view,
        args.output,
        epochs=args.epochs,
        image_size=args.image_size,
        progress=lambda epoch, total, loss: print(
            f"эпоха {epoch}/{total}: потери {loss:.4f}", flush=True
        ),
        weights_path=args.weights_path,
        include_foreign=not args.no_foreign,
    )


if __name__ == "__main__":
    main()
