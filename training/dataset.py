import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

ALLOWED_VIEWS = {"face", "side"}
ALLOWED_SPLITS = {"train", "validation", "test"}


@dataclass(frozen=True)
class DatasetItem:
    event_id: str
    captured_at: str
    view: str
    image_path: str
    label: str
    split: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DatasetItem":
        required = ("event_id", "captured_at", "view", "image_path", "label")
        missing = [key for key in required if not value.get(key)]
        if missing:
            raise ValueError(f"missing fields: {', '.join(missing)}")
        item = cls(
            event_id=str(value["event_id"]),
            captured_at=str(value["captured_at"]),
            view=str(value["view"]),
            image_path=str(value["image_path"]),
            label=str(value["label"]),
            split=str(value["split"]) if value.get("split") else None,
        )
        item.validate()
        return item

    def validate(self) -> None:
        datetime.fromisoformat(self.captured_at)
        if self.view not in ALLOWED_VIEWS:
            raise ValueError(f"unsupported view: {self.view}")
        if self.split is not None and self.split not in ALLOWED_SPLITS:
            raise ValueError(f"unsupported split: {self.split}")
        path = PurePosixPath(self.image_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("image_path must be a safe relative POSIX path")


def load_manifest(path: Path) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            items.append(DatasetItem.from_dict(json.loads(line)))
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not items:
        raise ValueError("manifest is empty")
    return items


def validate_manifest(items: list[DatasetItem], dataset_root: Path | None = None) -> dict[str, Any]:
    event_splits: dict[str, set[str]] = defaultdict(set)
    missing_files: list[str] = []
    for item in items:
        if item.split:
            event_splits[item.event_id].add(item.split)
        if dataset_root and not (dataset_root / Path(item.image_path)).is_file():
            missing_files.append(item.image_path)
    leaking = sorted(event_id for event_id, splits in event_splits.items() if len(splits) > 1)
    if leaking:
        raise ValueError(f"events occur in multiple splits: {', '.join(leaking)}")
    if missing_files:
        raise ValueError(f"missing image files: {', '.join(sorted(missing_files)[:10])}")
    return {
        "items": len(items),
        "events": len({item.event_id for item in items}),
        "labels": dict(sorted(Counter(item.label for item in items).items())),
        "views": dict(sorted(Counter(item.view for item in items).items())),
        "splits": dict(sorted(Counter(item.split or "unassigned" for item in items).items())),
    }


def split_for_event(event_id: str, seed: str = "catdoor-v1") -> str:
    value = int.from_bytes(hashlib.sha256(f"{seed}:{event_id}".encode()).digest()[:8], "big")
    bucket = value % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def assign_splits(items: list[DatasetItem], seed: str = "catdoor-v1") -> list[DatasetItem]:
    return [
        DatasetItem(
            event_id=item.event_id,
            captured_at=item.captured_at,
            view=item.view,
            image_path=item.image_path,
            label=item.label,
            split=split_for_event(item.event_id, seed),
        )
        for item in items
    ]


def write_manifest(items: list[DatasetItem], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(asdict(item), ensure_ascii=False) + "\n" for item in items),
        encoding="utf-8",
    )
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate or split a CatDoor dataset manifest")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--assign-splits", type=Path, metavar="OUTPUT")
    parser.add_argument("--seed", default="catdoor-v1")
    args = parser.parse_args()
    items = load_manifest(args.manifest)
    if args.assign_splits:
        items = assign_splits(items, args.seed)
        write_manifest(items, args.assign_splits)
    print(json.dumps(validate_manifest(items, args.dataset_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
