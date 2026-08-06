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


def _event_order(event_id: str, seed: str = "catdoor-v1") -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{event_id}".encode()).digest()[:8], "big")


def _split_quota(events: int) -> tuple[int, int]:
    """Сколько проходов метки уходит в test и validation.

    Ровно один проход остаётся только в train: по нему нечего проверять.
    Из двух один откладывается в test — иначе метка не участвует в приёмке
    вовсе. От трёх и больше — примерно по 15% туда и туда, но не меньше
    одного прохода в каждый.
    """
    if events <= 1:
        return 0, 0
    if events == 2:
        return 1, 0
    portion = max(1, round(events * 0.15))
    return portion, portion


def assign_splits(items: list[DatasetItem], seed: str = "catdoor-v1") -> list[DatasetItem]:
    """Разложить проходы по train/validation/test, не оставляя меток без test.

    Раньше сплит выбирался хэшем каждого прохода независимо, и малочисленные
    метки целиком проваливались мимо test: шести проходов чужих котов хватило,
    чтобы ни один не отложился, FAR стало не из чего считать, и приёмка не
    могла пройти ни при какой модели. Теперь доли распределяются внутри каждой
    метки: проходы выстраиваются по тому же хэшу и первые из них уходят в test
    и validation по квоте `_split_quota`.

    Расклад детерминирован для фиксированного набора проходов, но при росте
    набора отдельный проход может сменить сплит. Это честная цена: модель
    каждый раз обучается заново на своём train и не видит свой test, а
    сравнивать приёмки разных наборов между собой и так нельзя.
    """
    events_by_label: dict[str, set[str]] = defaultdict(set)
    label_by_event: dict[str, str] = {}
    for item in items:
        events_by_label[item.label].add(item.event_id)
        label_by_event.setdefault(item.event_id, item.label)
        if label_by_event[item.event_id] != item.label:
            raise ValueError(f"event {item.event_id} carries multiple labels")

    split_by_event: dict[str, str] = {}
    for label in sorted(events_by_label):
        ordered = sorted(events_by_label[label], key=lambda event: _event_order(event, seed))
        test_quota, validation_quota = _split_quota(len(ordered))
        for index, event_id in enumerate(ordered):
            if index < test_quota:
                split_by_event[event_id] = "test"
            elif index < test_quota + validation_quota:
                split_by_event[event_id] = "validation"
            else:
                split_by_event[event_id] = "train"

    return [
        DatasetItem(
            event_id=item.event_id,
            captured_at=item.captured_at,
            view=item.view,
            image_path=item.image_path,
            label=item.label,
            split=split_by_event[item.event_id],
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
