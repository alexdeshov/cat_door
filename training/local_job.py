import argparse
import json
import traceback
from collections import defaultdict
from pathlib import Path

from training.dataset import load_manifest
from training.train_classifier import train
from training.workflow import build_dataset, write_status


def run_job(
    database_path: Path,
    data_dir: Path,
    ffmpeg_path: str,
    output_dir: Path,
    epochs: int,
) -> None:
    status_path = output_dir / "status.json"
    try:
        write_status(status_path, state="preparing", message="Подготовка кадров")
        dataset_dir = output_dir / "dataset"
        manifest, report = build_dataset(
            database_path,
            data_dir,
            ffmpeg_path,
            "local",
            dataset_dir,
        )
        items_by_view: dict[str, list[object]] = defaultdict(list)
        for item in load_manifest(manifest):
            items_by_view[item.view].append(item)
        trained: list[str] = []
        for view in ("face", "side"):
            if not items_by_view[view]:
                continue
            write_status(
                status_path,
                state="training",
                message=f"Обучение модели {view}",
                view=view,
                report=report,
            )
            train(
                manifest,
                dataset_dir,
                view,
                output_dir / f"{view}.pt",
                epochs,
                160,
            )
            trained.append(view)
        if not trained:
            raise ValueError("dataset has no face or side images")
        write_status(
            status_path,
            state="completed",
            message="Экспериментальное обучение завершено",
            models=trained,
            report=report,
        )
    except Exception as exc:  # noqa: BLE001 - the detached worker must persist failures
        write_status(
            status_path,
            state="failed",
            message=str(exc),
            traceback=traceback.format_exc(),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an isolated local CatDoor training job")
    parser.add_argument("database_path", type=Path)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("ffmpeg_path")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    run_job(
        args.database_path,
        args.data_dir,
        args.ffmpeg_path,
        args.output_dir,
        args.epochs,
    )
    print(json.dumps({"status": str(args.output_dir / "status.json")}))


if __name__ == "__main__":
    main()
