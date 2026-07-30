import argparse
import json
import traceback
from collections import defaultdict
from pathlib import Path

from training.dataset import load_manifest
from training.train_classifier import train
from training.workflow import StatusHeartbeat, build_dataset, write_status


def run_job(
    database_path: Path,
    data_dir: Path,
    ffmpeg_path: str,
    output_dir: Path,
    detector_model_path: Path,
    detector_device: str,
    weights_path: Path | None,
    epochs: int,
) -> None:
    status_path = output_dir / "status.json"
    heartbeat = StatusHeartbeat(status_path)
    heartbeat.start()
    try:
        write_status(status_path, state="preparing", message="Подготовка кадров")
        dataset_dir = output_dir / "dataset"
        manifest, report = build_dataset(
            database_path,
            data_dir,
            ffmpeg_path,
            "local",
            dataset_dir,
            detector_model_path=detector_model_path,
            detector_device=detector_device,
        )
        items_by_view: dict[str, list[object]] = defaultdict(list)
        for item in load_manifest(manifest):
            items_by_view[item.view].append(item)
        trained: list[str] = []
        metrics: dict[str, object] = {}
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
            def report_epoch(
                epoch: int,
                total_epochs: int,
                loss: float,
                training_view: str = view,
            ) -> None:
                write_status(
                    status_path,
                    state="training",
                    message=(
                        f"Обучение модели {training_view}: эпоха {epoch} из {total_epochs}"
                    ),
                    view=training_view,
                    epoch=epoch,
                    epochs=total_epochs,
                    loss=round(loss, 5),
                    report=report,
                    models=trained,
                )

            metrics[view] = train(
                manifest,
                dataset_dir,
                view,
                output_dir / f"{view}.pt",
                epochs,
                160,
                progress=report_epoch,
                weights_path=weights_path,
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
            metrics=metrics,
        )
    except Exception as exc:  # noqa: BLE001 - the detached worker must persist failures
        write_status(
            status_path,
            state="failed",
            message=str(exc),
            traceback=traceback.format_exc(),
        )
    finally:
        heartbeat.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an isolated local CatDoor training job")
    parser.add_argument("database_path", type=Path)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("ffmpeg_path")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--detector-model", type=Path, required=True)
    parser.add_argument("--detector-device", default="CPU")
    parser.add_argument("--weights-path", type=Path)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    run_job(
        args.database_path,
        args.data_dir,
        args.ffmpeg_path,
        args.output_dir,
        args.detector_model,
        args.detector_device,
        args.weights_path,
        args.epochs,
    )
    print(json.dumps({"status": str(args.output_dir / "status.json")}))


if __name__ == "__main__":
    main()
