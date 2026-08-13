"""Полное обучение опознания — прямо на узле, без макбука.

До 2026-08-13 этот модуль обучал экспериментальный классификатор и годился
только для проверки конвейера. Теперь он делает то же, что цикл на макбуке, и
тем же кодом: набор → пространство эмбеддингов → ONNX → приёмка. Разница только
в железе — процессор вместо MPS — и во времени: часы вместо десятков минут.
Узлу это по силам, потому что обучение идёт ночью и никуда не торопится.

Чем автономный прогон отличается от макбучного по существу:

- **Каких котов учить, решает установленная модель.** На макбуке владелец
  передаёт `--cats` руками; здесь имена берутся из меток стоящей модели — узел
  переобучает тех, кого уже знает. Остальные именованные коты уезжают в
  отрицательные примеры, как и на макбуке.
- **Сравнение входит в прогон.** Установленная модель прогоняется через тот же
  отложенный набор, что и новая, — иначе отчёт показывал бы цифры двух разных
  экзаменов, а они несравнимы (`docs/model-benchmarks.md`).
- **Модель не применяется сама.** Прогон оставляет пакет и отчёт в своём
  каталоге; решение о замене — за владельцем. Это отличие от пересчёта
  эталонов намеренное: эталоны — обратимая правка метаданных с проверкой «не
  хуже», а здесь меняется сама сеть.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.common.metrics import FOREIGN_LABEL, REJECT_LABELS
from training.dataset import assign_splits, load_manifest, write_manifest
from training.workflow import StatusHeartbeat, build_dataset, write_status

# Пределы приёмки — те же, что в пакете макбука (model-manifests/thresholds).
# Второго источника не заводим: цена ошибки не зависит от того, где считали.
ACCEPTANCE_THRESHOLDS = {
    "accept_score": 0.9,
    "max_false_accept_rate": 0.01,
    "max_false_reject_rate": 0.2,
    "max_mean_inference_ms": 200.0,
}


def _known_cats(model_paths: dict[str, Path]) -> set[str]:
    """Кого узел уже знает. Пусто — учим всех именованных из разметки."""
    names: set[str] = set()
    for path in model_paths.values():
        metadata = path.with_suffix(".json")
        try:
            payload = json.loads(metadata.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        names.update(
            str(label)
            for label in payload.get("labels", [])
            if str(label) not in REJECT_LABELS
        )
    return names


def _restrict_manifest(manifest: Path, cats: set[str]) -> None:
    """Оставить классами только известных котов, остальных — в «чужие».

    Тот же приём, что у `restrict_to_cats` конвейера: настоящие коты не того
    имени — лучшие отрицательные примеры из возможных.
    """
    lines = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item["label"] not in cats and item["label"] not in REJECT_LABELS:
            item["label"] = FOREIGN_LABEL
        lines.append(json.dumps(item, ensure_ascii=False))
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metrics_dict(metrics: Any) -> dict[str, object]:
    return {
        "accuracy": round(metrics.accuracy, 3),
        "false_accept_rate": round(metrics.false_accept_rate, 3),
        "false_reject_rate": round(metrics.false_reject_rate, 3),
        "samples": metrics.samples,
        "passed": metrics.passed,
    }


def _not_worse(new: Any, installed: Any) -> bool:
    """Годится ли новая модель взамен стоящей — на одном и том же экзамене.

    Ложные пропуски не прощаются: пустить чужого кота дороже, чем не узнать
    своего. Узнаваемость может просесть на пару сотых — это шум выборки.
    """
    return (
        new.false_accept_rate <= installed.false_accept_rate
        and new.false_reject_rate <= installed.false_reject_rate + 0.02
    )


def run_job(
    database_path: Path,
    data_dir: Path,
    ffmpeg_path: str,
    output_dir: Path,
    detector_model_path: Path,
    detector_device: str,
    weights_path: Path | None,
    epochs: int,
    image_size: int,
    installed: dict[str, Path],
) -> None:
    status_path = output_dir / "status.json"
    # Веса ImageNet torchvision скачивает в ~/.cache — а у службы нет ни дома
    # (`/home/catdoor` не существует), ни доступа к нему (`ProtectHome=true` в
    # юните). Кеш направляется в каталог данных: он у службы точно писуемый, и
    # скачивание случается один раз, а не на каждый прогон.
    os.environ.setdefault("TORCH_HOME", str(data_dir / "training" / "torch-cache"))
    # Временные файлы — тоже в каталог данных, и это не причуда. У веб-службы
    # `PrivateTmp=true`: её /tmp — приватное пространство имён, и ребёнок его
    # наследует. `KillMode=process` спасает прогон от перезапуска админки, но
    # приватный /tmp умирает вместе со старым процессом службы — и 13 августа
    # прогон, переживший три перезапуска, упал через два часа обучения на
    # первом же tempfile в приёмке: «No usable temporary directory found».
    scratch = data_dir / "training" / "tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(scratch)
    tempfile.tempdir = None  # заставить tempfile перечитать TMPDIR
    heartbeat = StatusHeartbeat(status_path)
    heartbeat.start()
    try:
        write_status(status_path, state="preparing", message="Готовлю кадры размеченных проходов")
        dataset_dir = output_dir / "dataset"
        manifest, report = build_dataset(
            database_path,
            data_dir,
            ffmpeg_path,
            "all",
            dataset_dir,
            detector_model_path=detector_model_path,
            detector_device=detector_device,
        )
        cats = _known_cats(installed)
        if cats:
            _restrict_manifest(manifest, cats)
        items = assign_splits(load_manifest(manifest))
        split_manifest = output_dir / "manifest.split.jsonl"
        write_manifest(items, split_manifest)
        items_by_view: dict[str, list[Any]] = defaultdict(list)
        for item in items:
            items_by_view[item.view].append(item)

        from training.acceptance import run_onnx_acceptance, select_test_items
        from training.export_onnx import export_checkpoint
        from training.train_embeddings import train

        views: dict[str, dict[str, object]] = {}
        for view in ("face", "side"):
            if not items_by_view[view]:
                continue

            def report_epoch(
                epoch: int, total: int, loss: float, training_view: str = view
            ) -> None:
                write_status(
                    status_path,
                    state="training",
                    message=f"Обучение {training_view}: эпоха {epoch} из {total}",
                    view=training_view,
                    epoch=epoch,
                    epochs=total,
                    loss=round(loss, 5),
                    report=report,
                    views=views,
                )

            write_status(
                status_path,
                state="training",
                message=f"Обучение {view}: подготовка",
                report=report,
                views=views,
            )
            checkpoint = output_dir / f"{view}.pt"
            train(
                split_manifest,
                dataset_dir,
                view,
                checkpoint,
                epochs,
                image_size,
                progress=report_epoch,
                weights_path=weights_path,
            )
            write_status(
                status_path,
                state="training",
                message=f"Приёмка {view} на отложенных кропах",
                report=report,
                views=views,
            )
            onnx_path = output_dir / f"{view}_model.onnx"
            model_info = export_checkpoint(checkpoint, onnx_path)
            (output_dir / f"{view}_model.json").write_text(
                json.dumps(model_info, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            test_items = select_test_items(items, view)
            new_metrics = run_onnx_acceptance(
                onnx_path, test_items, dataset_dir, model_info, ACCEPTANCE_THRESHOLDS
            )
            comparison: dict[str, object] = {
                "new": _metrics_dict(new_metrics),
                "image_size": image_size,
                "threshold": model_info.get("threshold"),
            }
            stale_path = installed.get(view)
            if stale_path is not None and stale_path.is_file():
                # Стоящая модель сдаёт тот же экзамен: цифры разных наборов
                # несравнимы, и без этой строки отчёт был бы враньём.
                try:
                    stale_info = json.loads(
                        stale_path.with_suffix(".json").read_text(encoding="utf-8")
                    )
                    stale_metrics = run_onnx_acceptance(
                        stale_path, test_items, dataset_dir, stale_info,
                        ACCEPTANCE_THRESHOLDS,
                    )
                    comparison["installed"] = _metrics_dict(stale_metrics)
                    comparison["better"] = _not_worse(new_metrics, stale_metrics)
                except (OSError, ValueError, RuntimeError) as exc:
                    comparison["installed_error"] = str(exc)
            views[view] = comparison

        if not views:
            raise ValueError("в наборе нет кропов ни одного ракурса")
        better = [view for view, row in views.items() if row.get("better")]
        write_status(
            status_path,
            state="completed",
            message=(
                "Обучение завершено. Лучше стоящей: "
                + (", ".join(better) if better else "ничего — модели остаются прежними")
            ),
            report=report,
            views=views,
            package_dir=str(output_dir),
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database_path", type=Path)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("ffmpeg_path")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--detector-model", type=Path, required=True)
    parser.add_argument("--detector-device", default="CPU")
    parser.add_argument("--weights-path", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    # 224 — вывод замера 2026-08-13: на 160 ухо занимает считанные пиксели, и
    # коты, отличимые формой ушей, сливаются.
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--face-model", type=Path, default=None)
    parser.add_argument("--side-model", type=Path, default=None)
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
        args.image_size,
        {
            role: path
            for role, path in (("face", args.face_model), ("side", args.side_model))
            if path is not None
        },
    )


if __name__ == "__main__":
    main()
