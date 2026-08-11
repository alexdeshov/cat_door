"""Пересчёт эталонов котов на узле — то, что в админке зовётся «Запомнить заново».

Сеть здесь не обучается и torch не нужен. Берётся модель, которая уже стоит на
узле, через неё прогоняются все размеченные проходы, и эталон каждого кота
пересчитывается как среднее его векторов. Ровно так добавляет лица Frigate:
обучения нет, есть запоминание.

Отсюда и главное свойство: **новый кот заводится без переобучения**. Разметили
дюжину проходов Джуна, нажали кнопку — у него появился эталон. У классификатора
то же самое означало бы полный цикл на макбуке.

Чего пересчёт **не** делает: он не улучшает зрение. Если модель в принципе не
различает двух чёрных котов в инфракрасном свете, их векторы лежат рядом, и
сколько среднее ни уточняй, они рядом и останутся. Эту границу двигает только
обучение backbone — редкая работа на макбуке.

Отсюда же и правило по умолчанию: **обновляются только те коты, которых
пространство знает**. Измерено 2026-08-11 на живом узле: пересчёт завёл эталоны
всем шести размеченным котам, хотя пространство обучено на двух, — и
узнаваемость упала с 0.72 до 0.34. Четверо остальных при обучении были
«чужими», их векторы свалены в одну область и между собой неразличимы; порог,
вынужденный отсечь такое, отсекает заодно и своих.

Нового кота можно добавить явно (`--include`), и отчёт покажет цену. Но честный
путь для него — обучение, где он станет отдельным классом.

Новые эталоны применяются, только если они **не пускают больше чужих**, чем
прежние. Иначе они остаются в отчёте, а на узле продолжает работать то, что
работало: молча ухудшить опознание — худшее, что может сделать кнопка,
нажатая ради улучшения.
"""

from __future__ import annotations

import argparse
import json
import logging
import traceback
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.common.durable_write import write_text_durably
from src.common.metrics import REJECT_LABELS
from src.common.prototypes import (
    MatchQuality,
    centroid,
    choose_threshold,
    evaluate,
    unit,
)
from src.inference.cat_identity import OpenVinoCatIdentity
from training.dataset import load_manifest
from training.workflow import StatusHeartbeat, build_dataset, write_status

logger = logging.getLogger("catdoor.enrolment")

# Сколько кропов кота нужно, чтобы эталон был эталоном, а не случайностью.
# Десяток — та граница, ниже которой среднее ловит освещение и позу, а не кота.
MIN_CROPS_PER_CAT = 10


class EnrolmentError(RuntimeError):
    pass


def _embeddings(
    model: OpenVinoCatIdentity,
    dataset_root: Path,
    rows: list[Any],
) -> list[tuple[str, str, np.ndarray]]:
    """Векторы кропов: сплит, метка и сам вектор."""
    vectors = []
    for row in rows:
        with Image.open(dataset_root / row.image_path) as image:
            frame = np.asarray(image.convert("RGB"), dtype=np.uint8)
        vectors.append((row.split or "train", row.label, unit(model.embed(frame))))
    return vectors


def rebuild_for_view(
    model_path: Path,
    dataset_root: Path,
    rows: list[Any],
    *,
    device: str = "CPU",
    include: Sequence[str] = (),
) -> dict[str, object]:
    """Пересчитать эталоны одного ракурса. Возвращает отчёт, файл не трогает.

    По умолчанию обновляются только коты, которых модель уже знает. Кот,
    которого пространство не обучалось различать, эталона не получает: его
    векторы лежат там же, где все чужие, и добавление такого эталона рушит
    опознание остальных — измерено, а не предположено.
    """
    model = OpenVinoCatIdentity(model_path, device)
    if model.kind != "embedding":
        raise EnrolmentError(
            "на узле стоит классификатор: эталонов у него нет, и пересчитывать нечего. "
            "Сперва нужна модель эмбеддингов"
        )
    allowed = set(model.prototypes) | set(include)
    vectors = _embeddings(model, dataset_root, rows)
    by_label: dict[str, list[np.ndarray]] = defaultdict(list)
    for split, label, vector in vectors:
        if split == "train" and label not in REJECT_LABELS:
            by_label[label].append(vector)

    prototypes: dict[str, list[float]] = {}
    thin: dict[str, int] = {}
    unknown: dict[str, int] = {}
    for label, values in sorted(by_label.items()):
        if label not in allowed:
            unknown[label] = len(values)
            continue
        if len(values) < MIN_CROPS_PER_CAT:
            # Кот, показанный трижды, эталона не получает: среднее по трём
            # кадрам описывает освещение, а не кота, и пускать по нему в дом
            # хуже, чем не узнать.
            thin[label] = len(values)
            continue
        prototypes[label] = centroid([value.tolist() for value in values])
    if not prototypes:
        raise EnrolmentError(
            "ни у одного кота не набралось "
            f"{MIN_CROPS_PER_CAT} кропов — размечать пока нечего"
        )

    held_out = [(label, vector) for split, label, vector in vectors if split != "train"]
    same = [
        float(np.dot(unit(prototypes[label]), vector))
        for label, vector in held_out
        if label in prototypes
    ]
    other = [
        max(float(np.dot(unit(value), vector)) for value in prototypes.values())
        for label, vector in held_out
        if label not in prototypes
    ]
    threshold, stats = choose_threshold(same, other)

    fresh = evaluate(prototypes, held_out, threshold=threshold)
    stale = evaluate(model.prototypes, held_out, threshold=model.threshold)
    return {
        "view": rows[0].view if rows else "",
        "prototypes": prototypes,
        "threshold": threshold,
        "crops": {label: len(values) for label, values in sorted(by_label.items())},
        "too_thin": thin,
        # Коты, размеченные в базе, но незнакомые пространству: им нужно
        # обучение, а не эталон.
        "unknown_to_model": unknown,
        "similarity": {key: round(value, 3) for key, value in stats.items()},
        "fresh": fresh,
        "stale": stale,
        "better": fresh.not_worse_than(stale),
    }


def apply_prototypes(model_path: Path, report: dict[str, object]) -> None:
    """Переписать эталоны и порог рядом с моделью.

    Сама модель не трогается: меняются только эталоны в метаданных, и потому
    применение стоит миллисекунды, а откат — это прежний файл. Запись атомарная:
    рекордер читает этот файл на ходу, и половина эталонов хуже, чем старые.
    """
    metadata_path = model_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["prototypes"] = report["prototypes"]
    metadata["threshold"] = report["threshold"]
    metadata["labels"] = sorted(report["prototypes"])  # type: ignore[arg-type]
    write_text_durably(
        metadata_path,
        json.dumps(metadata, ensure_ascii=False, indent=2),
        mode=0o644,
    )


def _quality(value: MatchQuality) -> dict[str, object]:
    return {
        "recognised": round(value.recognised, 3),
        "false_accepts": round(value.false_accepts, 3),
        "samples": value.samples,
        "strangers": value.strangers,
    }


def run_job(
    database_path: Path,
    data_dir: Path,
    ffmpeg_path: str,
    output_dir: Path,
    *,
    detector_model_path: Path,
    detector_device: str,
    identity_models: dict[str, Path],
    apply: bool = True,
    include: Sequence[str] = (),
    dataset_dir: Path | None = None,
) -> dict[str, object]:
    status_path = output_dir / "status.json"
    heartbeat = StatusHeartbeat(status_path)
    heartbeat.start()
    report: dict[str, object] = {}
    try:
        # Готовый набор кропов можно переиспользовать: распаковка кадров стоит
        # час процессора, и повторять её ради других настроек пересчёта — та же
        # работа впустую, что и заново скачивать набор на макбук.
        if dataset_dir is not None and (dataset_dir / "dataset.jsonl").is_file():
            write_status(
                status_path, state="preparing", message="Беру готовые кадры прошлого прогона"
            )
            manifest = dataset_dir / "dataset.jsonl"
            dataset_report = json.loads(
                (dataset_dir / "dataset-report.json").read_text(encoding="utf-8")
            )
        else:
            write_status(
                status_path, state="preparing", message="Готовлю кадры размеченных проходов"
            )
            dataset_dir = output_dir / "dataset"
            manifest, dataset_report = build_dataset(
                database_path,
                data_dir,
                ffmpeg_path,
                "all",
                dataset_dir,
                detector_model_path=detector_model_path,
                detector_device=detector_device,
            )
        rows_by_view: dict[str, list[Any]] = defaultdict(list)
        for item in load_manifest(manifest):
            rows_by_view[item.view].append(item)

        views: dict[str, object] = {}
        applied: list[str] = []
        for role, model_path in identity_models.items():
            rows = rows_by_view.get(role, [])
            if not rows:
                views[role] = {"skipped": "ни одного кропа этого ракурса"}
                continue
            write_status(
                status_path,
                state="running",
                message=f"Считаю эталоны: {role}, кропов {len(rows)}",
            )
            view_report = rebuild_for_view(
                model_path, dataset_dir, rows, device=detector_device, include=include
            )
            if apply and view_report["better"]:
                apply_prototypes(model_path, view_report)
                applied.append(role)
            views[role] = {
                "prototypes": sorted(view_report["prototypes"]),  # type: ignore[arg-type]
                "threshold": view_report["threshold"],
                "crops": view_report["crops"],
                "too_thin": view_report["too_thin"],
                "unknown_to_model": view_report["unknown_to_model"],
                "similarity": view_report["similarity"],
                "before": _quality(view_report["stale"]),  # type: ignore[arg-type]
                "after": _quality(view_report["fresh"]),  # type: ignore[arg-type]
                "applied": role in applied,
            }
        report = {
            "events": dataset_report.get("events"),
            "views": views,
            "applied": applied,
        }
        (output_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_status(
            status_path,
            state="completed",
            message=(
                f"Эталоны обновлены: {', '.join(applied)}"
                if applied
                else "Новые эталоны оказались не лучше прежних — оставил как было"
            ),
            report=report,
        )
    except Exception as exc:
        # Кнопка не должна оставлять человека наедине с пустым статусом.
        logger.exception("Пересчёт эталонов не удался")
        write_status(
            status_path,
            state="failed",
            message=str(exc),
            traceback=traceback.format_exc()[-2000:],
        )
    finally:
        heartbeat.stop()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database_path", type=Path)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("ffmpeg_path")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--detector-model", type=Path, required=True)
    parser.add_argument("--detector-device", default="CPU")
    parser.add_argument("--face-model", type=Path, default=None)
    parser.add_argument("--side-model", type=Path, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="посчитать и показать, но эталоны не подменять",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        default=(),
        metavar="ИМЯ",
        help=(
            "завести эталон коту, которого модель пока не знает. Работает "
            "только если пространство его различает — проверяйте по отчёту"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="взять готовые кропы прошлого прогона вместо часовой распаковки",
    )
    args = parser.parse_args()
    models = {
        role: path
        for role, path in (("face", args.face_model), ("side", args.side_model))
        if path is not None
    }
    if not models:
        raise SystemExit("не задана ни одна модель опознания")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    run_job(
        args.database_path,
        args.data_dir,
        args.ffmpeg_path,
        args.output_dir,
        detector_model_path=args.detector_model,
        detector_device=args.detector_device,
        identity_models=models,
        apply=not args.dry_run,
        include=args.include,
        dataset_dir=args.dataset,
    )


if __name__ == "__main__":
    main()
