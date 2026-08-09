"""Сравнение двух и более детекторов на одних и тех же кадрах.

Штатная офлайн-оценка гоняет одну модель за запуск, и сравнение двух стоит двух
полных прогонов — а вырезание кадров дороже инференса примерно в десять раз.
Здесь запись декодируется **один раз**, и по одним и тем же кадрам проходят все
модели сразу. Побочно это убирает целый класс сомнений: разница не может
объясняться тем, что моделям достались разные кадры.

Главное же — как сравнивать. Проект уже дважды чуть не сделал неверный вывод,
сравнивая модели **на одном пороге**: 640-е откалиброваны иначе и уверены во всём
подряд сильнее, поэтому на общем пороге выглядят хуже, чем есть. Честное
сравнение — **на равной доле ложных срабатываний**: для каждой модели берётся
самый мягкий порог, укладывающийся в заданную долю ложных, и сравниваются
пропуски при нём. Именно это здесь и считается.

Запускается где угодно, где есть база, записи и OpenVINO, — в том числе на маке,
не занимая узел.
"""

from __future__ import annotations

import argparse
import json
import queue
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.database import create_session_factory
from src.common.detection_config import (
    DetectionConfig,
    detection_config_path,
    load_detection_config,
)
from src.common.evaluation import (
    DetectorViewEvaluation,
    EvaluatedDetectorEvent,
    build_detector_evaluation_report,
    evaluate_detector_frames,
)
from src.common.settings import Settings
from src.inference.cat_detector import OpenVinoCatDetector
from training.evaluation_job import event_query, load_frames

VIEWS = ("face", "side")


@dataclass
class EventRow:
    """Снимок события: после закрытия сессии ORM-объект трогать нельзя."""

    id: int
    label: str
    training_target: str
    created_at: Any
    media: dict[str, str | None]


def load_events(database: Path, labels: list[str] | None, limit: int | None) -> list[EventRow]:
    factory = create_session_factory(database)
    with factory() as session:
        events = session.scalars(event_query({"labels": labels} if labels else {})).all()
        rows = [
            EventRow(
                id=event.id,
                label=event.label,
                training_target=event.training_target,
                created_at=event.created_at,
                media={"face": event.media_path, "side": event.side_media_path},
            )
            for event in events
        ]
    return rows[:limit] if limit else rows


def miss_at_false_rate(
    sweep: list[dict[str, Any]],
    target_false_rate: float,
) -> tuple[float | None, float | None]:
    """Пропуски при самом мягком пороге, укладывающемся в долю ложных.

    Самый мягкий — потому что из порогов, удовлетворяющих ограничению, он даёт
    меньше всего пропусков; именно так модель и настраивали бы вручную.
    Возвращает `(порог, пропуски)`; `(None, None)` — ограничение недостижимо.
    """
    best: tuple[float, float] | None = None
    for point in sweep:
        false_rate = point.get("false_trigger_rate")
        miss = point.get("miss_rate")
        if false_rate is None or miss is None:
            continue
        if false_rate <= target_false_rate and (best is None or point["threshold"] < best[0]):
            best = (float(point["threshold"]), float(miss))
    return best if best is not None else (None, None)


def compare(
    database: Path,
    data_dir: Path,
    models: list[Path],
    *,
    device: str,
    ffmpeg: str,
    labels: list[str] | None,
    limit: int | None,
    workers: int,
    false_rates: list[float],
) -> dict[str, Any]:
    runtime_config = load_detection_config(
        detection_config_path(data_dir),
        DetectionConfig.from_settings(Settings(data_dir=data_dir.resolve())),
    )
    events = load_events(database, labels, limit)
    if not events:
        raise ValueError("фильтр не выбрал ни одного прохода")
    root = data_dir.resolve()

    # По набору детекторов на поток: у скомпилированной модели один запрос на
    # инференс, и делить её между потоками — приглашение к гонке. Модель весит
    # десяток мегабайт, копии дешевле отладки.
    pool: queue.Queue[dict[str, OpenVinoCatDetector]] = queue.Queue()
    for _ in range(workers):
        pool.put({path.name: OpenVinoCatDetector(path, device) for path in models})
    names = [path.name for path in models]
    print(f"моделей: {', '.join(names)} | устройство {device} | потоков {workers}")
    print(f"проходов: {len(events)}")

    results: dict[str, list[EvaluatedDetectorEvent]] = {name: [] for name in names}
    lock = threading.Lock()
    done = 0
    started_at = time.monotonic()
    scratch = data_dir / "runtime" / "compare"

    def process(event: EventRow) -> dict[str, EvaluatedDetectorEvent]:
        nonlocal done
        detectors = pool.get()
        views: dict[str, dict[str, DetectorViewEvaluation]] = {name: {} for name in names}
        try:
            for role in VIEWS:
                relative = event.media[role]
                frames = []
                if relative:
                    source = (root / relative).resolve()
                    if root in source.parents and source.is_file():
                        # Декодируем один раз — дальше кадры идут во все модели.
                        frames = load_frames(ffmpeg, source, scratch / f"{event.id:08d}" / role)
                camera_config = getattr(runtime_config, role)
                for name, detector in detectors.items():
                    views[name][role] = evaluate_detector_frames(frames, detector, camera_config)
                del frames
        finally:
            pool.put(detectors)
        with lock:
            done += 1
            if done % 10 == 0 or done == len(events):
                elapsed = time.monotonic() - started_at
                rate = done / elapsed if elapsed else 0
                left = (len(events) - done) / rate if rate else 0
                print(
                    f"  {done}/{len(events)} проходов, {elapsed:.0f} с, осталось ~{left:.0f} с",
                    flush=True,
                )
        return {
            name: EvaluatedDetectorEvent(
                event_id=event.id,
                label=event.label,
                training_target=event.training_target,
                created_at=event.created_at,
                views=views[name],
            )
            for name in names
        }

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for evaluated in executor.map(process, events):
                for name, item in evaluated.items():
                    results[name].append(item)
    finally:
        if scratch.exists():
            shutil.rmtree(scratch, ignore_errors=True)

    reports = {
        name: build_detector_evaluation_report(
            results[name],
            list(VIEWS),
            {role: getattr(runtime_config, role).confidence_threshold for role in VIEWS},
        )
        for name in names
    }
    comparison = {
        role: {
            f"{rate:.0%}": {
                name: dict(
                    zip(
                        ("threshold", "miss_rate"),
                        miss_at_false_rate(reports[name]["threshold_sweep"][role], rate),
                        strict=True,
                    )
                )
                for name in names
            }
            for rate in false_rates
        }
        for role in VIEWS
    }
    return {
        "models": names,
        "device": device,
        "events": len(events),
        "reports": reports,
        "equal_false_rate": comparison,
    }


def print_summary(payload: dict[str, Any]) -> None:
    names = payload["models"]
    print("\n=== На равной доле ложных срабатываний ===")
    print("Сравнивать модели на одном пороге нельзя: откалиброваны они по-разному.")
    for role, by_rate in payload["equal_false_rate"].items():
        title = "дверца" if role == "face" else "подход"
        print(f"\n{title}:")
        header = "  доля ложных " + "".join(f"{name:>26}" for name in names)
        print(header)
        for rate, models in by_rate.items():
            cells = ""
            for name in names:
                point = models[name]
                if point["miss_rate"] is None:
                    cells += f"{'недостижимо':>26}"
                else:
                    cells += f"{point['miss_rate']:>18.0%} при {point['threshold']:.2f}"
            print(f"  {rate:>12} {cells}")
    print("\n=== Ночью ===")
    for name in names:
        night = payload["reports"][name]["sections"]["night"]
        parts = []
        for role in VIEWS:
            metrics = night.get(role) or {}
            miss = metrics.get("miss_rate")
            parts.append(f"{role}: {miss:.0%}" if miss is not None else f"{role}: —")
        print(f"  {name}: " + ", ".join(parts))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Сравнить детекторы на одних и тех же кадрах, на равной доле ложных"
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, action="append", dest="models", required=True)
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--label", action="append", dest="labels")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--false-rate",
        type=float,
        action="append",
        dest="false_rates",
        help="доли ложных, на которых сравнивать (по умолчанию 5, 10 и 20 процентов)",
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    payload = compare(
        arguments.database,
        arguments.data_dir,
        arguments.models,
        device=arguments.device,
        ffmpeg=arguments.ffmpeg,
        labels=arguments.labels,
        limit=arguments.limit,
        workers=max(1, arguments.workers),
        false_rates=arguments.false_rates or [0.05, 0.10, 0.20],
    )
    print_summary(payload)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\nотчёт: {arguments.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
