"""Обучение моделей идентификации одной командой: узел → MacBook → узел.

Раньше путь от разметки до имени на экране состоял из десятка команд в трёх
документах: выгрузить набор, разложить split, обучить два ракурса, выгрузить в
ONNX, собрать пакет, перенести на узел, прописать пути. Каждый шаг помнился
по-своему, и переобучение после исправленного имени стоило вечера.

Здесь тот же путь, но один вызов. Активацию модели скрипт по-прежнему не
делает: пути к моделям лежат в `/etc/catdoor/catdoor.env`, который принадлежит
root, а раздавать права ради удобства нельзя — админка работает без
аутентификации. Вместо этого печатается готовая команда.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

VIEWS = ("face", "side")
DEFAULT_ROOT = Path.home() / "CatDoorTraining"
DEFAULT_ADMIN = "http://192.168.2.152:8080"
DEFAULT_NODE = "alex@192.168.2.152"
NODE_MODELS_DIR = "/opt/catdoor/models"

# Порог, ниже которого модель не считается пригодной даже для эксперимента.
# Две недостижимые цифры лучше одной: набор, где кот встречается трижды, даёт
# высокую точность на валидации и не значит ничего.
MIN_EVENTS_PER_CAT = 100
MIN_CATS = 2


class PipelineError(RuntimeError):
    """Ошибка, о которой достаточно сказать словами, без трассировки."""


def run(command: Sequence[str], *, cwd: Path | None = None) -> None:
    print(f"  $ {' '.join(str(part) for part in command)}", flush=True)
    result = subprocess.run(command, cwd=cwd, check=False)
    if result.returncode != 0:
        raise PipelineError(f"команда завершилась с кодом {result.returncode}")


def fetch(root: Path, admin: str) -> Path:
    """Забрать с узла свежий экспорт размеченных проходов."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    incoming = root / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    archive = incoming / f"macbook-{stamp}.zip"
    url = f"{admin.rstrip('/')}/api/training/macbook/export"
    print(f"Забираю набор: {url}")
    try:
        with urllib.request.urlopen(url, timeout=3600) as response:
            archive.write_bytes(response.read())
    except OSError as error:
        raise PipelineError(f"узел не отдал набор: {error}") from error

    dataset = root / "datasets" / "export"
    if dataset.exists():
        shutil.rmtree(dataset)
    dataset.mkdir(parents=True)
    with zipfile.ZipFile(archive) as source:
        source.extractall(dataset)
    print(f"Набор распакован: {dataset} ({archive.stat().st_size // 1024} КБ)")
    return dataset


def manifest_path(dataset: Path) -> Path:
    """Манифест экспорта. Имя разошлось с документацией, поэтому оба варианта.

    Экспорт узла кладёт `dataset.jsonl`, а training/README.md зовёт его
    `manifest.jsonl`. Угадывать не нужно — берём то, что есть.
    """
    for name in ("dataset.jsonl", "manifest.jsonl"):
        candidate = dataset / name
        if candidate.is_file():
            return candidate
    raise PipelineError(f"в наборе нет ни dataset.jsonl, ни manifest.jsonl: {dataset}")


def read_report(dataset: Path) -> dict[str, object]:
    path = dataset / "dataset-report.json"
    if not path.is_file():
        raise PipelineError(f"в наборе нет {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def describe(report: dict[str, object]) -> None:
    crops = report.get("detector_crops", {})
    included = crops.get("included", 0) if isinstance(crops, dict) else 0
    excluded = crops.get("excluded_no_confident_box", 0) if isinstance(crops, dict) else 0
    total = included + excluded
    share = f"{included / total:.0%}" if total else "—"
    print(f"\nКропов: {included} из {total} кадров ({share} с уверенной рамкой кота)")
    print(f"Проходов: {report.get('events')}; по именам: {report.get('events_per_label')}")
    for warning in report.get("warnings", []) or []:
        print(f"  ! {warning}")


def usable_for_production(report: dict[str, object]) -> bool:
    per_label = report.get("events_per_label")
    if not isinstance(per_label, dict):
        return False
    home = {name: count for name, count in per_label.items() if name != "__foreign__"}
    return len(home) >= MIN_CATS and all(count >= MIN_EVENTS_PER_CAT for count in home.values())


def train(root: Path, dataset: Path, *, epochs: int, image_size: int) -> Path:
    """Обучить оба ракурса и собрать пакет модели. Вернуть каталог релиза."""
    work = root / "work"
    work.mkdir(parents=True, exist_ok=True)
    split = work / "manifest.split.jsonl"
    run(
        [
            sys.executable, "-m", "training.dataset",
            str(manifest_path(dataset)),
            "--dataset-root", str(dataset),
            "--assign-splits", str(split),
        ]
    )

    release = root / "release"
    if release.exists():
        shutil.rmtree(release)
    release.mkdir(parents=True)

    trained: list[str] = []
    for view in VIEWS:
        checkpoint = work / f"{view}.pt"
        try:
            run(
                [
                    sys.executable, "-m", "training.train_classifier",
                    str(split), str(dataset), str(checkpoint),
                    "--view", view,
                    "--epochs", str(epochs),
                    "--image-size", str(image_size),
                ]
            )
        except PipelineError:
            # Ракурс без единого кропа — обычное дело на маленьком наборе, и
            # это не повод терять второй. Молчать нельзя: без предупреждения
            # половина модели выглядела бы как целая.
            print(f"  ! ракурс {view} обучить не удалось — пропускаю")
            continue
        run(
            [
                sys.executable, "-m", "training.export_onnx",
                str(checkpoint),
                str(release / f"{view}_model.onnx"),
                "--model-info", str(release / f"{view}-model-info.json"),
            ]
        )
        trained.append(view)

    if not trained:
        raise PipelineError("ни один ракурс не обучился — смотрите отчёт набора выше")

    # Пакет требует ровно одного model_info.json; берём первый обученный ракурс
    # как основной, остальные едут рядом отдельными файлами.
    shutil.copy(release / f"{trained[0]}-model-info.json", release / "model_info.json")
    for view in trained:
        (release / f"{view}-model-info.json").unlink()
    thresholds = Path(__file__).resolve().parents[1] / "model-manifests" / "thresholds.example.json"
    shutil.copy(thresholds, release / "thresholds.json")

    packages = root / "packages"
    packages.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    package = packages / f"catdoor-identity-{stamp}.zip"
    run([sys.executable, "-m", "training.model_package", "build", str(release), str(package)])
    run([sys.executable, "-m", "training.model_package", "verify", str(package)])
    print(f"\nПакет собран: {package}")
    print(f"Обучены ракурсы: {', '.join(trained)}")
    accept(root, package, dataset, split, trained)
    return release


def accept(root: Path, package: Path, dataset: Path, split: Path, views: Sequence[str]) -> bool:
    """Прогнать приёмку на отложенном наборе. Вернуть, прошли ли все ракурсы.

    Смотреть надо именно сюда, а не на validation accuracy. Валидация состоит
    из кадров тех же проходов, что и обучение, и потому льстит: 5 августа она
    показала 0.81 и 0.84 там, где приёмка на других проходах дала 0.57 и 0.30.
    """
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    verdicts: list[bool] = []
    print("\nПриёмка на отложенных проходах:")
    for view in views:
        report_path = reports / f"{view}-report.json"
        # Прошлый отчёт стирается заранее: иначе упавшая приёмка выглядела бы
        # как вчерашняя удачная.
        report_path.unlink(missing_ok=True)
        try:
            run(
                [
                    sys.executable, "-m", "training.acceptance",
                    str(package), str(split), str(dataset),
                    "--model", f"{view}_model.onnx",
                    "--view", view,
                    "--report", str(report_path),
                ]
            )
        except PipelineError:
            # Ненулевой код — это и «не прошла по порогам» (отчёт записан), и
            # настоящая поломка (отчёта нет). Отчёт надёжнее кода возврата.
            if not report_path.is_file():
                print(f"  {view}: приёмка не отработала")
                verdicts.append(False)
                continue
        summary = json.loads(report_path.read_text(encoding="utf-8"))
        metrics = summary.get("metrics", summary)
        passed = bool(metrics.get("passed"))
        verdicts.append(passed)
        far = metrics.get("false_accept_rate")
        far_text = f"{far:.0%}" if isinstance(far, int | float) else "не посчитан"
        print(
            f"  {view}: точность {metrics.get('accuracy', 0):.2f}, "
            f"не распознано {metrics.get('false_reject_rate', 0):.0%}, "
            f"«не уверена» {metrics.get('uncertain_rate', 0):.0%}, "
            f"чужие приняты {far_text} — "
            f"{'прошла' if passed else 'НЕ прошла'}"
        )
        for warning in summary.get("warnings", []) or []:
            print(f"    ! {warning}")
    return all(verdicts) and bool(verdicts)


def acceptance_passed(root: Path, views: Sequence[str]) -> bool:
    """Прошла ли последняя приёмка. Читается из отчётов, а не из памяти."""
    reports = root / "reports"
    found = False
    for view in views:
        path = reports / f"{view}-report.json"
        if not path.is_file():
            continue
        found = True
        summary = json.loads(path.read_text(encoding="utf-8"))
        metrics = summary.get("metrics", summary)
        if not metrics.get("passed"):
            return False
    return found


def install(root: Path, node: str, *, views: Sequence[str]) -> None:
    """Отвезти модели на узел. Прежние остаются рядом с суффиксом .bak."""
    release = root / "release"
    files = [release / f"{view}_model.onnx" for view in views]
    files = [path for path in files if path.is_file()]
    if not files:
        raise PipelineError("в релизе нет ни одной модели")

    print(f"\nОтвожу модели на {node}:{NODE_MODELS_DIR}")
    backup = (
        f"for f in {' '.join(f'{NODE_MODELS_DIR}/{path.name}' for path in files)}; "
        'do [ -f "$f" ] && cp -f "$f" "$f.bak"; done; true'
    )
    run(["ssh", node, backup])
    run(["scp", *[str(path) for path in files], f"{node}:{NODE_MODELS_DIR}/"])
    run(["ssh", node, f"ls -l {NODE_MODELS_DIR}"])

    print("\nОсталось активировать — команда требует root, поэтому выполните её сами:")
    settings = " ".join(
        f"CATDOOR_IDENTITY_{view.upper()}_MODEL_PATH=models/{view}_model.onnx"
        for view in views
        if (release / f"{view}_model.onnx").is_file()
    )
    print(
        f"\n  ssh -t {node} 'sudo sed -i \"/CATDOOR_IDENTITY_/d\" /etc/catdoor/catdoor.env "
        f"&& printf \"%s\\n\" {settings} | sudo tee -a /etc/catdoor/catdoor.env >/dev/null "
        f"&& sudo systemctl restart catdoor-recorder && sleep 5 && "
        f"systemctl is-active catdoor-recorder'\n"
    )
    print("Откат: убрать те же строки из catdoor.env и перезапустить recorder.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("fetch", "train", "install", "all"),
        help="fetch — забрать набор, train — обучить, install — отвезти, all — всё подряд",
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--admin", default=DEFAULT_ADMIN)
    parser.add_argument("--node", default=DEFAULT_NODE)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument(
        "--allow-experimental",
        action="store_true",
        help=(
            "ставить модель, обученную на наборе меньше рекомендованного "
            f"({MIN_CATS} кота по {MIN_EVENTS_PER_CAT} проходов). Такая модель годится "
            "для проверки конвейера, но не для решений"
        ),
    )
    args = parser.parse_args()

    try:
        dataset = args.root / "datasets" / "export"
        if args.stage in ("fetch", "all"):
            dataset = fetch(args.root, args.admin)
        report = read_report(dataset)
        describe(report)

        if args.stage in ("train", "all"):
            train(args.root, dataset, epochs=args.epochs, image_size=args.image_size)

        if args.stage in ("install", "all"):
            if not usable_for_production(report) and not args.allow_experimental:
                raise PipelineError(
                    "набор меньше рекомендованного — модель не установлена. "
                    "Повторите с --allow-experimental, если это проверка конвейера"
                )
            if not acceptance_passed(args.root, VIEWS) and not args.allow_experimental:
                raise PipelineError(
                    "приёмка на отложенных проходах не пройдена — модель не установлена. "
                    "Смотрите отчёты в reports/; --allow-experimental ставит вопреки этому"
                )
            install(args.root, args.node, views=VIEWS)
    except PipelineError as error:
        print(f"\nОстановился: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
