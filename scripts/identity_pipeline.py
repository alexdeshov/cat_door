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
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from src.common.metrics import FOREIGN_LABEL, REJECT_LABELS

VIEWS = ("face", "side")
DEFAULT_ROOT = Path.home() / "CatDoorTraining"
DEFAULT_ADMIN = "http://192.168.2.137:8080"
DEFAULT_NODE = "alex@192.168.2.137"
NODE_MODELS_DIR = "/opt/catdoor/models"

# Порог, ниже которого модель не считается пригодной даже для эксперимента.
# Две недостижимые цифры лучше одной: набор, где кот встречается трижды, даёт
# высокую точность на валидации и не значит ничего.
MIN_EVENTS_PER_CAT = 100
MIN_CATS = 2


ACTIVE_EXPORT_STATES = {"starting", "preparing", "training"}
# Два часа: живой набор режется около часа, и запас нужен на рост разметки.
# Сборка набора на узле режет ролики по кадру в секунду и прогоняет детектор
# на процессоре: 650 проходов 07.09 не уложились в два часа, и конвейер ушёл
# по таймауту, хотя сборка продолжалась. Четырёх часов хватает с запасом.
EXPORT_WAIT_SECONDS = 4 * 3600.0
EXPORT_POLL_SECONDS = 30.0
# Сколько готовый набор считается свежим. Шесть часов — это «собран сегодня и
# с тех пор разметка почти не менялась»; за большее время владелец успевает
# разметить заметную порцию проходов, и брать старый набор значило бы молча
# выбросить его работу.
EXPORT_REUSE_HOURS = 6.0


class PipelineError(RuntimeError):
    """Ошибка, о которой достаточно сказать словами, без трассировки."""


def run(command: Sequence[str], *, cwd: Path | None = None) -> None:
    print(f"  $ {' '.join(str(part) for part in command)}", flush=True)
    result = subprocess.run(command, cwd=cwd, check=False)
    if result.returncode != 0:
        raise PipelineError(f"команда завершилась с кодом {result.returncode}")


def _ask(url: str, *, method: str = "GET", timeout: float = 30.0) -> dict[str, object]:
    request = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:200]
        raise PipelineError(f"узел ответил {error.code}: {detail}") from error
    except OSError as error:
        raise PipelineError(f"узел недоступен: {error}") from error
    return payload if isinstance(payload, dict) else {}


def fetch(root: Path, admin: str, scope: str = "macbook", *, rebuild: bool = False) -> Path:
    """Забрать с узла свежий экспорт размеченных проходов.

    Сборка идёт на узле отдельным процессом, а здесь мы её запускаем и ждём.
    Прежний способ — «запросить файл и ждать, пока соберётся» — 11 августа
    отвалился по таймауту через час, и обучение пошло на недельном наборе.
    Ожидание опросом переживает и час, и разрыв связи: состояние лежит на узле.
    """
    base = admin.rstrip("/")
    status = _ask(f"{base}/api/training/macbook/export/status")
    state = status.get("state")
    ready = status.get("ready")
    if not rebuild and _is_fresh_export(ready):
        # Готовый набор уже есть — берём его, даже если рядом идёт сборка.
        # Ждать её незачем: она делает следующий набор, а этот лежит целиком.
        #
        # 13 августа конвейер сперва запустил вторую сборку поверх законченной,
        # а потом полтора часа её ждал — и то и другое ради того же набора.
        name = ready.get("archive", "") if isinstance(ready, dict) else ""
        print(f"Беру готовый набор: {name}")
        return _download(root, base)
    if state in ACTIVE_EXPORT_STATES:
        print("Сборка уже идёт на узле — жду её")
    else:
        started = _ask(f"{base}/api/training/macbook/export?scope={scope}", method="POST")
        print(f"Сборка набора запущена на узле: {started.get('run_id', '?')}")
    deadline = time.monotonic() + EXPORT_WAIT_SECONDS
    while True:
        status = _ask(f"{base}/api/training/macbook/export/status")
        state = status.get("state")
        if state == "completed":
            print(f"  {status.get('message', 'готово')}")
            break
        if state in {"failed", "idle"}:
            raise PipelineError(f"сборка набора не удалась: {status.get('message', state)}")
        if time.monotonic() > deadline:
            raise PipelineError("сборка набора не закончилась за четыре часа")
        print(f"  {status.get('message', state)} …", flush=True)
        time.sleep(EXPORT_POLL_SECONDS)

    return _download(root, base)


def _is_fresh_export(ready: object) -> bool:
    """Годится ли уже собранный набор.

    Разметка пополняется каждый день, и вчерашний набор — это вчерашняя работа
    владельца, выброшенная молча. Поэтому у готового набора есть срок годности,
    а не просто «есть или нет».
    """
    if not isinstance(ready, dict):
        return False
    raw = ready.get("finished_at")
    if not isinstance(raw, str):
        return False
    try:
        finished = datetime.fromisoformat(raw)
    except ValueError:
        return False
    if finished.tzinfo is None:
        finished = finished.replace(tzinfo=UTC)
    age_hours = (datetime.now(UTC) - finished).total_seconds() / 3600
    return 0 <= age_hours <= EXPORT_REUSE_HOURS


def _download(root: Path, base: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    incoming = root / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    archive = incoming / f"macbook-{stamp}.zip"
    url = f"{base}/api/training/macbook/export"
    print(f"Забираю набор: {url}")
    try:
        with urllib.request.urlopen(url, timeout=1800) as response:
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
    crowded = crops.get("excluded_two_cats", 0) if isinstance(crops, dict) else 0
    print(f"\nКропов: {included} из {total} кадров ({share} с уверенной рамкой кота)")
    if crowded:
        # Отдельной строкой, а не в общей куче отброшенных: кадр без кота —
        # шум съёмки, а кадр с двумя котами — повод пересмотреть проход.
        print(
            f"Кадров с двумя котами отброшено: {crowded} — "
            "имя у прохода одно, а котов в кадре несколько"
        )
    print(f"Проходов: {report.get('events')}; по именам: {report.get('events_per_label')}")
    for warning in report.get("warnings", []) or []:
        print(f"  ! {warning}")


def usable_for_production(report: dict[str, object]) -> bool:
    per_label = report.get("events_per_label")
    if not isinstance(per_label, dict):
        return False
    home = {
        name: count
        for name, count in per_label.items()
        if name not in REJECT_LABELS
    }
    return len(home) >= MIN_CATS and all(count >= MIN_EVENTS_PER_CAT for count in home.values())


def split_cats(cats: Sequence[str]) -> set[str]:
    """Имена котов из аргументов, хоть через пробел, хоть через запятую.

    `--cats Черныш Яся` и `--cats Черныш,Яся` человек пишет с одинаковой
    уверенностью, и разбираться, какая форма верна, ему незачем: 13 августа
    запятая стоила прогона, который отказал со словами «в наборе нет проходов
    котов: Черныш,Яся» — рядом с отчётом, где Черныша 142 прохода.
    """
    names = set()
    for value in cats:
        names.update(part.strip() for part in value.split(",") if part.strip())
    return names


def restrict_to_cats(source: Path, cats: Sequence[str], destination: Path) -> None:
    """Оставить классами только целевых котов, остальных — в отрицательные.

    Прочие именованные коты переезжают в `__foreign__`: это лучшие
    отрицательные примеры из возможных — настоящие коты не того имени.
    Первый прогон на шести классах, где у четырёх было по 10–18 проходов,
    дал 69% принятых чужих: крошечные классы не учат, а размывают.
    """
    targets = split_cats(cats)
    present = {
        json.loads(line)["label"]
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    unknown = targets - present
    if unknown:
        # Имена набора — в тексте отказа. Без них человек видит «нет проходов
        # котов: Черныш» рядом с отчётом, где Черныша 142 прохода, и ищет
        # ошибку не там: 13 августа причиной была запятая вместо пробела.
        known = sorted(present - REJECT_LABELS - {FOREIGN_LABEL})
        raise PipelineError(
            f"в наборе нет проходов котов: {', '.join(sorted(unknown))}. "
            f"Есть: {', '.join(known) if known else 'ни одного имени'}"
        )
    lines = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item["label"] not in targets and item["label"] not in REJECT_LABELS:
            item["label"] = FOREIGN_LABEL
        lines.append(json.dumps(item, ensure_ascii=False))
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train(
    root: Path,
    dataset: Path,
    *,
    epochs: int,
    image_size: int,
    views: Sequence[str],
    cats: Sequence[str] = (),
    train_foreign: bool = False,
    head: str = "embedding",
) -> Path:
    """Обучить выбранные ракурсы и собрать пакет. Вернуть каталог релиза."""
    work = root / "work"
    work.mkdir(parents=True, exist_ok=True)
    manifest = manifest_path(dataset)
    if cats:
        restricted = work / "manifest.cats.jsonl"
        restrict_to_cats(manifest, cats, restricted)
        manifest = restricted
    split = work / "manifest.split.jsonl"
    run(
        [
            sys.executable, "-m", "training.dataset",
            str(manifest),
            "--dataset-root", str(dataset),
            "--assign-splits", str(split),
        ]
    )

    release = root / "release"
    if release.exists():
        shutil.rmtree(release)
    release.mkdir(parents=True)

    trained: list[str] = []
    for view in views:
        checkpoint = work / f"{view}.pt"
        # У эмбеддингов чужие в обучении всегда: они не «ещё один кот», а сила,
        # растаскивающая пространство. Флаг `--train-foreign` остаётся про
        # классификатор, где чужой — отдельный класс со своей ценой.
        trainer = (
            ["-m", "training.train_embeddings"]
            if head == "embedding"
            else ["-m", "training.train_classifier", *(["--train-foreign"] if train_foreign else [])]
        )
        try:
            run(
                [
                    sys.executable, *trainer[:2],
                    str(split), str(dataset), str(checkpoint),
                    "--view", view,
                    "--epochs", str(epochs),
                    "--image-size", str(image_size),
                    *trainer[2:],
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
    # как основной. Пер-ракурсные метаданные НЕ удаляются, а переименовываются
    # в `<ракурс>_model.json` — ровно так их ищет рекордер рядом с моделью
    # (`model_path.with_suffix(".json")`). Первая боевая установка 2026-08-10
    # уехала без них, и recorder не смог бы поднять ни одну модель.
    shutil.copy(release / f"{trained[0]}-model-info.json", release / "model_info.json")
    for view in trained:
        (release / f"{view}-model-info.json").rename(release / f"{view}_model.json")
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
    files: list[Path] = []
    for view in views:
        model = release / f"{view}_model.onnx"
        metadata = release / f"{view}_model.json"
        if not model.is_file():
            continue
        if not metadata.is_file():
            raise PipelineError(
                f"у модели {model.name} нет метаданных {metadata.name} — "
                "рекордер без них её не поднимет"
            )
        files.extend([model, metadata])
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
    parser.add_argument(
        "--head",
        choices=("embedding", "softmax"),
        default="embedding",
        help=(
            "чем опознавать: embedding — эталон кота и косинусная близость "
            "(нового кота можно добавить без переобучения, «чужой» получается "
            "порогом); softmax — прежний классификатор с классом «чужой»"
        ),
    )
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument(
        "--views",
        nargs="+",
        choices=VIEWS,
        default=list(VIEWS),
        metavar="РАКУРС",
        help=(
            "какие ракурсы обучать, принимать и ставить (по умолчанию оба). "
            "Ракурсы независимы: у камеры подхода кот дальше, мельче и в "
            "пятнистом свете, и её модель может не годиться, когда модель "
            "дверцы уже хороша. Без этого выбора негодный ракурс валил приёмку "
            "целиком и не давал поставить годный"
        ),
    )
    parser.add_argument(
        "--train-foreign",
        action="store_true",
        help=(
            "обучать класс отказа «чужой» явно; включать вместе с --cats, "
            "когда пул не-целевых котов богат"
        ),
    )
    parser.add_argument(
        "--rebuild-export",
        action="store_true",
        help=(
            f"собрать набор заново, даже если готовый моложе "
            f"{EXPORT_REUSE_HOURS:.0f} ч. Нужно, когда разметку правили "
            "только что и она обязана попасть в обучение"
        ),
    )
    parser.add_argument(
        "--cats",
        nargs="+",
        default=(),
        metavar="ИМЯ",
        help=(
            "обучать только этих котов; остальные именованные переезжают в "
            "отрицательные примеры. Без флага обучаются все имена набора"
        ),
    )
    parser.add_argument(
        "--scope",
        choices=("macbook", "all"),
        default="macbook",
        help=(
            "какие проходы забирать: macbook — только помеченные точкой "
            "обучения «макбук» (по умолчанию), all — все размеченные без "
            "разбора точки. Разметка по умолчанию идёт в «не для обучения», "
            "поэтому «all» обычно в разы больше"
        ),
    )
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
            dataset = fetch(args.root, args.admin, args.scope, rebuild=args.rebuild_export)
        report = read_report(dataset)
        describe(report)

        if args.stage in ("train", "all"):
            train(
                args.root,
                dataset,
                epochs=args.epochs,
                image_size=args.image_size,
                head=args.head,
                views=args.views,
                cats=args.cats,
                train_foreign=args.train_foreign,
            )

        if args.stage in ("install", "all"):
            if not usable_for_production(report) and not args.allow_experimental:
                raise PipelineError(
                    "набор меньше рекомендованного — модель не установлена. "
                    "Повторите с --allow-experimental, если это проверка конвейера"
                )
            if not acceptance_passed(args.root, args.views) and not args.allow_experimental:
                raise PipelineError(
                    "приёмка на отложенных проходах не пройдена — модель не установлена. "
                    "Смотрите отчёты в reports/; --allow-experimental ставит вопреки этому"
                )
            install(args.root, args.node, views=args.views)
    except PipelineError as error:
        print(f"\nОстановился: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
