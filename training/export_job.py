"""Сборка набора для макбука — отдельным процессом.

Раньше архив собирался прямо в обработчике запроса. На игрушечных наборах это
работало, а на живом развалилось: 11 августа узел резал кадры больше часа,
веб-процесс висел на 145% процессора, клиент отвалился по таймауту — и обучение
пришлось вести на наборе недельной давности.

Дело не в медленном железе, а в том, что часовую работу нельзя делать внутри
запроса: HTTP не умеет ждать час, а веб-служба в это время не отвечает ни на
что другое.

Поэтому здесь то же устройство, что у пересчёта эталонов: отдельный процесс,
файл состояния, heartbeat. Админка спрашивает «готово?», а забирает готовый
архив обычной отдачей файла — быстро и без сборки.
"""

from __future__ import annotations

import argparse
import logging
import traceback
from pathlib import Path

from src.common.detection_config import DetectionConfig
from src.common.settings import Settings
from training.workflow import StatusHeartbeat, create_macbook_export, write_status

logger = logging.getLogger("catdoor.export-job")


def run_job(
    database_path: Path,
    data_dir: Path,
    ffmpeg_path: str,
    output_dir: Path,
    detector_model_path: Path,
    detector_device: str,
    detection_config: DetectionConfig,
    scope: str,
) -> Path | None:
    """Собрать архив и записать в состояние путь к нему.

    Путь пишется именно в состояние, а не выводится в лог: админке нужно уметь
    отдать архив после собственного перезапуска, а лог она не читает.
    """
    status_path = output_dir / "status.json"
    heartbeat = StatusHeartbeat(status_path)
    heartbeat.start()
    try:
        write_status(
            status_path,
            state="preparing",
            message="Готовлю кадры размеченных проходов",
        )
        archive = create_macbook_export(
            database_path,
            data_dir,
            ffmpeg_path,
            detector_model_path,
            detector_device,
            detection_config,
            scope=scope,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        logger.exception("Сборка набора не удалась")
        write_status(
            status_path,
            state="failed",
            message=f"Сборка набора не удалась: {exc}",
            traceback=traceback.format_exc(),
        )
        return None
    finally:
        heartbeat.stop()
    size_mb = round(archive.stat().st_size / (1024 * 1024), 1)
    write_status(
        status_path,
        state="completed",
        message=f"Набор собран: {archive.name}, {size_mb} МБ",
        archive=str(archive),
        size_mb=size_mb,
    )
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database_path", type=Path)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("ffmpeg_path")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--detector-model", type=Path, required=True)
    parser.add_argument("--detector-device", default="CPU")
    parser.add_argument("--scope", default="macbook", choices=("macbook", "all"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    # Пороги детекции берутся из окружения службы, а не из аргументов: процесс
    # запускается админкой и наследует её окружение, а два источника одних и тех
    # же порогов однажды разойдутся.
    run_job(
        args.database_path,
        args.data_dir,
        args.ffmpeg_path,
        args.output_dir,
        args.detector_model,
        args.detector_device,
        DetectionConfig.from_settings(Settings()),
        args.scope,
    )


if __name__ == "__main__":
    main()
