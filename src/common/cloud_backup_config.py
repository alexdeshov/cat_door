"""Настройки облачной копии событий: что, куда и под каким ключом.

Живут в `runtime/cloud-backup.json` с правами 0600 — внутри секретный ключ
бакета, тот же режим, что у паролей камер в инвентаре. В репозиторий и журнал
ключи не попадают; админка отдаёт секрет замаскированным и принимает пустое
поле как «оставить прежний».

Ключ у бакета нарочно только на запись и перечисление (роли IAM:
`storage.uploader` на каталог + `kms.keys.encrypter` на ключ шифрования;
bucket policy не используется — «пустая» политика запрещает всё): украденный с
сервера ключ не умеет ни читать архив, ни стирать его. Схема и две ловушки —
`docs/cloud-backup.md` и разбор в истории сессий 2026-08-14.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from src.common.durable_write import write_text_durably

logger = logging.getLogger("catdoor.cloud-backup")


def cloud_backup_config_path(data_dir: Path) -> Path:
    return data_dir / "runtime" / "cloud-backup.json"


def cloud_backup_state_path(data_dir: Path) -> Path:
    """Состояние выгрузчика — его пишет служба, читает страница «Состояние»."""
    return data_dir / "runtime" / "cloud-backup-state.json"


def cats_cloud_backup_state_path(data_dir: Path) -> Path:
    """Состояние кошачьей копии: своя служба — свой журнал залитого.

    Общий файл на две службы означал бы гонку за перезапись: догон кошачьих
    проходов идёт часами и пишет отметки по ходу, а калиточный цикл
    перезаписывает состояние каждые полминуты.
    """
    return data_dir / "runtime" / "cats-cloud-backup-state.json"


class CloudBackupConfig(BaseModel):
    enabled: bool = False
    bucket: str = ""
    endpoint: str = "https://storage.yandexcloud.net"
    region: str = "ru-central1"
    access_key_id: str = ""
    secret_key: str = ""
    # Префикс ключей в бакете: события лягут как
    # <prefix>/ГГГГ-ММ-ДД/<камера>/<событие>.mp4
    prefix: str = "events"
    # Насколько назад смотреть за незалитыми событиями. Сутки — чтобы дневной
    # простой облака или узла не терял события: выгрузчик догонит.
    lookback_hours: int = Field(default=24, ge=1, le=168)
    # Пустой список — все камеры. Имена — как во Frigate.
    cameras: list[str] = Field(default_factory=list)
    # Сколько дней события живут в облаке. Удаляет НЕ служба (её ключ не умеет
    # — украденный вместе с сервером ключ не должен стирать архив), а lifecycle
    # бакета, который владелец настраивает один раз командой из панели. Число
    # здесь — для этой команды и для сторожа: выгрузчик меряет бакет и говорит
    # вслух, если старейший объект пережил срок — значит, lifecycle не работает.
    retention_days: int = Field(default=30, ge=0, le=3650)
    # Кошачьи проходы. Бакет свой (изоляция квоты: разросшаяся кошачья заливка
    # не должна упереться в лимит и заблокировать калиточную — она страховка от
    # кражи и приоритетом выше), а endpoint, регион и ключи те же: секрет
    # ротируется в одном месте, а не в двух.
    cats_enabled: bool = False
    # Пусто — значит тот же бакет, что у калитки, и разделение только префиксом.
    cats_bucket: str = ""
    cats_prefix: str = "cat-events"
    # Свой срок: кошачьи проходы легче калиточных (около 2 ГБ в сутки против
    # 5.5), поэтому история может быть длиннее при той же цене. Как и у
    # калитки, число здесь — для команды lifecycle и для сторожа, а не для
    # службы: её ключ удалять не умеет.
    cats_retention_days: int = Field(default=90, ge=0, le=3650)

    @property
    def ready(self) -> bool:
        return bool(self.bucket and self.access_key_id and self.secret_key)

    @property
    def cats_target_bucket(self) -> str:
        return self.cats_bucket or self.bucket

    @property
    def cats_ready(self) -> bool:
        return bool(self.cats_target_bucket and self.access_key_id and self.secret_key)


def load_cloud_backup_config(path: Path, defaults: CloudBackupConfig) -> CloudBackupConfig:
    """Настройки с диска; при негодном файле — прежние значения и громкая жалоба."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return defaults
    except (OSError, ValueError) as exc:
        logger.warning("Настройки облачной копии не прочитаны (%s), беру прежние", exc)
        return defaults
    try:
        return CloudBackupConfig.model_validate(payload)
    except ValueError as exc:
        logger.warning(
            "Настройки облачной копии не приняты, работаю на прежних: %s",
            str(exc).replace("\n", " ")[:200],
        )
        return defaults


def save_cloud_backup_config(path: Path, config: CloudBackupConfig) -> None:
    write_text_durably(
        path,
        json.dumps(config.model_dump(), ensure_ascii=False, indent=2),
        mode=0o600,
    )
