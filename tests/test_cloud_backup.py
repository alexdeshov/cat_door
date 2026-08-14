"""Облачная копия событий: настройки, отбор, маскировка секрета."""

from __future__ import annotations

from pathlib import Path

from src.cloud_backup.uploader import eligible_events
from src.common.cloud_backup_config import (
    CloudBackupConfig,
    load_cloud_backup_config,
    save_cloud_backup_config,
)


def test_settings_round_trip_and_secret_stays_on_disk(tmp_path: Path) -> None:
    path = tmp_path / "cloud-backup.json"
    config = CloudBackupConfig(
        enabled=True, bucket="catdoor-events", access_key_id="AKID1234", secret_key="s3cr3t"
    )

    save_cloud_backup_config(path, config)
    restored = load_cloud_backup_config(path, CloudBackupConfig())

    assert restored == config
    assert oct(path.stat().st_mode)[-3:] == "600"  # внутри секрет


def test_eligible_events_take_finished_and_skip_uploaded() -> None:
    """Незавершённое событие пропускается БЕЗ пометки: клип ещё растёт.

    Пометить его «сделанным» значило бы никогда не залить целое.
    """
    config = CloudBackupConfig(cameras=["gate"])
    events = [
        {"id": "a", "camera": "gate", "end_time": 2.0, "start_time": 1.0},
        {"id": "b", "camera": "gate", "end_time": None, "start_time": 1.0},
        {"id": "c", "camera": "terrace", "end_time": 2.0, "start_time": 1.0},
        {"id": "a2", "camera": "gate", "end_time": 2.0, "start_time": 1.0},
        {"id": "d", "camera": "gate", "end_time": 2.0, "start_time": 1.0,
         "has_clip": False},
    ]

    selected = eligible_events(events, config, uploaded={"a2"})

    ids = [event["id"] for event in selected]
    assert ids == ["a", "d"]
    # Событие без записи заливать нечего — оно помечается пропуском.
    assert selected[1].get("_skip") is True


def test_the_admin_masks_the_secret(runtime_path: Path, monkeypatch) -> None:
    """Секрет наружу не выходит; пустые поля означают «оставить прежние»."""
    from tests.test_admin import make_client

    client = make_client(runtime_path, monkeypatch)

    saved = client.put(
        "/api/cloud-backup/settings",
        json={"enabled": True, "bucket": "b1", "access_key_id": "AKIDXYZW",
              "secret_key": "verysecret"},
    ).json()
    kept = client.put(
        "/api/cloud-backup/settings",
        json={"enabled": True, "bucket": "b2"},
    ).json()
    page = client.get("/health")

    assert "verysecret" not in str(saved) and "verysecret" not in page.text
    assert saved["has_secret"] is True
    assert saved["access_key_hint"] == "AKID…"
    # Пустые ключи во втором запросе не стёрли сохранённые.
    assert kept["bucket"] == "b2" and kept["has_secret"] is True
    assert "Облачная копия событий" in page.text
