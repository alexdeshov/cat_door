"""Сторож здоровья: дребезг молчит, беда говорит один раз, выздоровление слышно.

Машина состояний — чистая функция, поэтому вся спам-защита проверяется здесь
таблицей случаев, а не ночёвкой у калитки с выключателем света.
"""

from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

from src.gate.gate_state import GateStateWatcher
from src.monitor.healthwatch import (
    CONFIRM_CYCLES,
    REMIND_SECONDS,
    Trouble,
    boot_message,
    health_troubles,
    reconcile,
)

MOSCOW = ZoneInfo("Europe/Moscow")
NOON = 1_787_200_000.0


def _cycle(observed, state, now):  # type: ignore[no-untyped-def]
    return reconcile(observed, state, now=now, timezone=MOSCOW)


def test_a_new_trouble_is_confirmed_before_it_wakes_anyone() -> None:
    trouble = [Trouble("check:x", "Камера мертва")]

    first, state = _cycle(trouble, {}, NOON)
    second, state = _cycle(trouble, state, NOON + 60)

    assert first == []  # один цикл — ещё не беда
    assert second == ["🔴 Камера мертва"]
    assert "check:x" in state["problems"]


def test_a_flicker_never_reaches_the_owner() -> None:
    """Мигание на границе порога — не повод будить среди ночи."""
    lines1, state = _cycle([Trouble("check:x", "На грани")], {}, NOON)
    lines2, state = _cycle([], state, NOON + 60)
    lines3, state = _cycle([Trouble("check:x", "На грани")], state, NOON + 120)
    lines4, state = _cycle([], state, NOON + 180)

    assert lines1 == lines2 == lines3 == lines4 == []
    assert state["problems"] == {}


def test_a_lasting_trouble_reminds_not_spams() -> None:
    trouble = [Trouble("check:x", "Камера мертва")]
    state: dict = {}
    for tick in range(CONFIRM_CYCLES):
        _, state = _cycle(trouble, state, NOON + tick * 60)

    quiet, state = _cycle(trouble, state, NOON + 600)
    reminded, state = _cycle(trouble, state, NOON + 600 + REMIND_SECONDS)

    assert quiet == []  # между напоминаниями — тишина
    assert len(reminded) == 1 and reminded[0].startswith("⚠️ Всё ещё: Камера мертва")


def test_recovery_is_confirmed_and_reported() -> None:
    trouble = [Trouble("check:x", "Камера мертва")]
    state: dict = {}
    for tick in range(CONFIRM_CYCLES):
        _, state = _cycle(trouble, state, NOON + tick * 60)

    first_clean, state = _cycle([], state, NOON + 200)
    second_clean, state = _cycle([], state, NOON + 260)

    assert first_clean == []  # один чистый цикл выздоровления не доказывает
    assert len(second_clean) == 1 and second_clean[0].startswith("✅ Прошло: Камера мертва")
    assert state["problems"] == {}


def test_health_checks_become_named_troubles() -> None:
    """Владельцу в чат нужно имя камеры и суть, а не «узел нездоров»."""
    payload = {
        "groups": [
            {
                "title": "Контур калитки",
                "checks": [
                    {"key": "gate_alive_test_cam", "title": "Кадры идут, test_cam",
                     "status": "fail", "actual": "0.0 к/с"},
                    {"key": "gate_skipped_test_cam", "title": "Пропуск кадров",
                     "status": "ok", "actual": "0"},
                    {"key": "misc", "title": "Что-то не собралось",
                     "status": "unknown", "actual": None},
                ],
            },
            {
                "title": "Железо",
                "checks": [
                    {"key": "cpu_temp", "title": "Температура CPU",
                     "status": "warn", "actual": "88 °C"},
                ],
            },
        ]
    }

    troubles = health_troubles(payload)

    assert [item.id for item in troubles] == ["check:gate_alive_test_cam", "check:cpu_temp"]
    assert troubles[0].title == "Контур калитки: Кадры идут, test_cam — 0.0 к/с"


def test_a_silent_admin_is_a_trouble_itself() -> None:
    assert [item.id for item in health_troubles(None)] == ["health-api"]


def test_the_boot_message_fires_once_per_reboot(monkeypatch) -> None:
    """Первая запись — установка сторожа, а не падение узла: молчим.

    «Один раз» означает «один раз доставленное»: заметка снимается циклом
    после удачной отправки, и только тогда наступает тишина.
    """
    monkeypatch.setattr("src.monitor.healthwatch.boot_id", lambda: "boot-1")
    monkeypatch.setattr("src.monitor.healthwatch.uptime_seconds", lambda: 90.0)

    first, state = boot_message({}, timezone=MOSCOW)
    again, state = boot_message(state, timezone=MOSCOW)
    monkeypatch.setattr("src.monitor.healthwatch.boot_id", lambda: "boot-2")
    rebooted, state = boot_message(state, timezone=MOSCOW)
    state.pop("boot_note", None)  # цикл доставил сообщение
    calm, state = boot_message(state, timezone=MOSCOW)

    assert first is None and again is None
    assert rebooted is not None and rebooted.startswith("🔄 Узел перезагружался")
    assert calm is None


def test_the_gate_service_writes_its_pulse(runtime_path: Path) -> None:
    """Файл здоровья службы: слепоту по створке видно снаружи."""
    from src.gate.access import gate_health_path, write_gate_health

    watcher = GateStateWatcher("")  # брокер не настроен — подписки нет
    target = gate_health_path(runtime_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    write_gate_health(target, watcher, now=NOON)

    import json

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["broker_connected"] is False
    assert payload["leaf"]["value"] == "unknown"
    assert payload["updated_at"] == NOON


def test_the_reference_notices_a_dead_camera() -> None:
    """2026-08-21: test_cam с нулём кадров выглядела «ok» — пропускать нечего."""
    from src.common.health import gate_checks

    stats = {"cameras": {"test_cam": {"camera_fps": 0.0, "skipped_fps": 0.0}}}

    group = gate_checks(stats, "GPU")

    alive = next(check for check in group.checks if check.key == "gate_alive_test_cam")
    assert alive.status == "fail"


def test_the_reference_reads_the_gate_service_pulse() -> None:
    from src.common.health import gate_service_checks

    stale = gate_service_checks({"updated_at": NOON - 300, "broker_connected": True}, now=NOON)
    deaf = gate_service_checks({"updated_at": NOON - 1, "broker_connected": False}, now=NOON)
    fine = gate_service_checks({"updated_at": NOON - 1, "broker_connected": True}, now=NOON)
    absent = gate_service_checks(None, now=NOON)

    assert stale.status == "fail"  # служба молчит полминуты — стоит
    assert deaf.status == "fail"  # брокер потерян — правило стороны разоружено
    assert fine.status == "ok"
    assert absent.status == "unknown"


def test_latent_bombs_are_found_before_the_reboot(monkeypatch) -> None:
    """Мину, стреляющую только при ребуте, надо находить, пока узел жив.

    2026-08-21: .bak в /etc/dnsmasq.d и disabled-юнит frigate — обе беды
    молчали до перезагрузки. `dnsmasq --test` и `is-enabled` находят их сразу.
    """
    import subprocess as sp
    from types import SimpleNamespace

    from src.monitor.healthwatch import latent_troubles

    def fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
        if args[0].endswith("dnsmasq"):
            return SimpleNamespace(
                returncode=1, stdout="",
                stderr="dnsmasq: duplicate dhcp-host IP address 192.168.1.170\n",
            )
        unit = args[-1]
        status = "disabled" if unit == "frigate.service" else "enabled"
        return SimpleNamespace(returncode=0, stdout=status + "\n", stderr="")

    monkeypatch.setattr(sp, "run", fake_run)

    troubles = latent_troubles(("catdoor-web", "frigate"))

    ids = [item.id for item in troubles]
    assert ids == ["latent:dnsmasq-config", "latent:enable-frigate"]
    assert "не переживёт перезагрузку" in troubles[0].title
    assert "duplicate dhcp-host" in troubles[0].title
    assert "не в автозагрузке" in troubles[1].title


def test_an_undelivered_boot_note_is_retried_next_cycle(monkeypatch) -> None:
    """«🔄» не имеет права теряться: сторож просыпается раньше сети.

    Учебный ребут 2026-08-21: DNS ещё не поднялся, первая отправка упала — и
    сообщение о перезагрузке пропало бы навсегда. Заметка живёт в состоянии,
    пока цикл не доставит её; доставка снимает заметку.
    """
    monkeypatch.setattr("src.monitor.healthwatch.boot_id", lambda: "boot-1")
    monkeypatch.setattr("src.monitor.healthwatch.uptime_seconds", lambda: 60.0)

    _, state = boot_message({}, timezone=MOSCOW)
    monkeypatch.setattr("src.monitor.healthwatch.boot_id", lambda: "boot-2")
    first, state = boot_message(state, timezone=MOSCOW)
    # Отправка не удалась — заметка осталась; следующий цикл говорит её снова.
    retried, state = boot_message(state, timezone=MOSCOW)
    # Отправка удалась — цикл снимает заметку, дальше тишина.
    state.pop("boot_note", None)
    calm, state = boot_message(state, timezone=MOSCOW)

    assert first is not None and first.startswith("🔄")
    assert retried == first
    assert calm is None
