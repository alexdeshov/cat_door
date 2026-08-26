from __future__ import annotations

from scripts.gate_audit import Audit, check_openings

# Открытие створки без импульса: 2026-08-21 09:19:09 и 2026-08-23 10:13:04 —
# обычный выход со двора, люди видны на дворовых камерах за секунды до.
OPENED_AT = 1787293149.0


def audit(**extra: object) -> Audit:
    values: dict[str, object] = {
        "since": OPENED_AT - 3600,
        "until": OPENED_AT + 3600,
        "leaf": [(OPENED_AT, "open"), (OPENED_AT + 4, "closed")],
    }
    values.update(extra)
    return Audit(**values)  # type: ignore[arg-type]


def person(camera: str, start: float, end: float) -> dict:
    return {"camera": camera, "label": "person", "start_time": start, "end_time": end}


def test_someone_leaving_the_yard_explains_an_opening() -> None:
    """Выход со двора — не фантом.

    До 2026-08-26 проверка смотрела только камеру калитки, и человек, который
    открыл створку изнутри, оставался невидимым: он попадает в кадр `camera` и
    `nvr_2`, а у калитки появляется уже за створкой или не появляется вовсе.
    Два из трёх «необъяснённых открытий» окна 20–26 августа были ровно этим.
    """
    yard = [person("nvr_2", OPENED_AT - 4, OPENED_AT + 8)]

    assert check_openings(audit(nearby=yard)) == []


def test_a_gate_camera_person_still_explains_an_opening() -> None:
    """Человек у самой калитки объясняет открытие как и раньше."""
    at_gate = [person("test_cam", OPENED_AT - 10, OPENED_AT + 5)]

    assert check_openings(audit(events=at_gate)) == []


def test_an_opening_with_nobody_anywhere_stays_a_finding() -> None:
    """Пустая ночь и открытая створка — это находка.

    2026-08-21 22:11:54 так выглядело мигание классификатора: камера `nvr_2`
    переключилась в ИК, картинка сменилась целиком, и створка «открылась» на
    три секунды, оставаясь закрытой на записи. Проверка обязана такое
    показывать — отличать мигание от настоящего фантома реле она не умеет и не
    должна.
    """
    (finding,) = check_openings(audit())

    assert finding.check == "unexplained-open"
    assert finding.severity == "medium"


def test_a_pulse_explains_an_opening() -> None:
    """Открытие после импульса объясняется импульсом, а не человеком."""
    passes = [{"pulses": 1, "last_pulse_at": OPENED_AT - 6}]

    assert check_openings(audit(passes=passes)) == []


def test_a_person_long_gone_does_not_explain_an_opening() -> None:
    """Человек, ушедший десять минут назад, ничего не объясняет."""
    stale = [person("nvr_2", OPENED_AT - 900, OPENED_AT - 600)]

    (finding,) = check_openings(audit(nearby=stale))

    assert finding.check == "unexplained-open"
