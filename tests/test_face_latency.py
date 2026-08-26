from __future__ import annotations

from scripts.face_latency import (
    Attempt,
    measure,
    parse_attempt,
    render,
    size_report,
    summarise,
)

# Реальный проход 2026-08-25 21:12: человек шёл к калитке в темноте, лицо
# нашлось шесть раз, узналось дважды — и второе совпадение пришло через 9.6
# секунды после первого. Импульс ушёл через 0.4 с после кворума: наша часть
# ожидания — доли секунды, всё остальное собрал Frigate.
NIGHT_EVENT = "1787681518.600044-hjeiu2"
NIGHT_FILES = [
    f"{NIGHT_EVENT}-1787681520.973847-unknown-0.53.webp",
    f"{NIGHT_EVENT}-1787681521.330763-unknown-0.44.webp",
    f"{NIGHT_EVENT}-1787681521.932986-Леша-0.98.webp",
    f"{NIGHT_EVENT}-1787681525.384474-unknown-0.49.webp",
    f"{NIGHT_EVENT}-1787681531.524098-Леша-0.93.webp",
]
NIGHT_PASS = {
    "frigate_event_id": NIGHT_EVENT,
    "person_name": "Леша",
    "last_pulse_at": 1787681531.8874168,
}


def attempts(files: list[str]) -> list[Attempt]:
    parsed = [parse_attempt(name) for name in files]
    return [item for item in parsed if item is not None]


def test_an_attempt_is_read_out_of_its_file_name() -> None:
    """Имя файла — единственное место, где живёт попытка распознавания.

    Журнал знает только «имя появилось», и по нему нельзя отличить «лицо не
    нашлось» от «нашлось, но не узналось».
    """
    attempt = parse_attempt(f"{NIGHT_EVENT}-1787681521.932986-Леша-0.98.webp")

    assert attempt is not None
    assert attempt.event_id == NIGHT_EVENT
    assert attempt.at == 1787681521.932986
    assert attempt.name == "Леша"
    assert attempt.score == 0.98
    assert attempt.matched


def test_an_unrecognised_attempt_is_still_an_attempt() -> None:
    """`unknown` — это найденное лицо, которое не узнали.

    Считать такие попытки обязательно: их доля и есть качество кадра.
    """
    attempt = parse_attempt(f"{NIGHT_EVENT}-1787681520.973847-unknown-0.53.webp")

    assert attempt is not None
    assert not attempt.matched
    assert attempt.score == 0.53


def test_reference_images_are_not_attempts() -> None:
    """Рядом лежит библиотека эталонов — `Леша_1785936800.690284.webp`.

    Её файлы попыток не описывают, и принять их за попытки значит посчитать
    узнавания, которых не было.
    """
    assert parse_attempt("Леша_1785936800.690284.webp") is None
    assert parse_attempt("thumbnail.jpg") is None


def test_the_quorum_is_the_second_match_not_the_first() -> None:
    """Имя Frigate даёт по кворуму, и ждать приходится второго совпадения.

    Ровно эта разница и стоила человеку 9.6 секунды у закрытой калитки.
    """
    (row,) = measure(attempts(NIGHT_FILES), [NIGHT_PASS])

    first = row.first_match
    quorum = row.quorum(2)

    assert first is not None and quorum is not None
    assert round(first.at - row.started, 1) == 3.3
    assert round(quorum.at - row.started, 1) == 12.9
    assert round(row.pulse_at - quorum.at, 1) == 0.4, "наша доля ожидания — доли секунды"


def test_two_different_names_are_not_a_quorum() -> None:
    """Два кадра, узнавших разных людей, не кворум ни для кого.

    На том же ночном треке нашёлся кадр «Алиса 0.95»: одиночное совпадение выше
    порога умеет назвать не того человека, и складывать такие кадры между собой
    нельзя — список разрешённых имён открывает калитку каждому из них.
    """
    files = [
        f"{NIGHT_EVENT}-1787681521.932986-Леша-0.98.webp",
        f"{NIGHT_EVENT}-1787681531.794338-Алиса-0.95.webp",
    ]

    (row,) = measure(attempts(files), [])

    assert row.first_match is not None
    assert row.quorum(2) is None


def test_an_event_without_a_pulse_is_still_measured() -> None:
    """Отказ по кулдауну или по правилу стороны не отменяет замера.

    Frigate узнаёт лицо независимо от того, дали ли импульс, и половина
    измерений живёт именно в таких событиях.
    """
    (row,) = measure(attempts(NIGHT_FILES), [])

    assert row.pulse_at is None
    assert row.quorum(2) is not None
    assert row.as_dict(2)["our_lag"] is None


def test_manual_openings_do_not_pretend_to_be_events() -> None:
    """У ручного открытия идентификатор `manual-<uuid>` — начала события нет."""
    handmade = parse_attempt("manual-1a2b3c4d-1787681531.5-Леша-0.93.webp")

    assert handmade is None or measure([handmade], []) == []


def test_the_summary_splits_the_wait_into_its_three_owners() -> None:
    """Итог обязан показывать, чья секунда: света, кворума или наша.

    Иначе замер «до и после» превращается в спор о том, что именно чинили.
    """
    summary = summarise(measure(attempts(NIGHT_FILES), [NIGHT_PASS]), min_faces=2)

    assert summary["events"] == 1
    assert summary["with_quorum"] == 1
    assert summary["median_first_match"] == 3.3
    assert summary["median_quorum_wait"] == 9.6
    assert summary["median_our_lag"] == 0.4
    assert summary["median_total"] == 13.3
    assert summary["median_matched_share"] == 0.4


def test_crop_sizes_answer_whether_pixels_are_the_problem() -> None:
    """Узнанный кроп 36×47 и неузнанный 193×208 — дело не в пикселях.

    Замер размеров нужен затем, чтобы не поднимать разрешение потока детекции
    в надежде на чужую беду: если неузнанные кропы крупнее узнанных, мешает
    резкость, а не размер.
    """
    sizes = {
        NIGHT_FILES[0]: (61, 81),
        NIGHT_FILES[2]: (36, 47),
        NIGHT_FILES[3]: (193, 208),
        NIGHT_FILES[4]: (282, 324),
    }

    report = size_report(attempts(NIGHT_FILES), sizes)

    assert report is not None
    assert report["smallest_matched_area"] == 36 * 47
    assert report["largest_missed_area"] == 193 * 208
    assert report["largest_missed_area"] > report["smallest_matched_area"]


def test_missing_numbers_are_dashes_and_not_crashes() -> None:
    """Событие без кворума и без импульса печатается, а не роняет отчёт."""
    rows = measure(attempts(NIGHT_FILES[:2]), [])

    text = render(rows, summarise(rows, min_faces=2), min_faces=2)

    assert "—" in text
    assert "кворум набран у 0" in text
