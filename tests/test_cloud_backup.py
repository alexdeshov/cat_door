"""Облачная копия событий: настройки, отбор, маскировка секрета, обрыв канала."""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self
from urllib.error import URLError

import pytest

from src.cloud_backup.uploader import REMEMBER_IDS, eligible_events, run
from src.common.cloud_backup_config import (
    CloudBackupConfig,
    cloud_backup_config_path,
    cloud_backup_state_path,
    load_cloud_backup_config,
    save_cloud_backup_config,
)
from src.common.settings import Settings


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
    assert saved["access_key_id"] == "AKIDXYZW"  # id не секрет, сверяется с policy
    # Пустые ключи во втором запросе не стёрли сохранённые.
    assert kept["bucket"] == "b2" and kept["has_secret"] is True
    assert "Облачная копия событий" in page.text


def test_the_queue_weight_is_estimated_from_durations() -> None:
    """«Осталось N файлов» ничего не говорит: события бывают по три секунды и
    по три минуты. Вес считается по длительности и наблюдаемому весу секунды.
    """
    from src.cloud_backup.uploader import human_size, video_seconds

    events = [
        {"start_time": 100.0, "end_time": 130.0},
        {"start_time": 200.0, "end_time": 200.0},
        {"start_time": 300.0, "end_time": 360.0},
    ]

    assert video_seconds(events) == 90.0
    # Гигабайты, когда их больше одного; иначе мегабайты.
    assert human_size(2.5e9) == "2.5 ГБ"
    assert human_size(4e7) == "40 МБ"


def test_new_events_go_first() -> None:
    """Свежие минуты важнее позавчерашних: ради них копия и заводилась.

    Канал узла 1.8–3.6 Мбит/с, суточный хвост едет часами — порядок решает,
    что окажется в облаке к моменту, когда оборудование вынесли.
    """
    from src.cloud_backup.uploader import eligible_events

    events = [
        {"id": "старое", "camera": "gate", "start_time": 100.0, "end_time": 110.0},
        {"id": "свежее", "camera": "gate", "start_time": 900.0, "end_time": 910.0},
    ]

    order = [event["id"] for event in eligible_events(events, CloudBackupConfig(), set())]

    assert order == ["свежее", "старое"]


# ---------------------------------------------------------------------------
# Обрыв канала: что служба помнит, что теряет и как догоняет.
#
# Проверяется целый цикл `run(settings, once=True)` на поддельных Frigate и S3:
# именно на стыке «что залилось» и «что записано в журнал» живёт цена ошибки —
# отметка «залито» без объекта в бакете означает потерю навсегда, а её
# отсутствие после успешной заливки — лишнюю оплаченную перезапись.
# ---------------------------------------------------------------------------


class CloudUnreachable(Exception):
    """Отказ botocore, каким его видит служба при пропаже канала.

    `EndpointConnectionError` наследует `BotoCoreError` → `Exception`, а НЕ
    `OSError`: поддельный отказ обязан быть таким же, иначе тест разошёлся бы
    с боевым поведением ровно там, где код ловит только `OSError`.
    """


class _Response:
    """Ответ `urlopen`: контекстный менеджер с `read()` — больше службе не надо."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._payload


class _Frigate:
    """Frigate под рукой: список событий, клипы и снимки, журнал запросов."""

    def __init__(self, events: list[dict[str, Any]], *, down: bool = False) -> None:
        self.events = events
        self.down = down
        self.urls: list[str] = []

    def urlopen(self, url: str, timeout: float = 0.0) -> _Response:
        self.urls.append(url)
        if "/api/events?" in url:
            if self.down:
                raise URLError("Network is unreachable")
            return _Response(json.dumps(self.events).encode("utf-8"))
        if url.endswith("/clip.mp4"):
            return _Response(b"c" * 1024)
        return _Response(b"j" * 64)


class _FakeS3:
    """Бакет: помнит ключи и отказывает по правилу — так изображается обрыв."""

    def __init__(self, fail: Callable[[str], bool] | None = None) -> None:
        self.stored: list[str] = []  # то, что действительно легло в бакет
        self.attempted: list[str] = []
        self.fail: Callable[[str], bool] = fail or (lambda key: False)

    # Имена параметров — как у boto3: служба зовёт их по именам.
    def put_object(self, *, Bucket: str, Key: str, Body: bytes, ContentType: str) -> None:
        self.attempted.append(Key)
        if self.fail(Key):
            raise CloudUnreachable(f"Could not connect to the endpoint URL for {Key}")
        self.stored.append(Key)


def _event(event_id: str, start: float, length: float = 10.0) -> dict[str, Any]:
    return {
        "id": event_id,
        "camera": "gate",
        "start_time": start,
        "end_time": start + length,
        "has_clip": True,
    }


def _install(
    monkeypatch: pytest.MonkeyPatch,
    data_dir: Path,
    frigate: _Frigate,
    s3: _FakeS3,
) -> Settings:
    """Служба на поддельном окружении: настройки на диске, boto3 и HTTP — наши."""
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    save_cloud_backup_config(
        cloud_backup_config_path(data_dir),
        CloudBackupConfig(
            enabled=True, bucket="catdoor-events", access_key_id="AKID", secret_key="s3cr3t"
        ),
    )
    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=lambda *args, **kwargs: s3))
    monkeypatch.setattr(urllib.request, "urlopen", frigate.urlopen)
    return Settings(data_dir=data_dir, gate_api_url="http://frigate:5000")


def _state(data_dir: Path) -> dict[str, Any]:
    return json.loads(cloud_backup_state_path(data_dir).read_text(encoding="utf-8"))


def _age_attempts(data_dir: Path, seconds: float) -> None:
    """Состарить отметки попыток: «прошло N секунд» без настоящего ожидания.

    Бэкофф откладывает повтор от `last_at`; тесты не спят, а сдвигают отметку.
    """
    path = cloud_backup_state_path(data_dir)
    state = json.loads(path.read_text(encoding="utf-8"))
    for entry in (state.get("attempts") or {}).values():
        entry["last_at"] -= seconds
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def _uploaded_clips(s3: _FakeS3) -> Counter[str]:
    return Counter(key for key in s3.stored if key.endswith(".mp4"))


def test_a_dead_uplink_leaves_the_journal_empty(
    runtime_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Канала нет весь цикл: служба живёт, но НИ ОДНО событие не помечено залитым.

    Отметка «залито» без объекта в бакете — потеря навсегда: следующий цикл
    такое событие уже не выберет, а ради этих минут копия и заводилась.
    Счётчик попыток при этом обязан расти — на нём держится вся дальнейшая
    судьба события.
    """
    now = time.time()
    frigate = _Frigate([_event("a", now - 600), _event("b", now - 500), _event("c", now - 400)])
    s3 = _FakeS3(fail=lambda key: True)
    settings = _install(monkeypatch, runtime_path, frigate, s3)

    run(settings, once=True)

    state = _state(runtime_path)
    assert state["uploaded"] == []
    assert {key: value["count"] for key, value in state["attempts"].items()} == {
        "a": 1, "b": 1, "c": 1
    }
    assert state["day_events"] == 0
    # После правки 14.08 полное отсутствие доставки — это не «ok, отложено»,
    # а честная ошибка: зелёная строка неделю врала бы при пустом бакете.
    assert state["status"] == "error"
    assert "ждут 3" in state["message"]
    # Служба не упала и снимки после провалившегося клипа не спрашивала.
    assert not any(url.endswith("snapshot.jpg") for url in frigate.urls)
    # На странице «Состояние» это честная ошибка: отметка последнего успеха не
    # обновляется, пока в бакет ничего не легло.
    assert "last_success_at" not in state


def test_only_delivered_events_enter_the_journal_when_the_link_dies_mid_queue(
    runtime_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Канал упал на середине очереди: залитое помнится, незалитое — нет.

    Ровно эта граница отделяет «догоним в следующий цикл» от двух видов убытка:
    забытый успех — оплаченная перезапись версии в бакете с блокировкой,
    записанный неуспех — событие, которого в облаке не будет никогда.
    """
    now = time.time()
    events = [_event(f"e{index}", now - 600 + index * 60) for index in range(6)]
    delivered = {"e5", "e4", "e3"}  # три самых свежих успели уехать
    s3 = _FakeS3(fail=lambda key: not any(f"/{event_id}." in key for event_id in delivered))
    frigate = _Frigate(events)
    settings = _install(monkeypatch, runtime_path, frigate, s3)

    run(settings, once=True)

    state = _state(runtime_path)
    assert set(state["uploaded"]) == delivered
    assert {key: value["count"] for key, value in state["attempts"].items()} == {
        "e2": 1, "e1": 1, "e0": 1
    }
    # У доехавших событий в бакете и клип, и снимок.
    assert sorted(_uploaded_clips(s3).values()) == [1, 1, 1]
    assert len([key for key in s3.stored if key.endswith(".jpg")]) == 3
    assert state["day_events"] == 3


def test_an_interrupted_catch_up_forgets_at_most_one_delivery(
    runtime_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Догоняющий цикл убит на середине — на диске всё, что успело доехать.

    Такое уже было дважды: сперва два первых догона умерли до конца и 460
    клипов поехали по второму разу, потом 14 августа сторож трижды прислал
    SIGABRT посреди работающей заливки. Отметка раз в двадцать клипов стоила
    двадцати перезаливок за смерть — при версионировании с блокировкой это
    оплаченные запертые версии. Теперь журнал пишется на каждом событии, и
    цена смерти — не больше одного клипа.
    """
    now = time.time()
    events = [_event(f"e{index:02d}", now - 3600 + index * 60) for index in range(30)]
    s3 = _FakeS3()

    def die_on_the_twentysecond(key: str) -> bool:
        if len(_uploaded_clips(s3)) > 21:
            raise KeyboardInterrupt("служба остановлена")
        return False

    s3.fail = die_on_the_twentysecond
    settings = _install(monkeypatch, runtime_path, _Frigate(events), s3)

    with pytest.raises(KeyboardInterrupt):
        run(settings, once=True)

    state = _state(runtime_path)
    delivered = {key.rsplit("/", 1)[-1].removesuffix(".mp4") for key in _uploaded_clips(s3)}
    forgotten = delivered - set(state["uploaded"])
    assert len(forgotten) <= 1  # цена смерти — не больше одного клипа
    assert len(state["uploaded"]) >= 21  # не двадцатками, как было
    # Помнится именно то, что доехало, — сверху очереди, новые вперёд.
    assert set(state["uploaded"]) <= {f"e{index:02d}" for index in range(9, 30)}
    assert state["progress"]["total"] == 30


def test_the_returning_link_catches_up_without_paying_twice(
    runtime_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Связь вернулась: хвост догоняется, а уже залитое второй раз не едет.

    Перезаливка не безобидна: ключи детерминированы, и при версионировании с
    блокировкой каждая лишняя запись — запертая версия и деньги за неё.
    """
    now = time.time()
    events = [_event(f"e{index}", now - 600 + index * 60) for index in range(6)]
    delivered = {"e5", "e4", "e3"}
    s3 = _FakeS3(fail=lambda key: not any(f"/{event_id}." in key for event_id in delivered))
    settings = _install(monkeypatch, runtime_path, _Frigate(events), s3)

    run(settings, once=True)  # канала нет на хвосте очереди
    s3.fail = lambda key: False  # канал вернулся
    _age_attempts(runtime_path, 3600)  # бэкофф выждан
    run(settings, once=True)

    state = _state(runtime_path)
    assert set(state["uploaded"]) == {event["id"] for event in events}
    assert state["attempts"] == {}
    # Каждый клип уехал ровно один раз за оба цикла.
    assert set(_uploaded_clips(s3).values()) == {1}
    assert len(_uploaded_clips(s3)) == 6
    assert state["status"] == "ok"


def test_the_lookback_window_is_the_only_memory_of_a_pending_event(
    runtime_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Незалитое событие живёт только в выборке Frigate за окно оглядки.

    Своего списка «должен, но не смог» служба не ведёт: следующий цикл видит
    ровно то, что Frigate отдал за `lookback_hours` и не больше `limit`. Всё,
    что вышло за эту границу во время простоя, не догонит уже никто — тест
    держит обе цифры на виду, потому что от них зависит длина переживаемого
    простоя.
    """
    from src.cloud_backup.uploader import events_limit

    now = time.time()
    frigate = _Frigate([_event("свежее", now - 300)])
    settings = _install(monkeypatch, runtime_path, frigate, _FakeS3())

    run(settings, once=True)

    query = next(url for url in frigate.urls if "/api/events?" in url)
    after = float(query.split("after=")[1].split("&")[0])
    assert abs(after - (now - 24 * 3600)) < 5  # сутки по умолчанию
    # Предел растёт вместе с окном: жёсткие 500 резали бы окно шире суток молча.
    assert f"limit={events_limit(24)}" in query
    # Незалитого в состоянии не остаётся списком: только счётчик отложенных и
    # журнал залитого — догонять служба будет по выборке Frigate, не по памяти.
    state = _state(runtime_path)
    assert set(state) & {"queue", "debts", "failed"} == set()
    assert state["pending"] == 0


def test_no_amount_of_failures_writes_the_event_off(
    runtime_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Сколько бы попыток ни провалилось — событие не хоронится никогда.

    Списание при однонаправленном ключе необратимо (чтения нет — проверить,
    залито ли, нельзя), а его сигналы на рваной связи оказались ложными:
    попытки копились в мёртвых циклах, «живость» канала доказывалась одним
    мелким объектом (находка ревью 15.08). Теперь неудача лишь отодвигает
    следующую попытку; из памяти событие выходит только с окном оглядки.
    """
    now = time.time()
    frigate = _Frigate([_event("вор", now - 3600)])
    s3 = _FakeS3(fail=lambda key: True)
    settings = _install(monkeypatch, runtime_path, frigate, s3)

    for _ in range(6):
        run(settings, once=True)
        _age_attempts(runtime_path, 7200)  # каждый цикл — с выжданным бэкоффом

    state = _state(runtime_path)
    assert state["uploaded"] == []  # ничего не похоронено
    assert state["attempts"]["вор"]["count"] == 6  # попытки честно копятся
    assert _uploaded_clips(s3) == Counter()


def test_a_failure_defers_the_retry_instead_of_hammering(
    runtime_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Упавшее событие отдыхает по бэкоффу, а не долбится каждый цикл.

    Ядовитый клип, который узел не может отдать, раньше тратил таймаут в
    каждом 30-секундном цикле. Теперь после неудачи событие ждёт своей
    очереди — минута, две, четыре… потолок час — и не мешает остальным.
    """
    now = time.time()
    frigate = _Frigate([_event("яд", now - 3600)])
    s3 = _FakeS3(fail=lambda key: True)
    settings = _install(monkeypatch, runtime_path, frigate, s3)

    run(settings, once=True)  # первая неудача
    tried = len(s3.attempted)
    run(settings, once=True)  # сразу следом: событие отдыхает
    assert len(s3.attempted) == tried  # второй попытки не было
    state = _state(runtime_path)
    assert state["attempts"]["яд"]["count"] == 1
    assert "отложено 1" in state["message"] or "Жду повтора" in state["message"]

    _age_attempts(runtime_path, 3600)  # бэкофф вышел
    run(settings, once=True)
    assert len(s3.attempted) > tried  # попытка вернулась

def test_the_returning_link_must_catch_up_events_that_waited_out_the_outage(
    runtime_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Связь пропала на пять минут — событие обязано уехать, когда она вернулась.

    Сценарий отказа ровно тот, ради которого служба написана: вор перерезает
    провод или глушит канал, служба пять раз подряд не достаёт до облака,
    списывает клип подхода к калитке — и когда канал восстановят, в облаке
    будет всё, кроме единственного нужного события.
    """
    now = time.time()
    s3 = _FakeS3(fail=lambda key: True)
    settings = _install(monkeypatch, runtime_path, _Frigate([_event("вор", now - 300)]), s3)

    for _ in range(5):
        run(settings, once=True)
        _age_attempts(runtime_path, 7200)
    s3.fail = lambda key: False  # канал вернулся
    run(settings, once=True)

    # Клип обязан оказаться в бакете, как только канал восстановился.
    assert any(key.endswith("/вор.mp4") for key in s3.stored)


def test_a_failing_snapshot_neither_stops_the_queue_nor_forgets_the_clip(
    runtime_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Снимок не уехал — клип всё равно залит, и очередь идёт дальше.

    Снимок — приложение к клипу, и в коде это записано прямо: «клип важнее, не
    срываем цикл». Но ловится там только `OSError`, а отказ бакета им не
    является: при устойчивом отказе на снимке служба каждую минуту заливает
    один и тот же первый клип заново и не доходит до остальных никогда.
    """
    now = time.time()
    events = [_event(f"e{index}", now - 600 + index * 60) for index in range(3)]
    s3 = _FakeS3(fail=lambda key: key.endswith(".jpg"))
    settings = _install(monkeypatch, runtime_path, _Frigate(events), s3)

    run(settings, once=True)

    state = _state(runtime_path)
    assert set(state["uploaded"]) == {"e2", "e1", "e0"}
    assert len(_uploaded_clips(s3)) == 3


def test_frigate_outage_neither_crashes_the_service_nor_wipes_the_journal(
    runtime_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Frigate недоступен: цикл переживает отказ и не теряет память о залитом.

    Журнал залитого — единственная защита от повторной оплаченной заливки; цикл,
    который не смог даже спросить события, не имеет права его переписать.
    """
    now = time.time()
    frigate = _Frigate([_event("a", now - 300)], down=True)
    settings = _install(monkeypatch, runtime_path, frigate, _FakeS3())
    cloud_backup_state_path(runtime_path).parent.mkdir(parents=True, exist_ok=True)
    cloud_backup_state_path(runtime_path).write_text(
        json.dumps({"uploaded": ["старое"], "status": "ok"}), encoding="utf-8"
    )

    run(settings, once=True)

    state = _state(runtime_path)
    assert state["status"] == "error"
    assert "Network is unreachable" in state["last_error"]
    assert state["uploaded"] == ["старое"]


def test_the_journal_stops_growing_at_the_remembered_limit(
    runtime_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Файл состояния не растёт бесконечно — но помнит не меньше двух выборок.

    Журнал в 4000 при выборке в 4800 ронял отметки о свежезалитом и перезаливал
    излишек по кругу (находка ревью 15.08): потолок обязан быть не меньше
    удвоенного предела выборки текущего окна.
    """
    from src.cloud_backup.uploader import events_limit

    cap = max(REMEMBER_IDS, 2 * events_limit(24))
    now = time.time()
    settings = _install(monkeypatch, runtime_path, _Frigate([_event("a", now - 300)]), _FakeS3())
    cloud_backup_state_path(runtime_path).parent.mkdir(parents=True, exist_ok=True)
    cloud_backup_state_path(runtime_path).write_text(
        json.dumps({"uploaded": [f"old-{index:05d}" for index in range(cap + 400)]}),
        encoding="utf-8",
    )

    run(settings, once=True)

    assert len(_state(runtime_path)["uploaded"]) == cap
    assert cap >= 2 * events_limit(24)


def test_trimming_the_journal_keeps_the_freshest_marks(
    runtime_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Обрезка обязана срезать старое: свежие отметки нужны как раз сейчас.

    Отметка живёт ради окна оглядки — событие суточной давности ещё вернётся
    в выборке Frigate. Потерянная сегодняшняя отметка означает повторную
    заливку того же клипа, то есть лишнюю запертую версию и оплату за неё;
    выброшенная старая не стоит ничего.
    """
    from src.cloud_backup.uploader import events_limit

    cap = max(REMEMBER_IDS, 2 * events_limit(24))
    now = time.time()
    fresh = [f"fresh-{index:05d}" for index in range(400)]
    settings = _install(monkeypatch, runtime_path, _Frigate([_event("a", now - 300)]), _FakeS3())
    cloud_backup_state_path(runtime_path).parent.mkdir(parents=True, exist_ok=True)
    cloud_backup_state_path(runtime_path).write_text(
        json.dumps({"uploaded": [f"old-{index:05d}" for index in range(cap)] + fresh}),
        encoding="utf-8",
    )

    run(settings, once=True)

    kept = set(_state(runtime_path)["uploaded"])
    assert set(fresh) <= kept


def test_orphaned_attempts_do_not_mute_the_health_stamps(
    runtime_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Сирота: событие числится и залитым, и ждущим попытки — одновременно.

    Такие записи оставлял код до #267 (ветка `_skip` метила событие залитым,
    не убирая из попыток). Отбор залитое не видит, повтора не будет никогда —
    а вечно «непустая» очередь глушила отметку здоровья, и первая же тихая
    ночь будила человека ложной тревогой (15.08). Сирота обязана вычищаться
    первым же кругом.
    """
    now = time.time()
    frigate = _Frigate([_event("orphan", now - 600)])
    s3 = _FakeS3()
    settings = _install(monkeypatch, runtime_path, frigate, s3)
    cloud_backup_state_path(runtime_path).write_text(
        json.dumps({"uploaded": ["orphan"], "attempts": {"orphan": 3}}),
        encoding="utf-8",
    )

    run(settings, once=True)

    state = _state(runtime_path)
    assert state["attempts"] == {}
    assert "oldest_pending_at" not in state
    # Очередь честно пуста, круг прошёл — обе отметки здоровья на месте.
    assert "last_cycle_ok_at" in state
    assert "last_success_at" in state


def test_a_stuck_event_is_visible_as_oldest_pending(
    runtime_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Незалитое событие видно тревоге возрастом, а не отсутствием успеха.

    `oldest_pending_at` — конец старейшего ждущего события: по нему тревога
    отличает «ночью нечего заливать» от «клип час не может уехать».
    """
    now = time.time()
    event = _event("stuck", now - 600)
    frigate = _Frigate([event])
    s3 = _FakeS3(fail=lambda key: True)
    settings = _install(monkeypatch, runtime_path, frigate, s3)

    run(settings, once=True)

    state = _state(runtime_path)
    assert state["attempts"]["stuck"]["count"] == 1
    assert state["oldest_pending_at"] == pytest.approx(event["end_time"])
    # Круг при этом завершился: тревога придёт про застрявшее событие, а не
    # про «циклы не проходят».
    assert "last_cycle_ok_at" in state


# ── Тревога о простое копии ────────────────────────────────────────


def test_a_quiet_night_is_not_an_alarm() -> None:
    """Ночь без событий — не беда, сколько бы часов ни шло без заливок.

    Прежний критерий «нет успеха час» не отличал «нечего заливать» от «не
    может залить»: первая же тихая ночь после выката будила человека ложным
    «копия не выгружает 7 ч» (15.08). Пока циклы проходят и очередь пуста —
    тишина.
    """
    from src.cloud_backup.uploader import alert_decision

    now = 100_000.0
    state = {"last_cycle_ok_at": now - 10.0, "last_success_at": now - 9 * 3600}

    assert alert_decision(state, now=now) is None


def test_an_hour_of_failing_cycles_raises_the_alarm() -> None:
    """Круг не завершается час — Frigate лежит или фильтр камер не совпал.

    Час выбран как примерно два десятка событий обычного дня: достаточно,
    чтобы переждать перезагрузку роутера, и мало, чтобы узнать о беде вечером.
    """
    from src.cloud_backup.uploader import alert_decision

    now = 100_000.0
    fresh = {"last_cycle_ok_at": now - 600}
    stale = {"last_cycle_ok_at": now - 3700}

    assert alert_decision(fresh, now=now) is None
    assert alert_decision(stale, now=now) == "alarm:cycles"


def test_a_stuck_event_raises_the_alarm() -> None:
    """Циклы идут, но событие час не заливается: канал мёртв или оно ядовито.

    «Молчания» тут нет — беду доказывает конкретный клип, которого нет в
    облаке дольше часа.
    """
    from src.cloud_backup.uploader import alert_decision

    now = 100_000.0
    waiting = {"last_cycle_ok_at": now - 10.0, "oldest_pending_at": now - 600}
    stuck = {"last_cycle_ok_at": now - 10.0, "oldest_pending_at": now - 3700}

    assert alert_decision(waiting, now=now) is None
    assert alert_decision(stuck, now=now) == "alarm:stuck"


def test_a_backup_that_never_worked_alarms_too() -> None:
    """«Ни разу не заработало» — такая же беда, как «перестало».

    Отсчёт идёт от момента включения, иначе неверный ключ молчал бы вечно.
    """
    from src.cloud_backup.uploader import alert_decision

    now = 100_000.0

    assert alert_decision({"watching_since": now - 3700}, now=now) == "alarm:cycles"
    assert alert_decision({"watching_since": now - 60}, now=now) is None
    # Копия ещё не включалась — тревожить не о чем.
    assert alert_decision({}, now=now) is None


def test_clock_jumps_do_not_confuse_the_alarm() -> None:
    """Отметка из «будущего» (часы прыгнули назад) читается как «только что».

    Иначе обрыв питания то глушил бы тревогу на часы вперёд, то взводил её
    отрицательным возрастом.
    """
    from src.cloud_backup.uploader import alert_decision

    now = 100_000.0

    assert alert_decision({"last_cycle_ok_at": now + 9999}, now=now) is None
    assert (
        alert_decision(
            {"last_cycle_ok_at": now - 10.0, "oldest_pending_at": now + 9999}, now=now
        )
        is None
    )


def test_the_alarm_does_not_repeat_every_minute() -> None:
    """Тревога раз в минуту превращается в шум, который перестают читать."""
    from src.cloud_backup.uploader import alert_decision

    now = 100_000.0
    state = {"last_cycle_ok_at": now - 7200, "alerted_at": now - 600}

    assert alert_decision(state, now=now) is None
    # Через шесть часов напомнить стоит: беда не рассосалась.
    assert alert_decision(state, now=now + 6 * 3600) == "alarm:cycles"


def test_recovery_is_announced_once() -> None:
    """О возвращении надо сказать — иначе человек ждёт вторую тревогу зря."""
    from src.cloud_backup.uploader import _notify_about_trouble, alert_decision

    now = 100_000.0
    state = {"last_cycle_ok_at": now - 60, "alerted_at": now - 7200}
    sent: list[str] = []

    assert alert_decision(state, now=now) == "recovered"
    _notify_about_trouble(state, sent.append, now=now)

    assert "снова работает" in sent[0]
    assert "alerted_at" not in state  # второй раз о том же не скажет
    assert alert_decision(state, now=now) is None


def test_the_alarm_text_carries_the_reason() -> None:
    """Сообщение обязано отвечать «почему», иначе человек идёт искать вслепую."""
    from src.cloud_backup.uploader import _notify_about_trouble

    now = 100_000.0
    cycles = {
        "last_cycle_ok_at": now - 7200,
        "last_error": "AccessDenied при PutObject",
    }
    stuck = {
        "last_cycle_ok_at": now - 10.0,
        "oldest_pending_at": now - 7200,
        "pending": 3,
        "last_error": "Could not connect to the endpoint URL",
    }
    sent: list[str] = []

    _notify_about_trouble(cycles, sent.append, now=now)
    _notify_about_trouble(stuck, sent.append, now=now)

    assert "ни одного круга 2 ч" in sent[0]
    assert "AccessDenied" in sent[0]
    assert cycles["alerted_at"] == now
    # Застрявшие — не «копия молчит»: сказано, сколько их и как давно ждут.
    assert "3 событий" in sent[1]
    assert "2 ч" in sent[1]
    assert "endpoint" in sent[1]
    assert stuck["alerted_at"] == now


def test_a_three_hour_event_does_not_stall_the_queue() -> None:
    """14 августа машина простояла у калитки 162 минуты.

    Клип такого события — около гигабайта: Frigate собирает его дольше любого
    разумного таймаута, а канал на 300 КБ/с отдавал бы час. Событие встало
    первым в очереди (новые вперёд) и остановило выгрузку целиком при живом
    канале. От таких событий уезжает снимок — факт и кадр важнее трёх часов
    записи стоящей машины, а ролик остаётся на узле и в ночной копии.
    """
    from src.cloud_backup.uploader import MAX_CLIP_SECONDS, eligible_events, video_seconds

    events = [
        {"id": "машина", "camera": "gate", "start_time": 0.0,
         "end_time": MAX_CLIP_SECONDS + 60},
        {"id": "человек", "camera": "gate", "start_time": 0.0, "end_time": 30.0},
    ]

    selected = eligible_events(events, CloudBackupConfig(), set())

    assert [event["id"] for event in selected] == ["машина", "человек"]
    assert selected[0]["_snapshot_only"] is True
    assert "_snapshot_only" not in selected[1]
    # Вес очереди считается без неподъёмного ролика: иначе «осталось N ГБ»
    # состояло бы из того, что никто не собирается заливать.
    assert video_seconds(selected) == 30.0


def test_the_broker_wakes_the_cycle_only_when_an_event_ends() -> None:
    """Событие человека живёт 8 секунд (замер 14.08), а опрос занимал минуту.

    Почти вся задержка между «человек ушёл» и «клип в облаке» приходилась на
    ожидание следующего круга. Будильник от брокера её снимает — но только на
    конце события: у начатого клип ещё растёт, и будить цикл на него значит
    гонять его вхолостую.
    """
    import json as json_module
    from types import SimpleNamespace

    from src.cloud_backup.uploader import EventTrigger

    trigger = EventTrigger("127.0.0.1")

    def send(payload: object) -> None:
        trigger._on_message(
            None, None, SimpleNamespace(payload=json_module.dumps(payload).encode())
        )

    send({"type": "new", "after": {"id": "a"}})
    assert not trigger._wake.is_set()  # начало события цикл не будит

    send({"type": "end", "after": {"id": "a"}})
    assert trigger._wake.is_set()

    trigger.wait(0)
    assert not trigger._wake.is_set()  # разбудились один раз, флаг снят
    # Мусор в теме не должен ронять подписчика.
    trigger._on_message(None, None, SimpleNamespace(payload="не json".encode()))


def test_without_a_broker_the_cycle_falls_back_to_the_timer() -> None:
    """Подписка дополняет опрос, а не заменяет: сообщение, пришедшее в мёртвую
    службу, потеряно навсегда, а опрос переживает и перезапуск, и падение
    брокера.
    """
    from src.cloud_backup.uploader import EventTrigger

    assert EventTrigger("").start() is False
    # Ожидание без брокера просто выдерживает таймаут — и говорит, что разбудил
    # именно он: после будильника брокера цикл обязан выждать сборку клипа.
    assert EventTrigger("").wait(0.01) is False


def test_a_just_finished_event_waits_for_frigate_to_build_the_clip() -> None:
    """Будильник брокера приходит в ту же секунду, в которую событие кончилось.

    Замер 14 августа: у шести событий подряд время отказа `HTTP 400` совпало с
    их `end_time` до секунды — клипа в этот момент у Frigate ещё нет. Ранний
    вопрос ничего не ускорял (событие всё равно уезжало следующим кругом), зато
    копил попытки, а пять попыток при живом канале списывают событие как
    «залитое», не залив его.
    """
    from src.cloud_backup.uploader import CLIP_READY_SECONDS, eligible_events

    now = 10_000.0
    events = [
        {"id": "только что", "camera": "gate", "start_time": now - 8, "end_time": now},
        {"id": "созрело", "camera": "gate", "start_time": now - 60,
         "end_time": now - CLIP_READY_SECONDS},
    ]

    selected = eligible_events(events, CloudBackupConfig(), set(), now=now)

    assert [event["id"] for event in selected] == ["созрело"]
    # Пропуск без пометки — как у незавершённого события: следующий круг обязан
    # увидеть его снова, иначе клип не уедет никогда.
    later = eligible_events(
        events, CloudBackupConfig(), set(), now=now + CLIP_READY_SECONDS
    )
    assert [event["id"] for event in later] == ["только что", "созрело"]


def test_an_empty_queue_says_so_instead_of_looking_broken(
    runtime_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ноль на графике бакета — это либо «всё залито», либо «всё встало».

    14 августа догон закончился в 21:16, график лёг в ноль, и строка «За сутки:
    134 событий, 1660 МБ» прочиталась как поломка. Пустая очередь обязана
    говорить о себе прямо, а свежее событие — не попадать в журнал залитого.
    """
    now = time.time()
    frigate = _Frigate([_event("свежее", now - 5, 5.0)])  # закончилось только что
    s3 = _FakeS3()
    settings = _install(monkeypatch, runtime_path, frigate, s3)

    run(settings, once=True)

    state = _state(runtime_path)
    assert s3.attempted == []  # клип ещё собирается — не дёргаем
    assert state["uploaded"] == [] and state["attempts"] == {}
    assert state["status"] == "ok" and state["pending"] == 0
    assert "Очередь пуста" in state["message"]

    # А непустая очередь отчитывается по-прежнему — суточным итогом.
    frigate.events = [_event("вчерашнее", now - 3600)]
    run(settings, once=True)

    assert "За сутки" in _state(runtime_path)["message"]


def test_the_watchdog_is_fed_while_a_clip_is_uploading() -> None:
    """Пинги между событиями не спасают: один клип уезжает восемь минут.

    14 августа сторож (на узле всё ещё 300 с) трижды убил выгрузчик посреди
    работающей заливки — SIGABRT, потерянные отметки, повторно залитое окно.
    Кормить его надо изнутри передачи.
    """
    from src.cloud_backup.uploader import _heartbeat

    pings: list[int] = []
    notifier = SimpleNamespace(
        watchdog_interval_seconds=0.01, watchdog=lambda: pings.append(1)
    )

    with _heartbeat(notifier):  # type: ignore[arg-type]  # хватает утиной совместимости
        time.sleep(0.05)

    assert len(pings) >= 2
    stopped = len(pings)
    time.sleep(0.03)
    assert len(pings) == stopped  # нить не переживает выход из передачи

    # Без сторожа (служба запущена из оболочки) нить не заводится вовсе.
    with _heartbeat(SimpleNamespace(watchdog_interval_seconds=None)):  # type: ignore[arg-type]
        pass


def test_the_events_limit_grows_with_the_lookback_window() -> None:
    """Жёсткий предел в 500 событий тихо резал окно шире суток.

    Frigate отдаёт новые первыми: с оглядкой в неделю служба видела бы только
    свежие пятьсот, а остальное окно не залилось бы никогда — и панель
    показывала бы «очередь пуста». Замер 14 августа: 465 событий в сутки.
    """
    from src.cloud_backup.uploader import EVENTS_LIMIT_CEILING, events_limit

    assert events_limit(1) == 500  # пол не ниже прежнего предела
    assert events_limit(24) == 4800  # десятикратный запас к замеренным 465
    assert events_limit(168) == EVENTS_LIMIT_CEILING  # неделя упирается в потолок


# ── Находки адверсарного ревью 15.08 ───────────────────────────────


def test_an_unreadable_state_file_does_not_wipe_the_journal(
    runtime_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Сбойное чтение состояния — пропущенный цикл, а не стёртый журнал.

    Раньше любой сбойный read выглядел как чистый старт: цикл работал с пустым
    журналом и в конце перезаписывал файл без отметок — всё окно ехало по
    второму разу. Нечитаемый файл обязан оставлять журнал нетронутым.
    """
    now = time.time()
    s3 = _FakeS3()
    settings = _install(monkeypatch, runtime_path, _Frigate([_event("a", now - 300)]), s3)
    path = cloud_backup_state_path(runtime_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"uploaded": ["a"]', encoding="utf-8")  # обрезанный json

    run(settings, once=True)

    # Файл не перезаписан «чистым листом», клип не поехал по второму разу.
    assert path.read_text(encoding="utf-8") == '{"uploaded": ["a"]'
    assert s3.attempted == []


def test_a_garbage_event_does_not_stall_the_whole_queue(
    runtime_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Кривое событие в выдаче Frigate пропускается, остальные едут.

    Одна мусорная строка (нечисловое время) раньше роняла отбор целиком —
    и каждые 30 секунд, пока событие не выйдет из окна оглядки.
    """
    now = time.time()
    events = [
        {"id": "мусор", "camera": "gate", "start_time": "не число", "end_time": now - 60},
        _event("живое", now - 300),
    ]
    s3 = _FakeS3()
    settings = _install(monkeypatch, runtime_path, _Frigate(events), s3)

    run(settings, once=True)

    assert any(key.endswith("/живое.mp4") for key in s3.stored)
    assert _state(runtime_path)["status"] == "ok"


def test_a_transient_has_clip_false_is_not_a_death_sentence() -> None:
    """`has_clip=False` сразу после конца события — ещё не «записи нет».

    Флаг у Frigate выставляется с задержкой: событие, увиденное без него в
    первые секунды, раньше помечалось пропуском навсегда — и появившийся
    клип не уезжал уже никогда.
    """
    from src.cloud_backup.uploader import HAS_CLIP_GRACE_SECONDS, eligible_events

    now = 100_000.0
    fresh = {"id": "свежее", "camera": "gate", "start_time": now - 40,
             "end_time": now - 30, "has_clip": False}
    stale = {"id": "старое", "camera": "gate", "start_time": now - 4000,
             "end_time": now - 3990, "has_clip": False}

    selected = eligible_events([fresh, stale], CloudBackupConfig(), set(), now=now)

    ids = {event["id"]: event for event in selected}
    assert "свежее" not in ids  # без пометки: вернёмся, когда флаг дозреет
    assert ids["старое"].get("_skip") is True  # старому ждать нечего
    # Флаг дозрел — событие снова в отборе, как обычное.
    ripened = {**fresh, "has_clip": True}
    later = eligible_events(
        [ripened], CloudBackupConfig(), set(), now=now + HAS_CLIP_GRACE_SECONDS
    )
    assert [event["id"] for event in later] == ["свежее"]


def test_a_duplicate_id_in_the_feed_uploads_once() -> None:
    """Дубликат id в выдаче Frigate не уезжает дважды за цикл."""
    now = 100_000.0
    twice = [_event("одно", now - 300), _event("одно", now - 300)]

    selected = eligible_events(twice, CloudBackupConfig(), set(), now=now)

    assert len(selected) == 1


def test_a_mismatched_camera_filter_is_an_error_not_an_empty_queue(
    runtime_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Фильтр камер, не совпавший ни с чем, — честная ошибка.

    Прецедент `test-cam` против `test_cam`: события шли, фильтр их отсекал,
    панель писала «Очередь пуста» со свежим успехом — и тревога молчала
    неделями при пустом бакете.
    """
    now = time.time()
    frigate = _Frigate([_event("a", now - 300)])  # камера gate
    settings = _install(monkeypatch, runtime_path, frigate, _FakeS3())
    save_cloud_backup_config(
        cloud_backup_config_path(runtime_path),
        CloudBackupConfig(
            enabled=True, bucket="catdoor-events", access_key_id="AKID",
            secret_key="s3cr3t", cameras=["test-cam"],  # а во Frigate «gate»
        ),
    )

    run(settings, once=True)

    state = _state(runtime_path)
    assert state["status"] == "error"
    assert "Фильтр камер" in state["last_error"]
    assert "last_success_at" not in state


def test_waiting_out_backoff_does_not_refresh_the_health_mark(
    runtime_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Цикл, где вся очередь отдыхает по бэкоффу, не притворяется здоровым.

    Свежая отметка успеха в таком цикле глушила бы тревогу при неделями
    стоящих клипах: канал ничего не доказал.
    """
    now = time.time()
    s3 = _FakeS3(fail=lambda key: True)
    settings = _install(monkeypatch, runtime_path, _Frigate([_event("a", now - 300)]), s3)

    run(settings, once=True)  # неудача: события в бэкоффе
    state = _state(runtime_path)
    assert "last_success_at" not in state

    run(settings, once=True)  # очередь отдыхает — успех не рисуется
    state = _state(runtime_path)
    assert "last_success_at" not in state
    assert "Жду повтора" in state["message"]


def test_attempts_for_events_outside_the_window_are_forgotten(
    runtime_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Память о попытках живёт вместе с окном оглядки.

    Событие, ушедшее из окна, Frigate больше не отдаст — его запись только
    копила бы файл состояния навсегда.
    """
    now = time.time()
    s3 = _FakeS3()
    settings = _install(monkeypatch, runtime_path, _Frigate([_event("a", now - 300)]), s3)
    path = cloud_backup_state_path(runtime_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"attempts": {"давно-ушедшее": {"count": 3, "last_at": now - 90000}}}),
        encoding="utf-8",
    )

    run(settings, once=True)

    assert _state(runtime_path)["attempts"] == {}


def test_the_heartbeat_thread_does_not_feed_the_watchdog_forever() -> None:
    """У нити-кормильца есть потолок жизни: зависшую передачу нельзя прятать.

    Таймауты передач по-операционные: полуживое соединение, отдающее байт раз
    в сто секунд, держит put_object вечно — и бесконечный пинг делал бы это
    зависание невидимым для systemd навсегда.
    """
    from src.cloud_backup.uploader import _heartbeat

    pings: list[int] = []
    notifier = SimpleNamespace(
        watchdog_interval_seconds=0.01, watchdog=lambda: pings.append(1)
    )
    # Потолок пересчитывается от интервала: подменять константу не нужно —
    # хватает того, что нить считает прожитое как beats * interval.
    import src.cloud_backup.uploader as uploader_module

    original = uploader_module.HEARTBEAT_LIFETIME_SECONDS
    uploader_module.HEARTBEAT_LIFETIME_SECONDS = 0.03
    try:
        with _heartbeat(notifier):  # type: ignore[arg-type]
            time.sleep(0.15)  # «передача» живёт дольше потолка
    finally:
        uploader_module.HEARTBEAT_LIFETIME_SECONDS = original

    assert len(pings) <= 4  # ~3 пинга до потолка, дальше тишина


def test_the_bucket_is_measured_and_a_dead_lifecycle_is_called_out(
    runtime_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Служба меряет бакет и говорит вслух, если уборка не работает.

    Удалять она не умеет по построению — уборку делает lifecycle, который
    настраивает владелец; забытая настройка означала бы бесконечный счёт.
    """
    from datetime import UTC, timedelta
    from datetime import datetime as dt

    now = time.time()
    settings = _install(monkeypatch, runtime_path, _Frigate([_event("a", now - 300)]), _FakeS3())
    save_cloud_backup_config(
        cloud_backup_config_path(runtime_path),
        CloudBackupConfig(
            enabled=True, bucket="catdoor-events", access_key_id="AKID",
            secret_key="s3cr3t", retention_days=30,
        ),
    )
    stale = dt.now(UTC) - timedelta(days=40)  # старше срока + люфта

    class _ListingS3(_FakeS3):
        def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "Contents": [
                    {"Size": 1000, "LastModified": stale},
                    {"Size": 2000, "LastModified": dt.now(UTC)},
                ],
                "IsTruncated": False,
            }

    s3 = _ListingS3()
    monkeypatch.setitem(
        sys.modules, "boto3", SimpleNamespace(client=lambda *args, **kwargs: s3)
    )

    run(settings, once=True)

    cloud = _state(runtime_path)["cloud"]
    assert cloud["bytes"] == 3000 and cloud["objects"] == 2
    assert cloud["retention_working"] is False  # уборка не убирает — сказано вслух


def test_the_panel_prints_the_lifecycle_command_for_the_owner(
    runtime_path: Path, monkeypatch
) -> None:
    """Панель печатает готовую команду настройки уборки — владелец выполняет."""
    from tests.test_admin import make_client

    client = make_client(runtime_path, monkeypatch)
    saved = client.put(
        "/api/cloud-backup/settings",
        json={"enabled": True, "bucket": "b1", "retention_days": 30,
              "access_key_id": "AKIDXYZW", "secret_key": "verysecret"},
    ).json()

    command = saved["lifecycle_command"]
    assert "put-bucket-lifecycle-configuration" in command
    assert '"Days":30' in command and "--bucket b1" in command
    assert "verysecret" not in command  # команда — без секретов


def test_shared_bucket_gets_one_command_with_both_rules(
    runtime_path: Path, monkeypatch
) -> None:
    """`put-bucket-lifecycle-configuration` заменяет конфигурацию целиком.

    Две команды на один бакет — мина: вторая молча снесла бы правило первой и
    остановила уборку. Пока бакет общий, правила печатаются одной командой, а
    отдельной кошачьей не существует.
    """
    from tests.test_admin import make_client

    client = make_client(runtime_path, monkeypatch)

    saved = client.put(
        "/api/cloud-backup/settings",
        json={"enabled": True, "bucket": "b1", "retention_days": 30,
              "cats_enabled": True, "cats_retention_days": 90,
              "access_key_id": "AKIDXYZW", "secret_key": "verysecret"},
    ).json()

    assert "cats_lifecycle_command" not in saved
    command = saved["lifecycle_command"]
    assert command.count("--bucket") == 1 and "--bucket b1" in command
    assert "catdoor-events-retention" in command
    assert "catdoor-cats-retention" in command
    assert '"Days":30' in command and '"Days":90' in command


def test_own_cats_bucket_gets_its_own_command(runtime_path: Path, monkeypatch) -> None:
    """Разные бакеты — разные конфигурации, затирать друг друга нечем."""
    from tests.test_admin import make_client

    client = make_client(runtime_path, monkeypatch)

    saved = client.put(
        "/api/cloud-backup/settings",
        json={"enabled": True, "bucket": "b1", "retention_days": 30,
              "cats_enabled": True, "cats_bucket": "b1-cats",
              "cats_retention_days": 90,
              "access_key_id": "AKIDXYZW", "secret_key": "verysecret"},
    ).json()

    gate = saved["lifecycle_command"]
    cats = saved["cats_lifecycle_command"]
    assert "--bucket b1 " in gate and "catdoor-cats-retention" not in gate
    assert "--bucket b1-cats " in cats and "catdoor-events-retention" not in cats
    assert '"Prefix":"cat-events/"' in cats
    assert "verysecret" not in gate and "verysecret" not in cats
