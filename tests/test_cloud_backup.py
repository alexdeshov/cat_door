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
    assert state["attempts"] == {"a": 1, "b": 1, "c": 1}
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
    assert state["attempts"] == {"e2": 1, "e1": 1, "e0": 1}
    # У доехавших событий в бакете и клип, и снимок.
    assert sorted(_uploaded_clips(s3).values()) == [1, 1, 1]
    assert len([key for key in s3.stored if key.endswith(".jpg")]) == 3
    assert state["day_events"] == 3


def test_an_interrupted_catch_up_keeps_the_first_twenty_deliveries(
    runtime_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Догоняющий цикл убит на середине — журнал промежуточных сохранений цел.

    Такое уже было: два первых догоняющих цикла умерли до конца, час выгрузки
    не оставил ни одной отметки, и 460 клипов поехали по второму разу. Здесь
    цикл обрывается на 22-м событии так, как его обрывает systemd, — и первая
    двадцатка обязана остаться на диске.
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
    assert len(state["uploaded"]) == 20
    # Помнится именно то, что доехало, — сверху очереди, новые вперёд.
    assert set(state["uploaded"]) == {f"e{index:02d}" for index in range(10, 30)}
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
    now = time.time()
    frigate = _Frigate([_event("свежее", now - 300)])
    settings = _install(monkeypatch, runtime_path, frigate, _FakeS3())

    run(settings, once=True)

    query = next(url for url in frigate.urls if "/api/events?" in url)
    after = float(query.split("after=")[1].split("&")[0])
    assert abs(after - (now - 24 * 3600)) < 5  # сутки по умолчанию
    assert "limit=500" in query
    # Незалитого в состоянии не остаётся ни строкой — только журнал залитого.
    state = _state(runtime_path)
    assert set(state) & {"pending", "queue", "failed"} == set()


def test_five_offline_cycles_do_not_write_the_event_off(
    runtime_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Пять минут без интернета не должны хоронить очередь.

    Порог в пять попыток задумывался против «ядовитого» клипа, который узел не
    может отдать. Но счётчик рос по циклам, а цикл — минута: пять минут обрыва
    списывали ВСЮ очередь как залитую, и вернувшаяся связь уже не догоняла ни
    одного клипа — включая тот, ради которого копия и заводилась. Списание
    допустимо только когда канал в том же цикле доказал, что жив.
    """
    frigate = _Frigate([_event("вор", start=1000.0)])
    s3 = _FakeS3(fail=lambda key: True)
    settings = _install(monkeypatch, runtime_path, frigate, s3)

    for _ in range(6):
        run(settings, once=True)

    state = _state(runtime_path)
    assert state["uploaded"] == []  # ничего не похоронено
    assert state["attempts"]["вор"] == 6  # попытки честно копятся
    assert _uploaded_clips(s3) == Counter()

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
    """Файл состояния не растёт бесконечно: помним ровно REMEMBER_IDS отметок."""
    now = time.time()
    settings = _install(monkeypatch, runtime_path, _Frigate([_event("a", now - 300)]), _FakeS3())
    cloud_backup_state_path(runtime_path).parent.mkdir(parents=True, exist_ok=True)
    cloud_backup_state_path(runtime_path).write_text(
        json.dumps({"uploaded": [f"old-{index:05d}" for index in range(REMEMBER_IDS + 400)]}),
        encoding="utf-8",
    )

    run(settings, once=True)

    assert len(_state(runtime_path)["uploaded"]) == REMEMBER_IDS


def test_trimming_the_journal_keeps_the_freshest_marks(
    runtime_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Обрезка обязана срезать старое: свежие отметки нужны как раз сейчас.

    Отметка живёт ради окна оглядки — событие суточной давности ещё вернётся
    в выборке Frigate. Потерянная сегодняшняя отметка означает повторную
    заливку того же клипа, то есть лишнюю запертую версию и оплату за неё;
    выброшенная старая не стоит ничего. Сейчас код выбирает произвольные:
    хвост берётся у списка, собранного из множества.
    """
    now = time.time()
    fresh = [f"fresh-{index:05d}" for index in range(400)]
    settings = _install(monkeypatch, runtime_path, _Frigate([_event("a", now - 300)]), _FakeS3())
    cloud_backup_state_path(runtime_path).parent.mkdir(parents=True, exist_ok=True)
    cloud_backup_state_path(runtime_path).write_text(
        json.dumps({"uploaded": [f"old-{index:05d}" for index in range(REMEMBER_IDS)] + fresh}),
        encoding="utf-8",
    )

    run(settings, once=True)

    kept = set(_state(runtime_path)["uploaded"])
    assert set(fresh) <= kept


# ── Тревога о простое копии ────────────────────────────────────────


def test_an_hour_of_silence_raises_the_alarm() -> None:
    """Копия ломается молча: бакет пуст, а на узле всё «работает».

    Час выбран как примерно два десятка событий обычного дня: достаточно,
    чтобы переждать перезагрузку роутера, и мало, чтобы узнать о беде вечером.
    """
    from src.cloud_backup.uploader import alert_decision

    now = 100_000.0
    fresh = {"last_success_at": now - 600}
    stale = {"last_success_at": now - 3700}

    assert alert_decision(fresh, now=now) is None
    assert alert_decision(stale, now=now) == "alarm"


def test_a_backup_that_never_worked_alarms_too() -> None:
    """«Ни разу не заработало» — такая же беда, как «перестало».

    Отсчёт идёт от момента включения, иначе неверный ключ молчал бы вечно.
    """
    from src.cloud_backup.uploader import alert_decision

    now = 100_000.0

    assert alert_decision({"watching_since": now - 3700}, now=now) == "alarm"
    assert alert_decision({"watching_since": now - 60}, now=now) is None
    # Копия ещё не включалась — тревожить не о чем.
    assert alert_decision({}, now=now) is None


def test_the_alarm_does_not_repeat_every_minute() -> None:
    """Тревога раз в минуту превращается в шум, который перестают читать."""
    from src.cloud_backup.uploader import alert_decision

    now = 100_000.0
    state = {"last_success_at": now - 7200, "alerted_at": now - 600}

    assert alert_decision(state, now=now) is None
    # Через шесть часов напомнить стоит: беда не рассосалась.
    assert alert_decision(state, now=now + 6 * 3600) == "alarm"


def test_recovery_is_announced_once() -> None:
    """О возвращении надо сказать — иначе человек ждёт вторую тревогу зря."""
    from src.cloud_backup.uploader import _notify_about_silence, alert_decision

    now = 100_000.0
    state = {"last_success_at": now - 60, "alerted_at": now - 7200}
    sent: list[str] = []

    assert alert_decision(state, now=now) == "recovered"
    _notify_about_silence(state, sent.append, now=now)

    assert "снова работает" in sent[0]
    assert "alerted_at" not in state  # второй раз о том же не скажет
    assert alert_decision(state, now=now) is None


def test_the_alarm_text_carries_the_reason(runtime_path: Path) -> None:
    """Сообщение обязано отвечать «почему», иначе человек идёт искать вслепую."""
    from src.cloud_backup.uploader import _notify_about_silence

    now = 100_000.0
    state = {
        "last_success_at": now - 7200,
        "last_error": "AccessDenied при PutObject",
    }
    sent: list[str] = []

    _notify_about_silence(state, sent.append, now=now)

    assert "2 ч" in sent[0]
    assert "AccessDenied" in sent[0]
    assert state["alerted_at"] == now


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
