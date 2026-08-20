"""Сценарные проверки калитки: живые случаи, прогнанные через настоящий цикл.

Появились после инцидентов 2026-08-20 (21:39 — калитка открылась вслед
вошедшему и осталась бы открытой на ночь; 22:08 — вышедшему и развернувшемуся
пришлось выходить из кадра). Разбор показал: точечные юнит-тесты проверяли
каждую защиту порознь, а стреляло взаимодействие — повтор после ошибки платы
попал в закрывшуюся створку, «второй шанс» щёлкал мёртвому треку через грацию
`ACT_WITHIN_SECONDS`.

Здесь каждый сценарий — таймлайн: створка ходит, треки рождаются и умирают,
имя приезжает с опозданием, плата иногда молчит. Симулятор крутит тот же цикл,
что `run()`: опрос → `passes_from_events` → `PriorPass` из базы →
`execute_openings` → `record_passes`, — и сценарий утверждает ровно одно:
когда реле щёлкнуло, а когда обязано было промолчать. Новые правила логики
проверяются добавлением таймлайна, а не набором статистики на живых проходах.

Система координат: секунды от начала сценария, тик = секунда = один опрос.
Створка описывается переходами; `closed_at` симулятор ведёт так же, как
`GateStateWatcher` — по последнему переходу в «закрыта».
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.common.database import GatePass, create_session_factory
from src.common.gate_access_config import GateAccessConfig
from src.gate.access import (
    DECISION_LEAF_OPEN,
    LastOpening,
    PriorPass,
    execute_openings,
    passes_from_events,
    record_passes,
)
from src.gate.gate_state import CLOSED as LEAF_CLOSED
from src.gate.gate_state import OPEN as LEAF_OPEN
from src.gate.gate_state import UNKNOWN as LEAF_UNKNOWN
from src.gate.notify import GateEvent
from src.gate.opener import PulseResult

MOSCOW = ZoneInfo("Europe/Moscow")
# Дневное «сейчас»: ночью окно автооткрытия закрыто, и все сценарии молчали бы
# по посторонней причине.
START = datetime(2026, 8, 20, 12, 0, tzinfo=MOSCOW).timestamp()


@dataclass(frozen=True)
class Track:
    """Один трек Frigate: когда человек в кадре и когда у него появилось имя.

    `named_from` может быть позже `disappears`: имя приезжает и после ухода из
    кадра — событие перечитывается всё окно наблюдения.
    """

    id: str
    appears: float
    disappears: float | None = None
    named_from: float | None = None
    name: str = "Алексей"
    score: float = 0.96


@dataclass(frozen=True)
class LeafMove:
    """Переход створки в указанную секунду. Публикуются только изменения."""

    at: float
    state: str


@dataclass
class Pulse:
    """Что и когда ушло на плату."""

    at: float
    ok: bool


@dataclass
class World:
    """Симулятор цикла службы. Один экземпляр — один сценарий."""

    runtime_path: Path
    tracks: list[Track]
    leaf_moves: list[LeafMove] = field(default_factory=list)
    # Состояние створки до начала сценария: (state, когда закрылась). None —
    # состояние неизвестно (рваная цепочка доставки, свежий рестарт).
    initial_leaf: tuple[str, float | None] | None = (LEAF_CLOSED, -60.0)
    # Тики, в которые плата отвечает битым эхом.
    board_fails_at: frozenset[float] = frozenset()
    duration: int = 60

    def __post_init__(self) -> None:
        self._factory = create_session_factory(self.runtime_path / "catdoor.sqlite3")
        self._config = GateAccessConfig.model_validate(
            {
                "mode": "on",
                "gates": [{"camera": "gate", "relay": 1}],
                "allowed_names": ["Алексей", "Дарья"],
                "cooldown_seconds": 10,
            }
        )
        self.pulses: list[Pulse] = []

    def run(self) -> list[Pulse]:
        opening: LastOpening | None = None
        for tick in range(self.duration):
            opening = self._poll(float(tick), opening)
        return self.pulses

    def decisions(self) -> dict[str, str]:
        """Итог журнала: событие — решение. Для утверждений о причинах."""
        with self._factory() as session:
            return {
                row.frigate_event_id: row.decision for row in session.query(GatePass)
            }

    def _leaf(self, now: float) -> tuple[str, float | None]:
        """Состояние створки и время последнего закрытия — как у наблюдателя."""
        state, closed_at = (
            self.initial_leaf if self.initial_leaf is not None else (LEAF_UNKNOWN, None)
        )
        for move in sorted(self.leaf_moves, key=lambda item: item.at):
            if move.at > now:
                break
            state = move.state
            if move.state == LEAF_CLOSED:
                closed_at = move.at
        return state, closed_at

    def _events(self, now: float) -> list[GateEvent]:
        events = []
        for track in self.tracks:
            if now < track.appears:
                continue
            gone = track.disappears is not None and now >= track.disappears
            named = track.named_from is not None and now >= track.named_from
            events.append(
                GateEvent(
                    id=track.id,
                    camera="gate",
                    label="person",
                    sub_label=track.name if named else None,
                    started_at=START + track.appears,
                    has_snapshot=False,
                    name_score=track.score if named else None,
                    ended_at=START + track.disappears if gone else None,
                )
            )
        return events

    def _poll(self, tick: float, opening: LastOpening | None) -> LastOpening | None:
        """Один опрос — точное зеркало тела цикла `run()` службы."""
        moment = START + tick
        state, closed_at = self._leaf(tick)
        records, opening = passes_from_events(
            self._events(tick),
            cameras={"gate": "Калитка"},
            config=self._config,
            timezone=MOSCOW,
            last_opening=opening,
            now=datetime.fromtimestamp(moment, UTC),
            gate_state=state,
            gate_just_closed=(
                closed_at is not None
                and 0
                <= tick - closed_at
                <= self._config.movement.return_window_seconds
            ),
        )
        if not records:
            return opening
        with self._factory() as session:
            already = {
                row.frigate_event_id: PriorPass(
                    row.decision, row.pulses or 0, row.last_pulse_at, row.pulse_gate_state
                )
                for row in session.query(GatePass).filter(
                    GatePass.frigate_event_id.in_(
                        [record.frigate_event_id for record in records]
                    )
                )
            }
            opener = _Board(self, tick)
            records, pulsed = execute_openings(
                records,
                config=self._config,
                opener=opener,
                already=already,
                gate_state=state,
                gate_closed_at=START + closed_at if closed_at is not None else None,
                now=moment,
            )
            if pulsed is not None:
                opening = pulsed
            record_passes(session, records)
            session.commit()
        return opening


class _Board:
    """Плата: щёлкает или молчит по сценарию, каждый щелчок — в протокол."""

    def __init__(self, world: World, tick: float) -> None:
        self._world = world
        self._tick = tick

    def pulse(self, channel: int, milliseconds: int) -> PulseResult:
        ok = self._tick not in self._world.board_fails_at
        self._world.pulses.append(Pulse(self._tick, ok))
        return (
            PulseResult(True, "ок", milliseconds)
            if ok
            else PulseResult(False, "ответ короче эха", milliseconds)
        )


def _times(pulses: list[Pulse]) -> list[float]:
    return [pulse.at for pulse in pulses]


def test_scenario_a_normal_arrival_opens_once(runtime_path: Path) -> None:
    """Обычный приход: створка закрыта, человек подошёл, узнан — один импульс."""
    world = World(
        runtime_path,
        tracks=[Track("arrive", appears=0, disappears=20, named_from=3)],
    )

    world.run()

    assert _times(world.pulses) == [3.0]
    assert world.decisions() == {"arrive": "opened"}


def test_scenario_the_entry_ghost_stays_silent(runtime_path: Path) -> None:
    """Реплей призрака 21:39–21:40: вошёл, закрыл за собой — ни одного щелчка.

    Реальный таймлайн: створка открылась в 21:39:58 и в ней родился новый трек
    вошедшего; имя приехало на 11-й секунде, трек умер вместе с закрытием
    створки на 18-й, а попытка импульса на 11-й секунде сорвалась на эхе
    платы. До правила `DECISION_LEAF_OPEN` повтор после ошибки догнал калитку
    через 10 секунд — по уже закрытой створке, за которой никого не было;
    замок вытолкнул створку, и она осталась бы открытой на ночь. Теперь трек,
    родившийся в открытой створке, не получает импульса вовсе.
    """
    world = World(
        runtime_path,
        tracks=[Track("ghost", appears=0, disappears=18, named_from=11)],
        leaf_moves=[LeafMove(0, LEAF_OPEN), LeafMove(18, LEAF_CLOSED)],
        board_fails_at=frozenset({11.0}),
        duration=45,
    )

    world.run()

    assert world.pulses == []  # плату не трогали ни разу — даже неудачно
    assert world.decisions() == {"ghost": DECISION_LEAF_OPEN}


def test_scenario_an_entered_person_lingering_in_frame_gets_nothing(
    runtime_path: Path,
) -> None:
    """Вошёл и остался стоять у калитки в кадре — импульса вслед не будет.

    Выбор владельца 2026-08-20: кадр не отличает «стоит снаружи перед
    закрывшейся створкой» от «вошёл и стоит во дворе у калитки», и цена ошибки
    несимметрична — калитка нараспашку на ночь против полуминуты ожидания.
    Поэтому долгое присутствие в кадре после закрытия не аргумент за импульс.
    """
    world = World(
        runtime_path,
        tracks=[Track("linger", appears=0, disappears=None, named_from=5)],
        leaf_moves=[LeafMove(0, LEAF_OPEN), LeafMove(10, LEAF_CLOSED)],
        duration=90,
    )

    world.run()

    assert world.pulses == []
    assert world.decisions() == {"linger": DECISION_LEAF_OPEN}


def test_scenario_exit_and_return_needs_a_fresh_track(runtime_path: Path) -> None:
    """Реплей 22:08–22:09: вышел, развернулся — впустят с нового трека.

    Трек вышедшего родился в открытой им створке, и по правилу стороны он
    неотличим от вошедшего: пока он стоит снаружи в кадре, импульса нет.
    Отойдя от камеры, он убивает трек; возвращение — новый трек при створке,
    закрытой до его рождения, и обычный путь открывает калитку сразу. Это
    осознанная цена правила: полминуты против калитки, открытой вслед
    вошедшему.
    """
    world = World(
        runtime_path,
        tracks=[
            # Вышел: трек и открытая створка появляются в одну секунду, как в
            # живом журнале (22:08:45), имя приезжает на третьей.
            Track("exit", appears=0, disappears=34, named_from=3),
            # Вернулся из-за кадра: свежий трек при давно закрытой створке.
            Track("return", appears=39, disappears=52, named_from=44),
        ],
        leaf_moves=[LeafMove(0, LEAF_OPEN), LeafMove(5, LEAF_CLOSED)],
        duration=60,
    )

    world.run()

    assert _times(world.pulses) == [44.0]
    assert world.decisions() == {"exit": DECISION_LEAF_OPEN, "return": "opened"}


def test_scenario_a_standing_open_leaf_needs_no_pulse(runtime_path: Path) -> None:
    """Створка стоит открытой (разгрузка дров 21:33–21:36) — импульсы молчат.

    Человек проходит в открытую створку сам; импульс в неё бессмыслен — и
    именно такой потраченный впустую импульс 12 августа породил «второй шанс»
    со всеми его призраками. Закрытие створки после ухода человека тоже не
    даёт щелчка: грация `ACT_WITHIN_SECONDS` ещё держит право на импульс, но
    правило стороны видит, что створка ходила во время события.
    """
    world = World(
        runtime_path,
        initial_leaf=(LEAF_OPEN, None),
        tracks=[Track("through", appears=0, disappears=20, named_from=5)],
        leaf_moves=[LeafMove(25, LEAF_CLOSED)],
        duration=45,
    )

    world.run()

    assert world.pulses == []
    assert world.decisions() == {"through": DECISION_LEAF_OPEN}


def test_scenario_a_follower_through_the_open_leaf_gets_no_click(
    runtime_path: Path,
) -> None:
    """Второй человек следом через открытую створку — щелчка вслед нет.

    Реплей 21:35–21:36: первому импульс положен (створка закрыта с рождения
    его трека), он входит, второй проходит за ним в открытую створку. Трек
    второго родился при открытой створке — по правилу стороны он, возможно,
    уже во дворе, и после закрытия створки калитка ему не пикает. До правила
    здесь стрелял «второй шанс» (21:36:16, импульс мёртвому треку).
    """
    world = World(
        runtime_path,
        tracks=[
            Track("first", appears=0, disappears=12, named_from=2),
            Track("follower", appears=6, disappears=30, named_from=8, name="Дарья"),
        ],
        leaf_moves=[LeafMove(5, LEAF_OPEN), LeafMove(12, LEAF_CLOSED)],
        duration=45,
    )

    world.run()

    assert _times(world.pulses) == [2.0]
    assert world.decisions() == {"first": "opened", "follower": DECISION_LEAF_OPEN}


def test_scenario_a_board_error_at_a_closed_leaf_earns_a_retry(
    runtime_path: Path,
) -> None:
    """Честный отказ платы: створка закрыта и не шевелилась — повтор положен.

    Человек стоит перед закрытой калиткой, плата сглючила. Створка закрыта
    непрерывно с рождения трека, правило стороны не возражает, и через
    `ERROR_RETRY_SECONDS` человек получает вторую попытку — те самые
    четырнадцать секунд из 17:04, которые нельзя вернуть.
    """
    world = World(
        runtime_path,
        tracks=[Track("patient", appears=0, disappears=30, named_from=2)],
        board_fails_at=frozenset({2.0}),
        duration=30,
    )

    world.run()

    assert _times(world.pulses) == [2.0, 12.0]
    assert not world.pulses[0].ok and world.pulses[1].ok
    assert world.decisions() == {"patient": "opened"}


def test_scenario_an_unknown_leaf_does_not_lock_the_family_out(
    runtime_path: Path,
) -> None:
    """Состояние створки неизвестно (рестарт, брокер) — приход открывает.

    Правило стороны держится на классификаторе, но его молчание — не повод
    запереть семью: неизвестное состояние оставляет старое поведение. Дыра на
    время рваной цепочки осознана и принята — её страхует повтор после ошибки
    платы, не гоняющийся за закрывшейся створкой.
    """
    world = World(
        runtime_path,
        initial_leaf=None,
        tracks=[Track("blind", appears=0, disappears=20, named_from=3)],
    )

    world.run()

    assert _times(world.pulses) == [3.0]
    assert world.decisions() == {"blind": "opened"}


def test_scenario_a_flicker_at_the_gate_keeps_the_grace(runtime_path: Path) -> None:
    """Мигание детекции вплотную к камере: имя после смерти трека — импульс есть.

    Грация `ACT_WITHIN_SECONDS` существует для человека, который стоит у самой
    калитки и выпадает из детекции, пока решение дозревает. Створка при этом
    закрыта с рождения трека — правило стороны грацию не отнимает.
    """
    world = World(
        runtime_path,
        tracks=[Track("flicker", appears=0, disappears=4, named_from=6)],
        duration=20,
    )

    world.run()

    assert _times(world.pulses) == [6.0]
    assert world.decisions() == {"flicker": "opened"}


def test_scenario_a_late_name_after_the_grace_is_stale(runtime_path: Path) -> None:
    """Имя приехало сильно после ухода — решение опоздало, импульса нет.

    Контроль к грации: человек ушёл из кадра за двадцать секунд до того, как
    дозрело имя. Открытая калитка позади ушедшего не нужна никому.
    """
    world = World(
        runtime_path,
        tracks=[Track("late", appears=0, disappears=4, named_from=25)],
        duration=40,
    )

    world.run()

    assert world.pulses == []
    assert world.decisions() == {"late": "denied_stale"}
