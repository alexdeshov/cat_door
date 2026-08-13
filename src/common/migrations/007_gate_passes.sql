-- Журнал проходов калитки: кто подошёл, узнала ли его система и что она
-- решила. Своя таблица, а не `events`: там кошачьи проходы со своими клипами,
-- метками и обучением, и общего у них — только слово «проход».
--
-- `frigate_event_id` уникален, потому что одно и то же событие перечитывается
-- несколько раз подряд: имя лицу Frigate приписывает не сразу, а через
-- секунды после появления человека. Строка обновляется, а не добавляется.
--
-- `reed_before`/`reed_after` заводятся пустыми осознанно: геркона на калитке
-- нет (решение владельца 2026-08-10), и «открыл» в журнале означает «импульс
-- ушёл», а не «калитка открылась». Добавить столбец потом дороже, чем не
-- заполнять его сейчас.
--
-- `pulse_ms`, `modbus_ok` тоже пока пусты: механика не подключена и в коде её
-- нет вовсе.

CREATE TABLE gate_passes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    frigate_event_id VARCHAR(100) NOT NULL UNIQUE,
    created_at DATETIME NOT NULL,
    camera VARCHAR(100) NOT NULL,
    person_name VARCHAR(100),
    score FLOAT,
    zone VARCHAR(100),
    decision VARCHAR(40) NOT NULL,
    reason TEXT,
    snapshot_path VARCHAR(500),
    factors_json TEXT,
    pulse_ms INTEGER,
    modbus_ok BOOLEAN,
    reed_before BOOLEAN,
    reed_after BOOLEAN
);

CREATE INDEX ix_gate_passes_created_at ON gate_passes(created_at);
CREATE INDEX ix_gate_passes_person_name ON gate_passes(person_name);
CREATE INDEX ix_gate_passes_decision ON gate_passes(decision);
