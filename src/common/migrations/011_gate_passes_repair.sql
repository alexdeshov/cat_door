-- Восстановить столбцы журнала калитки, недобавленные обрывом питания.
--
-- Миграции 009 и 010 состоят из нескольких `ALTER` подряд. До 2026-08-12
-- команды выполнялись скриптом целиком: обрыв между ними оставлял часть
-- столбцов недобавленными, а повтор упирался в «duplicate column» на первой же
-- команде — это считалось признаком сделанной работы, и остальные команды не
-- выполнялись уже никогда. Версия при этом помечалась применённой, а служба
-- падала на «no such column» при каждой записи.
--
-- Теперь команды выполняются по одной, и «уже применено» решается для каждой
-- отдельно. Эта миграция досоздаёт всё, чего могло не хватить: на здоровой базе
-- каждая команда упрётся в «duplicate column» и будет пропущена.

ALTER TABLE gate_passes ADD COLUMN approach VARCHAR(20);
ALTER TABLE gate_passes ADD COLUMN first_box_bottom FLOAT;
ALTER TABLE gate_passes ADD COLUMN gate_state_before VARCHAR(20);
ALTER TABLE gate_passes ADD COLUMN gate_state_after VARCHAR(20);
ALTER TABLE gate_passes ADD COLUMN movement VARCHAR(20);
ALTER TABLE gate_passes ADD COLUMN area_ratio FLOAT;
