#!/bin/sh
# Резервная копия на внешний диск.
#
# Что и зачем копируется:
#   база проходов   — вся разметка, восстановить её нечем;
#   events/         — видео проходов, из них режутся кропы для обучения;
#   runtime/        — инвентарь камер с паролями и настройки детекции;
#   база Frigate    — события калитки и привязка лиц;
#   clips/faces/    — библиотека лиц, накоплена руками;
#   recordings/     — записи старше недели: Frigate сам сносит их по сроку
#                     хранения (14 дней), и без копии они исчезают навсегда.
#
# Записи копируются, а не переносятся. Перенос освободил бы место, но в базе
# Frigate остались бы строки, указывающие на исчезнувшие файлы, и
# воспроизведение за дни между порогом копирования и сроком хранения сломалось
# бы прямо в его интерфейсе.
#
# Запускается от root таймером `catdoor-backup.timer`; вручную — просто
# `sudo /opt/catdoor/scripts/backup_to_archive.sh`.

set -eu

# Каталог самого скрипта: рядом лежит подрезчик архива.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

ARCHIVE=${CATDOOR_ARCHIVE_DIR:-/srv/archive}
DATA=${CATDOOR_DATA_DIR:-/var/lib/catdoor}
FRIGATE=${CATDOOR_FRIGATE_DIR:-/var/lib/frigate}
# Ниже этого возраста записи Frigate ещё живы у него самого и копировать их
# незачем: он их отдаёт из своего интерфейса и сам же чистит по сроку.
RECORDINGS_AGE_DAYS=${CATDOOR_ARCHIVE_RECORDINGS_AGE_DAYS:-7}
# Снимки базы датированные, а не один перезаписываемый файл: копия, которая
# перезаписывается поверх, повторяет за оригиналом и порчу, и случайное
# удаление разметки. Место позволяет — база весит сотни килобайт.
KEEP_DB_SNAPSHOTS=${CATDOOR_ARCHIVE_DB_SNAPSHOTS:-30}
# Сколько суток держать клипы проходов в архиве. Ноль — не подрезать вовсе, и
# это умолчание: зеркало событий росло без предела с самого начала (`rsync` без
# `--delete` и без отсечки), и включать удаление само по себе, обновлением, —
# последнее, что здесь можно делать.
#
# Подрезка **никогда не трогает проход, которого нет в облаке**: она сверяется
# с журналом залитого у `catdoor-cloud-backup-cats`. Нет журнала или он не
# читается — не удаляется ничего. Клипы проходов существуют в единственном
# экземпляре (камеры контура котов Frigate не пишет), и «удалили по возрасту, а
# копия не уехала» — это потеря навсегда.
EVENTS_AGE_DAYS=${CATDOOR_ARCHIVE_EVENTS_AGE_DAYS:-0}
CATS_STATE=${CATDOOR_CATS_BACKUP_STATUS:-$DATA/runtime/cats-cloud-backup-state.json}

# Отметка о копии для страницы состояния. Лежит **на внутреннем диске**, а не в
# архиве, и это главное в ней: чаще всего копия не проходит из-за переполненного
# архивного диска, и записать отчёт о такой неудаче было бы некуда.
STATUS=${CATDOOR_BACKUP_STATUS:-$DATA/runtime/backup-status.json}
reported=0

# Отчёт пишется и при удаче, и при провале, и при обрыве посреди работы — иначе
# «копия не прошла» и «копия давно не запускалась» на странице выглядели бы
# одинаково, а ведут они к разным действиям.
report() {
    reported=1
    free_gb=$(df -Pk "$ARCHIVE" 2>/dev/null | awk 'NR==2 {print int($4 / 1048576)}')
    mkdir -p "$(dirname "$STATUS")" 2>/dev/null || return 0
    printf '{"finished_at": %s, "ok": %s, "detail": "%s", "free_gb": %s}\n' \
        "$(date +%s)" "$1" "$2" "${free_gb:-0}" > "$STATUS.tmp" 2>/dev/null || return 0
    mv -f "$STATUS.tmp" "$STATUS" 2>/dev/null || true
}

say() { echo "архив: $1"; }
fail() { report false "$1"; echo "архив: $1" >&2; exit 1; }
# Обрыв на полпути — тоже неудача: копия, прерванная выключением питания,
# оставила бы отчёт об успехе прошлого раза и выглядела бы свежей.
trap 'report false "копирование прервано"; exit 130' INT TERM
trap 'code=$?; [ "$code" = 0 ] || [ "$reported" = 1 ] || report false "оборвалось с кодом $code"' EXIT

# Внешний диск объявлен в fstab с `nofail`, то есть его может не быть. Писать
# в этом случае некуда: без проверки копия ушла бы в каталог-заглушку на корне
# и незаметно забила бы системный том — ровно та поломка, ради которой данные
# с корня и уносили.
mountpoint -q "$ARCHIVE" || fail "$ARCHIVE не смонтирован, копировать некуда"
[ -d "$DATA" ] || fail "нет каталога данных $DATA"

root="$ARCHIVE/catdoor"
stamp=$(date +%Y-%m-%d)
mkdir -p "$root/db" "$root/runtime" "$root/events" \
         "$root/frigate/db" "$root/frigate/faces" "$root/frigate/recordings"

# --- базы ---------------------------------------------------------------
# Копируются не файлом, а через API резервного копирования SQLite: recorder и
# Frigate пишут прямо сейчас, и `cp` дал бы базу с оборванной транзакцией и
# отставшим WAL. Открытие исходника только на чтение — чтобы копия не смогла
# ничего изменить в рабочей базе даже при ошибке.
snapshot_db() {
    src=$1
    dst=$2
    [ -f "$src" ] || return 0
    python3 - "$src" "$dst" <<'PY'
import sqlite3
import sys

src, dst = sys.argv[1], sys.argv[2]
source = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
target = sqlite3.connect(dst)
with target:
    source.backup(target)
target.close()
source.close()
PY
    chmod 600 "$dst"
}

snapshot_db "$DATA/catdoor.sqlite3" "$root/db/catdoor-$stamp.sqlite3"
say "снимок базы проходов: $(du -h "$root/db/catdoor-$stamp.sqlite3" | cut -f1)"

if [ -f "$FRIGATE/config/frigate.db" ]; then
    snapshot_db "$FRIGATE/config/frigate.db" "$root/frigate/db/frigate-$stamp.sqlite3"
    say "снимок базы Frigate: $(du -h "$root/frigate/db/frigate-$stamp.sqlite3" | cut -f1)"
fi

prune_snapshots() {
    dir=$1
    # Лишние снимки убираются по имени, а не по времени файла: имя содержит
    # дату снятия, а mtime сбивается любым касанием каталога.
    ls -1 "$dir" 2>/dev/null | sort | head -n -"$KEEP_DB_SNAPSHOTS" | while read -r old; do
        rm -f "$dir/$old"
    done
}
prune_snapshots "$root/db"
prune_snapshots "$root/frigate/db"

# --- видео проходов и инвентарь -----------------------------------------
# Без `--delete`: это резервная копия, а не зеркало. Удаление прохода в
# админке не должно стирать его из архива — иначе одна ошибка в разметке
# уносит обучающий материал безвозвратно. Архив при необходимости чистится
# руками.
rsync -a --info=stats2 "$DATA/events/" "$root/events/" | grep -E "transferred|Total" | head -2
rsync -a --exclude=ring "$DATA/runtime/" "$root/runtime/"
say "проходов в архиве: $(find "$root/events" -mindepth 1 -maxdepth 1 -type d | wc -l)"

# Подрезка зеркала событий по возрасту. Комментарий выше объясняет, почему у
# `rsync` нет `--delete` и почему архив «чистится руками»: руки до него не
# дошли ни разу, и к 30 августа зеркало весило 84 ГБ на диске, который Frigate
# держит полным. Это та самая ручная чистка, только с двумя предохранителями:
# возрастом и подтверждением, что проход уже в облаке.
if [ "$EVENTS_AGE_DAYS" -gt 0 ]; then
    python3 "$SCRIPT_DIR/prune_archived_events.py" \
        --events-dir "$root/events" \
        --database "$DATA/catdoor.sqlite3" \
        --state "$CATS_STATE" \
        --days "$EVENTS_AGE_DAYS"
else
    say "подрезка событий выключена (CATDOOR_ARCHIVE_EVENTS_AGE_DAYS=0)"
fi

# --- Frigate: лица и старые записи --------------------------------------
if [ -d "$FRIGATE/media/clips/faces" ]; then
    rsync -a "$FRIGATE/media/clips/faces/" "$root/frigate/faces/"
    say "снимков лиц: $(find "$root/frigate/faces" -type f | wc -l)"
fi

# Записи могут уже лежать на этом же диске: неделя записи по движению не
# помещается на внутренний том, и media Frigate переносят на архивный. Копия
# внутри того же тома не защищает ни от чего и вдвое ускоряет его заполнение —
# а заполнение архивного тома означает, что Frigate начнёт вытеснять свои же
# записи раньше срока.
recordings_source_device=$(stat -c %d "$FRIGATE/media/recordings" 2>/dev/null || echo x)
archive_device=$(stat -c %d "$ARCHIVE" 2>/dev/null || echo y)

if [ "$recordings_source_device" = "$archive_device" ]; then
    say "записи Frigate уже на архивном диске — копировать некуда и незачем"
elif [ -d "$FRIGATE/media/recordings" ]; then
    cutoff=$(date -d "$RECORDINGS_AGE_DAYS days ago" +%Y-%m-%d)
    copied=0
    # Frigate раскладывает записи как recordings/ГГГГ-ММ-ДД/ЧЧ/камера/*.mp4,
    # поэтому возраст читается прямо из имени каталога. Это надёжнее, чем
    # mtime: копирование и перенос тома его меняют, а дата в имени — нет.
    for day in "$FRIGATE"/media/recordings/*/; do
        [ -d "$day" ] || continue
        name=$(basename "$day")
        [ "$name" \< "$cutoff" ] || continue
        rsync -a "$day" "$root/frigate/recordings/$name/"
        copied=$((copied + 1))
    done
    say "суток записей скопировано: $copied (старше $cutoff)"
fi

occupied=$(du -sh "$root" | cut -f1)
free_human=$(df -h "$ARCHIVE" | awk 'NR==2 {print $4}')
say "занято в архиве: $occupied, свободно $free_human"
report true "копия прошла: занято $occupied, свободно $free_human"
