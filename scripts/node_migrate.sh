#!/bin/sh
# Перенос состояния CatDoor со старого узла на новый.
#
# Запускается с третьей машины (мака), у которой есть SSH к обоим узлам:
#
#   ./scripts/node_migrate.sh alex@192.168.2.152 alex@<адрес нового>
#
# Что делает:
#   1. На старом узле останавливает записывающие службы, собирает архив
#      состояния и запускает службы обратно — старый узел продолжает работать
#      до самого переключения.
#   2. Перевозит архив через эту машину (узлы друг про друга не знают).
#   3. На новом узле раскладывает данные по местам, а машинно-зависимые
#      конфигурации (сеть, dnsmasq, chrony, ssh) складывает в
#      /root/catdoor-migrate-staged — их надо просмотреть и применить руками:
#      имена интерфейсов и адреса на новом железе другие.
#
# Что едет в архив:
#   по местам: база и видео проходов (вся разметка), наборы для обучения,
#     инвентарь камер и прочие runtime-файлы, /etc/catdoor/catdoor.env,
#     /opt/gate_access, а также состояние контура калитки целиком —
#     /var/lib/frigate (библиотека лиц, frigate.db, записи) и /etc/frigate;
#   в staged: netplan, dnsmasq (включая аренды камер), chrony, ~/.ssh,
#     снимок правил ufw.
#   не едет (пересоздаётся само): кольцевой буфер, .venv, каталог тестового
#     запуска. Файл конфигурации Frigate приезжает, но его в любом случае
#     пересобирает примиритель из инвентаря камер.
#
# Данные Frigate едут, а не пересоздаются: библиотека лиц и обученная модель
# распознавания — та же накопленная разметка, что и проходы котов, и повторить
# её нечем. Ради согласованного снимка frigate.db контейнер останавливается на
# время сборки, а примиритель придерживается, чтобы не поднял его посреди tar.
#
# Обе стороны спросят пароль sudo — это ожидаемо. Скрипты фаз передаются
# файлами, а не через stdin: sudo читает пароль с терминала, и занимать stdin
# нельзя.

set -eu

OLD=${1:-}
NEW=${2:-}
ARCHIVE=/tmp/catdoor-migrate.tar.gz

usage() {
    echo "использование: $0 <пользователь@старый-узел> <пользователь@новый-узел>" >&2
    exit 2
}

fail() {
    echo "перенос: $1" >&2
    exit 1
}

[ -n "$OLD" ] && [ -n "$NEW" ] || usage
[ "$OLD" != "$NEW" ] || fail "старый и новый узлы совпадают"

workdir=$(mktemp -d)
trap 'rm -rf "$workdir"' EXIT

echo "== Проверка доступа =="
ssh -o ConnectTimeout=10 -o BatchMode=yes "$OLD" true \
    || fail "нет доступа по ключу к старому узлу $OLD"
ssh -o ConnectTimeout=10 -o BatchMode=yes "$NEW" true \
    || fail "нет доступа по ключу к новому узлу $NEW"

# Защита от перепутанных аргументов: узел с работающим recorder — старый.
# Новому узлу здесь работать не с чего: данных на нём ещё нет.
if ssh -o BatchMode=yes "$NEW" 'systemctl is-active --quiet catdoor-recorder' 2>/dev/null; then
    fail "на «новом» узле работает catdoor-recorder — аргументы перепутаны?"
fi
if ssh -o BatchMode=yes "$NEW" 'test -f /var/lib/catdoor/catdoor.sqlite3' 2>/dev/null; then
    fail "на «новом» узле уже есть база — отказываюсь затирать. Уберите её сами, если это осознанно"
fi

cat > "$workdir/phase-old.sh" <<'OLDSCRIPT'
set -eu
staged=/tmp/catdoor-migrate-staged
rm -rf "$staged" /tmp/catdoor-migrate.tar.gz
mkdir -p "$staged"

# Машинно-зависимое — только в staged, на новом узле это нельзя класть в /etc
# вслепую: имена интерфейсов и адреса другие.
cp -a /etc/netplan "$staged/netplan" 2>/dev/null || true
mkdir -p "$staged/dnsmasq.d"
cp -a /etc/dnsmasq.d/. "$staged/dnsmasq.d/" 2>/dev/null || true
cp -a /etc/default/dnsmasq "$staged/dnsmasq.default" 2>/dev/null || true
mkdir -p "$staged/chrony"
cp -a /etc/chrony/conf.d/. "$staged/chrony/" 2>/dev/null || true
home=$(getent passwd "${SUDO_USER:-root}" | cut -d: -f6)
cp -a "$home/.ssh" "$staged/ssh" 2>/dev/null || true
ufw status verbose > "$staged/ufw-status.txt" 2>/dev/null || true

# Базы копируются без пишущих служб: остановка на время tar — цена
# согласованного снимка. Дальше службы возвращаются, узел живёт как жил.
#
# Примиритель придерживается отдельно: он следит за состоянием контуров и
# поднял бы Frigate обратно посреди сборки, а снимок frigate.db на живом
# контейнере — это база с недописанным WAL.
gate_units=""
for unit in catdoor-gate-sync.path catdoor-gate-sync.timer; do
    if systemctl is-active --quiet "$unit"; then
        gate_units="$gate_units $unit"
    fi
done
frigate_was_up=no
if systemctl is-active --quiet frigate; then
    frigate_was_up=yes
fi

restore() {
    systemctl start catdoor-recorder catdoor-web
    if [ "$frigate_was_up" = yes ]; then
        systemctl start frigate
    fi
    if [ -n "$gate_units" ]; then
        systemctl start $gate_units
    fi
}
trap restore EXIT

if [ -n "$gate_units" ]; then
    systemctl stop $gate_units
fi
if [ "$frigate_was_up" = yes ]; then
    systemctl stop frigate
fi
systemctl stop catdoor-recorder catdoor-web

# Пути перечисляются по одному и только существующие: контур калитки на узле
# может быть не развёрнут вовсе, а tar на отсутствующем пути валит всю сборку.
set -- /etc/catdoor/catdoor.env
for path in /var/lib/catdoor/catdoor.sqlite3 /var/lib/catdoor/catdoor.sqlite3-wal \
            /var/lib/catdoor/catdoor.sqlite3-shm /var/lib/catdoor/events \
            /var/lib/catdoor/runtime /var/lib/catdoor/training \
            /var/lib/catdoor/exports /var/lib/catdoor/evaluations \
            /opt/gate_access /etc/frigate /var/lib/frigate; do
    if [ -e "$path" ]; then
        set -- "$@" "$path"
    fi
done

tar czf /tmp/catdoor-migrate.tar.gz \
    --exclude=/var/lib/catdoor/runtime/ring \
    --exclude=/var/lib/catdoor/test-runtime \
    "$@" \
    "$staged"

chown "${SUDO_USER:-root}" /tmp/catdoor-migrate.tar.gz
chmod 600 /tmp/catdoor-migrate.tar.gz
ls -lh /tmp/catdoor-migrate.tar.gz
OLDSCRIPT

cat > "$workdir/phase-new.sh" <<'NEWSCRIPT'
set -eu
# Служебный пользователь нужен до распаковки: права на данные назначаются по
# имени, а не по числовому uid со старого узла — там он мог быть другим.
if ! id catdoor >/dev/null 2>&1; then
    useradd --system --no-create-home --shell /usr/sbin/nologin catdoor
fi

tar xzf /tmp/catdoor-migrate.tar.gz -C /

chown -R catdoor:catdoor /var/lib/catdoor
chmod 750 /var/lib/catdoor
chown root:catdoor /etc/catdoor/catdoor.env
chmod 640 /etc/catdoor/catdoor.env

# Состояние Frigate остаётся за root: контейнер работает от него, а в /etc
# лежит файл с паролем камер.
if [ -d /var/lib/frigate ]; then
    chown -R root:root /var/lib/frigate
fi
if [ -d /etc/frigate ]; then
    chown -R root:root /etc/frigate
    chmod 750 /etc/frigate
    chmod 640 /etc/frigate/frigate.env 2>/dev/null || true
fi

rm -rf /root/catdoor-migrate-staged
mv /tmp/catdoor-migrate-staged /root/catdoor-migrate-staged
rm -f /tmp/catdoor-migrate.tar.gz /tmp/catdoor-phase.sh

echo
echo "Данные разложены. Дальше по docs/deployment-ubuntu.md, раздел «Переезд»:"
echo "  1. git clone cat_door, .venv, sudo ./install/install.sh"
echo "  2. Docker и юнит frigate.service — /var/lib/frigate и /etc/frigate уже"
echo "     на местах, автозапуск Frigate не включать: контур включает админка"
echo "  3. /root/catdoor-migrate-staged — netplan/dnsmasq/chrony/ssh/ufw:"
echo "     просмотреть, поправить имена интерфейсов, применить руками"
echo "  4. systemctl start catdoor-recorder catdoor-web"
echo "  5. Проверка перезагрузкой — как требует дока"
NEWSCRIPT

echo "== Старый узел: сбор архива (службы записи остановятся на время сборки) =="
scp -q "$workdir/phase-old.sh" "$OLD:/tmp/catdoor-phase.sh"
ssh -t "$OLD" 'sudo sh /tmp/catdoor-phase.sh && rm -f /tmp/catdoor-phase.sh'

# Место проверяется по готовому архиву, а не заранее: до sudo размер данных
# с узла не прочитать, а видео в архиве почти не жмётся — распакованное
# займёт примерно столько же. Двойной запас покрывает архив рядом с данными.
size=$(ssh -o BatchMode=yes "$OLD" "stat -c %s $ARCHIVE" 2>/dev/null || echo 0)
need=$(( size / 1024 / 1024 * 2 + 512 ))
free=$(ssh -o BatchMode=yes "$NEW" 'df -m /var/lib 2>/dev/null | awk "NR==2 {print \$4}"')
[ "${free:-0}" -gt "$need" ] \
    || fail "на новом узле мало места: нужно ~${need} МБ, свободно ${free:-?} МБ"
echo "архив $(( size / 1024 / 1024 )) МБ, на новом узле свободно ${free} МБ"

echo "== Перевозка через эту машину =="
# Архив идёт трубой, а не через диск этой машины: видео проходов — десятки
# гигабайт, и складывать их на ноутбук ради пересылки незачем. Узлы по-прежнему
# друг про друга не знают, весь трафик идёт через отсюда открытые соединения.
ssh -o BatchMode=yes "$OLD" "cat $ARCHIVE" \
    | ssh -o BatchMode=yes "$NEW" "cat > /tmp/catdoor-migrate.tar.gz"
new_size=$(ssh -o BatchMode=yes "$NEW" 'stat -c %s /tmp/catdoor-migrate.tar.gz')
[ "$new_size" = "$size" ] \
    || fail "архив доехал не целиком: было $size байт, стало $new_size"
ssh -o BatchMode=yes "$OLD" "rm -f $ARCHIVE"

echo "== Новый узел: раскладка =="
scp -q "$workdir/phase-new.sh" "$NEW:/tmp/catdoor-phase.sh"
ssh -t "$NEW" 'sudo sh /tmp/catdoor-phase.sh'

echo "== Готово =="
echo "Старый узел работает дальше; новый ждёт шагов из подсказки выше."
