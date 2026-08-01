#!/bin/sh
# Install the CatDoor systemd units on an Ubuntu Server runtime host.
#
# Safe to run repeatedly: an existing configuration file is never overwritten,
# and the services are enabled but not started, so an upgrade cannot restart a
# recorder in the middle of an event by surprise.

set -eu

REPO_DIR=${REPO_DIR:-/opt/catdoor}
DATA_DIR=${DATA_DIR:-/var/lib/catdoor}
CONFIG_DIR=/etc/catdoor
CONFIG_FILE="$CONFIG_DIR/catdoor.env"
UNIT_DIR=/etc/systemd/system
SERVICE_USER=catdoor
MINIMUM_SYSTEMD=254

fail() {
    echo "install: $1" >&2
    exit 1
}

[ "$(id -u)" -eq 0 ] || fail "нужны права root: sudo ./install/install.sh"

# RestartSteps and RestartMaxDelaySec, which give the exponential backoff in the
# units, landed in systemd 254. Ubuntu 24.04 carries 255.
systemd_version=$(systemctl --version | head -1 | awk '{print $2}')
case "$systemd_version" in
    ''|*[!0-9]*) fail "не удалось определить версию systemd" ;;
esac
[ "$systemd_version" -ge "$MINIMUM_SYSTEMD" ] || fail \
    "нужен systemd >= $MINIMUM_SYSTEMD, установлен $systemd_version"

[ -x "$REPO_DIR/.venv/bin/python" ] || fail \
    "нет виртуального окружения $REPO_DIR/.venv — сначала python3 -m venv .venv"

if ! id "$SERVICE_USER" >/dev/null 2>&1; then
    # No home and no shell: this account exists to own files and run two
    # services, never to log in.
    useradd --system --no-create-home --shell /usr/sbin/nologin "$SERVICE_USER"
    echo "создан системный пользователь $SERVICE_USER"
fi

mkdir -p "$DATA_DIR"
chown "$SERVICE_USER:$SERVICE_USER" "$DATA_DIR"
chmod 750 "$DATA_DIR"

mkdir -p "$CONFIG_DIR"
if [ -f "$CONFIG_FILE" ]; then
    echo "конфигурация $CONFIG_FILE оставлена без изменений"
else
    cp "$REPO_DIR/.env.example" "$CONFIG_FILE"
    echo "создан $CONFIG_FILE — заполните RTSP-адреса камер"
fi
# The file carries camera credentials, so it stays unreadable to everyone but
# root and the service account.
chown "root:$SERVICE_USER" "$CONFIG_FILE"
chmod 640 "$CONFIG_FILE"

for unit in catdoor-recorder.service catdoor-web.service; do
    install -m 644 "$REPO_DIR/install/$unit" "$UNIT_DIR/$unit"
done

systemctl daemon-reload
systemctl enable catdoor-recorder.service catdoor-web.service >/dev/null

cat <<'NEXT'

Юниты установлены и включены, но не запущены.

  1. Заполните камеры и пути:   sudo nano /etc/catdoor/catdoor.env
  2. Запустите:                 sudo systemctl start catdoor-recorder catdoor-web
  3. Проверьте:                 systemctl status catdoor-recorder --no-pager
                                journalctl -u catdoor-recorder -n 30

Признак рабочего sd_notify — состояние active (running). Зависание в
activating (start) означает, что READY=1 не дошёл.
NEXT
