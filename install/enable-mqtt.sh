#!/usr/bin/env bash
# Поднять брокер MQTT рядом с Frigate и включить публикацию.
#
# Зачем это нужно — в docs/mqtt-broker.md. Коротко: состояние обученного
# классификатора калитки и обновления трека по ходу прохода Frigate отдаёт
# только в MQTT, по HTTP их нет.
#
# Скрипт можно запускать повторно: он проверяет, что уже сделано, и не делает
# этого дважды. Frigate не перезапускается без явного согласия — перезапуск
# прерывает запись обоих контуров.
#
#   sudo ./install/enable-mqtt.sh            поднять и включить
#   sudo ./install/enable-mqtt.sh --disable  выключить публикацию (брокер оставить)

set -euo pipefail

COMPOSE_FILE=${COMPOSE_FILE:-/opt/gate_access/docker-compose.yml}
MOSQUITTO_CONF=${MOSQUITTO_CONF:-/etc/mosquitto/catdoor.conf}
CATDOOR_ENV=${CATDOOR_ENV:-/etc/catdoor/catdoor.env}
FRIGATE_UNIT=${FRIGATE_UNIT:-frigate}
BROKER=${BROKER:-catdoor-mqtt}

say() { printf '\n== %s\n' "$1"; }
fail() { printf '\nОстановился: %s\n' "$1" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || fail "нужен root: sudo $0"
[ -f "$COMPOSE_FILE" ] || fail "не нашёл $COMPOSE_FILE"
[ -f "$CATDOOR_ENV" ] || fail "не нашёл $CATDOOR_ENV"

set_env() {
    # Одна строка настройки, без дублей при повторном запуске.
    sed -i "/^${1}=/d" "$CATDOOR_ENV"
    printf '%s=%s\n' "$1" "$2" >> "$CATDOOR_ENV"
}

apply_config() {
    # Примиритель пересоберёт config.yml из инвентаря и настроек. Это тот же
    # путь, которым конфигурация меняется всегда, — своего у скрипта нет.
    systemctl start catdoor-gate-sync.service || true
    sleep 2
}

if [ "${1:-}" = "--disable" ]; then
    say "Выключаю публикацию в MQTT"
    set_env CATDOOR_GATE_MQTT_ENABLED false
    apply_config
    printf 'Готово. Контейнер брокера оставлен — он никому не мешает.\n'
    printf 'Осталось перезапустить Frigate, когда будет удобно:\n'
    printf '  sudo systemctl restart %s\n' "$FRIGATE_UNIT"
    exit 0
fi

say "Настройка брокера"
mkdir -p "$(dirname "$MOSQUITTO_CONF")"
if [ -f "$MOSQUITTO_CONF" ]; then
    printf 'уже есть: %s\n' "$MOSQUITTO_CONF"
else
    cat > "$MOSQUITTO_CONF" <<'CONF'
listener 1883
# Порт опубликован только на loopback, наружу брокер не смотрит — поэтому без
# паролей. Понадобится открыть его в сеть — пароли становятся обязательными в
# тот же день.
allow_anonymous true
# Сохранять сообщения незачем: нас интересует происходящее сейчас, а не архив.
persistence false
CONF
    printf 'создан: %s\n' "$MOSQUITTO_CONF"
fi

say "Сервис брокера в compose"
if grep -qE '^\s{2}mqtt:' "$COMPOSE_FILE"; then
    printf 'уже есть в %s\n' "$COMPOSE_FILE"
else
    backup="${COMPOSE_FILE}.$(date +%Y%m%d-%H%M%S).bak"
    cp -a "$COMPOSE_FILE" "$backup"
    cat >> "$COMPOSE_FILE" <<COMPOSE

  # Брокер для Frigate и служб узла. Живёт здесь, а не пакетом на узле, потому
  # что Frigate работает в мостовой сети: 127.0.0.1 внутри контейнера — это он
  # сам, и брокера на узле он бы не увидел.
  mqtt:
    container_name: ${BROKER}
    image: eclipse-mosquitto:2
    restart: unless-stopped
    volumes:
      - ${MOSQUITTO_CONF}:/mosquitto/config/mosquitto.conf:ro
    ports:
      # Только loopback: службы узла ходят сюда, снаружи порт не виден.
      - "127.0.0.1:1883:1883"
COMPOSE
    printf 'добавлен, прежний файл сохранён: %s\n' "$backup"
fi

say "Поднимаю брокер"
# Только сервис брокера: Frigate при этом не перезапускается.
(cd "$(dirname "$COMPOSE_FILE")" && docker compose -f "$(basename "$COMPOSE_FILE")" up -d mqtt)

say "Проверяю, что брокер отвечает"
for attempt in 1 2 3 4 5; do
    if docker exec "$BROKER" mosquitto_pub -h 127.0.0.1 -t catdoor/ping -m ok 2>/dev/null; then
        printf 'брокер принимает подключения\n'
        break
    fi
    [ "$attempt" = 5 ] && fail "брокер не отвечает: docker logs $BROKER"
    sleep 2
done
if timeout 3 bash -c "cat < /dev/null > /dev/tcp/127.0.0.1/1883" 2>/dev/null; then
    printf 'порт 127.0.0.1:1883 доступен службам узла\n'
else
    fail "порт 1883 на loopback закрыт — проверьте ports в $COMPOSE_FILE"
fi

say "Включаю публикацию в настройках CatDoor"
set_env CATDOOR_GATE_MQTT_ENABLED true
apply_config

say "Что получилось в конфигурации Frigate"
sed -n '/^mqtt:/,/^[a-z]/p' /var/lib/frigate/config/config.yml | head -12

say "Остался перезапуск Frigate"
cat <<'WARN'
Он прерывает запись обоих контуров — кошачьего и регистратора — на полминуты.
До перезапуска Frigate брокер не трогает: раздел mqtt он прочитает только при
старте.
WARN
read -r -p "Перезапустить Frigate сейчас? [y/N] " answer
if [ "${answer:-N}" = "y" ] || [ "${answer:-N}" = "Y" ]; then
    systemctl restart "$FRIGATE_UNIT"
    sleep 10
    systemctl is-active "$FRIGATE_UNIT"
    say "Подключился ли Frigate к брокеру"
    docker logs --since 2m frigate 2>&1 | grep -iE "mqtt" | tail -5 || printf 'о MQTT в логе пока тихо\n'
else
    printf '\nПерезапустите, когда будет удобно:\n  sudo systemctl restart %s\n' "$FRIGATE_UNIT"
fi

say "Откат"
printf 'sudo %s --disable  — вернуть enabled: false и перезапустить Frigate\n' "$0"
