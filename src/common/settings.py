import re
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.common.metrics import DEFAULT_IDENTITY_QUORUM
from src.common.telegram import TELEGRAM_API


class ConfigurationError(RuntimeError):
    """The configured paths cannot be used, so starting up would be pointless."""


# Camera ids end up inside stream URLs, so they stay to characters that survive
# a path segment without escaping.
_CAMERA_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

# Опечатка в имени режима не должна молча означать «выключено»: аппаратное
# декодирование включают ради экономии ядра и потом ищут, куда она делась.
_HWACCEL_MODES = frozenset({"none", "vaapi"})


class CameraConfig(BaseModel):
    """One camera as it can be declared in the environment.

    Only a seed: on first start `src.common.camera_registry` copies these into
    `data/runtime/cameras.json` and everything afterwards edits that file. A
    camera that is moved, re-addressed or renamed must not require a service
    restart, which is exactly what an environment variable would require.
    """

    id: str
    rtsp: str
    rotate_180: bool = False
    title: str = ""

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        if not _CAMERA_ID.match(value):
            raise ValueError(
                "идентификатор камеры может содержать только латиницу в нижнем "
                f"регистре, цифры, дефис и подчёркивание, задано: {value!r}"
            )
        return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CATDOOR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    camera_face_rtsp: str = ""
    camera_side_rtsp: str = ""
    camera_face_rotate_180: bool = True
    camera_side_rotate_180: bool = True
    cameras: list[CameraConfig] = Field(default_factory=list)
    # Источники поиска камер. Файл аренд и ARP-таблица читаются как есть;
    # подсеть задаётся отдельно, потому что обход по портам должен идти по сети
    # камер, а не по той, через которую администратор пришёл на сервер.
    camera_dhcp_leases_path: Path = Path("/var/lib/misc/dnsmasq.leases")
    camera_arp_table_path: Path = Path("/proc/net/arp")
    camera_discovery_subnet: str = ""
    camera_discovery_timeout_seconds: float = Field(default=3.0, gt=0.0, le=15.0)
    # Порог свободного места, ниже которого новые записи не начинаются.
    # Заполненный до конца диск глушит и сами записи, и всю отчётность о них:
    # статус, журнал и базу. Отказаться от новой записи заранее — единственный
    # способ сохранить систему говорящей.
    disk_min_free_gb: float = Field(default=2.0, ge=0.0, le=500.0)
    # Сколько recorder ждёт кадр, прежде чем счесть поток застрявшим и убить
    # ffmpeg. Должно превышать самую медленную настроенную частоту кадров с
    # запасом на переподключение, иначе живой поток будут рвать зря.
    camera_stall_seconds: float = Field(default=30.0, ge=5.0, le=600.0)
    # Превью в админке живёт отдельным процессом ffmpeg на каждую открытую
    # вкладку; без этого предела забытая вкладка с мёртвой камерой держала бы
    # его до перезапуска веб-службы.
    preview_idle_timeout_seconds: float = Field(default=20.0, ge=5.0, le=300.0)
    # Сколько превью держится за оборванную камеру, показывая «нет сигнала»,
    # прежде чем закончить ответ. Всё это время оно переоткрывает источник, так
    # что картинка оживает сама; предел нужен забытой вкладке перед камерой,
    # которая уже не вернётся.
    preview_silence_limit_seconds: float = Field(default=300.0, ge=10.0, le=3600.0)
    # Контур калитки: служба systemd, поднимающая Frigate, и адрес его
    # интерфейса для ссылки с пульта. Пустой адрес — ссылка не показывается.
    # Устройство детектора Frigate. Раньше было зашито `CPU`, и это стоило узлу
    # почти пяти ядер: OpenVINO на процессоре распараллеливает каждый инференс
    # по всем ядрам, а Frigate гоняет детектор по областям движения — около ста
    # раз в секунду при четырёх камерах. Умолчание оставлено прежним: перенос на
    # видеоядро не бесплатен, там уже считает детектор котов, и очередь у
    # видеоядра одна на всех.
    gate_detector_device: str = "CPU"
    gate_service_unit: str = "frigate.service"
    gate_container_name: str = "frigate"
    gate_ui_url: str = ""
    # Конфигурация Frigate собирается из инвентаря камер и пишется примирителем
    # от root. Путь — тот каталог, который смонтирован в контейнер как
    # `/config`: Frigate мигрирует конфигурацию между версиями сам и держит там
    # же свою базу, поэтому монтируется каталог целиком, а не файл.
    gate_config_path: Path = Path("/var/lib/frigate/config/config.yml")
    # Копия того, что примиритель положил в конфигурацию в прошлый раз.
    # Сравнивать нужно именно с ней, а не с рабочим файлом: Frigate переписывает
    # его при миграции, и сравнение с живым файлом после первой же миграции
    # расходилось бы навсегда — примиритель перезапускал бы контур по кругу.
    gate_applied_config_path: Path = Path("/etc/frigate/catdoor-applied.yml")
    # Пароли камер: отдельный файл с отдельными правами, его же читает Docker
    # как env_file.
    gate_env_path: Path = Path("/etc/frigate/frigate.env")
    # База и записи Frigate — только для чтения, ради глубины архива на вкладке
    # регистратора. Записи могут лежать на отдельном томе: недельный архив по
    # движению не помещается на системный диск, и его выносят на внешний.
    gate_database_path: Path = Path("/var/lib/frigate/config/frigate.db")
    gate_recordings_path: Path = Path("/var/lib/frigate/media/recordings")
    data_dir: Path = Path("data")
    database_path: Path | None = None
    ffmpeg_path: str = "ffmpeg"
    preview_fps: int = Field(default=5, ge=1, le=10)
    capture_duration_seconds: int = Field(default=8, ge=3, le=30)
    automatic_capture_enabled: bool = False
    automatic_capture_dry_run: bool = True
    detector_model_path: Path = Path("models/cat-detector.onnx")
    detector_device: str = "CPU"
    # Устройство офлайн-оценки держится отдельно от рабочего намеренно. Во-первых,
    # на узле с живым recorder на видеоядре оценка на том же видеоядре испортит
    # оба числа сразу — и `inference_ms` в журнале, и `mean_detector_inference_ms`
    # в отчёте. Во-вторых, подстановка сюда `detector_device` тихо переключила бы
    # все будущие оценки в тот день, когда recorder переехал на видеоядро, и
    # прежние отчёты стали бы несравнимы задним числом.
    evaluation_detector_device: str = "CPU"
    # Декодирование потока камер медиадвижком. Замер на узле acer: поток дверцы
    # (4 Мп, 30 секунд, то же масштабирование в 960×540) стоил 12.2 с
    # процессорного времени программно и 1.2 с через VA-API. Выключено по
    # умолчанию: на узле без видеоядра путь обязан остаться прежним. Речь
    # только о декодировании — инференс на видеоядре замерен и отброшен.
    recorder_hwaccel: str = "none"
    recorder_hwaccel_device: Path = Path("/dev/dri/renderD128")
    identity_face_model_path: Path | None = None
    identity_side_model_path: Path | None = None
    identity_pretrained_weights_path: Path | None = None
    identity_confidence_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    # Какая доля кадров прохода должна сойтись на одном имени. Прежнее правило
    # «хватит одного кадра» пускало чужих котов в журнал и в уведомление —
    # см. `aggregate_frame_predictions` и разбор 17 августа.
    identity_quorum: float = Field(default=DEFAULT_IDENTITY_QUORUM, ge=0.0, le=1.0)
    detector_face_sample_fps: float = Field(default=8.0, gt=0.0, le=20.0)
    detector_sample_fps: float = Field(default=3.0, gt=0.0, le=20.0)
    ring_segment_seconds: int = Field(default=2, ge=1, le=10)
    ring_retention_seconds: int = Field(default=180, ge=30, le=1800)
    automatic_prebuffer_seconds: int = Field(default=12, ge=0, le=60)
    detection_confidence_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    detection_box_min_confidence: float = Field(default=0.20, ge=0.0, le=1.0)
    detection_confirmation_frames: int = Field(default=2, ge=1, le=20)
    detection_window_frames: int = Field(default=5, ge=1, le=30)
    detection_candidate_timeout_seconds: float = Field(default=3.0, gt=0.0, le=30.0)
    detection_absence_seconds: float = Field(default=3.0, gt=0.0, le=60.0)
    detection_tail_seconds: float = Field(default=10.0, ge=0.0, le=60.0)
    detection_cooldown_seconds: float = Field(default=5.0, ge=0.0, le=300.0)
    # Потолок длительности одного события. Кот, заснувший у дверцы, детектору
    # виден так же уверенно, как идущий, и без потолка одна детекция писала бы
    # диск досуха. Ноль — без лимита.
    detection_max_event_seconds: float = Field(default=60.0, ge=0.0, le=600.0)
    detection_log_max_bytes: int = Field(default=10_000_000, ge=100_000, le=100_000_000)
    detection_log_backup_count: int = Field(default=5, ge=1, le=20)
    # Уведомления в Телеграм. Токен — секрет: он живёт в /etc/catdoor/catdoor.env
    # и не должен попадать ни в журнал, ни в инвентарь камер, ни в снимки
    # состояния. Роли по умолчанию — только дверца: на подходе детектор пока
    # не отличает кота от тени, и телефон звонил бы листвой.
    # Адрес API Телеграма. Меняется на свой обратный прокси там, где провайдер
    # закрывает прямые адреса дата-центров: 2026-08-09 у владельца
    # `web.telegram.org` отвечал, а `149.154.166.110:443` молчал до таймаута —
    # уведомления встали, хотя интернет работал.
    # Реле-контроллер дверцы (Waveshare Modbus POE ETH Relay (B)) в камерной
    # сети. Пустой host — панель дверцы и опрос выключены целиком: контроллер
    # есть не на каждой установке, и админка без него обязана жить как раньше.
    door_controller_host: str = ""
    door_controller_port: int = Field(default=4196, ge=1, le=65535)
    door_reed_input: int = Field(default=1, ge=1, le=8)
    door_lock_relay: int = Field(default=1, ge=1, le=8)
    # «Геркон замкнут 2–3 секунды — дверца закрыта» — требование владельца;
    # берём середину. Разомкнутый или дребезжащий геркон — открыта сразу.
    door_position_stable_seconds: float = Field(default=2.5, ge=0.5, le=60.0)
    telegram_api_base: str = TELEGRAM_API
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    notify_roles: list[str] = Field(default_factory=lambda: ["face"])
    notify_cooldown_seconds: float = Field(default=300.0, ge=0.0, le=86400.0)
    # Порог уверенности для уведомления — выше порога записи намеренно: писать
    # стоит всё сомнительное, а будить человека только уверенным.
    notify_min_confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    # Уведомления контура калитки. Источник — HTTP API самого Frigate на
    # локальном порту: наружу он не смотрит и пароля не спрашивает. Пустой
    # список камер означает «все», пустой список меток — «любые».
    gate_api_url: str = "http://127.0.0.1:5000"
    # Брокер MQTT. Адресов два, и это не дублирование: Frigate живёт в
    # контейнере с мостовой сетью и видит брокер по имени сервиса в compose, а
    # наши службы работают на самом узле и ходят на опубликованный порт
    # loopback. Один адрес на двоих означал бы, что кто-то из них не подключится.
    gate_mqtt_enabled: bool = False
    # Как брокер зовётся из контейнера Frigate.
    gate_mqtt_host: str = "mqtt"
    gate_mqtt_port: int = Field(default=1883, ge=1, le=65535)
    # Как до брокера дотягиваются службы узла.
    mqtt_host: str = "127.0.0.1"
    mqtt_port: int = Field(default=1883, ge=1, le=65535)
    gate_notify_cameras: list[str] = Field(default_factory=list)
    gate_notify_labels: list[str] = Field(default_factory=lambda: ["person"])
    # О своих — с именем, о чужих — отдельным сообщением «неизвестный».
    # `faces_only` возвращает прежнее поведение: молчать обо всех, кого не
    # узнали. Выдержка нужна потому, что Frigate приписывает имя событию не
    # сразу: без неё один гость давал бы два сообщения — сперва «неизвестный»,
    # потом имя. Пауза у чужих своя: знакомых дома ждут, а чужой у калитки
    # может оказаться прохожим.
    gate_notify_faces_only: bool = False
    gate_notify_face_grace_seconds: float = Field(default=20.0, ge=0.0, le=300.0)
    gate_notify_unknown_cooldown_seconds: float = Field(default=600.0, ge=0.0, le=86400.0)
    gate_notify_cooldown_seconds: float = Field(default=120.0, ge=0.0, le=86400.0)
    gate_notify_poll_seconds: float = Field(default=10.0, ge=1.0, le=600.0)
    # Журнал проходов калитки. Опрос втрое чаще уведомлений: сообщению
    # безразлично, придёт оно секундой позже, а решению об открытии — нет,
    # человек за четыре секунды успевает набрать код руками.
    gate_access_poll_seconds: float = Field(default=1.0, ge=0.5, le=60.0)
    # Сколько дней хранить строки журнала и снимки к ним. Ноль — не удалять.
    gate_access_retention_days: int = Field(default=30, ge=0, le=3650)
    # До 2026-08-13 здесь стояло 2: обучение на узле было проверкой конвейера.
    # Теперь это полный цикл, и умолчание — рабочее число эпох макбучного
    # обучения. На процессоре узла это часы, и это нормально: прогон идёт
    # фоном и не трогает ни запись, ни детекцию.
    local_training_epochs: int = Field(default=20, ge=1, le=100)
    web_host: str = "127.0.0.1"
    web_port: int = Field(default=8080, ge=1, le=65535)
    # Часовой пояс, в котором админка показывает время, а Frigate ведёт свой
    # архив. В базе метки хранятся в UTC — так они сравнимы между узлами и
    # переживают перевод часов, — но на экране нужен тот час, который человек
    # видит на своих часах у дверцы.
    display_timezone: str = "Europe/Moscow"

    @field_validator("display_timezone")
    @classmethod
    def _check_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(
                f"неизвестный часовой пояс: {value!r}, ожидается имя из базы "
                "IANA, например Europe/Moscow"
            ) from exc
        return value

    @field_validator("recorder_hwaccel")
    @classmethod
    def _check_hwaccel(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in _HWACCEL_MODES:
            raise ValueError(
                "аппаратное декодирование принимает только "
                f"{' и '.join(sorted(_HWACCEL_MODES))}, задано: {value!r}"
            )
        return normalized

    @property
    def resolved_database_path(self) -> Path:
        return self.database_path or self.data_dir / "catdoor.sqlite3"


def _require_writable_directory(directory: Path) -> None:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfigurationError(f"Не удалось создать каталог {directory}") from exc
    probe = directory / f".catdoor-write-check.{uuid4().hex}"
    try:
        probe.touch()
    except OSError as exc:
        raise ConfigurationError(f"Каталог недоступен для записи: {directory}") from exc
    finally:
        probe.unlink(missing_ok=True)


def validate_runtime_paths(settings: Settings) -> None:
    """Fail fast when the configured paths cannot work.

    The recorder's main loop swallows every exception and retries forever, so an
    unusable data directory would otherwise turn into a silent spin instead of a
    visible failure. Under systemd a relative path also resolves against /, which
    makes the default value unusable for a service.
    """
    if not settings.data_dir.is_absolute():
        raise ConfigurationError(
            f"CATDOOR_DATA_DIR должен быть абсолютным путём, задано: {settings.data_dir}"
        )
    if "\\" in str(settings.detector_model_path):
        raise ConfigurationError(
            "CATDOOR_DETECTOR_MODEL_PATH содержит обратный слэш — "
            f"на Linux это часть имени файла, а не разделитель: {settings.detector_model_path}"
        )
    _require_writable_directory(settings.data_dir)
    _require_writable_directory(settings.resolved_database_path.parent)
