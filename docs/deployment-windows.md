# Развёртывание на Windows

> **Документ описывает замороженное состояние.** Runtime переносится на Ubuntu
> Server; `main` под Windows больше не работает. Всё изложенное ниже верно для
> тега `windows-runtime-final`, на котором живёт работающий моноблок.
> Раздел про OpenVINO GPU и ошибки CISA сохраняется намеренно: на него
> ссылаются `docs/expansion-plan.md` и промты Codex, а вывод про графику Gen9
> нужно перепроверить уже на Linux-драйвере.

Целевая машина:

- Intel Core i3-8109U;
- Intel Iris Plus Graphics 655;
- 16 ГБ RAM;
- SSD 512 ГБ.

## Ограничения

Постоянно анализируются субпотоки с частотой около 3–8 FPS. Полноразмерные кадры используются только во время события.

## Runtime

Предпочтительно OpenVINO. ONNX Runtime используется как резервный вариант.

## Явный запуск OpenVINO на Iris Plus 655

Сначала проверить, что установленный OpenVINO и драйвер Intel видят iGPU:

```powershell
.\.venv\Scripts\python.exe -c "import openvino as ov; print(ov.Core().available_devices)"
```

В списке должно присутствовать `GPU`. Само наличие устройства ещё не
подтверждает, что драйвер корректно выполнит модель. Только после успешного
smoke-теста явно задать в локальном `.env`:

```dotenv
CATDOOR_DETECTOR_DEVICE=GPU
```

После перезапуска recorder компилирует детектор именно для `GPU` с
`PERFORMANCE_HINT=LATENCY` и `INFERENCE_PRECISION_HINT=f16`. Проверить без
записи событий:

```powershell
.\.venv\Scripts\python.exe -m src.recorder.app --probe-seconds 10
Invoke-RestMethod http://127.0.0.1:8080/api/recorder/status |
  Select-Object detector_device_requested, detector_device_actual
```

При успешном запуске ожидается: запрошено `GPU`, фактически используется
`GPU.0`. Эти поля также видны в админке. Recorder не переключается на CPU
автоматически: если GPU недоступен или компиляция не удалась, статус переходит в
ошибку. Для возврата на CPU нужно явно установить
`CATDOOR_DETECTOR_DEVICE=CPU` и перезапустить recorder.

Проверка 2026-07-30 на текущей Iris Plus 655 обнаружила `CPU` и `GPU`, а
компиляция FP16 сообщила `GPU.0`, однако первый реальный инференс не завершился
за 120 секунд и драйвер создал `kernel.errors.txt` с ошибками CISA. Поэтому
рабочая конфигурация должна пока оставаться
`CATDOOR_DETECTOR_DEVICE=CPU`. GPU нельзя считать принятым до обновления
драйвера/OpenVINO и успешного повторного bounded smoke-теста с нормальной
задержкой. Служебный `kernel.errors.txt` в Git не добавляется.

## Локальный фоновый запуск

Web и recorder можно запускать как отделённые процессы:

```powershell
python -m scripts.start_web
python -m scripts.start_recorder --record
```

Логи сохраняются только локально в `CatDoorData/runtime/`.

Web-превью использует локальные кадры recorder и не открывает дополнительные
RTSP-подключения к камерам.

## Автозапуск при входе в Windows

Автозапуск устанавливается в стандартный раздел `Run` текущего пользователя и
не требует прав администратора:

```powershell
.\.venv\Scripts\python.exe -m scripts.manage_autostart install
```

После следующего входа в Windows отдельно запустятся web и recorder. Когда web
начнёт отвечать, локальная админка автоматически откроется в Google Chrome.
Recorder стартует в режиме локальной записи подтверждённых событий. Этот режим
сохраняет только видео для последующей разметки и не управляет реальной дверью.
Дочерние процессы FFmpeg запускаются с Windows-флагом `CREATE_NO_WINDOW`,
поэтому старт и штатные переподключения камер не открывают консольные окна.
Та же команда создаёт задачу `CatDoorWatchdog` планировщика текущего
пользователя: она запускает проверку компонентов каждые 5 минут с ограниченным
уровнем прав.

Точка входа автозапуска сама переходит в корень проекта перед загрузкой `.env`,
поэтому запуск не зависит от рабочей папки, унаследованной от Windows.
Начало и результат каждого запуска записываются в
`CatDoorData/runtime/autostart.jsonl`. Если основной runtime-каталог нельзя
определить или открыть, используется резервный журнал
`%LOCALAPPDATA%\CatDoor\runtime\autostart.jsonl`. Сообщения исключений в этот
журнал не включаются, чтобы случайно не сохранить RTSP-пароль или другой секрет.

Проверить или удалить автозапуск:

```powershell
.\.venv\Scripts\python.exe -m scripts.manage_autostart status
.\.venv\Scripts\python.exe -m scripts.manage_autostart uninstall
```

Компоненты запускаются при входе текущего пользователя, а не до входа в учётную
запись. Для запуска до входа нужна системная служба и отдельное подтверждение на
действия с правами администратора.

## Watchdog

`scripts/watchdog.py` проверяет два независимых признака жизни:

- свежесть `CatDoorData/runtime/recorder-status.json`;
- HTTP-ответ web на loopback-адресе и настроенном `CATDOOR_WEB_PORT`.

Если признак жизни отсутствует, watchdog запускает только соответствующий
компонент через `scripts.start_recorder` или `scripts.start_web`. Живой процесс
он не завершает и не перезапускает. Повторный запуск одного компонента
ограничен паузой 10 минут. После трёх неудачных попыток watchdog записывает
переход в backoff и ждёт 30 минут.

Действия сохраняются в `CatDoorData/runtime/watchdog.jsonl`; сообщения
исключений и настройки камер туда не записываются. Состояние cooldown/backoff
хранится в локальном `watchdog-state.json` и в Git не попадает. При нормальной
работе компонентов журнал действий остаётся пустым.

Пороги можно изменить только локальной конфигурацией:

```dotenv
CATDOOR_WATCHDOG_RECORDER_STALE_SECONDS=180
CATDOOR_WATCHDOG_RESTART_COOLDOWN_SECONDS=600
CATDOOR_WATCHDOG_MAX_RESTART_ATTEMPTS=3
CATDOOR_WATCHDOG_FAILURE_BACKOFF_SECONDS=1800
CATDOOR_WATCHDOG_WEB_TIMEOUT_SECONDS=2
```

Для ручной безопасной проверки одного цикла (без остановки живых процессов):

```powershell
.\.venv\Scripts\python.exe -m scripts.watchdog
```

## Режим сна

Пока процесс web-сервера работает, он через Windows API запрещает автоматический
переход компьютера в спящий режим. Экран может выключаться согласно системным
настройкам. После остановки web-сервера ограничение автоматически снимается.
