# Развёртывание на Windows

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

Проверить или удалить автозапуск:

```powershell
.\.venv\Scripts\python.exe -m scripts.manage_autostart status
.\.venv\Scripts\python.exe -m scripts.manage_autostart uninstall
```

Компоненты запускаются при входе текущего пользователя, а не до входа в учётную
запись. Для запуска до входа нужна системная служба и отдельное подтверждение на
действия с правами администратора.

## Режим сна

Пока процесс web-сервера работает, он через Windows API запрещает автоматический
переход компьютера в спящий режим. Экран может выключаться согласно системным
настройкам. После остановки web-сервера ограничение автоматически снимается.
