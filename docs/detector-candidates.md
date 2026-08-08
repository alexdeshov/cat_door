# Кандидаты на замену детектора

> Написан до переезда камер на улицу. Имена камер историчны: Cute 2 — прежняя
> камера роли `face`, Ursa Minor выведена из проекта 2026-08-02.

## Область эксперимента

Текущий SSDLite320 MobileNetV3 остаётся активным baseline. Подготовлены
совместимые адаптеры для кандидатов:

| Кандидат | Вход | Лицензия | Выход ONNX | Postprocessing |
| --- | --- | --- | --- | --- |
| `nanodet-plus-m-416` | 416 | Apache-2.0 | сырые class/distribution predictions | decode и NMS в runtime |
| `yolox-nano-416` | 416 | Apache-2.0 | декодированные `cx, cy, w, h` | NMS в runtime |
| `yolo11n-416` | 416 | AGPL-3.0 | `xyxy, confidence, class` | NMS в ONNX-графе |
| `yolo11n-640` | 640 | AGPL-3.0 | то же | то же |
| `yolo11s-640` | 640 | AGPL-3.0 | то же | то же |
| SSDLite320 MobileNetV3 | 320 | BSD-3-Clause | `boxes, labels, scores` | встроен в экспорт |

**Вход 640 пробуется наравне с 416, и это не формальность.** У кота в дальнем
конце террасы после сжатия кадра до 320×320 остаются десятки точек; разрешение
здесь способно дать больше, чем смена архитектуры. Кандидаты на 640 стоят
дороже по времени, но запас есть: 11 инференсов в секунду по 26 мс — это 0.29
ядра из шестнадцати, и ограничение, ради которого выбирали самый лёгкий
SSDLite-320, исчезло вместе со старым двухъядерным узлом.

**Лицензии домашнему применению не мешают.** Репозиторий приватный, применение
некоммерческое, программу никому не передают и сетевого доступа посторонним не
дают — копилефт AGPL и GPL не срабатывает. Оговорка на будущее: публикация
репозитория с кодом или дообученными весами Ultralytics потребует сделать AGPL
весь проект. Поэтому YOLO11 держится как **быстрый опыт** (дообучение в десяток
строк, экспорт одной командой), а производственным кандидатом остаётся
**D-FINE**: Apache-2.0, OpenVINO CPU, нативная поддержка во Frigate 0.17.
**YOLO-NAS вычеркнут — проект заброшен.**

Все адаптеры возвращают один контракт: рамки в координатах квадратного входа
детектора и confidence класса `cat`. Архитектура, имена выходов, preprocessing,
лицензия и SHA-256 берутся из соседнего manifest. Путь к модели остаётся
настройкой `CATDOOR_DETECTOR_MODEL_PATH`.

Модель-кандидат не активируется автоматически. Реальная дверь по-прежнему не
управляется.

## Получение весов

Весов и ONNX-файлов в Git нет. Скачивание выполняется только явной командой по
ссылке из официального model zoo. SHA-256 нужно получить независимым способом
и передать полностью — placeholder скрипт отклонит:

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_detector_candidate download `
  --url "<официальная-прямая-ссылка>" `
  --expected-sha256 "<64 hex символа>" `
  --output "models\<имя-весов>"
```

Скрипт сначала пишет временный файл, проверяет SHA-256 и только затем публикует
его под итоговым именем. Существующие файлы он не перезаписывает.

Официальные источники:

- NanoDet: `https://github.com/RangiLyu/nanodet`;
- YOLOX: `https://github.com/Megvii-BaseDetection/YOLOX`;
- YOLO11: `https://github.com/ultralytics/ultralytics`.

## Локальный экспорт

NanoDet и YOLOX экспортируются официальными скриптами из локального checkout
соответствующего репозитория. Checkout следует зафиксировать на конкретном
commit и установить его зависимости в отдельное окружение подготовки.

```powershell
# NanoDet-Plus-m @416
python -m scripts.prepare_detector_candidate export nanodet-plus-m-416 `
  --weights "models\nanodet-plus-m-416.ckpt" `
  --weights-sha256 "<проверенный SHA-256>" `
  --source-root "<путь-к-checkout-nanodet>" `
  --python "<python-окружения-nanodet>" `
  --output "models\nanodet-plus-m-416.onnx"

# YOLOX-Nano @416
python -m scripts.prepare_detector_candidate export yolox-nano-416 `
  --weights "models\yolox_nano.pth" `
  --weights-sha256 "<проверенный SHA-256>" `
  --source-root "<путь-к-checkout-YOLOX>" `
  --python "<python-окружения-yolox>" `
  --output "models\yolox-nano-416.onnx"

# YOLO11n @416; ultralytics должен быть установлен только в окружении экспорта
python -m scripts.prepare_detector_candidate export yolo11n-416 `
  --weights "models\yolo11n.pt" `
  --weights-sha256 "<проверенный SHA-256>" `
  --output "models\yolo11n-416.onnx"
```

На узле (Ubuntu) те же команды без обратных кавычек; веса `yolo11s.pt` берутся
из того же релиза Ultralytics, что и `yolo11n.pt`:

```sh
.venv/bin/python -m scripts.prepare_detector_candidate export yolo11n-640 \
  --weights models/yolo11n.pt --weights-sha256 "<проверенный SHA-256>" \
  --output models/yolo11n-640.onnx

.venv/bin/python -m scripts.prepare_detector_candidate export yolo11s-640 \
  --weights models/yolo11s.pt --weights-sha256 "<проверенный SHA-256>" \
  --output models/yolo11s-640.onnx
```

Сторона входа берётся из записи кандидата, а не из ключа командной строки:
манифест и сам ONNX разойтись не могут — экспорт с чужим размером отклоняется
и не оставляет за собой ни файла, ни манифеста.

Экспорт проверяет статический вход из таблицы, вычисляет SHA-256 ONNX
и создаёт рядом JSON manifest. Для NanoDet используются BGR, растяжение до
квадрата и нормализация из официального `nanodet-plus-m_416.yml`; для YOLOX —
BGR без деления на 255 и letterbox от левого верхнего угла; для YOLO11 — RGB в
диапазоне `0..1` и центрированный letterbox. Способ resize, выравнивание и
значение заполнения хранятся в manifest, поэтому рамки правильно пересчитываются
обратно в координаты исходного кадра.

## Следующие измерения

Перед probe штатный recorder нужно остановить, чтобы не открывать по второму
RTSP-подключению к каждой камере. Сам bounded probe создаёт рабочую нагрузку
8 FPS для Cute 2 и 3 FPS для Ursa Minor:

```powershell
python -m scripts.benchmark_detector "models\nanodet-plus-m-416.onnx" --seconds 30
```

Команда принудительно использует OpenVINO CPU, ограничивает статистику точным
временным окном запуска и сохраняет локальный JSON в
`data/runtime/detector-probes/`. В отчёте есть число наблюдений, median, p95,
бюджет 70 мс и verdict по каждой камере и объединённому потоку.

После latency probe запускается detector-only replay из `/metrics`: в поле
«Режим» выбрать «Только сравнение detector», указать имя ONNX из `models/` и
оставить набор меток с `known_cat`, `foreign_cat` и `not_visible`. Replay:

- выполняется отдельным последовательным процессом только на OpenVINO CPU;
- сохраняет запуск и версию модели по SHA-256;
- считает miss rate отдельно для Cute 2 и Ursa Minor;
- считает false trigger rate на `not_visible`;
- показывает распределение максимального confidence и threshold sweep;
- отдельно показывает чистый и ночной срезы;
- позволяет сравнить два detector-запуска на `/metrics/compare`.

Кандидат с медианой выше 70 мс помечается как не прошедший latency budget, но
его метрики качества всё равно сохраняются. Особо сравниваются miss rate камеры
Cute 2, ночной срез и false trigger rate.

До этих измерений нельзя рекомендовать или активировать замену baseline.
