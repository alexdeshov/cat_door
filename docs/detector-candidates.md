# Кандидаты на замену детектора

## Область эксперимента

Текущий SSDLite320 MobileNetV3 остаётся активным baseline. Подготовлены
совместимые адаптеры для четырёх кандидатов с входом `416×416`:

| Кандидат | Лицензия | Выход ONNX | Postprocessing |
| --- | --- | --- | --- |
| NanoDet-Plus-m | Apache-2.0 | сырые class/distribution predictions | decode и NMS в runtime |
| YOLOX-Nano | Apache-2.0 | декодированные `cx, cy, w, h` | NMS в runtime |
| YOLO11n | AGPL-3.0 | `xyxy, confidence, class` | NMS в ONNX-графе |
| YOLO26n | AGPL-3.0 | `xyxy, confidence, class` | end-to-end, NMS не требуется |
| SSDLite320 MobileNetV3 | BSD-3-Clause | `boxes, labels, scores` | встроен в экспорт |

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
- YOLO11: `https://github.com/ultralytics/ultralytics`;
- YOLO26: `https://github.com/ultralytics/assets/releases/tag/v8.4.0`.

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

# YOLO26n @416, end-to-end NMS-free; Ultralytics только в отдельном окружении
python -m scripts.prepare_detector_candidate export yolo26n-416 `
  --weights "models\yolo26n.pt" `
  --weights-sha256 "<проверенный SHA-256>" `
  --output "models\yolo26n-416.onnx"
```

Экспорт проверяет статический вход `[1, 3, 416, 416]`, вычисляет SHA-256 ONNX
и создаёт рядом JSON manifest. Для NanoDet используются BGR, растяжение до
квадрата и нормализация из официального `nanodet-plus-m_416.yml`; для YOLOX —
BGR без деления на 255 и letterbox от левого верхнего угла; для YOLO11 — RGB в
диапазоне `0..1` и центрированный letterbox. Способ resize, выравнивание и
значение заполнения хранятся в manifest, поэтому рамки правильно пересчитываются
обратно в координаты исходного кадра. YOLO26 использует тот же конечный
контракт рамок, что end-to-end YOLO11, но его one-to-one head уже устраняет
дубликаты и возвращает `(1, 300, 6)` без NMS.

## YOLO26n @416 — подготовка и latency на 2026-07-31

Официальные веса `yolo26n.pt` получены из релиза Ultralytics assets `v8.4.0`
через проверяемый download-механизм проекта. GitHub release API опубликовал и
локальная проверка подтвердила:

```text
yolo26n.pt SHA-256:
9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef
```

Экспорт выполнен локально Ultralytics `8.4.55`, Torch `2.13.0+cpu`, torchvision
`0.28.0+cpu`, ONNX `1.22.0` и onnxslim `0.1.94`. Команда использовала
`format=onnx`, `imgsz=416`, `opset=18`, `simplify=True`, `end2end=True`,
`max_det=300`, `batch=1`, `device=cpu`. Фактический контракт:

```text
input:  images  [1, 3, 416, 416]
output: output0 [1, 300, 6]
row:    x1, y1, x2, y2, confidence, class_id
```

Метаданные загруженного checkpoint содержат 80 COCO-классов и явно задают
`names[15] == "cat"`. Проверка того же индекса на реальном размеченном кадре
остаётся обязательной перед replay и активацией.

Итоговый локальный `models/yolo26n-416.onnx` имеет SHA-256
`192283e7ebe8d804e4891c9406d253c71ad6afcac1453a4b71ce94cde219624e`.
Файлы весов, ONNX, manifest и локальные probe-отчёты намеренно не добавляются в
Git.

Одинаковые 30-секундные OpenVINO CPU probe на целевом Intel Core i3-8109U:

| Модель | Камера | Observations | Median, мс | p95, мс | Median ≤ 70 мс |
| --- | --- | ---: | ---: | ---: | --- |
| SSDLite320 | face | 217 | 52.50 | 98.02 | да |
| SSDLite320 | side | 85 | 68.40 | 92.62 | да |
| SSDLite320 | combined | 302 | 57.40 | 97.665 | да |
| YOLO26n @416 | face | 226 | 48.45 | 84.70 | да |
| YOLO26n @416 | side | 85 | 45.50 | 70.40 | да |
| YOLO26n @416 | combined | 311 | 47.40 | 83.05 | да |

YOLO26n быстрее baseline по combined median примерно на 17%, но не принят и не
активирован. На момент замера в локальной базе было 125 событий и все имели
метку `pending`; событий `known_cat`, `foreign_cat` и `not_visible` не было.
Поэтому обязательные проверки класса `cat` на реальных кадрах, miss rate,
false trigger rate и threshold sweep пока невыполнимы. До разметки минимум трёх
проходов с котом и трёх известных ложных событий нельзя рекомендовать замену,
менять `CATDOOR_DETECTOR_MODEL_PATH` или предлагать новые рабочие пороги.

После разметки повторить sanity и detector-only replay. Только если они пройдут,
инструкция ручной активации будет состоять в сохранении
`yolo26n-416.onnx` и `yolo26n-416.json` рядом в `models/` и установке:

```dotenv
CATDOOR_DETECTOR_MODEL_PATH=models\yolo26n-416.onnx
```

Пороги камер до threshold sweep должны остаться текущими; активация в этой
сессии не выполнялась.

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
