# Training

Полностью локальный конвейер обучения CatDoor AI.

## Среды

- обучение: MacBook M5, PyTorch с устройством `mps`;
- перенос: единый ONNX-файл без external data;
- приёмка: Windows-моноблок, ONNX Runtime CPU;
- активация модели этим конвейером не выполняется.

## Подготовка Mac

```bash
python3 -m venv .venv-training
source .venv-training/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[training,dev]"
```

## Локальный рабочий каталог

Датасет, checkpoints, ONNX, ZIP-пакеты и отчёты хранятся вне Git. Рекомендуемая
структура на MacBook:

```text
~/CatDoorTraining/
├── incoming/         # полученные ZIP-экспорты разметки
├── datasets/images/  # распакованные изображения
├── manifests/        # проверенные JSONL и split
├── work/             # промежуточные файлы
├── runs/
│   ├── face/         # checkpoints модели морды
│   └── side/         # checkpoints модели тела
├── release/          # только ONNX и метаданные для сборки
├── packages/         # итоговые ZIP-архивы моделей
├── reports/
└── logs/
```

Каталог `~/CatDoorTraining` не следует переносить внутрь репозитория или
добавлять в Git.

## Манифест датасета

Формат JSON Lines показан в `dataset-manifest.example.jsonl`. Один кадр — одна
строка. `image_path` всегда относителен к корню датасета. Специальная метка
`__foreign__` используется только для open-set проверки и не обучается как
домашний кот.

Экспорт с Windows уже содержит кропы кошек, подготовленные тем же локальным
детектором и контрактом, что используются runtime-идентификацией. Повторно
запускать детектор на MacBook не нужно. Статистика исключённых кадров без
уверенной рамки находится в `dataset-report.json`.

Проверка и назначение split:

```bash
python -m training.dataset export/manifest.jsonl \
  --dataset-root export/dataset \
  --assign-splits work/manifest.split.jsonl
```

Split вычисляется детерминированно из `event_id`: 70% train, 15% validation,
15% test. Все кадры события всегда получают один split. Перед обучением следует
также проверить представленность разных дат и условий съёмки в каждом split.

## Обучение

Отдельные модели обучаются для `face` и `side`:

```bash
python -m training.train_classifier work/manifest.split.jsonl export/dataset \
  work/face.pt --view face --epochs 20
```

При доступности MPS он выбирается автоматически, иначе используется CPU.
Скрипт не скачивает pretrained-веса и может работать в полностью изолированной
среде.

## ONNX и пакет

```bash
python -m training.export_onnx \
  ~/CatDoorTraining/runs/face/checkpoint.pt \
  ~/CatDoorTraining/release/face_model.onnx \
  --model-info ~/CatDoorTraining/release/model_info.json

cp model-manifests/thresholds.example.json \
  ~/CatDoorTraining/release/thresholds.json
python -m training.model_package build \
  ~/CatDoorTraining/release \
  ~/CatDoorTraining/packages/catdoor-face-model.zip
```

Порог нельзя считать окончательным до калибровки на validation и проверки на
независимых чужих кошках.

## Приёмка на Windows

```powershell
py -3.13 -m venv .venv-acceptance
.\.venv-acceptance\Scripts\python.exe -m pip install -e ".[acceptance]"

.\.venv-acceptance\Scripts\python.exe -m training.model_package verify `
  D:\Transfer\catdoor-face-model.zip

.\.venv-acceptance\Scripts\python.exe -m training.acceptance `
  D:\Transfer\catdoor-face-model.zip `
  D:\Acceptance\manifest.split.jsonl `
  D:\Acceptance\dataset `
  --model face_model.onnx `
  --report D:\Acceptance\reports\face-report.json
```

Приёмочный набор не должен участвовать в обучении или подборе порогов. Он обязан
содержать домашних и чужих кошек. Пакет проходит приёмку только если FAR и FRR не
выше порогов. Даже успешная приёмка не активирует модель и не управляет дверью.
