# Модель данных

Фактическая схема SQLite (`src/common/database.py`):

- `events` — проход: статус, метка разметки, имя кота, назначение для обучения,
  пути к видео обеих камер, уверенности детектора, какая камера начала событие;
- `model_versions` — версии моделей по SHA-256: детектор и identity-модели;
- `event_predictions` — предсказания identity по событиям, онлайн и офлайн;
- `evaluation_runs` — запуски офлайн-оценки для страницы метрик.

Видео лежит в файловой системе (`/var/lib/catdoor/events/`), в базе — метаданные
и пути.

Не в базе, а в runtime-файлах (`/var/lib/catdoor/runtime/`): инвентарь камер
`cameras.json`, настройки детекции `detection-settings.json`, состояние контуров
`circuits.json`, режим сбора `capture-mode.json`, журналы `detection.jsonl` и
`observations.jsonl`.
