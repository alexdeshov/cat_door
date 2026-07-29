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

## Локальный фоновый запуск

Web и recorder можно запускать как отделённые процессы:

```powershell
python -m scripts.start_web
python -m scripts.start_recorder --record
```

Логи сохраняются только локально в `CatDoorData/runtime/`.
