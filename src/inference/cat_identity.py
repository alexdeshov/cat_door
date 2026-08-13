from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.common.prototypes import DEFAULT_MARGIN, match
from src.inference.cat_detector import DetectorError
from src.inference.preprocessing import (
    IDENTITY_NORMALIZATION_MEAN,
    IDENTITY_NORMALIZATION_STD,
    prepare_identity_tensor,
)

logger = logging.getLogger("catdoor.identity")

# Порог косинусной близости, если модель своего не назвала. Осознанно
# консервативный: не узнать своего кота — неудобство, принять чужого за своего —
# ошибка, ради которой всё это и разбиралось.
DEFAULT_EMBEDDING_THRESHOLD = 0.55
# Насколько лучший должен опережать второго. Два кота, похожие в сумерках,
# дают близкие расстояния до обоих эталонов, и выбирать между ними по третьему
# знаку — значит выдавать монетку за решение.
DEFAULT_PROTOTYPE_MARGIN = DEFAULT_MARGIN


@dataclass(frozen=True)
class IdentityPrediction:
    name: str | None
    confidence: float
    candidate_name: str | None = None


class OpenVinoCatIdentity:
    """Опознание кота: либо классификатор, либо эмбеддинги с эталонами.

    Два вида моделей различаются полем `kind` в метаданных рядом с файлом.

    **Классификатор** отвечает вероятностями по обученным классам. Добавить
    кота — переобучить целиком, а «чужой» существует ровно постольку, поскольку
    его успели показать.

    **Эмбеддинги** отвечают вектором, который сравнивается с эталоном каждого
    кота по косинусу. Новый кот добавляется своим эталоном без переобучения, а
    «чужой» получается сам собой: незнакомец далёк от всех эталонов, и это
    свойство пространства, а не отдельный класс, который нужно чем-то наполнять.

    Порог у эмбеддингов берётся **из метаданных модели**, а не из настроек узла:
    граница «свой/чужой» — свойство обученного пространства, и она подобрана на
    отложенном наборе при обучении. Настройка узла осталась бы от softmax, где
    0.8 значит совсем другое, и модель молча перестала бы узнавать кого угодно.
    """

    def __init__(self, model_path: Path, device: str = "CPU") -> None:
        metadata_path = model_path.with_suffix(".json")
        if not model_path.is_file() or not metadata_path.is_file():
            raise DetectorError(f"identity model or metadata is missing: {model_path}")
        self._metadata_path = metadata_path
        self._metadata_mtime = metadata_path.stat().st_mtime
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.kind = str(metadata.get("kind", "logits"))
        self.image_size = int(metadata.get("image_size", 160))
        self.mean = np.asarray(
            metadata.get("normalization_mean", IDENTITY_NORMALIZATION_MEAN),
            dtype=np.float32,
        ).reshape(1, 3, 1, 1)
        self.std = np.asarray(
            metadata.get("normalization_std", IDENTITY_NORMALIZATION_STD),
            dtype=np.float32,
        ).reshape(1, 3, 1, 1)
        self.prototypes: dict[str, np.ndarray] = {}
        self.threshold = float(metadata.get("threshold", DEFAULT_EMBEDDING_THRESHOLD))
        self.prototype_margin = float(
            metadata.get("prototype_margin", DEFAULT_PROTOTYPE_MARGIN)
        )
        if self.kind == "embedding":
            prototypes = metadata.get("prototypes")
            if not isinstance(prototypes, dict) or not prototypes:
                raise DetectorError(
                    f"embedding model without prototypes is useless: {metadata_path}"
                )
            self.prototypes = {
                str(name): np.asarray(vector, dtype=np.float32)
                for name, vector in prototypes.items()
            }
            self.labels = sorted(self.prototypes)
            logger.info(
                "Опознание по эмбеддингам: эталоны %s, порог %.2f",
                ", ".join(self.labels),
                self.threshold,
            )
        else:
            self.labels = [str(label) for label in metadata["labels"]]
        try:
            from openvino import Core

            compiled = Core().compile_model(Core().read_model(model_path), device)
        except Exception as exc:
            raise DetectorError(f"could not load identity model {model_path}: {exc}") from exc
        self._compiled: Any = compiled
        self._output = compiled.output(0)

    def reload_prototypes_if_changed(self) -> bool:
        """Перечитать эталоны, если их пересчитали. Возвращает, менялось ли.

        Кнопка «Запомнить заново» переписывает только метаданные — сама сеть
        остаётся прежней, — и ради новых эталонов перезапускать recorder
        незачем: перезапуск оборвал бы запись прохода, который сейчас идёт.

        Битый или наполовину записанный файл не применяется: лучше работать на
        прежних эталонах, чем на половине новых. Запись атомарная, так что
        такого быть не должно, — но «не должно» не повод не проверить.
        """
        if self.kind != "embedding":
            return False
        try:
            mtime = self._metadata_path.stat().st_mtime
        except OSError:
            return False
        if mtime == self._metadata_mtime:
            return False
        try:
            metadata = json.loads(self._metadata_path.read_text(encoding="utf-8"))
            prototypes = {
                str(name): np.asarray(vector, dtype=np.float32)
                for name, vector in metadata["prototypes"].items()
            }
            threshold = float(metadata.get("threshold", self.threshold))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Новые эталоны не прочитаны (%s), работаю на прежних", exc)
            return False
        if not prototypes:
            logger.warning("Новые эталоны пусты — работаю на прежних")
            return False
        self._metadata_mtime = mtime
        self.prototypes = prototypes
        self.threshold = threshold
        self.labels = sorted(prototypes)
        logger.info(
            "Эталоны перечитаны: %s, порог %.2f", ", ".join(self.labels), threshold
        )
        return True

    def embed(self, frame: np.ndarray) -> np.ndarray:
        """Сырой ответ модели по кропу.

        Вынесено отдельно ради пересчёта эталонов на узле: там нужен сам
        вектор, а не решение о нём, и гонять кроп через `predict` только чтобы
        выбросить вердикт — значит считать одно и то же дважды.
        """
        tensor = prepare_identity_tensor(
            frame,
            self.image_size,
            mean=self.mean,
            std=self.std,
        )
        return np.asarray(self._compiled([tensor])[self._output])[0]

    def predict(self, frame: np.ndarray, threshold: float) -> IdentityPrediction:
        output = self.embed(frame)
        if self.kind == "embedding":
            return self._match(output)
        shifted = output - output.max()
        probabilities = np.exp(shifted) / np.exp(shifted).sum()
        index = int(probabilities.argmax())
        confidence = float(probabilities[index])
        candidate_name = self.labels[index]
        return IdentityPrediction(
            name=candidate_name if confidence >= threshold else None,
            confidence=confidence,
            candidate_name=candidate_name,
        )

    def _match(self, embedding: np.ndarray) -> IdentityPrediction:
        """Ближайший эталон по косинусу — если он достаточно близок и один такой.

        Уверенность здесь — косинусная близость, а не вероятность. Шкала другая:
        0.6 у эмбеддингов означает «уверенно свой», тогда как 0.6 у softmax
        значило «почти монетка». Пороги, настроенные для одной шкалы, к другой
        не применимы.
        """
        name, candidate, score = match(
            embedding,
            self.prototypes,
            threshold=self.threshold,
            margin=self.prototype_margin,
        )
        return IdentityPrediction(name=name, confidence=score, candidate_name=candidate)
