"""Sign recognition, narrowed to the answers the current question allows.

A question such as "known conditions?" has four valid signed answers. Asking
the model "which of these four?" is far more reliable than "which of all 22?",
so every prediction is renormalised over the allowed labels plus OTHER.

TensorFlow is imported only when a model is actually loaded. If the model
file is missing or unreadable the recogniser reports `available = False` and
the consultation carries on by tapping.
"""
from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from consultation.features import NoHandError, clip_features, frame_features
from consultation.landmarks import hand_present

OTHER = "OTHER"


@dataclass(frozen=True)
class Prediction:
    label: str | None    # None: not confident enough, or "none of these"
    confidence: float    # probability of the best allowed label after narrowing
    top3: list           # [(label, probability), ...], allowed labels only, best first


NO_PREDICTION = Prediction(None, 0.0, [])


def narrow(probs: np.ndarray, labels: list[str], allowed: list[str], threshold: float) -> Prediction:
    """Renormalise `probs` over `allowed` plus OTHER and pick the winner."""
    keep = [i for i, label in enumerate(labels) if label in allowed or label == OTHER]
    kept = np.asarray(probs, dtype=np.float64)[keep]
    total = float(kept.sum())
    if total <= 0 or not any(labels[i] != OTHER for i in keep):
        return NO_PREDICTION
    ranked = sorted(zip((labels[i] for i in keep), kept / total), key=lambda pair: -pair[1])
    top3 = [(label, float(p)) for label, p in ranked if label != OTHER][:3]
    best_label, best_p = ranked[0]
    if best_label == OTHER or best_p < threshold:
        return Prediction(None, top3[0][1], top3)
    return Prediction(best_label, float(best_p), top3)


@functools.lru_cache(maxsize=1)
def temporal_attention_layer():
    """The custom Keras layer the models use, built on first use."""
    import keras

    @keras.saving.register_keras_serializable(package="signbridge")
    class TemporalAttention(keras.layers.Layer):
        """Weighted sum over time: the model learns which frames matter."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.score = keras.layers.Dense(1, activation="tanh")

        def call(self, x):
            weights = keras.ops.softmax(self.score(x), axis=1)
            return keras.ops.sum(x * weights, axis=1)

    return TemporalAttention


def _load_keras_model(path: Path):
    import keras

    return keras.models.load_model(
        str(path), custom_objects={"TemporalAttention": temporal_attention_layer()}, compile=False)


class _Recognizer:
    def __init__(self, model_path: Path, meta_path: Path, model=None):
        self.available = False
        self.error: str | None = None
        self.labels: list[str] = []
        self.threshold = 0.6
        self.weak_labels: list[str] = []
        self.meta: dict = {}
        self._model = None
        try:
            self.meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
            self.labels = list(self.meta["labels"])
            self.threshold = float(self.meta.get("threshold", 0.6))
            self.weak_labels = list(self.meta.get("weak_labels", []))
            self._model = model if model is not None else _load_keras_model(Path(model_path))
            self.available = True
        except Exception as error:  # any failure means: answer by tapping
            self.error = f"{type(error).__name__}: {error}"

    def _features(self, clip: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, clip: np.ndarray, allowed: list[str]) -> Prediction:
        if not self.available:
            return NO_PREDICTION
        allowed = [label for label in allowed if label not in self.weak_labels]
        try:
            batch = self._features(np.asarray(clip, dtype=np.float32))
        except NoHandError:
            return NO_PREDICTION
        probs = np.asarray(self._model(batch, training=False)).mean(axis=0)
        return narrow(probs, self.labels, allowed, self.threshold)


class SignRecognizer(_Recognizer):
    """Moving signs: one clip in, one prediction out."""

    def _features(self, clip):
        return clip_features(clip)[None]


class StaticSignRecognizer(_Recognizer):
    """Signs learned from still images: classify each frame, then average."""

    def __init__(self, model_path: Path, meta_path: Path, model=None):
        super().__init__(model_path, meta_path, model)
        self.signing_enabled = bool(self.meta.get("signing_enabled", False)) and self.available
        self.to_answer: dict = dict(self.meta.get("to_answer", {}))

    def _features(self, clip):
        frames = [frame for frame in clip if hand_present(frame)]
        if not frames:
            raise NoHandError("no hand visible in any frame")
        return np.stack([frame_features(frame) for frame in frames])
