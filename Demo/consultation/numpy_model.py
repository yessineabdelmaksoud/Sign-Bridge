"""A small fully connected classifier that runs on NumPy alone.

The consultation page uses this so that recognising a sign needs no deep
learning framework: the file is a few megabytes and loads instantly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class DenseModel:
    """Standardise, then Dense+ReLU layers, then softmax. Inference only."""

    def __init__(self, weights: list, biases: list, mean: np.ndarray, std: np.ndarray):
        self.weights = [np.asarray(w, dtype=np.float32) for w in weights]
        self.biases = [np.asarray(b, dtype=np.float32) for b in biases]
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    def __call__(self, x, training: bool = False) -> np.ndarray:
        h = np.asarray(x, dtype=np.float32)
        h = (h.reshape(len(h), -1) - self.mean) / self.std
        last = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ w + b
            if i < last:
                np.maximum(h, 0.0, out=h)
        h -= h.max(axis=1, keepdims=True)
        np.exp(h, out=h)
        return h / h.sum(axis=1, keepdims=True)

    def save(self, path: Path | str) -> None:
        arrays = {"mean": self.mean, "std": self.std, "layers": np.array(len(self.weights))}
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            arrays[f"w{i}"], arrays[f"b{i}"] = w, b
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path | str) -> "DenseModel":
        with np.load(path, allow_pickle=False) as data:
            count = int(data["layers"])
            return cls([data[f"w{i}"] for i in range(count)], [data[f"b{i}"] for i in range(count)],
                       data["mean"], data["std"])
