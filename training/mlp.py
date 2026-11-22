"""Train the fully connected sign classifier with NumPy (Adam, dropout, early stopping)."""
from __future__ import annotations

import numpy as np

from consultation.numpy_model import DenseModel


def train_mlp(x, y, n_classes: int, hidden=(256, 128), epochs: int = 100, batch: int = 64,
              lr: float = 1e-3, weight_decay: float = 1e-4, dropout: float = 0.3,
              label_smoothing: float = 0.1, x_val=None, y_val=None, patience: int = 12,
              seed: int = 0) -> tuple[DenseModel, dict]:
    """Returns the model with the best validation accuracy (or the last one without validation)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float32).reshape(len(x), -1)
    y = np.asarray(y, dtype=np.int64)
    mean, std = x.mean(axis=0), x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    xs = (x - mean) / std

    sizes = [xs.shape[1], *hidden, n_classes]
    weights = [(rng.normal(0, np.sqrt(2.0 / a), (a, b))).astype(np.float32) for a, b in zip(sizes, sizes[1:])]
    biases = [np.zeros(b, dtype=np.float32) for b in sizes[1:]]
    params = weights + biases
    moment1 = [np.zeros_like(p) for p in params]
    moment2 = [np.zeros_like(p) for p in params]

    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    class_weight = np.where(counts > 0, len(y) / (n_classes * np.maximum(counts, 1)), 0.0)
    targets = np.full((len(y), n_classes), label_smoothing / n_classes, dtype=np.float32)
    targets[np.arange(len(y)), y] += 1.0 - label_smoothing

    def snapshot() -> DenseModel:
        return DenseModel([w.copy() for w in weights], [b.copy() for b in biases], mean, std)

    best, best_acc, best_epoch, since_best = snapshot(), -1.0, 0, 0
    history = {"val_accuracy": [], "best_epoch": 0}
    step = 0
    last = len(weights) - 1
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(xs))
        for start in range(0, len(xs), batch):
            rows = order[start:start + batch]
            activations, masks = [xs[rows]], []
            for i, (w, b) in enumerate(zip(weights, biases)):   # forward
                h = activations[-1] @ w + b
                if i < last:
                    h = np.maximum(h, 0.0)
                    mask = (rng.random(h.shape) >= dropout).astype(np.float32) / (1.0 - dropout)
                    h *= mask
                    masks.append(mask)
                activations.append(h)
            logits = activations[-1]
            logits = logits - logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            delta = (probs - targets[rows]) * class_weight[y[rows]][:, None] / len(rows)

            grads_w, grads_b = [None] * len(weights), [None] * len(weights)
            for i in range(last, -1, -1):                          # backward
                grads_w[i] = activations[i].T @ delta + weight_decay * weights[i]
                grads_b[i] = delta.sum(axis=0)
                if i > 0:
                    delta = (delta @ weights[i].T) * masks[i - 1] * (activations[i] > 0)

            step += 1
            for p, g, m1, m2 in zip(params, grads_w + grads_b, moment1, moment2):   # Adam
                m1 *= 0.9
                m1 += 0.1 * g
                m2 *= 0.999
                m2 += 0.001 * g * g
                p -= lr * (m1 / (1 - 0.9 ** step)) / (np.sqrt(m2 / (1 - 0.999 ** step)) + 1e-8)

        if x_val is None:
            best, best_epoch = snapshot(), epoch
            continue
        accuracy = float((snapshot()(x_val).argmax(axis=1) == np.asarray(y_val)).mean())
        history["val_accuracy"].append(accuracy)
        if accuracy > best_acc:
            best, best_acc, best_epoch, since_best = snapshot(), accuracy, epoch, 0
        else:
            since_best += 1
            if since_best >= patience:
                break
    history["best_epoch"] = best_epoch
    return best, history
