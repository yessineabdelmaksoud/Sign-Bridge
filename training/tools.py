"""NumPy-only helpers for training and scoring the sign models."""
from __future__ import annotations

import numpy as np

from consultation.landmarks import mirror_clip
from consultation.recognizer import OTHER, narrow


def augment_clip(clip: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One random variant of a canonical clip: other hand, size, tilt, jitter, speed."""
    out = np.array(clip, dtype=np.float32, copy=True)
    if rng.random() < 0.5:
        out = mirror_clip(out)
    xy = out[..., :2]
    centre = np.nanmean(xy[:, [11, 12]], axis=(0, 1))
    if not np.isfinite(centre).all():
        centre = np.array([0.5, 0.5], dtype=np.float32)
    angle = np.deg2rad(rng.uniform(-15.0, 15.0))
    scale = rng.uniform(0.85, 1.15)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32)
    xy = (xy - centre) @ rotation.T * scale + centre
    out[..., :2] = xy + rng.normal(0.0, 0.01, xy.shape).astype(np.float32)

    # Time-warp: read a window 80-120% as long as the clip, at the same frame count.
    last = len(out) - 1
    span = rng.uniform(0.8, 1.2) * last
    start = rng.uniform(min(0.0, last - span), max(0.0, last - span))
    index = np.clip(np.round(np.linspace(start, start + span, len(out))), 0, last).astype(int)
    return out[index]


def narrowed_accuracy(probs: np.ndarray, y: np.ndarray, labels: list[str], allowed: list[str]) -> tuple[float, int]:
    """Accuracy over the samples whose true label is in `allowed`, choosing among `allowed` only."""
    hits = total = 0
    for p, target in zip(probs, y):
        truth = labels[int(target)]
        if truth not in allowed:
            continue
        total += 1
        hits += narrow(p, labels, allowed, 0.0).label == truth
    return (hits / total if total else 0.0), total


def pick_threshold(probs: np.ndarray, y: np.ndarray, labels: list[str], sets: list[list[str]],
                   max_wrong_accept: float = 0.05, fallback: float = 0.60) -> float:
    """The lowest threshold whose wrong-accept rate is within the limit.

    A wrong accept is a confident answer that is not the true sign. Clips of
    OTHER signs are tested against every answer set: accepting one is always wrong.
    """
    for threshold in np.round(np.arange(0.40, 0.90 + 1e-9, 0.05), 2):
        wrong = total = 0
        for p, target in zip(probs, y):
            truth = labels[int(target)]
            for allowed in sets:
                if truth != OTHER and truth not in allowed:
                    continue
                total += 1
                predicted = narrow(p, labels, allowed, float(threshold)).label
                wrong += predicted is not None and predicted != truth
        if total and wrong / total <= max_wrong_accept:
            return float(threshold)
    return fallback


def weak_labels(correct_by_label: dict, floor: float = 0.70) -> list[str]:
    """Labels scored on at least one clip whose accuracy is under `floor`."""
    return sorted(label for label, (hits, total) in correct_by_label.items() if total and hits / total < floor)


def stratified_folds(y: np.ndarray, k: int, seed: int = 42) -> list[np.ndarray]:
    """Split sample indices into k folds with each class spread evenly."""
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    for cls in np.unique(y):
        members = rng.permutation(np.flatnonzero(y == cls))
        for position, index in enumerate(members):
            folds[position % k].append(int(index))
    return [np.array(sorted(fold)) for fold in folds]
