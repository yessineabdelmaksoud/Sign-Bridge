"""Train and score the health sign model on the SWL-LSE subset.

Run A (shipped): train on the official train split, stop on validation, score on test.
Run B: k-fold cross-validation over every clip, because the official test split
       holds only two clips per sign and one number from it is too noisy to trust.

Usage:
    PYTHONPATH=Demo python -m training.train_health --data data/swl22.npz \
        --out Demo/consultation/models --plots docs/metrics
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from consultation.features import FRAMES, WIDTH, NoHandError, clip_features
from consultation.flow import SIGN_SETS
from consultation.recognizer import OTHER, narrow
from training.mlp import train_mlp
from training.tools import augment_clip, narrowed_accuracy, pick_threshold, stratified_folds, weak_labels

TRAIN, VAL, TEST = 0, 1, 2


def featurise(clips: np.ndarray) -> np.ndarray:
    if len(clips) == 0:
        return np.empty((0, FRAMES, WIDTH), dtype=np.float32)
    return np.stack([clip_features(c) for c in clips])


def with_augmentation(clips: np.ndarray, y: np.ndarray, copies: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """The clips themselves plus `copies` random variants of each."""
    features, targets = [featurise(clips)], [y]
    extra_x, extra_y = [], []
    for clip, target in zip(clips, y):
        for _ in range(copies):
            try:
                extra_x.append(clip_features(augment_clip(clip, rng)))
                extra_y.append(target)
            except NoHandError:
                continue
    if extra_x:
        features.append(np.stack(extra_x))
        targets.append(np.asarray(extra_y, dtype=y.dtype))
    return np.concatenate(features), np.concatenate(targets)


def top_k(probs: np.ndarray, y: np.ndarray, k: int) -> float:
    if not len(y):
        return 0.0
    best = np.argsort(-probs, axis=1)[:, :k]
    return float((best == y[:, None]).any(axis=1).mean())


def score(probs: np.ndarray, y: np.ndarray, labels: list[str]) -> dict:
    narrowed = {}
    for name, allowed in SIGN_SETS.items():
        accuracy, count = narrowed_accuracy(probs, y, labels, allowed)
        narrowed[name] = {"top1": round(accuracy, 4), "n": count}
    return {"top1": round(top_k(probs, y, 1), 4), "top3": round(top_k(probs, y, 3), 4),
            "n": int(len(y)), "narrowed": narrowed}


def per_label_hits(probs: np.ndarray, y: np.ndarray, labels: list[str]) -> dict:
    """For each sign: (clips recognised within its own answer set, clips)."""
    hits = {}
    for allowed in SIGN_SETS.values():
        for label in allowed:
            rows = np.flatnonzero(y == labels.index(label))
            right = sum(narrow(probs[i], labels, allowed, 0.0).label == label for i in rows)
            hits[label] = (int(right), int(len(rows)))
    return hits


def confusion_image(matrix: np.ndarray, labels: list[str], path: Path) -> None:
    """A plain heat map of the confusion matrix (rows: true sign, columns: predicted)."""
    from PIL import Image, ImageDraw

    cell, margin = 26, 150
    size = margin + cell * len(labels) + 10
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    peak = max(1, int(matrix.max()))
    for r, row_label in enumerate(labels):
        draw.text((6, margin + r * cell + 7), row_label[:20], fill="black")
        for c in range(len(labels)):
            value = int(matrix[r, c])
            shade = 255 - int(215 * value / peak)
            colour = (shade, 255 - (255 - shade) // 2, shade) if r == c else (255, shade, shade)
            box = (margin + c * cell, margin + r * cell, margin + (c + 1) * cell - 1, margin + (r + 1) * cell - 1)
            draw.rectangle(box, fill=colour if value else "white", outline="#dddddd")
            if value:
                draw.text((box[0] + 7, box[1] + 7), str(value), fill="black")
    for c, label in enumerate(labels):
        draw.text((margin + c * cell + 2, margin - 14 - (c % 4) * 12), label[:8], fill="black")
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=Path("data/swl22.npz"))
    parser.add_argument("--out", type=Path, default=Path("Demo/consultation/models"))
    parser.add_argument("--plots", type=Path, default=Path("docs/metrics"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--copies", type=int, default=8, help="augmented variants per training clip")
    parser.add_argument("--hidden", default="256,128")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    hidden = tuple(int(v) for v in args.hidden.split(",") if v)

    with np.load(args.data, allow_pickle=False) as data:
        clips, y, split = data["clips"], data["y"].astype(np.int64), data["split"]
        labels = [str(v) for v in data["labels"]]
    n_classes = len(labels)
    rng = np.random.default_rng(args.seed)
    sets = list(SIGN_SETS.values())
    print(f"{len(clips)} clips, {n_classes} classes, split sizes "
          f"{[int((split == s).sum()) for s in (TRAIN, VAL, TEST)]}", flush=True)

    # Run A: the official split.
    x_train, y_train = with_augmentation(clips[split == TRAIN], y[split == TRAIN], args.copies, rng)
    x_val, y_val = featurise(clips[split == VAL]), y[split == VAL]
    x_test, y_test = featurise(clips[split == TEST]), y[split == TEST]
    model, history = train_mlp(x_train, y_train, n_classes, hidden=hidden, epochs=args.epochs,
                               x_val=x_val, y_val=y_val, seed=args.seed)
    probs_val, probs_test = model(x_val), model(x_test)
    official = score(probs_test, y_test, labels)
    validation = score(probs_val, y_val, labels)
    print("official test:", official, flush=True)

    threshold = pick_threshold(probs_val, y_val, labels, sets)
    held_probs, held_y = np.concatenate([probs_val, probs_test]), np.concatenate([y_val, y_test])
    hits = per_label_hits(held_probs, held_y, labels)
    weak = weak_labels(hits)

    # Run B: cross-validation over every clip, trained for the epoch count run A settled on.
    folds_report = []
    if args.folds > 1:
        for number, held in enumerate(stratified_folds(y, args.folds, seed=args.seed), start=1):
            rest = np.setdiff1d(np.arange(len(y)), held)
            x_fold, y_fold = with_augmentation(clips[rest], y[rest], args.copies, rng)
            fold_model, _ = train_mlp(x_fold, y_fold, n_classes, hidden=hidden,
                                      epochs=max(1, history["best_epoch"]), seed=args.seed + number)
            folds_report.append(score(fold_model(featurise(clips[held])), y[held], labels))
            print(f"fold {number}: {folds_report[-1]}", flush=True)

    def spread(pick) -> dict:
        values = [pick(report) for report in folds_report]
        return {"mean": round(float(np.mean(values)), 4), "std": round(float(np.std(values)), 4)} if values else {}

    metrics = {
        "official_test": official,
        "official_validation": validation,
        "cv_folds": len(folds_report),
        "cv_top1": spread(lambda r: r["top1"]),
        "cv_top3": spread(lambda r: r["top3"]),
        "cv_narrowed_top1": {name: spread(lambda r, n=name: r["narrowed"][n]["top1"]) for name in SIGN_SETS},
        "per_label_heldout": {label: {"correct": h, "n": n} for label, (h, n) in hits.items()},
        "best_epoch": history["best_epoch"],
        "split_signer_independent": "unknown",
        "note": "Scores are on SWL-LSE (Spanish Sign Language). 'narrowed' means choosing only among "
                "the answers one question allows. The official test split has two clips per sign.",
    }
    meta = {"labels": labels, "threshold": threshold, "frames": 32, "features": 184,
            "weak_labels": weak, "model": "numpy-mlp", "hidden": list(hidden), "metrics": metrics}

    args.out.mkdir(parents=True, exist_ok=True)
    model.save(args.out / "health_signs.npz")
    (args.out / "health_signs.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    args.plots.mkdir(parents=True, exist_ok=True)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for truth, guess in zip(held_y, held_probs.argmax(axis=1)):
        matrix[truth, guess] += 1
    confusion_image(matrix, labels, args.plots / "health_signs_confusion.png")
    print(json.dumps({"threshold": threshold, "weak_labels": weak, "cv_top1": metrics["cv_top1"],
                      "cv_narrowed_top1": metrics["cv_narrowed_top1"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
