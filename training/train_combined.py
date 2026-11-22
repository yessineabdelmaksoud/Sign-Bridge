"""Train one model on the 22 SWL-LSE health signs and the Tunisian everyday words.

Scoring, kept apart so neither flatters the other:
- Health signs: the SWL-LSE official split, as before (train, stop on validation, score on test).
- Tunisian words: leave one signer out. For each signer, a model trained without any of
  their takes is scored on them. A word scoring under 70% this way, or performed by a
  single signer (so it cannot be scored), is marked weak: it is never proposed to the
  patient, though the doctor's screen can still show it as a stick figure.

    PYTHONPATH=Demo python -m training.train_combined
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from consultation.features import FRAMES, NoHandError, resample, trim_to_hands
from consultation.flow import SIGN_SETS
from consultation.recognizer import OTHER
from training.mlp import train_mlp
from training.tools import pick_threshold, weak_labels
from training.train_health import TEST, TRAIN, VAL, confusion_image, featurise, per_label_hits, score, top_k, with_augmentation
from training.tunisian_signs import load_vocab, steady_clip

WEAK_FLOOR = 0.70


def load_tunisian(seq_dir: Path, codes: list[str]):
    """(clips, labels, signers) for every Tunisian take with a visible hand, body held steady."""
    index = json.loads((seq_dir / "index.json").read_text(encoding="utf-8"))
    clips, labels, signers = [], [], []
    for entry in index:
        if entry["code"] not in codes or entry["hand_frames"] < 5:
            continue
        try:
            clip = resample(trim_to_hands(steady_clip(np.load(seq_dir / entry["file"]))), FRAMES)
        except NoHandError:
            continue
        clips.append(clip)
        labels.append(entry["code"])
        signers.append(entry["signer"])
    return np.stack(clips).astype(np.float32), labels, np.asarray(signers)


def leave_one_signer_out(base_x, base_y, tn_clips, tn_y, tn_signers, n_classes, epochs, copies, hidden, seed):
    """Per Tunisian label: [correct, scored] over models that never saw the signer."""
    tally = defaultdict(lambda: [0, 0])
    rng = np.random.default_rng(seed)
    for signer in sorted(set(tn_signers.tolist())):
        held = tn_signers == signer
        if held.all():
            continue
        x_tn, y_tn = with_augmentation(tn_clips[~held], tn_y[~held], copies, rng)
        model, _ = train_mlp(np.concatenate([base_x, x_tn]), np.concatenate([base_y, y_tn]), n_classes,
                             hidden=hidden, epochs=epochs, seed=seed + int(signer))
        predicted = model(featurise(tn_clips[held])).argmax(axis=1)
        for truth, guess in zip(tn_y[held], predicted):
            tally[int(truth)][0] += int(truth == guess)
            tally[int(truth)][1] += 1
        print(f"signer {signer}: {int(held.sum())} takes held out, "
              f"{int((predicted == tn_y[held]).sum())} recognised", flush=True)
    return tally


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--swl", type=Path, default=Path("data/swl22.npz"))
    parser.add_argument("--tunisian", type=Path, default=Path("data/tunsl_seq"))
    parser.add_argument("--out", type=Path, default=Path("Demo/consultation/models"))
    parser.add_argument("--plots", type=Path, default=Path("docs/metrics"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--copies", type=int, default=8, help="augmented variants per SWL-LSE clip")
    parser.add_argument("--tn-copies", type=int, default=40, help="augmented variants per Tunisian take")
    parser.add_argument("--hidden", default="256,128")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    hidden = tuple(int(v) for v in args.hidden.split(",") if v)
    rng = np.random.default_rng(args.seed)

    with np.load(args.swl, allow_pickle=False) as data:
        swl_clips, swl_y, split = data["clips"], data["y"].astype(np.int64), data["split"]
        swl_labels = [str(v) for v in data["labels"]]
    health = swl_labels[:-1]                                   # OTHER is last
    vocab_codes = [v["code"] for v in load_vocab()]
    tn_clips, tn_codes, tn_signers = load_tunisian(args.tunisian, vocab_codes)
    everyday = [code for code in vocab_codes if code in set(tn_codes)]
    labels = health + everyday + [OTHER]
    index = {label: i for i, label in enumerate(labels)}
    swl_y = np.asarray([index[swl_labels[i]] for i in swl_y])
    tn_y = np.asarray([index[c] for c in tn_codes])
    n = len(labels)
    print(f"{len(swl_clips)} SWL-LSE clips, {len(tn_clips)} Tunisian takes of {len(everyday)} words "
          f"from {len(set(tn_signers.tolist()))} signers; {n} classes", flush=True)

    # Health signs on the official split, with every Tunisian take in training.
    x_swl, y_swl = with_augmentation(swl_clips[split == TRAIN], swl_y[split == TRAIN], args.copies, rng)
    x_tn, y_tn = with_augmentation(tn_clips, tn_y, args.tn_copies, rng)
    x_val, y_val = featurise(swl_clips[split == VAL]), swl_y[split == VAL]
    x_test, y_test = featurise(swl_clips[split == TEST]), swl_y[split == TEST]
    model, history = train_mlp(np.concatenate([x_swl, x_tn]), np.concatenate([y_swl, y_tn]), n, hidden=hidden,
                               epochs=args.epochs, x_val=x_val, y_val=y_val, seed=args.seed)
    probs_val, probs_test = model(x_val), model(x_test)
    official = score(probs_test, y_test, labels)
    print("health signs, official test:", official, flush=True)
    sets = list(SIGN_SETS.values())
    threshold = pick_threshold(probs_val, y_val, labels, sets)
    held_probs, held_y = np.concatenate([probs_val, probs_test]), np.concatenate([y_val, y_test])
    weak_health = weak_labels(per_label_hits(held_probs, held_y, labels), WEAK_FLOOR)

    # Tunisian words, leave one signer out.
    tally = leave_one_signer_out(x_swl, y_swl, tn_clips, tn_y, tn_signers, n,
                                 max(1, history["best_epoch"]), args.tn_copies, hidden, args.seed)
    per_word, weak_tn = {}, []
    for code in everyday:
        right, scored = tally.get(index[code], [0, 0])
        signers = len(set(tn_signers[tn_y == index[code]].tolist()))
        per_word[code] = {"correct": right, "n": scored, "signers": signers}
        if scored == 0 or right / scored < WEAK_FLOOR:
            weak_tn.append(code)
    right = sum(v["correct"] for v in per_word.values())
    scored = sum(v["n"] for v in per_word.values())

    metrics = {
        "official_test": official,
        "official_validation": score(probs_val, y_val, labels),
        "best_epoch": history["best_epoch"],
        "tunisian_leave_one_signer_out": {"top1": round(right / scored, 4) if scored else 0.0, "n": scored,
                                          "per_word": per_word},
        "per_label_heldout": {label: {"correct": h, "n": c}
                              for label, (h, c) in per_label_hits(held_probs, held_y, labels).items()},
        "split_signer_independent": "unknown",
        "note": "Health signs: SWL-LSE (Spanish Sign Language), official split. Tunisian words: Tunisian Sign "
                "Language Dataset, each signer held out in turn. A weak label is never proposed.",
    }
    meta = {"labels": labels, "threshold": threshold, "frames": FRAMES, "features": 184,
            "weak_labels": [label for label in labels if label in set(weak_health) | set(weak_tn)],   # model order "model": "numpy-mlp", "hidden": list(hidden),
            "groups": {"everyday": everyday, **SIGN_SETS}, "metrics": metrics}
    args.out.mkdir(parents=True, exist_ok=True)
    model.save(args.out / "health_signs.npz")
    (args.out / "health_signs.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    args.plots.mkdir(parents=True, exist_ok=True)
    matrix = np.zeros((n, n), dtype=int)
    for truth, guess in zip(held_y, held_probs.argmax(axis=1)):
        matrix[truth, guess] += 1
    confusion_image(matrix, labels, args.plots / "health_signs_confusion.png")
    print(json.dumps({"threshold": threshold, "official_top1": official["top1"],
                      "tunisian_loso_top1": metrics["tunisian_leave_one_signer_out"]["top1"],
                      "weak": meta["weak_labels"], "recognised_tunisian": [c for c in everyday if c not in weak_tn]},
                     indent=1, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
