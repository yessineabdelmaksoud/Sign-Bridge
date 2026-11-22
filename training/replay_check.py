"""Replay held-out SWL-LSE clips through the same capture logic the live screen uses.

Each raw clip (rest frames included) is pushed frame by frame into the capture
logic, and whatever sign it cuts out goes to the shipped recogniser, narrowed
to the question that sign belongs to. This checks capture, features, model
file and threshold together, without a camera.

Usage:
    PYTHONPATH=Demo python -m training.replay_check
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from consultation.capture import SignCapture
from consultation.flow import SIGN_SETS
from consultation.landmarks import empty_frame
from consultation.recognizer import SignRecognizer
from training.swl_convert import read_annotations, select_ids


def replay(raw: np.ndarray, fps: float = 30.0):
    """The sign the capture logic cuts out of a raw clip, or None."""
    capture = SignCapture()
    frames = list(raw) + [empty_frame()] * 12   # the hands drop out of view at the end
    found = None
    for index, frame in enumerate(frames):
        clip = capture.push(frame, index / fps)
        if clip is not None:
            found = clip   # keep the last finished sign, as the live screen would show it
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=Path("data/swl"))
    parser.add_argument("--models", type=Path, default=Path("Demo/consultation/models"))
    args = parser.parse_args()

    recogniser = SignRecognizer(args.models / "health_signs.npz", args.models / "health_signs.json")
    if not recogniser.available:
        raise SystemExit(f"model not available: {recogniser.error}")
    annotations, class_names = read_annotations(args.data / "ANNOTATIONS.zip", args.data / "videos_ref_annotations.csv")
    signs = [code for codes in SIGN_SETS.values() for code in codes]
    question_of = {code: name for name, codes in SIGN_SETS.items() for code in codes}

    outcome = {name: Counter() for name in SIGN_SETS}
    for clip_id, label, split in select_ids(annotations, class_names, signs):
        path = args.data / "clips" / f"{clip_id}.npy"
        if split == "train" or label not in question_of or not path.exists():
            continue
        name = question_of[label]
        clip = replay(np.load(path))
        if clip is None:
            outcome[name]["not captured"] += 1
            continue
        prediction = recogniser.predict(clip, SIGN_SETS[name])
        if prediction.label is None:
            outcome[name]["no proposal (asks again)"] += 1
        elif prediction.label == label:
            outcome[name]["proposed correctly"] += 1
        else:
            outcome[name]["proposed wrongly (patient must say no)"] += 1
    report = {name: dict(counts) | {"n": sum(counts.values())} for name, counts in outcome.items()}
    print(json.dumps({"threshold": recogniser.threshold, "weak_labels": recogniser.weak_labels, **report},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
