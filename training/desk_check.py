"""How well signs are caught when the hands stay visible at rest, as at a desk.

In the training recordings the hands leave the picture between signs. At a desk
they usually stay visible, resting low. This replays the held-out SWL-LSE clips
three ways through the capture and the shipped model, with every proposed sign
allowed (as on the patient screen):

  as recorded   hands out of the picture between signs
  at a desk     both hands visible at rest for 1 s before and after the sign,
                and a hand that is not signing stays visible at rest
  desk + fix    the same, with resting hands dropped before the capture

    PYTHONPATH=Demo python -m training.desk_check
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from consultation.capture import SignCapture
from consultation.flow import SIGN_SETS
from consultation.landmarks import LEFT, RIGHT, empty_frame
from consultation.recognizer import OTHER, SignRecognizer
from training.swl_convert import read_annotations, select_ids
from training.tools import drop_resting_hands

FPS = 20
REST_SECONDS = 1.0


def resting_hand(frame: np.ndarray, side: slice, template: np.ndarray) -> np.ndarray:
    """`template` (21, 3) moved so its wrist rests on the desk, low on that side of the body."""
    shoulders = frame[[11, 12], :2]
    middle, width = shoulders.mean(axis=0), np.linalg.norm(shoulders[0] - shoulders[1])
    side_x = 0.45 if side == LEFT else -0.45            # the signer's left is on the image right
    wrist = middle + np.array([side_x * width, 2.0 * width])
    hand = template.copy()
    hand[:, :2] = hand[:, :2] - hand[0, :2] + wrist
    return hand


def at_a_desk(clip: np.ndarray) -> np.ndarray:
    """The clip as it would look with the hands visible at rest."""
    clip = np.array(clip, dtype=np.float32, copy=True)
    has_hand = np.isfinite(clip[:, LEFT.start:RIGHT.stop]).any(axis=(1, 2))
    first = clip[np.flatnonzero(has_hand)[0]]
    template = first[LEFT] if np.isfinite(first[LEFT]).all() else first[RIGHT]
    body = first.copy()
    body[LEFT.start:RIGHT.stop] = np.nan
    rest = body.copy()
    rest[LEFT], rest[RIGHT] = resting_hand(body, LEFT, template), resting_hand(body, RIGHT, template)
    for t in range(len(clip)):
        for side in (LEFT, RIGHT):
            if not np.isfinite(clip[t, side]).all():
                clip[t, side] = rest[side]
    pad = np.stack([rest] * int(REST_SECONDS * FPS))
    return np.concatenate([pad, clip, pad])


def capture(frames, drop_rest: bool):
    cap = SignCapture()
    found = None
    for i, frame in enumerate(list(frames) + [empty_frame()] * 12):
        clip = cap.push(drop_resting_hands(frame) if drop_rest else frame, i / FPS)
        if clip is not None:
            found = clip
    return found


def main() -> None:
    models, data = Path("Demo/consultation/models"), Path("data/swl")
    recogniser = SignRecognizer(models / "health_signs.npz", models / "health_signs.json")
    allowed = [label for label in recogniser.labels if label != OTHER and label not in recogniser.weak_labels]
    annotations, names = read_annotations(data / "ANNOTATIONS.zip", data / "videos_ref_annotations.csv")
    signs = [code for codes in SIGN_SETS.values() for code in codes]
    scenarios = {"as recorded": (False, False), "as recorded + fix": (False, True),
                 "at a desk": (True, False), "desk + fix": (True, True)}
    results = {name: Counter() for name in scenarios}
    for clip_id, label, split in select_ids(annotations, names, signs):
        path = data / "clips" / f"{clip_id}.npy"
        if split == "train" or label == OTHER or not path.exists():
            continue
        raw = np.load(path)
        for name, (desk, fix) in scenarios.items():
            clip = capture(at_a_desk(raw) if desk else raw, fix)
            if clip is None:
                results[name]["not caught"] += 1
                continue
            prediction = recogniser.predict(clip, allowed)
            key = "asks again" if prediction.label is None else "right" if prediction.label == label else "wrong"
            results[name][key] += 1
    print(json.dumps({name: dict(counts) | {"n": sum(counts.values())} for name, counts in results.items()}, indent=1))


if __name__ == "__main__":
    main()
