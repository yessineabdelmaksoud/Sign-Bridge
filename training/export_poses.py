"""Turn each reference sign video into stick-figure data for the patient screen.

The patient never sees the dataset's videos of real people. Instead, MediaPipe
Holistic reads each reference video once, here, and only the movement of the
upper body and hands is kept: normalised to the signer's shoulders, gaps in
hand tracking bridged, jitter smoothed. web/lib/avatar.js draws it as lines.

Usage (needs mediapipe and opencv):
    PYTHONPATH=Demo python -m training.export_poses
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from consultation.landmarks import LEFT, POSE, RIGHT, canonical_from_holistic

POSE_POINTS = [0, 11, 12, 13, 14, 15, 16]   # nose (head), shoulders, elbows, wrists
DECIMALS = 3


def normalise(clip: np.ndarray):
    """(pose, left, right): x, y in shoulder widths from the shoulder midpoint, y pointing down.

    One transform for the whole clip (medians over frames), so the body's own
    movement is kept while where the signer stands and their size are removed.
    """
    xy = np.asarray(clip, dtype=np.float64)[:, :, :2]
    shoulders = xy[:, [11, 12]]
    origin = np.nanmedian(shoulders.mean(axis=1), axis=0)
    width = np.nanmedian(np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1))
    width = width if np.isfinite(width) and width > 1e-6 else 1.0
    body = (xy - origin) / width
    return body[:, POSE_POINTS], body[:, LEFT], body[:, RIGHT]


def fill_gaps(seq: np.ndarray, max_gap: int = 4) -> np.ndarray:
    """Linearly bridge runs of missing frames no longer than max_gap, between two present frames."""
    out = np.array(seq, dtype=np.float64, copy=True)
    present = np.isfinite(out).all(axis=tuple(range(1, out.ndim)))
    index = np.flatnonzero(present)
    for a, b in zip(index[:-1], index[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            for k in range(1, gap + 1):
                w = k / (gap + 1)
                out[a + k] = (1 - w) * out[a] + w * out[b]
    return out


def smooth(seq: np.ndarray, window: int = 3) -> np.ndarray:
    """Centred moving average over time, using only present frames; missing frames stay missing."""
    out = np.array(seq, dtype=np.float64, copy=True)
    present = np.isfinite(seq).all(axis=tuple(range(1, seq.ndim)))
    half = window // 2
    for t in np.flatnonzero(present):
        lo, hi = max(0, t - half), min(len(seq), t + half + 1)
        near = [i for i in range(lo, hi) if present[i]]
        out[t] = np.mean(seq[near], axis=0)
    return out


def trim(clip: np.ndarray, pad: int = 4) -> np.ndarray:
    """From `pad` frames before the first frame with a hand to `pad` frames after the last."""
    has_hand = np.isfinite(clip[:, LEFT.start:RIGHT.stop]).any(axis=(1, 2))
    index = np.flatnonzero(has_hand)
    if index.size == 0:
        return clip
    return clip[max(0, index[0] - pad):index[-1] + pad + 1]


def _flat(points: np.ndarray):
    if not np.isfinite(points).all():
        return None
    return [round(float(v), DECIMALS) for v in points.ravel()]


def encode_sign(clip: np.ndarray, pad: int = 4) -> dict:
    """One sign as {"frames": [{"p": [...], "l": [...] | null, "r": [...] | null}, ...]}."""
    pose, left, right = normalise(trim(clip, pad))
    pose = smooth(fill_gaps(pose, 6))
    left, right = smooth(fill_gaps(left)), smooth(fill_gaps(right))
    frames = []
    last_pose = None
    for p, l, r in zip(pose, left, right):
        flat_pose = _flat(p) or last_pose
        if flat_pose is None:
            continue
        last_pose = flat_pose
        frames.append({"p": flat_pose, "l": _flat(l), "r": _flat(r)})
    return {"frames": frames}


def track_video(path: Path) -> tuple[np.ndarray, float]:
    """Canonical frames of a video, read with MediaPipe Holistic, and its frame rate."""
    import cv2
    import mediapipe as mp

    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
    frames = []
    with mp.solutions.holistic.Holistic(model_complexity=1, smooth_landmarks=True) as holistic:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            frames.append(canonical_from_holistic(holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))))
    capture.release()
    return np.stack(frames), float(fps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--clips", type=Path, default=Path("web/clips"))
    parser.add_argument("--signs", type=Path, default=Path("web/data/signs.json"))
    parser.add_argument("--out", type=Path, default=Path("web/data/sign_poses.json"))
    args = parser.parse_args()

    codes = [s["code"] for s in json.loads(args.signs.read_text(encoding="utf-8"))["signs"]]
    signs, fps_seen = {}, set()
    for code in codes:
        clip, fps = track_video(args.clips / f"{code}.mp4")
        fps_seen.add(round(fps))
        signs[code] = encode_sign(clip)
        hands = sum(1 for f in signs[code]["frames"] if f["l"] or f["r"])
        print(f"{code}: {len(clip)} video frames -> {len(signs[code]['frames'])} kept, {hands} with a hand")
    data = {
        "fps": fps_seen.pop() if len(fps_seen) == 1 else 20,
        "pose_points": POSE_POINTS,
        "units": "x, y in shoulder widths from the shoulder midpoint; y points down; as seen facing the signer",
        "source": "SWL-LSE reference videos (Universidade de Vigo, CC BY 4.0), movement only",
        "signs": signs,
    }
    args.out.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {args.out} ({args.out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
