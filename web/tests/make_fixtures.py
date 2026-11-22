"""Write the fixtures the Node tests compare the JavaScript against.

Every expected value here is computed by the Python reference in
Demo/consultation: on the raw clips of the official SWL-LSE test split, and on
a few made-up edge cases. Two files are written next to this script:

  fixtures/parity.bin    little-endian float32: clips, frames and features
  fixtures/parity.json   everything else, and where each array sits in the .bin
                         (offsets and lengths count float32 values, not bytes)

Usage, from the repository root (needs data/swl, which is not in the repository):
    PYTHONPATH=Demo:. python web/tests/make_fixtures.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from consultation.capture import SignCapture
from consultation.features import FRAMES, clip_features, frame_features, resample, trim_to_hands
from consultation.flow import SIGN_SETS
from consultation.landmarks import LEFT, RIGHT, empty_frame, hand_present
from consultation.numpy_model import DenseModel
from consultation.recognizer import OTHER, SignRecognizer, narrow
from training.export_web import array_order
from training.swl_convert import read_annotations, select_ids

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "fixtures"
MODELS = ROOT / "Demo/consultation/models"
DATA = ROOT / "data/swl"

# The clips whose full feature arrays are stored, by what their hands do.
FEATURE_CLIPS = {
    "no hand in some frames": 3,
    "one hand comes and goes": 4,
    "left hand only": 2,
    "right hand only": 2,
    "both hands throughout": 1,
}


class Floats:
    """Collects float32 arrays for the .bin file and hands back their offsets."""

    def __init__(self):
        self._chunks, self._count = [], 0

    def add(self, array) -> int:
        flat = np.ascontiguousarray(array, dtype="<f4").ravel()
        offset, self._count = self._count, self._count + flat.size
        self._chunks.append(flat.tobytes())
        return offset

    def bytes(self) -> bytes:
        return b"".join(self._chunks)


def as_dict(prediction) -> dict:
    return {"label": prediction.label, "confidence": prediction.confidence,
            "top3": [[label, p] for label, p in prediction.top3]}


def frame(left=True, right=False, seed=0) -> np.ndarray:
    """A made-up frame, as in tests/test_features.py."""
    rng = np.random.default_rng(seed)
    f = rng.uniform(0.2, 0.8, (75, 3)).astype(np.float32)
    f[11, :2], f[12, :2] = (0.6, 0.4), (0.4, 0.4)
    if not left:
        f[LEFT] = np.nan
    if not right:
        f[RIGHT] = np.nan
    return f


# --- real clips -------------------------------------------------------------------

def hand_kind(span: np.ndarray) -> str:
    """What the hands do over the signed span of a clip."""
    left = np.isfinite(span[:, LEFT]).any(axis=(1, 2))
    right = np.isfinite(span[:, RIGHT]).any(axis=(1, 2))
    if (~left & ~right).any():
        return "no hand in some frames"
    if left.all() and right.all():
        return "both hands throughout"
    if left.all() and not right.any():
        return "left hand only"
    if right.all() and not left.any():
        return "right hand only"
    return "one hand comes and goes"


def first_and_stop(raw: np.ndarray) -> tuple[int, int]:
    """trim_to_hands as a pair of indices: raw[first:stop] is the signed span."""
    span = trim_to_hands(raw)
    first = next(i for i in range(len(raw)) if np.shares_memory(raw[i], span[0]))
    return first, first + len(span)


def replay(raw: np.ndarray, fps: float = 30.0):
    """The sign the capture logic cuts out of a raw clip, as training/replay_check.py does it."""
    capture, found = SignCapture(), None
    for index, f in enumerate(list(raw) + [empty_frame()] * 12):
        clip = capture.push(f, index / fps)
        if clip is not None:
            found = clip
    return found


def real_clips(floats: Floats, model: DenseModel, recogniser: SignRecognizer) -> list[dict]:
    annotations, class_names = read_annotations(DATA / "ANNOTATIONS.zip", DATA / "videos_ref_annotations.csv")
    signs = [code for codes in SIGN_SETS.values() for code in codes]
    group_of = {code: name for name, codes in SIGN_SETS.items() for code in codes}
    allowed_sets = {**SIGN_SETS, "all": signs}
    wanted = dict(FEATURE_CLIPS)

    clips = []
    for clip_id, label, split in select_ids(annotations, class_names, signs):
        if split != "test":
            continue
        raw = np.load(DATA / "clips" / f"{clip_id}.npy")
        features = clip_features(raw)
        first, stop = first_and_stop(raw)
        kind = hand_kind(raw[first:stop])
        captured = replay(raw)
        entry = {
            "id": clip_id, "label": label, "group": group_of.get(label), "hands": kind,
            "frames": len(raw), "offset": floats.add(raw),
            "trim": [first, stop],
            "resample": [int(i) for i in resample(np.arange(stop - first))],
            "features_offset": None,
            "probs": [float(p) for p in model(features[None])[0]],
            "predictions": {name: as_dict(recogniser.predict(raw, allowed)) for name, allowed in allowed_sets.items()},
            "replay": None if captured is None else {
                "frames": len(captured), "prediction": as_dict(recogniser.predict(captured, signs))},
        }
        if wanted.get(kind, 0) > 0:
            wanted[kind] -= 1
            entry["features_offset"] = floats.add(features)
        clips.append(entry)
    if any(wanted.values()):
        raise SystemExit(f"the test split has too few clips of these kinds: {wanted}")
    return clips


# --- single frames ----------------------------------------------------------------

def frame_cases(floats: Floats) -> list[dict]:
    cases = {
        "left hand only": frame(left=True, right=False, seed=1),
        "right hand only": frame(left=False, right=True, seed=2),
        "both hands": frame(left=True, right=True, seed=3),
        "no hand": frame(left=False, right=False, seed=4),
    }
    f = frame(left=True, right=True, seed=5)
    f[LEFT.start + 7, 0] = np.nan
    cases["one point of the left hand missing"] = f
    f = frame(left=True, right=True, seed=12)
    f[RIGHT.start + 20, 1] = np.nan
    cases["only the y of one right hand point missing"] = f
    f = frame(left=True, right=True, seed=6)
    f[:33] = np.nan
    cases["no body, both hands"] = f
    f = frame(left=True, right=True, seed=7)
    f[12, :2] = f[11, :2] + np.float32(1e-4)
    cases["shoulders closer than the floor"] = f
    f = frame(left=True, right=False, seed=8)
    f[LEFT] = f[LEFT.start]
    cases["hand collapsed to one point"] = f
    f = frame(left=True, right=False, seed=13)
    f[LEFT, :2] = f[LEFT.start, :2] + (f[LEFT, :2] - 0.5) * np.float32(1e-3)
    cases["hand smaller than the floor"] = f
    f = frame(left=True, right=True, seed=9)
    f[RIGHT, 2] = np.nan
    cases["hand with x and y but no depth"] = f
    f = frame(left=True, right=False, seed=10)
    f[LEFT, :2] = np.nan
    cases["hand with depth only"] = f
    f = frame(left=True, right=True, seed=11)
    f[12, :2] = f[11, :2]
    f[LEFT.start + 3, 0] = 1e36
    cases["a coordinate too large for float32 arithmetic"] = f
    f = frame(left=True, right=True, seed=14)
    f[11, 0], f[12, 0] = 1e20, -1e20
    cases["a shoulder width too large for float32 arithmetic"] = f

    out = []
    for name, f in cases.items():
        with np.errstate(over="ignore", invalid="ignore"):
            features = frame_features(f)
        out.append({"name": name, "hand_present": hand_present(f),
                    "frame_offset": floats.add(f), "features_offset": floats.add(features)})
    return out


# --- resampling -------------------------------------------------------------------

def resample_cases() -> list[dict]:
    pairs = [(length, FRAMES) for length in range(1, 81)]
    pairs += [(1, 5), (10, 1), (2, 3), (4, 3), (6, 3), (2, 4), (8, 16), (3, 2), (12, 5), (7, 4), (100, 9)]
    return [{"length": length, "frames": frames,
             "index": [int(i) for i in resample(np.arange(length), frames)]} for length, frames in pairs]


# --- narrowing --------------------------------------------------------------------

def narrow_cases() -> list[dict]:
    labels = ["DOLOR", "FIEBRE2", "TOSER", "CITA", OTHER]
    three = ["DOLOR", "FIEBRE2", "TOSER"]
    cases = [
        ("labels outside the question are ignored", [0.30, 0.10, 0.05, 0.50, 0.05], three, 0.5),
        ("below the threshold: no label, top 3 kept", [0.25, 0.20, 0.20, 0.0, 0.35], three, 0.6),
        ("OTHER wins: no label", [0.1, 0.1, 0.1, 0.0, 0.7], ["DOLOR", "FIEBRE2"], 0.3),
        ("nothing allowed", [0.30, 0.10, 0.05, 0.50, 0.05], [], 0.5),
        ("allowed labels the model does not know", [0.30, 0.10, 0.05, 0.50, 0.05], ["GRIPE"], 0.5),
        ("every allowed probability is zero", [0.0, 0.0, 0.0, 1.0, 0.0], ["DOLOR", "FIEBRE2"], 0.5),
        ("a tie keeps the order of the labels", [0.2, 0.2, 0.2, 0.2, 0.2], three, 0.2),
        ("a tie with OTHER goes to the sign", [0.4, 0.0, 0.0, 0.2, 0.4], ["DOLOR"], 0.3),
        ("exactly at the threshold is enough", [0.5, 0.25, 0.0, 0.0, 0.25], ["DOLOR", "FIEBRE2"], 0.5),
        ("just under the threshold is not", [0.49, 0.26, 0.0, 0.0, 0.25], ["DOLOR", "FIEBRE2"], 0.5),
        ("four allowed: the best three are kept", [0.1, 0.3, 0.2, 0.35, 0.05], three + ["CITA"], 0.3),
        ("the order of allowed does not matter", [0.1, 0.3, 0.2, 0.35, 0.05], ["CITA", "TOSER", "FIEBRE2"], 0.3),
        ("OTHER named in allowed changes nothing", [0.30, 0.10, 0.05, 0.50, 0.05], ["DOLOR", OTHER], 0.5),
        ("threshold zero always proposes the best sign", [0.01, 0.02, 0.0, 0.9, 0.01], three, 0.0),
    ]
    return [{"name": name, "probs": probs, "labels": labels, "allowed": allowed, "threshold": threshold,
             "expected": as_dict(narrow(np.array(probs), labels, allowed, threshold))}
            for name, probs, allowed, threshold in cases]


# --- capture ----------------------------------------------------------------------

def capture_cases() -> list[dict]:
    """Hand-there / hand-gone patterns, fed as tests/test_capture.py feeds them."""
    cases = [
        ("one sign, the trailing gap dropped", [0] * 5 + [1] * 20 + [0] * 10, 1 / 30),
        ("no hand at all", [0] * 300, 1 / 30),
        ("a short blip is discarded", [1] * 5 + [0] * 10, 1 / 30),
        ("cut at four seconds", [1] * 200, 1 / 30),
        ("recording starts with the third frame", [1] * 3, 1 / 30),
        ("recording stops once the hand has gone", [1] * 3 + [0] * 8, 1 / 30),
        ("two frames with a hand do not start a sign", [1, 1, 0] * 10, 1 / 30),
        ("a gap of seven frames stays inside the sign", [1] * 10 + [0] * 7 + [1] * 10 + [0] * 8, 1 / 30),
        ("two signs in a row", [0] * 2 + [1] * 15 + [0] * 9 + [1] * 12 + [0] * 8, 1 / 30),
        ("a sign still open when the frames stop", [0] * 4 + [1] * 30, 1 / 30),
        ("a slow camera is cut at four seconds too", [1] * 100, 0.1),
        ("a sign that ends just before the time limit", [1] * 110 + [0] * 12, 1 / 30),
        ("a frame exactly four seconds after the first ends the sign", [1] * 12, 0.5),
        ("eight frames are enough for a sign", [1] * 8 + [0] * 8, 1 / 30),
        ("seven frames are not", [1] * 7 + [0] * 8, 1 / 30),
    ]
    out = []
    for name, pattern, dt in cases:
        capture, clips, t = SignCapture(), [], 0.0
        for index, has_hand in enumerate(pattern):
            f = frame(left=bool(has_hand))
            f[0, 2] = index  # depth of the nose: not used by the capture, tells us which frames came back
            clip = capture.push(f, t)
            t += dt
            if clip is not None:
                kept = [int(i) for i in clip[:, 0, 2]]
                assert kept == list(range(kept[0], kept[0] + len(kept)))
                clips.append({"returned_at": index, "first": kept[0], "frames": len(kept)})
        out.append({"name": name, "pattern": pattern, "dt": dt, "clips": clips, "recording": capture.recording})
    return out


# --- writing ----------------------------------------------------------------------

def weights_sha256(npz_path: Path) -> str:
    """The checksum web/models/health_signs.bin must have to be this model."""
    with np.load(npz_path, allow_pickle=False) as data:
        names = array_order(int(data["layers"]))
        return hashlib.sha256(b"".join(np.ascontiguousarray(data[n], dtype="<f4").tobytes() for n in names)).hexdigest()


def dump(index: dict) -> str:
    """JSON with one line per case, so a change shows up as a readable diff."""
    parts = []
    for key, value in index.items():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            lines = ",\n".join("  " + json.dumps(item, ensure_ascii=False) for item in value)
            parts.append(f"{json.dumps(key)}: [\n{lines}\n]")
        else:
            parts.append(f"{json.dumps(key)}: {json.dumps(value, ensure_ascii=False)}")
    return "{\n" + ",\n".join(parts) + "\n}\n"


def main() -> None:
    recogniser = SignRecognizer(MODELS / "health_signs.npz", MODELS / "health_signs.json")
    if not recogniser.available:
        raise SystemExit(f"model not available: {recogniser.error}")
    model = DenseModel.load(MODELS / "health_signs.npz")
    floats = Floats()
    index = {
        "written_by": "web/tests/make_fixtures.py",
        "labels": recogniser.labels,
        "threshold": recogniser.threshold,
        "weak_labels": recogniser.weak_labels,
        "sign_sets": SIGN_SETS,
        "weights_sha256": weights_sha256(MODELS / "health_signs.npz"),
        "clips": real_clips(floats, model, recogniser),
        "frame_cases": frame_cases(floats),
        "resample_cases": resample_cases(),
        "narrow_cases": narrow_cases(),
        "capture_cases": capture_cases(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "parity.bin").write_bytes(floats.bytes())
    (OUT / "parity.json").write_text(dump(index), encoding="utf-8")
    stored = [clip for clip in index["clips"] if clip["features_offset"] is not None]
    print(json.dumps({"clips": len(index["clips"]), "with_features": len(stored),
                      "bin_bytes": (OUT / "parity.bin").stat().st_size,
                      "json_bytes": (OUT / "parity.json").stat().st_size}))


if __name__ == "__main__":
    main()
