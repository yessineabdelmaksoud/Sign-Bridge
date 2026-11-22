"""Build the health-sign training set from SWL-LSE (Zenodo record 13691887, CC BY 4.0).

Each SWL-LSE clip is a pickle: a list of frames, each a dict with the keys
`pose`, `hands` and `holistic_legacy`. Pickles can run code when opened with
the standard loader, so this module never does that. `load_frames` replaces
every class the pickle names with an inert stand-in, and the landmark lists
(serialised protobuf messages) are decoded by hand. Nothing from the file is
executed, and MediaPipe is not needed.

Usage:
    PYTHONPATH=Demo python -m training.swl_convert --out data/swl22.npz
"""
from __future__ import annotations

import argparse
import csv
import io
import pickle
import random
import struct
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from consultation.features import FRAMES, NoHandError, resample, trim_to_hands
from consultation.flow import SIGN_SETS
from consultation.landmarks import LEFT, POSE, RIGHT, empty_frame

RECORD = "https://zenodo.org/api/records/13691887/files"
OTHER = "OTHER"
OTHER_QUOTA = {"train": 120, "val": 12, "test": 12}
SPLITS = ("train", "val", "test")


# --- reading one pickle safely -------------------------------------------------

class _Inert:
    """Stands in for any class a pickle refers to. It stores state and does nothing."""

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj.state = None
        return obj

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.state = state


class _InertUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return type(name, (_Inert,), {})


def load_frames(data: bytes) -> list:
    """Unpickle an SWL-LSE clip without importing or calling anything it names."""
    return _InertUnpickler(io.BytesIO(data)).load()


def _varint(data: bytes, pos: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, pos
        shift += 7


def _fields(data: bytes):
    """Yield (field number, wire type, value) for one protobuf message."""
    pos = 0
    while pos < len(data):
        key, pos = _varint(data, pos)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _varint(data, pos)
        elif wire == 1:
            value, pos = data[pos:pos + 8], pos + 8
        elif wire == 2:
            length, pos = _varint(data, pos)
            value, pos = data[pos:pos + length], pos + length
        elif wire == 5:
            value, pos = data[pos:pos + 4], pos + 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield number, wire, value


def parse_landmark_list(data: bytes) -> np.ndarray:
    """Decode a serialised (Normalized)LandmarkList into an (n, 3) array of x, y, z."""
    points = []
    for number, wire, value in _fields(data):
        if number != 1 or wire != 2:
            continue
        xyz = [0.0, 0.0, 0.0]
        for field, field_wire, raw in _fields(value):
            if field_wire == 5 and 1 <= field <= 3:
                xyz[field - 1] = struct.unpack("<f", raw)[0]
        points.append(xyz)
    return np.asarray(points, dtype=np.float32).reshape(-1, 3)


def _serialized(obj) -> bytes | None:
    state = getattr(obj, "state", None)
    if isinstance(state, tuple):  # (dict state, slots state)
        state = next((s for s in state if isinstance(s, dict)), None)
    if isinstance(state, dict) and isinstance(state.get("serialized"), bytes):
        return state["serialized"]
    return None


def frame_to_canonical(frame: dict) -> np.ndarray:
    """One SWL-LSE frame (from `load_frames`) as a canonical (75, 3) frame."""
    holistic = frame.get("holistic_legacy") or {}
    out = empty_frame()
    for block, key in ((POSE, "pose_landmarks"), (LEFT, "left_hand_landmarks"), (RIGHT, "right_hand_landmarks")):
        data = _serialized(holistic.get(key))
        if data is None:
            continue
        points = parse_landmark_list(data)
        size = block.stop - block.start
        if len(points) == size:
            out[block] = points
    return out


def pickle_to_clip(data: bytes) -> np.ndarray:
    frames = [frame_to_canonical(f) for f in load_frames(data) if isinstance(f, dict)]
    return np.stack(frames) if frames else np.empty((0, 75, 3), dtype=np.float32)


# --- choosing which clips to fetch ----------------------------------------------

def select_ids(annotations: dict, class_names: dict, signs: list[str], other: dict = OTHER_QUOTA,
               seed: int = 42) -> list[tuple[str, str, str]]:
    """All clips of the chosen signs, plus a seeded sample of the others as OTHER.

    annotations: {"train"|"val"|"test": [(clip id, class id), ...]}
    Returns [(clip id, label, split), ...].
    """
    wanted = set(signs)
    picked: list[tuple[str, str, str]] = []
    rng = random.Random(seed)
    for split in SPLITS:
        rest = []
        for clip_id, class_id in annotations.get(split, []):
            label = class_names.get(class_id)
            if label in wanted:
                picked.append((clip_id, label, split))
            else:
                rest.append(clip_id)
        rest.sort()
        for clip_id in rng.sample(rest, min(other.get(split, 0), len(rest))):
            picked.append((clip_id, OTHER, split))
    return picked


def read_annotations(annotations_zip: Path, glosses_csv: Path) -> tuple[dict, dict]:
    with open(glosses_csv, encoding="utf-8") as handle:
        class_names = {int(row["CLASS_ID"]): row["LABEL"].strip() for row in csv.DictReader(handle)}
    annotations = {}
    with zipfile.ZipFile(annotations_zip) as archive:
        for split in SPLITS:
            text = archive.read(f"ANNOTATIONS/{split}_labels.csv").decode("utf-8")
            annotations[split] = [(clip_id, int(class_id)) for clip_id, class_id in csv.reader(io.StringIO(text))]
    return annotations, class_names


# --- fetching and assembling ------------------------------------------------------

def _download(name: str, target: Path) -> Path:
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(f"{RECORD}/{name}/content", headers={"User-Agent": "curl"})
        with urllib.request.urlopen(request, timeout=120) as response:
            target.write_bytes(response.read())
    return target


def fetch_clips(picked, cache: Path, workers: int = 4) -> None:
    """Fetch every missing clip into `cache` as <id>.npy (raw canonical frames)."""
    from training.swl_remote import HttpRangeReader, RemoteZip

    cache.mkdir(parents=True, exist_ok=True)
    todo = [clip_id for clip_id, _, _ in picked if not (cache / f"{clip_id}.npy").exists()]
    print(f"{len(picked)} clips wanted, {len(todo)} to fetch", flush=True)
    if not todo:
        return
    remote = RemoteZip(HttpRangeReader(f"{RECORD}/MEDIAPIPE.zip/content"))
    done = 0

    def one(clip_id: str) -> None:
        clip = pickle_to_clip(remote.read(f"MEDIAPIPE/{clip_id}.pkl"))
        partial = cache / f"{clip_id}.part.npy"  # written whole, then renamed, so a kill leaves no half file
        np.save(partial, clip)
        partial.rename(cache / f"{clip_id}.npy")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(one, todo):
            done += 1
            if done % 50 == 0 or done == len(todo):
                print(f"fetched {done}/{len(todo)}", flush=True)


def cached_only(picked, cache: Path) -> list:
    """The picked clips that are already in the cache (for working without a connection)."""
    return [item for item in picked if (cache / f"{item[0]}.npy").exists()]


def build_dataset(picked, cache: Path, out_path: Path) -> dict:
    """Trim and resample the cached clips and save the training file."""
    labels = [code for name in ("reason", "symptom", "conditions") for code in SIGN_SETS[name]] + [OTHER]
    index = {label: i for i, label in enumerate(labels)}
    clips, targets, splits, skipped = [], [], [], 0
    for clip_id, label, split in picked:
        raw = np.load(cache / f"{clip_id}.npy")
        try:
            clips.append(resample(trim_to_hands(raw), FRAMES))
        except NoHandError:
            skipped += 1
            continue
        targets.append(index[label])
        splits.append(SPLITS.index(split))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path, clips=np.stack(clips).astype(np.float32), y=np.asarray(targets, dtype=np.int16),
        split=np.asarray(splits, dtype=np.int8), labels=np.asarray(labels))
    counts = {label: int(sum(1 for t in targets if t == i)) for label, i in index.items()}
    return {"clips": len(clips), "skipped_no_hand": skipped, "per_label": counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=Path("data/swl"))
    parser.add_argument("--out", type=Path, default=Path("data/swl22.npz"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--offline", action="store_true", help="use only the clips already downloaded")
    args = parser.parse_args()

    annotations, class_names = read_annotations(
        _download("ANNOTATIONS.zip", args.data / "ANNOTATIONS.zip"),
        _download("videos_ref_annotations.csv", args.data / "videos_ref_annotations.csv"))
    signs = [code for codes in SIGN_SETS.values() for code in codes]
    missing = sorted(set(signs) - set(class_names.values()))
    if missing:
        raise SystemExit(f"signs not found in SWL-LSE: {missing}")
    picked = select_ids(annotations, class_names, signs)
    if args.offline:
        picked = cached_only(picked, args.data / "clips")
    else:
        fetch_clips(picked, args.data / "clips", args.workers)
    report = build_dataset(picked, args.data / "clips", args.out)
    print(report)
    if report["skipped_no_hand"] > 0.05 * len(picked):
        raise SystemExit("more than 5% of clips have no visible hand: inspect before training")


if __name__ == "__main__":
    main()
