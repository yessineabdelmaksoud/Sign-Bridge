"""Tunisian Sign Language everyday words: sequences, stick figures and training data.

The Tunisian dataset (Chakroun and Jerbi, with ATILS; CC BY 4.0) stores each
word as a folder of still frames from several signers, interleaved. Frames are
grouped by signer from their colours (clothing, background), and each run of
consecutive frames by one signer is taken as one performance ("take") of the word.

    PYTHONPATH=Demo python -m training.tunisian_signs extract --data data/tunsl/Data
    PYTHONPATH=Demo python -m training.tunisian_signs poses
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

VOCAB_PATH = Path(__file__).with_name("tunisian_vocab.json")
SEQ_DIR = Path("data/tunsl_seq")
POSE_FPS = 10   # the frames are a few per second of signing; played back at this rate
MIN_POSE_FRAMES = 8


def load_vocab(path: Path = VOCAB_PATH) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def takes(signers: np.ndarray, min_length: int = 6, bridge: int = 1) -> list[tuple[int, list[int]]]:
    """Runs of consecutive positions by one signer: [(signer, positions), ...].

    A run may jump over up to `bridge` frames of someone else; runs shorter
    than `min_length` are dropped.
    """
    runs: list[tuple[int, list[int]]] = []
    position = 0
    signers = list(signers)
    while position < len(signers):
        who = signers[position]
        members = [position]
        cursor = position + 1
        while cursor < len(signers):
            if signers[cursor] == who:
                members.append(cursor)
                cursor += 1
                continue
            ahead = signers[cursor + 1:cursor + 1 + bridge]
            if bridge and who in ahead:
                cursor += 1 + list(ahead).index(who)
                continue
            break
        if len(members) >= min_length:
            runs.append((int(who), members))
        position = members[-1] + 1
    return runs


SIGNERS = 7   # the dataset was signed by seven people; checked on "oui", where three appear


HAND_MODEL = Path("Demo/signtotext/hand_landmarker.task")


def attach_hands(frame: np.ndarray, hands: list[np.ndarray]) -> np.ndarray:
    """Put hands found by the hand detector into a canonical frame, each at the nearer pose wrist.

    The dedicated hand detector finds hands in these small, cropped images far
    more often than Holistic does; Holistic's pose says which hand is which.
    """
    from consultation.landmarks import LEFT, RIGHT

    out = np.array(frame, dtype=np.float32, copy=True)
    wrists = {LEFT: out[15, :2], RIGHT: out[16, :2]}
    free = [LEFT, RIGHT]
    ranked = []
    for hand in hands:
        wrist = np.asarray(hand, dtype=np.float32)[0, :2]
        distance = {side: float(np.linalg.norm(wrist - point)) if np.isfinite(point).all() else np.inf
                    for side, point in wrists.items()}
        ranked.append((min(distance.values()), hand, distance))
    for _, hand, distance in sorted(ranked, key=lambda item: item[0]):
        if not free:
            break
        side = min(free, key=lambda s: (distance[s], s.start))
        out[side] = np.asarray(hand, dtype=np.float32)
        free.remove(side)
    return out


# The Tunisian images are cropped at the chin, so MediaPipe guesses the shoulders and
# head badly while the hands are clear. steady_clip holds the body still for a take.
DEFAULT_SHOULDERS = ((0.75, 0.3), (0.25, 0.3))   # signer's left (image right), right; typical framing
HEAD_ABOVE = 0.85    # nose above the shoulder midpoint, in shoulder widths
LIMB = 0.95          # upper arm and forearm length, in shoulder widths


def _steady_shoulders(clip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left, right = clip[:, 11, :2], clip[:, 12, :2]
    width = np.linalg.norm(left - right, axis=1)
    level = np.abs(left[:, 1] - right[:, 1])
    inside = ((left >= 0) & (left <= 1)).all(axis=1) & ((right >= 0) & (right <= 1)).all(axis=1)
    ok = np.isfinite(width) & (width > 0.3) & (width < 0.9) & (level < 0.15) & inside & (left[:, 0] > right[:, 0])
    if ok.sum() == 0:
        return np.array(DEFAULT_SHOULDERS[0], np.float32), np.array(DEFAULT_SHOULDERS[1], np.float32)
    # The median of the plausible frames, then only the frames close to it (drops wild guesses).
    l_med, r_med = np.median(left[ok], axis=0), np.median(right[ok], axis=0)
    near = ok & (np.linalg.norm(left - l_med, axis=1) < 0.1) & (np.linalg.norm(right - r_med, axis=1) < 0.1)
    if near.any():
        l_med, r_med = np.median(left[near], axis=0), np.median(right[near], axis=0)
    return l_med.astype(np.float32), r_med.astype(np.float32)


def _keep_sides(clip: np.ndarray) -> np.ndarray:
    """Swap the two hands back when a frame has them the other way round."""
    from consultation.landmarks import LEFT, RIGHT

    out = clip.copy()
    last = {LEFT: None, RIGHT: None}
    for t in range(len(out)):
        l, r = out[t, LEFT.start, :2], out[t, RIGHT.start, :2]
        if last[LEFT] is not None and last[RIGHT] is not None and np.isfinite(l).all() and np.isfinite(r).all():
            straight = np.linalg.norm(l - last[LEFT]) + np.linalg.norm(r - last[RIGHT])
            swapped = np.linalg.norm(l - last[RIGHT]) + np.linalg.norm(r - last[LEFT])
            if swapped < straight:
                out[t, LEFT], out[t, RIGHT] = out[t, RIGHT].copy(), out[t, LEFT].copy()
        for side in (LEFT, RIGHT):
            if np.isfinite(out[t, side.start, :2]).all():
                last[side] = out[t, side.start, :2].copy()
    return out


def _elbow(shoulder: np.ndarray, wrist: np.ndarray, outward: float, limb: float) -> np.ndarray:
    """Two equal bones from shoulder to wrist; the elbow bends down and away from the body."""
    span = wrist - shoulder
    d = float(np.linalg.norm(span))
    if d < 1e-6:
        return shoulder + np.array([0.0, limb], np.float32)
    d = min(d, 2 * limb * 0.999)
    middle = shoulder + span / 2 if np.linalg.norm(span) <= 2 * limb else shoulder + span / np.linalg.norm(span) * d / 2
    rise = np.sqrt(max(limb ** 2 - (d / 2) ** 2, 0.0))
    normal = np.array([-span[1], span[0]]) / max(np.linalg.norm(span), 1e-6)
    candidates = (middle + normal * rise, middle - normal * rise)
    # Prefer the elbow that is lower (larger y) and on the outer side.
    return max(candidates, key=lambda e: e[1] + 0.5 * outward * (e[0] - shoulder[0])).astype(np.float32)


def fix_take_sides(clip: np.ndarray) -> np.ndarray:
    """Put a take's hands back on their sides when the whole take has them the wrong way.

    Seen facing the signer, the signer's left hand is on the image right. Used for the
    stick figures only (training mirrors clips anyway).
    """
    from consultation.landmarks import LEFT, RIGHT

    out = np.array(clip, dtype=np.float32, copy=True)
    middle = np.nanmedian((out[:, 11, 0] + out[:, 12, 0]) / 2) if np.isfinite(out[:, [11, 12], 0]).any() else 0.5
    width = np.nanmedian(np.abs(out[:, 11, 0] - out[:, 12, 0])) if np.isfinite(out[:, [11, 12], 0]).any() else 0.5
    left_x, right_x = out[:, LEFT.start, 0], out[:, RIGHT.start, 0]
    both = np.isfinite(left_x) & np.isfinite(right_x)
    if both.sum() >= max(3, len(out) // 3):
        if np.median(left_x[both]) < np.median(right_x[both]):      # crossed for the whole take
            out[:, [*range(LEFT.start, LEFT.stop), *range(RIGHT.start, RIGHT.stop)]] = \
                out[:, [*range(RIGHT.start, RIGHT.stop), *range(LEFT.start, LEFT.stop)]]
        return out
    margin = 0.25 * width
    if np.isfinite(left_x).any() and not np.isfinite(right_x).any() and np.nanmedian(left_x) < middle - margin:
        out[:, RIGHT], out[:, LEFT] = out[:, LEFT].copy(), np.nan
    elif np.isfinite(right_x).any() and not np.isfinite(left_x).any() and np.nanmedian(right_x) > middle + margin:
        out[:, LEFT], out[:, RIGHT] = out[:, RIGHT].copy(), np.nan
    return out


def steady_clip(clip: np.ndarray) -> np.ndarray:
    """A take with steady shoulders and head, hands kept on their side, and plausible elbows."""
    from consultation.landmarks import LEFT, RIGHT

    out = _keep_sides(np.array(clip, dtype=np.float32, copy=True))
    left_shoulder, right_shoulder = _steady_shoulders(out)
    width = float(np.linalg.norm(left_shoulder - right_shoulder))
    middle = (left_shoulder + right_shoulder) / 2
    out[:, 11, :2], out[:, 12, :2] = left_shoulder, right_shoulder
    out[:, 0, :2] = middle - np.array([0.0, HEAD_ABOVE * width], np.float32)
    limb = LIMB * width
    for shoulder, elbow_i, wrist_i, hand, outward in ((left_shoulder, 13, 15, LEFT, 1.0), (right_shoulder, 14, 16, RIGHT, -1.0)):
        for t in range(len(out)):
            hand_wrist = out[t, hand.start, :2]
            wrist = hand_wrist if np.isfinite(hand_wrist).all() else out[t, wrist_i, :2]
            if not np.isfinite(wrist).all():
                wrist = shoulder + np.array([0.15 * outward * width, 1.8 * width], np.float32)   # arm down
            out[t, wrist_i, :2] = wrist
            elbow = out[t, elbow_i, :2]
            plausible = (np.isfinite(elbow).all() and elbow[1] > shoulder[1] - 0.2 * width
                         and np.linalg.norm(elbow - shoulder) < 1.25 * limb
                         and np.linalg.norm(elbow - wrist) < 1.25 * limb)
            if not plausible:
                out[t, elbow_i, :2] = _elbow(shoulder, wrist, outward, limb)
    return out


def frame_landmarks(paths: list[Path], cache: Path) -> np.ndarray:
    """Canonical landmarks of every image, read once with MediaPipe and cached."""
    if cache.exists():
        return np.load(cache)
    import cv2
    import mediapipe as mp

    from consultation.landmarks import canonical_from_holistic

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3)
    frames = []
    with mp.solutions.holistic.Holistic(static_image_mode=True, model_complexity=1) as holistic, \
            mp.tasks.vision.HandLandmarker.create_from_options(options) as hand_detector:
        for path in paths:
            rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            frame = canonical_from_holistic(holistic.process(rgb))
            found = hand_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)).hand_landmarks
            hands = [np.array([(p.x, p.y, p.z) for p in hand], dtype=np.float32) for hand in found]
            frames.append(attach_hands(frame, hands) if hands else frame)
    clip = np.stack(frames)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, clip)
    return clip


def extract(data: Path, out: Path) -> dict:
    """Every take of every word in the vocabulary as landmarks; writes <out>/index.json."""
    from training.train_tunisl10 import _kmeans, colour_signature, frame_files

    vocab = load_vocab()
    files = {v["code"]: frame_files(data / v["folder"]) for v in vocab}
    everyone = _kmeans(np.stack([colour_signature(p) for v in vocab for p in files[v["code"]]]), SIGNERS)
    out.mkdir(parents=True, exist_ok=True)
    index, start = [], 0
    for v in vocab:
        paths = files[v["code"]]
        signers = everyone[start:start + len(paths)]
        start += len(paths)
        frames = frame_landmarks(paths, out / "frames" / f"{v['code']}.npy")
        for number, (who, positions) in enumerate(takes(signers)):
            clip = frames[positions]
            name = f"{v['code']}__s{who}_t{number}.npy"
            np.save(out / name, clip)
            hands = int(np.isfinite(clip[:, 33:]).any(axis=(1, 2)).sum())
            index.append({"code": v["code"], "signer": int(who), "file": name, "frames": len(clip), "hand_frames": hands})
        done = [e for e in index if e["code"] == v["code"]]
        print(f"{v['code']}: {len(paths)} images, {len(done)} takes, signers {sorted({e['signer'] for e in done})}", flush=True)
    (out / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    return {"takes": len(index), "signers": int(len(set(everyone)))}


def best_take(entries: list[dict]) -> dict | None:
    """The take to draw: the most frames with a hand, then the longest."""
    usable = [e for e in entries if e["hand_frames"] >= 5]
    return max(usable, key=lambda e: (e["hand_frames"], e["frames"])) if usable else None


def add_poses(seq_dir: Path, poses_path: Path) -> list[str]:
    """Add one stick-figure sign per Tunisian word to the sign poses file."""
    from training.export_poses import encode_sign

    index = json.loads((seq_dir / "index.json").read_text(encoding="utf-8"))
    data = json.loads(poses_path.read_text(encoding="utf-8"))
    added = []
    for v in load_vocab():
        candidates = sorted((e for e in index if e["code"] == v["code"] and e["hand_frames"] >= 5),
                            key=lambda e: (e["hand_frames"], e["frames"]), reverse=True)
        for take in candidates:   # the best take whose body was found too
            with np.errstate(all="ignore"):
                sign = encode_sign(steady_clip(fix_take_sides(np.load(seq_dir / take["file"]))), pad=1)
            if len(sign["frames"]) >= MIN_POSE_FRAMES:
                sign["fps"] = POSE_FPS
                data["signs"][v["code"]] = sign
                added.append(v["code"])
                break
        else:
            print(f"{v['code']}: no take with a body and hands, no stick figure")
    data["source"] = ("SWL-LSE reference videos (Universidade de Vigo, CC BY 4.0) and the Tunisian Sign Language "
                      "Dataset (Chakroun and Jerbi, with ATILS, CC BY 4.0); movement only")
    poses_path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=["extract", "poses"])
    parser.add_argument("--data", type=Path, default=Path("data/tunsl/Data"))
    parser.add_argument("--seq", type=Path, default=SEQ_DIR)
    parser.add_argument("--poses", type=Path, default=Path("web/data/sign_poses.json"))
    args = parser.parse_args()
    if args.step == "extract":
        print(extract(args.data, args.seq))
    else:
        added = add_poses(args.seq, args.poses)
        print(f"stick figures added for {len(added)} Tunisian signs")


if __name__ == "__main__":
    main()
