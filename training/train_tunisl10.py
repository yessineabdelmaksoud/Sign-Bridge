"""Train the Tunisian yes/no and day signs, and decide whether they may be used.

The Tunisian dataset is still images with no clip boundaries, so the model
classifies single frames. Its numbers are only meaningful if the held-out
frames come from a different person or take than the training frames, so a
gate decides whether signed yes/no is switched on at all:

    signing_enabled = yes/no held-out accuracy >= 0.90
                      and both "oui" and "non" were held out as a separate take

Frames of one sign come from several signers, interleaved. The script groups
frames by signer from their colours (clothing and background), then holds one
whole person out of every sign they appear in.

Usage (needs mediapipe and opencv installed):
    PYTHONPATH=Demo python -m training.train_tunisl10 --data <dataset>/Data
    python -m training.train_tunisl10 --data <dataset>/Data --sheets sheets/   # to look at the frames

Dataset: github.com/W7CH/First-ever-Tunisian-Sign-Language-Dataset (CC BY 4.0).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

SIGNS = {"oui": "Demandes", "non": "Demandes", "lyoum": "Demandes", "thnin": "Jours", "thleth": "Jours",
         "erb3a": "Jours", "5mis": "Jours", "jom3a": "Jours", "sebt": "Jours", "a7ad": "Jours"}
TO_ANSWER = {"oui": True, "non": False, "lyoum": "TODAY", "thnin": "MON", "thleth": "TUE", "erb3a": "WED",
             "5mis": "THU", "jom3a": "FRI", "sebt": "SAT", "a7ad": "SUN"}
GATE_ACCURACY = 0.90


def passes_gate(acc_yes_no: float, weak_holdout: dict) -> bool:
    return acc_yes_no >= GATE_ACCURACY and not weak_holdout.get("oui", True) and not weak_holdout.get("non", True)


def holdout_split(n_frames: int, blocks: list) -> tuple[np.ndarray, np.ndarray, bool]:
    """(training positions, held-out positions, weak). Weak means same take, later frames."""
    if len(blocks) >= 2:
        first, last = blocks[-1]
        held = np.arange(first, last + 1)
        weak = False
    else:
        held = np.arange(n_frames - max(1, n_frames // 4), n_frames)
        weak = True
    return np.setdiff1d(np.arange(n_frames), held), held, weak


def colour_signature(path: Path) -> np.ndarray:
    """Mean colours of the whole frame, both top corners and the torso: enough to tell signers apart."""
    from PIL import Image

    pixels = np.asarray(Image.open(path).convert("RGB").resize((32, 32)), dtype=np.float64)
    regions = (pixels, pixels[:6, :6], pixels[:6, -6:], pixels[12:28, 10:22])
    return np.concatenate([region.mean(axis=(0, 1)) for region in regions])


def _kmeans(x: np.ndarray, k: int, rounds: int = 30) -> np.ndarray:
    centres = [x[0]]
    for _ in range(1, k):   # start from points far from each other
        distance = np.min([np.linalg.norm(x - c, axis=1) for c in centres], axis=0)
        centres.append(x[int(distance.argmax())])
    centres = np.asarray(centres)
    labels = np.zeros(len(x), dtype=int)
    for _ in range(rounds):
        labels = np.linalg.norm(x[:, None] - centres[None], axis=2).argmin(axis=1)
        centres = np.asarray([x[labels == j].mean(axis=0) if (labels == j).any() else centres[j] for j in range(k)])
    return labels


def _silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    distances = np.linalg.norm(x[:, None] - x[None], axis=2)
    scores = []
    for i, own in enumerate(labels):
        same = labels == own
        if same.sum() < 2:
            scores.append(0.0)
            continue
        inside = distances[i, same].sum() / (same.sum() - 1)
        outside = min(distances[i, labels == other].mean() for other in set(labels) if other != own)
        scores.append((outside - inside) / max(inside, outside))
    return float(np.mean(scores))


def cluster_signers(signatures: np.ndarray, max_k: int = 6) -> np.ndarray:
    """A signer number for each frame; one group if the frames do not split cleanly."""
    x = np.asarray(signatures, dtype=np.float64)
    best_labels, best_score = np.zeros(len(x), dtype=int), 0.5   # below 0.5: not clearly separate people
    for k in range(2, min(max_k, len(x) - 1) + 1):
        labels = _kmeans(x, k)
        if len(set(labels)) < k:
            continue
        score = _silhouette(x, labels)
        if score > best_score:
            best_labels, best_score = labels, score
    return best_labels


def choose_held_out_signer(signers: dict) -> int | None:
    """Who to keep out of training: someone seen signing both yes and no, in as many signs as possible,
    and with the fewest frames among those (so the most data stays for training)."""
    candidates = set(signers.get("oui", [])) & set(signers.get("non", []))
    if not candidates:
        return None

    def rank(person):
        covered = sum(person in set(ids) for ids in signers.values())
        frames = sum(int((np.asarray(ids) == person).sum()) for ids in signers.values())
        return (-covered, frames, person)

    return int(min(candidates, key=rank))


def frame_files(folder: Path) -> list[Path]:
    def number(path: Path) -> int:
        found = re.findall(r"\((\d+)\)", path.name)
        return int(found[0]) if found else 0

    return sorted((p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}), key=number)


def write_sheets(data: Path, out: Path) -> None:
    """One contact sheet per sign, frames numbered by position, for marking the takes by hand."""
    from PIL import Image, ImageDraw

    out.mkdir(parents=True, exist_ok=True)
    for sign, category in SIGNS.items():
        files = frame_files(data / category / sign)
        columns, size = 10, 112
        sheet = Image.new("RGB", (columns * size, ((len(files) + columns - 1) // columns) * size), "white")
        draw = ImageDraw.Draw(sheet)
        for position, path in enumerate(files):
            x, y = (position % columns) * size, (position // columns) * size
            sheet.paste(Image.open(path).convert("RGB").resize((size, size)), (x, y))
            draw.rectangle((x, y, x + 26, y + 13), fill="white")
            draw.text((x + 2, y + 1), str(position), fill="black")
        sheet.save(out / f"{sign}.jpg", quality=80)
        print(f"{sign}: {len(files)} frames -> {out / (sign + '.jpg')}")


def extract_features(files: list[Path]) -> tuple[np.ndarray, list[int]]:
    """Landmark features for the frames where a hand was found, and their positions."""
    import cv2
    import mediapipe as mp

    from consultation.features import frame_features
    from consultation.landmarks import canonical_from_holistic, hand_present

    features, positions = [], []
    with mp.solutions.holistic.Holistic(static_image_mode=True, model_complexity=1) as holistic:
        for position, path in enumerate(files):
            image = cv2.imread(str(path))
            if image is None:
                continue
            frame = canonical_from_holistic(holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
            if hand_present(frame):
                features.append(frame_features(frame))
                positions.append(position)
    return np.asarray(features, dtype=np.float32).reshape(-1, 184), positions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, required=True, help="the dataset's Data folder")
    parser.add_argument("--sheets", type=Path, help="write contact sheets here and stop")
    parser.add_argument("--out", type=Path, default=Path("Demo/consultation/models"))
    args = parser.parse_args()
    if args.sheets:
        write_sheets(args.data, args.sheets)
        return

    from training.mlp import train_mlp

    labels = list(SIGNS)
    files = {sign: frame_files(args.data / category / sign) for sign, category in SIGNS.items()}
    everyone = cluster_signers(np.stack([colour_signature(p) for sign in labels for p in files[sign]]))
    signers, start = {}, 0
    for sign in labels:
        signers[sign] = everyone[start:start + len(files[sign])]
        start += len(files[sign])
    held_signer = choose_held_out_signer(signers)
    print(f"signers found: {len(set(everyone))}; held out of training: signer {held_signer}")

    train_x, train_y, held_x, held_y, weak = [], [], [], [], {}
    for index, sign in enumerate(labels):
        features, positions = extract_features(files[sign])
        held_pos = np.flatnonzero(signers[sign] == held_signer) if held_signer is not None else np.array([], dtype=int)
        weak[sign] = len(held_pos) == 0
        if weak[sign]:   # this person never signs it: fall back to the last quarter of the frames
            _, held_pos, _ = holdout_split(len(files[sign]), [[0, len(files[sign]) - 1]])
        in_held = np.isin(positions, held_pos)
        train_x.append(features[~in_held])
        held_x.append(features[in_held])
        train_y += [index] * int((~in_held).sum())
        held_y += [index] * int(in_held.sum())
        counts = {int(person): int((signers[sign] == person).sum()) for person in sorted(set(signers[sign]))}
        print(f"{sign}: {len(files[sign])} frames by signer {counts}, {len(positions)} with a hand, "
              f"{int(in_held.sum())} held out{' (weak: same people as training)' if weak[sign] else ''}")
    train_x, held_x = np.concatenate(train_x), np.concatenate(held_x)
    train_y, held_y = np.asarray(train_y), np.asarray(held_y)

    model, _ = train_mlp(train_x, train_y, len(labels), hidden=(64,), epochs=150, seed=42)
    predicted = model(held_x).argmax(axis=1) if len(held_x) else np.array([], dtype=int)
    yes_no = np.isin(held_y, [labels.index("oui"), labels.index("non")])
    acc_all = float((predicted == held_y).mean()) if len(held_y) else 0.0
    acc_yes_no = float((predicted[yes_no] == held_y[yes_no]).mean()) if yes_no.any() else 0.0
    enabled = passes_gate(acc_yes_no, weak)

    args.out.mkdir(parents=True, exist_ok=True)
    model.save(args.out / "tunisl10.npz")
    meta = {"labels": labels, "threshold": 0.6, "features": 184, "signing_enabled": enabled, "to_answer": TO_ANSWER,
            "metrics": {"heldout_top1": round(acc_all, 4), "acc_yes_no": round(acc_yes_no, 4),
                        "n_heldout": int(len(held_y)), "weak_holdout": weak}}
    (args.out / "tunisl10.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta["metrics"], indent=2), "\nsigning_enabled:", enabled)


if __name__ == "__main__":
    main()
