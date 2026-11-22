"""Train the Tunisian yes/no and day signs, and decide whether they may be used.

The Tunisian dataset is still images with no clip boundaries, so the model
classifies single frames. Its numbers are only meaningful if the held-out
frames come from a different person or take than the training frames, so a
gate decides whether signed yes/no is switched on at all:

    signing_enabled = yes/no held-out accuracy >= 0.90
                      and both "oui" and "non" were held out as a separate take

Steps (needs mediapipe and opencv installed):
    1. python -m training.train_tunisl10 --data <dataset>/Data --sheets sheets/
       Look at each contact sheet and write the frame ranges of each person or
       take into training/tunisl_blocks.json, e.g. {"oui": [[0, 11], [12, 27]]}.
    2. PYTHONPATH=Demo python -m training.train_tunisl10 --data <dataset>/Data \
           --out Demo/consultation/models

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
    parser.add_argument("--blocks", type=Path, default=Path("training/tunisl_blocks.json"))
    parser.add_argument("--out", type=Path, default=Path("Demo/consultation/models"))
    args = parser.parse_args()
    if args.sheets:
        write_sheets(args.data, args.sheets)
        return

    from training.mlp import train_mlp

    blocks = json.loads(args.blocks.read_text(encoding="utf-8")) if args.blocks.exists() else {}
    labels = list(SIGNS)
    train_x, train_y, held_x, held_y, weak = [], [], [], [], {}
    for index, (sign, category) in enumerate(SIGNS.items()):
        files = frame_files(args.data / category / sign)
        features, positions = extract_features(files)
        train_pos, held_pos, weak[sign] = holdout_split(len(files), blocks.get(sign, [[0, len(files) - 1]]))
        in_train = np.isin(positions, train_pos)
        train_x.append(features[in_train])
        held_x.append(features[~in_train])
        train_y += [index] * int(in_train.sum())
        held_y += [index] * int((~in_train).sum())
        print(f"{sign}: {len(files)} frames, {len(positions)} with a hand, "
              f"{int((~in_train).sum())} held out{' (weak)' if weak[sign] else ''}")
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
