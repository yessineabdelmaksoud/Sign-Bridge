"""Export the health sign model and the sign list for the static web app.

Writes three files under web/:
  models/health_signs.bin   every model array as little-endian float32, back to
                            back, in the order the manifest lists them
  models/health_signs.json  the manifest: labels, threshold, weak labels, where
                            each array sits in the .bin, and the headline metrics
  data/signs.json           the 22 signs: code, group, pictogram, French and
                            Arabic names, reference clip, weak flag

Usage:
    PYTHONPATH=Demo python -m training.export_web
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from consultation.flow import SIGN_SETS, Flow
from consultation.recognizer import OTHER

ROOT = Path(__file__).resolve().parents[1]

# flow.json has a pictogram for every sign but none for the three questions.
GROUP_ICONS = {"reason": "🏥", "symptom": "🩺", "conditions": "📋"}

# Copied from the Python meta file as they are. The per-sign table stays behind.
HEADLINE_METRICS = ("official_test", "official_validation", "cv_folds", "cv_top1", "cv_top3",
                    "cv_narrowed_top1", "split_signer_independent", "note")


def array_order(layers: int) -> list[str]:
    """The order of the arrays in the .bin file."""
    return ["mean", "std"] + [f"{kind}{i}" for i in range(layers) for kind in ("w", "b")]


def pack_model(npz_path: Path) -> tuple[bytes, list[dict], int]:
    """The .bin content, the manifest's description of it, and the layer count."""
    with np.load(npz_path, allow_pickle=False) as data:
        layers = int(data["layers"])
        arrays = {name: np.ascontiguousarray(data[name], dtype="<f4") for name in array_order(layers)}
    chunks, described, offset = [], [], 0
    for name, array in arrays.items():
        described.append({"name": name, "shape": list(array.shape), "byte_offset": offset, "length": int(array.size)})
        chunks.append(array.tobytes())
        offset += array.nbytes
    return b"".join(chunks), described, layers


def build_manifest(meta: dict, described: list[dict], layers: int, blob: bytes) -> dict:
    return {
        "labels": meta["labels"],
        "threshold": meta["threshold"],
        "weak_labels": meta.get("weak_labels", []),
        "frames": meta["frames"],
        "features": meta["features"],
        "model": meta.get("model"),
        "hidden": meta.get("hidden"),
        "layers": layers,
        "weights": "health_signs.bin",
        "dtype": "float32",
        "byte_order": "little",
        "byte_length": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "arrays": described,
        "metrics": {key: meta["metrics"][key] for key in HEADLINE_METRICS if key in meta.get("metrics", {})},
    }


def build_signs(labels: list[str], weak: list[str]) -> dict:
    """The sign list, in the model's label order, grouped as flow.json groups them."""
    flow = Flow.load()
    signs = [
        {"code": code, "group": group, "icon": flow.icon(code), "fr": flow.label(code, "fr"),
         "ar": flow.label(code, "ar"), "clip": f"clips/{code}.mp4", "weak": code in weak}
        for group, codes in SIGN_SETS.items() for code in codes
    ]
    codes = [sign["code"] for sign in signs]
    if codes + [OTHER] != labels:
        raise SystemExit("the signs in flow.json are not the labels the model was trained on")
    if set(SIGN_SETS) != set(GROUP_ICONS):
        raise SystemExit(f"GROUP_ICONS must name exactly these groups: {sorted(SIGN_SETS)}")
    return {"other": OTHER, "groups": [{"id": group, "icon": GROUP_ICONS[group]} for group in SIGN_SETS],
            "signs": signs}


def check_shapes(meta: dict, described: list[dict]) -> None:
    shapes = {item["name"]: item["shape"] for item in described}
    width = meta["frames"] * meta["features"]
    last = max(int(name[1:]) for name in shapes if name.startswith("w"))
    if shapes["mean"] != [width] or shapes["std"] != [width] or shapes["w0"][0] != width:
        raise SystemExit(f"the model does not take {meta['frames']} x {meta['features']} features")
    if shapes[f"w{last}"][1] != len(meta["labels"]):
        raise SystemExit("the model's last layer does not have one output per label")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--models", type=Path, default=ROOT / "Demo/consultation/models")
    parser.add_argument("--out", type=Path, default=ROOT / "web")
    args = parser.parse_args()

    meta = json.loads((args.models / "health_signs.json").read_text(encoding="utf-8"))
    blob, described, layers = pack_model(args.models / "health_signs.npz")
    check_shapes(meta, described)
    manifest = build_manifest(meta, described, layers, blob)
    signs = build_signs(manifest["labels"], manifest["weak_labels"])

    (args.out / "models").mkdir(parents=True, exist_ok=True)
    (args.out / "models" / "health_signs.bin").write_bytes(blob)
    write_json(args.out / "models" / "health_signs.json", manifest)
    write_json(args.out / "data" / "signs.json", signs)
    print(json.dumps({"bin_bytes": len(blob), "arrays": [item["name"] for item in described],
                      "signs": len(signs["signs"]), "weak": manifest["weak_labels"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
