"""The QR codes of web/lib/qr.js decode: Node draws each matrix, OpenCV reads it back.

Each code is drawn black on white with a 4-module quiet zone, 8 px per module,
and must decode to exactly the text that went in.
"""
import json
import random
import re
import shutil
import string
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
QR_MODULE = ROOT / "web" / "lib" / "qr.js"
LINK = "http://192.168.1.20:8520/patient.html?room=4821"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node to run web/lib/qr.js")


def ascii_text(length: int) -> str:
    sentence = "Consultation SignBridge, room 4821: the doctor joins in a minute. "
    return (sentence * (length // len(sentence) + 1))[:length]


# (text, level, version the encoder must pick)
CASES = [
    ("HELLO", "M", 1),
    ("https://example.com", "M", 2),
    (LINK, "L", 3),
    (LINK, "M", 4),
    (LINK, "Q", 5),
    (LINK, "H", 6),
    (ascii_text(100), "M", 6),
    (ascii_text(120), "M", 7),  # first version with version information
    (ascii_text(180), "M", 9),  # exactly the capacity of 9-M
    (ascii_text(250), "M", 11),  # 16-bit character count from version 10
    ("Médecin : j'ai de la fièvre depuis hier.", "M", 3),
    ("عندي سخانة من البارح", "Q", 4),  # 37 bytes in UTF-8
]

# Bytes each version holds in byte mode, versions 1 to 40 (ISO/IEC 18004 capacity table).
CAPACITY = {
    "L": (17, 32, 53, 78, 106, 134, 154, 192, 230, 271, 321, 367, 425, 458, 520, 586, 644, 718, 792, 858,
          929, 1003, 1091, 1171, 1273, 1367, 1465, 1528, 1628, 1732, 1840, 1952, 2068, 2188, 2303, 2431, 2563, 2699, 2809, 2953),
    "M": (14, 26, 42, 62, 84, 106, 122, 152, 180, 213, 251, 287, 331, 362, 412, 450, 504, 560, 624, 666,
          711, 779, 857, 911, 997, 1059, 1125, 1190, 1264, 1370, 1452, 1538, 1628, 1722, 1809, 1911, 1989, 2099, 2213, 2331),
    "Q": (11, 20, 32, 46, 60, 74, 86, 108, 130, 151, 177, 203, 241, 258, 292, 322, 364, 394, 442, 482,
          509, 565, 611, 661, 715, 751, 805, 868, 908, 982, 1030, 1112, 1168, 1228, 1283, 1351, 1423, 1499, 1579, 1663),
    "H": (7, 14, 24, 34, 44, 58, 64, 84, 98, 119, 137, 155, 177, 194, 220, 250, 280, 310, 338, 382,
          403, 439, 461, 511, 535, 593, 625, 658, 698, 742, 790, 842, 898, 958, 983, 1051, 1093, 1139, 1219, 1273),
}


def full_text(version: int, level: str) -> str:
    """Random printable ASCII filling the version to its last byte (the same text on every run)."""
    rng = random.Random(f"{level}{version}")
    alphabet = string.ascii_letters + string.digits + " -_./:?=&"
    return "".join(rng.choice(alphabet) for _ in range(CAPACITY[level][version - 1]))


FULL = [(full_text(v, "LMQH"[v % 4]), "LMQH"[v % 4], v) for v in range(1, 41)]  # each level every 4th version
SVG_CASES = [(LINK, {}), ("HELLO", {"ecl": "H", "quiet": 6, "dark": "#000000"})]

SCRIPT = """
import { readFileSync } from "node:fs";
import { qrMatrix, qrSvg } from %s;
const { matrices, svgs } = JSON.parse(readFileSync(0, "utf8"));
process.stdout.write(JSON.stringify({
  matrices: matrices.map(([text, ecl]) => {
    const { version, size, modules } = qrMatrix(text, { ecl });
    return { version, size, modules: modules.map((row) => row.map(Number)) };
  }),
  svgs: svgs.map(([text, options]) => qrSvg(text, options)),
}));
"""


@pytest.fixture(scope="module")
def encoded():
    """Every case encoded by one Node run: {"matrices": {(text, level): ...}, "svgs": [...]}."""
    pairs = [[text, level] for text, level, _ in CASES + FULL]
    request = {"matrices": pairs, "svgs": [list(case) for case in SVG_CASES]}
    done = subprocess.run(
        ["node", "--input-type=module", "-e", SCRIPT % json.dumps(QR_MODULE.as_uri())],
        cwd=ROOT, input=json.dumps(request), capture_output=True, text=True, encoding="utf-8", timeout=120,
    )
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout)
    return {"matrices": {tuple(p): m for p, m in zip(pairs, out["matrices"])}, "svgs": out["svgs"]}


def draw(modules, quiet: int = 4, scale: int = 8) -> np.ndarray:
    """Black on white, `quiet` light modules around, `scale` pixels per module."""
    dark = np.pad(np.asarray(modules, dtype=bool), quiet, constant_values=False)
    return np.where(dark.repeat(scale, axis=0).repeat(scale, axis=1), 0, 255).astype(np.uint8)


def case_id(case) -> str:
    text, level, version = case
    return f"{level}-v{version}-{len(text)}chars"


@pytest.mark.parametrize("text, level, version", CASES, ids=[case_id(c) for c in CASES])
def test_opencv_decodes_the_code(encoded, text, level, version):
    matrix = encoded["matrices"][(text, level)]
    assert matrix["version"] == version
    assert matrix["size"] == 17 + 4 * version == len(matrix["modules"])
    decoded, _, _ = cv2.QRCodeDetector().detectAndDecode(draw(matrix["modules"]))
    assert decoded == text


# OpenCV's default detector does not find the finder patterns of some dense codes
# (decoding the same image with the known corners works); its ArUco-based detector
# finds them, so the full range is read with it.
@pytest.mark.parametrize("text, level, version", FULL, ids=[case_id(c) for c in FULL])
def test_versions_1_to_40_decode_at_full_capacity(encoded, text, level, version):
    matrix = encoded["matrices"][(text, level)]
    assert matrix["version"] == version
    decoded, _, _ = cv2.QRCodeDetectorAruco().detectAndDecode(draw(matrix["modules"]))
    assert decoded == text


@pytest.mark.parametrize("index", range(len(SVG_CASES)), ids=["link", "hello-H-quiet6"])
def test_the_svg_decodes(encoded, index):
    """Redraws the svg's runs ("M x y h n v1 h-n z") on its viewBox grid; the quiet zone comes from the svg."""
    text, options = SVG_CASES[index]
    svg = encoded["svgs"][index]
    quiet = options.get("quiet", 4)
    n = int(re.search(r'viewBox="0 0 (\d+) \1"', svg).group(1))
    assert (n - 2 * quiet - 17) % 4 == 0
    dark = np.zeros((n, n), dtype=bool)
    path = re.search(r'<path fill="[^"]*" d="([^"]*)"', svg).group(1)
    for x, y, run in re.findall(r"M(\d+) (\d+)h(\d+)v1h-\d+z", path):
        dark[int(y), int(x):int(x) + int(run)] = True
    inner = dark[quiet:n - quiet, quiet:n - quiet]
    assert dark.sum() == inner.sum() > 0, "nothing dark in the quiet zone"
    decoded, _, _ = cv2.QRCodeDetector().detectAndDecode(draw(dark, quiet=0))
    assert decoded == text
