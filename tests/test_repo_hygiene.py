import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def tracked(pattern: str) -> list[str]:
    out = subprocess.run(["git", "ls-files", pattern], cwd=ROOT, capture_output=True, text=True, check=True)
    return [line for line in out.stdout.splitlines() if line]


def test_no_hardcoded_api_key():
    for name in tracked("*.py"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert not re.search(r"sk_[0-9a-f]{20,}", text), name


def test_no_bytecode_tracked():
    assert tracked("*.pyc") == []


def test_single_copy_of_sign_images():
    assert tracked("Tun_TTSi/data/*") == []
