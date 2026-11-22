"""The consultation must work with the network off and must never store camera frames."""
import re
from pathlib import Path

DEMO = Path(__file__).resolve().parents[1] / "Demo"
FILES = [DEMO / "pages" / "6_Consultation.py", *sorted((DEMO / "consultation").glob("*.py"))]


def test_the_page_exists():
    assert FILES[0].exists()


def test_no_network_code_in_consultation():
    for path in FILES:
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"elevenlabs|\brequests\b|urllib|https?://", text), path


def test_no_frame_is_written_to_disk():
    for path in FILES:
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"imwrite|VideoWriter|\.save\(|tempfile", text), path
