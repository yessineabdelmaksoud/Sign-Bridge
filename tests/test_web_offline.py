"""The web app must work with the network off: no screen may load anything from the internet.

The vendored MediaPipe bundle is excluded (it is the official package, pinned to a
version that sends nothing; see web/vendor/mediapipe/VENDORED.txt).
"""
import re
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web"
SOURCES = sorted(
    [*WEB.glob("*.html"), *WEB.glob("*.js"), *WEB.glob("*.css"), *(WEB / "lib").glob("*.js")]
)
URL = re.compile(r"""(?:https?:)?//[a-z0-9.-]+\.[a-z]{2,}[/\w.-]*""", re.IGNORECASE)
ALLOWED_HOSTS = ("localhost", "127.0.0.1", "www.w3.org")  # w3.org: XML namespaces in inline SVG, never fetched


def external_urls(text: str) -> list[str]:
    found = []
    for match in URL.finditer(text):
        url = match.group(0)
        if not any(host in url for host in ALLOWED_HOSTS):
            found.append(url)
    return found


def test_there_are_web_sources_to_check():
    names = {p.name for p in SOURCES}
    assert {"tracker.js", "recognizer.js"} <= names


def test_no_screen_loads_anything_from_the_internet():
    offenders = {}
    for path in SOURCES:
        urls = external_urls(path.read_text(encoding="utf-8"))
        if urls:
            offenders[str(path.relative_to(WEB))] = urls
    assert offenders == {}


def test_the_mediapipe_bundle_is_the_pinned_version():
    note = (WEB / "vendor" / "mediapipe" / "VENDORED.txt").read_text(encoding="utf-8")
    assert "0.10.35" in note


def test_the_check_itself_catches_a_remote_font():
    assert external_urls("@import url('https://fonts.googleapis.com/css2?family=Inter');")
    assert external_urls('<script src="//cdn.example.com/x.js"></script>')
    assert not external_urls("fetch('/relay/ping'); fetch('http://localhost:8520/x')")
