import http.client

import pytest

from training import swl_remote
from training.swl_remote import HttpRangeReader

DATA = bytes(range(256)) * 40


class FakeResponse:
    def __init__(self, body, fail=None):
        self.headers = {"Content-Length": str(len(DATA))}
        self._body, self._fail = body, fail

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def geturl(self):
        return "https://files.example/final"

    def read(self):
        if self._fail:
            raise self._fail
        return self._body


def install(monkeypatch, failures):
    """Serve DATA by range; the first `failures` body reads break off mid-transfer."""
    state = {"left": failures, "gets": 0}

    def fake_urlopen(request, timeout=None):
        if request.get_method() == "HEAD":
            return FakeResponse(b"")
        state["gets"] += 1
        start, end = request.get_header("Range").removeprefix("bytes=").split("-")
        body = DATA[int(start):int(end) + 1]
        if state["left"] > 0:
            state["left"] -= 1
            return FakeResponse(body, fail=http.client.IncompleteRead(body[:10], len(body) - 10))
        return FakeResponse(body)

    monkeypatch.setattr(swl_remote.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(swl_remote.time, "sleep", lambda seconds: None)
    return state


def test_a_transfer_that_breaks_off_is_retried(monkeypatch):
    state = install(monkeypatch, failures=2)
    reader = HttpRangeReader("https://zenodo.example/file", min_interval=0)
    assert reader.read_range(100, 600) == DATA[100:600]
    assert state["gets"] == 3


def test_gives_up_with_a_clear_error_after_repeated_failures(monkeypatch):
    install(monkeypatch, failures=99)
    reader = HttpRangeReader("https://zenodo.example/file", min_interval=0)
    with pytest.raises(IOError, match="could not read bytes 0-50"):
        reader.read_range(0, 50)
