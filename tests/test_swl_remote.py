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


def install(monkeypatch, failures, partial=10, cap=None):
    """Serve DATA by range.

    The first `failures` body reads break off after `partial` bytes. With `cap`,
    every response is cut to at most `cap` bytes, however often it is asked.
    """
    state = {"left": failures, "gets": 0}

    def fake_urlopen(request, timeout=None):
        if request.get_method() == "HEAD":
            return FakeResponse(b"")
        state["gets"] += 1
        start, end = request.get_header("Range").removeprefix("bytes=").split("-")
        body = DATA[int(start):int(end) + 1]
        if cap is not None and len(body) > cap:
            return FakeResponse(body, fail=http.client.IncompleteRead(body[:cap], len(body) - cap))
        if state["left"] > 0:
            state["left"] -= 1
            return FakeResponse(body, fail=http.client.IncompleteRead(body[:partial], len(body) - partial))
        return FakeResponse(body)

    monkeypatch.setattr(swl_remote.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(swl_remote.time, "sleep", lambda seconds: None)
    return state


def test_a_transfer_that_breaks_off_is_retried(monkeypatch):
    state = install(monkeypatch, failures=2)
    reader = HttpRangeReader("https://zenodo.example/file", min_interval=0)
    assert reader.read_range(100, 600) == DATA[100:600]
    assert state["gets"] == 3


def test_a_server_that_always_cuts_responses_short_is_read_in_pieces(monkeypatch):
    state = install(monkeypatch, failures=0, cap=300)
    reader = HttpRangeReader("https://zenodo.example/file", min_interval=0)
    assert reader.read_range(0, 5000) == DATA[:5000]
    assert state["gets"] == 17  # 16 cut-off pieces of 300 bytes, then the last 200


def test_gives_up_with_a_clear_error_when_no_bytes_arrive(monkeypatch):
    install(monkeypatch, failures=99, partial=0)
    reader = HttpRangeReader("https://zenodo.example/file", min_interval=0)
    with pytest.raises(IOError, match="could not read bytes 0-50"):
        reader.read_range(0, 50)
