"""The conversation relay in web/serve.py (see web/CONVERSATION.md).

Each test starts the real server on a free port in a thread and talks to it
over HTTP, the way the browser does.
"""
import http.client
import importlib.util
import json
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"


def _load_serve():
    spec = importlib.util.spec_from_file_location("signbridge_serve", WEB / "serve.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


serve = _load_serve()


@pytest.fixture
def server():
    srv = serve.make_server("127.0.0.1", 0)
    thread = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()
    thread.join(5)


def _address(srv):
    host, port = srv.server_address[:2]
    return host, port


def request(srv, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection(*_address(srv), timeout=5)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        response = conn.getresponse()
        return response.status, response.headers, response.read()
    finally:
        conn.close()


def send(srv, room, event):
    body = event if isinstance(event, bytes) else json.dumps(event).encode()
    status, _, data = request(
        srv, "POST", f"/relay/send?room={room}", body=body, headers={"Content-Type": "application/json"}
    )
    return status, json.loads(data) if data else None


def message(text, sender="patient", **extra):
    return {"type": "message", "from": sender, "via": "sign", "text": text, "signs": [], **extra}


class Listener:
    """A minimal EventSource: one open GET /relay/events, read block by block."""

    def __init__(self, srv, room, timeout=5):
        self.conn = http.client.HTTPConnection(*_address(srv), timeout=timeout)
        self.conn.request("GET", f"/relay/events?room={room}")
        self.response = self.conn.getresponse()

    @property
    def status(self):
        return self.response.status

    def block(self):
        """The next block: ("event", dict) or ("comment", text)."""
        data, comments = [], []
        while True:
            line = self.response.readline()
            if not line:
                raise EOFError("stream closed")
            line = line.decode("utf-8").rstrip("\r\n")
            if line == "":
                if data:
                    return "event", json.loads("\n".join(data))
                if comments:
                    return "comment", "\n".join(comments)
                continue
            if line.startswith(":"):
                comments.append(line[1:].strip())
            elif line.startswith("data:"):
                data.append(line[5:].lstrip(" "))

    def event(self):
        while True:
            kind, value = self.block()
            if kind == "event":
                return value

    def events(self, count):
        return [self.event() for _ in range(count)]

    def close(self):
        self.response.close()  # http.client hands the socket to a response without length
        self.conn.close()


def wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


# --- ping and events ---------------------------------------------------------


def test_ping(server):
    status, headers, body = request(server, "GET", "/relay/ping")
    assert status == 200
    assert headers["Content-Type"].startswith("application/json")
    assert headers["Cache-Control"] == "no-store"
    assert json.loads(body) == {"ok": True}


def test_events_stream_headers(server):
    listener = Listener(server, "headers")
    try:
        assert listener.status == 200
        assert listener.response.getheader("Content-Type").startswith("text/event-stream")
        assert listener.response.getheader("Cache-Control") == "no-store"
    finally:
        listener.close()


def test_a_message_reaches_two_listeners_with_id_and_ts(server):
    a, b = Listener(server, "room1"), Listener(server, "room1")
    try:
        assert wait_until(lambda: server.relay.listener_count("room1") == 2)
        before = time.time() * 1000
        status, answer = send(server, "room1", message("Je suis malade.", signs=["ENFERMO"]))
        after = time.time() * 1000
        assert status == 200 and answer["ok"] is True and answer["id"]
        for listener in (a, b):
            event = listener.event()
            assert event["id"] == answer["id"]
            assert isinstance(event["ts"], int) and before - 1 <= event["ts"] <= after + 1
            assert event["type"] == "message" and event["from"] == "patient" and event["via"] == "sign"
            assert event["text"] == "Je suis malade." and event["signs"] == ["ENFERMO"]
    finally:
        a.close()
        b.close()


def test_an_id_given_by_the_sender_is_kept(server):
    status, answer = send(server, "keep", message("x", id="abc-123"))
    assert status == 200 and answer["id"] == "abc-123"
    assert [e["id"] for e in server.relay.history("keep")] == ["abc-123"]


def test_ids_are_unique(server):
    ids = {send(server, "unique", message(str(i)))[1]["id"] for i in range(20)}
    assert len(ids) == 20


def test_rooms_are_separate(server):
    other = Listener(server, "other")
    try:
        send(server, "mine", message("for mine"))
        send(server, "other", message("for other"))
        assert other.event()["text"] == "for other"
    finally:
        other.close()


def test_a_late_listener_gets_the_history_in_order(server):
    for text in ("one", "two", "three"):
        assert send(server, "late", message(text))[0] == 200
    listener = Listener(server, "late")
    try:
        assert [e["text"] for e in listener.events(3)] == ["one", "two", "three"]
        send(server, "late", message("four", sender="doctor"))
        assert listener.event()["text"] == "four"
    finally:
        listener.close()


def test_status_is_passed_on_but_not_stored(server):
    early = Listener(server, "st")
    try:
        assert wait_until(lambda: server.relay.listener_count("st") == 1)
        status, answer = send(server, "st", {"type": "status", "from": "doctor", "state": "typing"})
        assert status == 200 and answer["ok"] is True
        event = early.event()
        assert event["type"] == "status" and event["state"] == "typing" and event["from"] == "doctor"
    finally:
        early.close()
    assert server.relay.history("st") == []
    send(server, "st", message("after the status"))
    late = Listener(server, "st")
    try:
        assert late.event()["text"] == "after the status"
    finally:
        late.close()


def test_clear_empties_the_history_and_is_passed_on(server):
    send(server, "clr", message("old 1"))
    send(server, "clr", message("old 2"))
    live = Listener(server, "clr")
    try:
        live.events(2)
        status, answer = send(server, "clr", {"type": "clear"})
        assert status == 200 and answer["ok"] is True
        event = live.event()
        assert event["type"] == "clear" and event["id"] == answer["id"] and "ts" in event
    finally:
        live.close()
    send(server, "clr", message("new"))
    late = Listener(server, "clr")
    try:
        assert [e["type"] for e in late.events(2)] == ["clear", "message"]
    finally:
        late.close()
    assert [e["type"] for e in server.relay.history("clr")] == ["clear", "message"]


def test_the_history_keeps_the_last_200_messages(server):
    conn = http.client.HTTPConnection(*_address(server), timeout=5)
    try:
        for i in range(205):
            conn.request("POST", "/relay/send?room=cap", body=json.dumps(message(str(i))).encode())
            response = conn.getresponse()
            response.read()
            assert response.status == 200
    finally:
        conn.close()
    history = server.relay.history("cap")
    assert len(history) == 200
    assert history[0]["text"] == "5" and history[-1]["text"] == "204"
    listener = Listener(server, "cap")
    try:
        received = listener.events(200)
        assert [e["text"] for e in received] == [str(i) for i in range(5, 205)]
        send(server, "cap", message("marker"))
        assert listener.event()["text"] == "marker"  # nothing else was replayed
    finally:
        listener.close()


def test_a_ping_comment_keeps_an_idle_stream_open(server):
    server.relay.ping_seconds = 0.2
    listener = Listener(server, "quiet")
    try:
        assert listener.block() == ("comment", "ping")
    finally:
        listener.close()


# --- validation ----------------------------------------------------------------


@pytest.mark.parametrize("room", ["", "a" * 33, "bad%20room", "caf%C3%A9", "a/b", "x.y"])
def test_a_bad_room_name_is_refused(server, room):
    status, answer = send(server, room, message("x"))
    assert status == 400 and answer["ok"] is False and "room" in answer["error"]
    status, _, body = request(server, "GET", f"/relay/events?room={room}")
    assert status == 400 and "room" in json.loads(body)["error"]


def test_a_missing_room_is_refused(server):
    status, _, body = request(server, "POST", "/relay/send", body=json.dumps(message("x")).encode())
    assert status == 400 and "room" in json.loads(body)["error"]


def test_room_names_may_use_letters_digits_dash_and_underscore(server):
    room = "Room_1-" + "z" * 25
    assert len(room) == 32
    assert send(server, room, message("x"))[0] == 200


def test_an_oversize_body_is_refused(server):
    event = message("x" * (16 * 1024))
    status, answer = send(server, "big", event)
    assert status == 400 and answer["ok"] is False and "large" in answer["error"]
    assert server.relay.history("big") == []


def test_a_body_of_exactly_16_kb_is_accepted(server):
    event = message("")
    padding = 16 * 1024 - len(json.dumps(event).encode())
    event["text"] = "x" * padding
    body = json.dumps(event).encode()
    assert len(body) == 16 * 1024
    assert send(server, "edge", body)[0] == 200


@pytest.mark.parametrize(
    "event, reason",
    [
        ({"type": "shout", "from": "patient"}, "type"),
        ({"from": "patient", "text": "x"}, "type"),
        ({"type": "message", "via": "text", "text": "x"}, "from"),
        ({"type": "message", "from": "nurse", "via": "text", "text": "x"}, "from"),
        ({"type": "status", "state": "typing"}, "from"),
        ({"type": "status", "from": "doctor", "state": "dancing"}, "state"),
        ({"type": "message", "from": "doctor", "via": "fax", "text": "x"}, "via"),
        ({"type": "message", "from": "doctor", "via": "text", "text": 3}, "text"),
        ({"type": "message", "from": "doctor", "via": "text", "text": "x", "signs": "FIEBRE2"}, "signs"),
    ],
)
def test_a_malformed_event_is_refused(server, event, reason):
    status, answer = send(server, "bad", event)
    assert status == 400 and answer["ok"] is False and reason in answer["error"]
    assert server.relay.history("bad") == []


@pytest.mark.parametrize("body", [b"", b"not json", b"[1, 2]", b'"message"'])
def test_a_body_that_is_not_a_json_object_is_refused(server, body):
    status, answer = send(server, "bad", body)
    assert status == 400 and answer["ok"] is False


def test_get_on_send_and_unknown_relay_paths(server):
    assert request(server, "GET", "/relay/send?room=x")[0] == 405
    assert request(server, "GET", "/relay/nothing")[0] == 404
    assert request(server, "POST", "/relay/ping", body=b"{}")[0] == 405


# --- connections ----------------------------------------------------------------


def test_a_refused_request_leaves_the_connection_usable(server):
    conn = http.client.HTTPConnection(*_address(server), timeout=5)
    try:
        for path, expected in (("/relay/send?room=bad.room", 400), ("/relay/send?room=good", 200)):
            conn.request("POST", path, body=json.dumps(message("x")).encode())
            response = conn.getresponse()
            response.read()
            assert response.status == expected
    finally:
        conn.close()


def test_answers_on_a_kept_alive_connection_are_not_delayed(server):
    conn = http.client.HTTPConnection(*_address(server), timeout=5)
    try:
        started = time.monotonic()
        for i in range(20):
            conn.request("POST", "/relay/send?room=fast", body=json.dumps(message(str(i))).encode())
            response = conn.getresponse()
            response.read()
        assert time.monotonic() - started < 0.5  # 20 x 40 ms if small writes wait for an ACK
    finally:
        conn.close()


def test_a_listener_that_disconnects_is_cleaned_up_quietly(server, capfd):
    listener = Listener(server, "gone")
    assert wait_until(lambda: server.relay.listener_count("gone") == 1)
    listener.close()
    assert wait_until(lambda: server.relay.listener_count("gone") == 0, timeout=3)
    send(server, "gone", message("nobody hears this"))
    time.sleep(0.2)
    assert "Traceback" not in capfd.readouterr().err


def test_open_streams_do_not_block_other_requests(server):
    listeners = [Listener(server, "busy") for _ in range(4)]
    try:
        assert wait_until(lambda: server.relay.listener_count("busy") == 4)
        started = time.monotonic()
        assert request(server, "GET", "/relay/ping")[0] == 200
        assert request(server, "GET", "/data/signs.json")[0] == 200
        assert time.monotonic() - started < 1.0
    finally:
        for listener in listeners:
            listener.close()


def test_closing_the_server_ends_open_streams(server):
    listener = Listener(server, "bye")
    assert wait_until(lambda: server.relay.listener_count("bye") == 1)
    server.shutdown()
    server.server_close()
    with pytest.raises((EOFError, OSError)):
        listener.event()
    listener.close()


# --- the static server still works ------------------------------------------


def test_a_static_file_is_still_served(server):
    status, headers, body = request(server, "GET", "/data/signs.json")
    assert status == 200
    assert headers["Content-Type"] == "application/json"
    assert headers["Cache-Control"] == "no-store"
    assert len(json.loads(body)["signs"]) >= 22


def test_es_modules_get_a_javascript_mime_type(server):
    status, headers, _ = request(server, "GET", "/lib/capture.js")
    assert status == 200 and headers["Content-Type"].startswith("text/javascript")


def test_a_video_range_request_still_works(server):
    video = WEB / "clips" / "DOLOR.mp4"
    expected = video.read_bytes()[100:200]
    status, headers, body = request(server, "GET", "/clips/DOLOR.mp4", headers={"Range": "bytes=100-199"})
    assert status == 206
    assert headers["Content-Type"] == "video/mp4"
    assert headers["Content-Range"] == f"bytes 100-199/{video.stat().st_size}"
    assert body == expected


def test_host_option():
    assert serve.parse_args([]).host == "127.0.0.1"
    assert serve.parse_args([]).port == 8520
    args = serve.parse_args(["--host", "0.0.0.0", "--port", "8701"])
    assert (args.host, args.port) == ("0.0.0.0", 8701)
