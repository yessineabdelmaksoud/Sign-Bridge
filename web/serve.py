#!/usr/bin/env python3
"""Serve the web/ folder for local use, with the conversation relay.

    python web/serve.py [--port 8520] [--host 127.0.0.1]

Listens on 127.0.0.1 by default; `--host 0.0.0.0` lets a doctor join from
another device on the same network. Sends the MIME types the browser needs for
ES modules, WebAssembly and video, answers HTTP range requests (so videos can
seek) and tells the browser not to cache anything.

The relay (web/CONVERSATION.md) passes conversation events between the patient
and doctor screens of a room:

    GET  /relay/ping              -> {"ok": true}
    GET  /relay/address           -> {"ok": true, "urls": [...]}: how another device on the
                                     network reaches this server (empty when only this
                                     computer can); used for the room's QR code
    GET  /relay/events?room=R     -> text/event-stream: the stored events, then new ones
    POST /relay/send?room=R       -> one JSON event; answers {"ok": true, "id": ...}

Events live in memory only (the last 200 per room); nothing is written to disk.

Tunisian speech for the doctor screen (web/speech_service.py), when set up:

    GET  /speech/status           -> {"ok": true, "stt": bool, "tts": bool}
    POST /speech/stt              -> 16-bit mono WAV, at most a minute; answers {"ok": true, "text": ...}
    POST /speech/tts              -> {"text": ...}; answers audio/mpeg

Standard library only (dictation also needs `pip install vosk` and the model).
"""
import argparse
import importlib.util
import json
import os
import queue
import re
import secrets
import select
import socket
import threading
import time
from collections import deque
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parent
HOST = "127.0.0.1"
DEFAULT_PORT = 8520

MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".json": "application/json",
    ".map": "application/json",
    ".wasm": "application/wasm",
    ".task": "application/octet-stream",
    ".bin": "application/octet-stream",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
    ".txt": "text/plain; charset=utf-8",
    ".md": "text/plain; charset=utf-8",
    ".ts": "text/plain; charset=utf-8",
}

_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")
UNSATISFIABLE = object()


def _load_speech_service():
    spec = importlib.util.spec_from_file_location("signbridge_speech_service", ROOT / "speech_service.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


speech_service = _load_speech_service()
MAX_AUDIO = 60 * 16000 * 2 + 4096   # a minute of 16 kHz, 16-bit mono WAV
MAX_SPEECH_TEXT = 4096              # bytes of JSON for one sentence to read aloud


def parse_range(header: str, size: int):
    """Turn a Range header into (first, last) byte positions, both inclusive.

    Returns None when the header should be ignored (malformed, or several
    ranges: the whole file is sent instead) and UNSATISFIABLE when the range
    lies outside the file.
    """
    match = _RANGE.match(header.strip())
    if not match or match.group(0) == "bytes=-":
        return None
    first, last = match.groups()
    if first == "":  # "bytes=-500": the last 500 bytes
        length = int(last)
        if length == 0 or size == 0:
            return UNSATISFIABLE
        return max(0, size - length), size - 1
    start = int(first)
    end = min(int(last), size - 1) if last else size - 1
    if start >= size or start > end:
        return UNSATISFIABLE
    return start, end


# --- the relay ------------------------------------------------------------------

ROOM = re.compile(r"[A-Za-z0-9_-]{1,32}")
MAX_BODY = 16 * 1024
KEEP = 200
PING_SECONDS = 15.0
TYPES = {"message", "status", "clear"}
SENDERS = {"patient", "doctor"}
VIA = {"sign", "text", "voice"}
STATES = {"online", "idle", "signing", "typing", "listening"}
_CLOSED = object()  # put in a listener's queue when the server stops


class BadRequest(ValueError):
    """The short reason sent back with a 400."""


LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def lan_addresses(host: str) -> list:
    """The addresses other devices can use to reach a server listening on `host`."""
    if host in LOOPBACK or host.startswith("127."):
        return []
    if host not in ("0.0.0.0", "", "::"):
        return [host]
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("10.255.255.255", 1))   # UDP: nothing is sent; this picks the network interface
            address = probe.getsockname()[0]
    except OSError:
        return []
    return [] if address.startswith("127.") or address == "0.0.0.0" else [address]


def check_room(value) -> str:
    if not isinstance(value, str) or not ROOM.fullmatch(value):
        raise BadRequest("bad room name: use 1 to 32 letters, digits, - or _")
    return value


def parse_event(body: bytes) -> dict:
    """Decode and check one event; raises BadRequest with a short reason."""
    try:
        event = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise BadRequest("body is not valid JSON") from None
    if not isinstance(event, dict):
        raise BadRequest("body must be a JSON object")
    kind = event.get("type")
    if kind not in TYPES:
        raise BadRequest("type must be message, status or clear")
    if kind in ("message", "status") and event.get("from") not in SENDERS:
        raise BadRequest("from must be patient or doctor")
    if kind == "status" and event.get("state") not in STATES:
        raise BadRequest("state must be online, idle, signing, typing or listening")
    if kind == "message":
        if "via" in event and event["via"] not in VIA:
            raise BadRequest("via must be sign, text or voice")
        if "text" in event and not isinstance(event["text"], str):
            raise BadRequest("text must be a string")
        signs = event.get("signs", [])
        if not isinstance(signs, list) or not all(isinstance(code, str) for code in signs):
            raise BadRequest("signs must be a list of sign codes")
    ident = event.get("id")
    if ident is not None and (isinstance(ident, bool) or not isinstance(ident, (str, int))):
        raise BadRequest("id must be a string")
    return event


class Relay:
    """Rooms of events, in memory. Safe to use from many handler threads."""

    def __init__(self, keep: int = KEEP, ping_seconds: float = PING_SECONDS):
        self.keep = keep
        self.ping_seconds = ping_seconds
        self._lock = threading.Lock()
        self._history = {}  # room -> deque of stored events
        self._listeners = {}  # room -> set of queue.Queue
        self._closed = False

    def publish(self, room: str, event: dict) -> dict:
        """Stamp, store (messages and clears) and pass on one checked event."""
        event = dict(event)
        if event.get("id") in (None, ""):
            event["id"] = secrets.token_hex(8)  # random: ids stay unique across restarts
        ts = event.get("ts")
        if isinstance(ts, bool) or not isinstance(ts, (int, float)):
            event["ts"] = int(time.time() * 1000)
        with self._lock:
            if event["type"] == "clear":
                self._history[room] = deque([event], maxlen=self.keep)
            elif event["type"] == "message":
                self._history.setdefault(room, deque(maxlen=self.keep)).append(event)
            for listener in self._listeners.get(room, ()):
                listener.put(event)
        return event

    def subscribe(self, room: str):
        """Register a listener; returns (stored events, queue of new events).

        Both happen under one lock, so no event is missed or seen twice
        between the history and the live stream.
        """
        listener = queue.Queue()
        with self._lock:
            self._listeners.setdefault(room, set()).add(listener)
            if self._closed:
                listener.put(_CLOSED)
            return list(self._history.get(room, ())), listener

    def unsubscribe(self, room: str, listener) -> None:
        with self._lock:
            listeners = self._listeners.get(room)
            if listeners is not None:
                listeners.discard(listener)
                if not listeners:
                    del self._listeners[room]

    def history(self, room: str) -> list:
        with self._lock:
            return list(self._history.get(room, ()))

    def listener_count(self, room: str) -> int:
        with self._lock:
            return len(self._listeners.get(room, ()))

    def close(self) -> None:
        """Wake every open stream so its thread ends and closes the connection."""
        with self._lock:
            self._closed = True
            for listeners in self._listeners.values():
                for listener in listeners:
                    listener.put(_CLOSED)


# --- HTTP -----------------------------------------------------------------------


class Handler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "SignBridgeServe/1"
    disable_nagle_algorithm = True  # small relay answers and events go out at once, not after 40 ms
    poll_seconds = 0.5  # how often an idle stream checks that its client is still there

    def guess_type(self, path):
        return MIME_TYPES.get(Path(path).suffix.lower(), "application/octet-stream")

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        if not str(getattr(self, "path", "") or "").startswith("/relay/"):
            self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    # routing

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/relay/ping":
            self._json(HTTPStatus.OK, {"ok": True})
        elif path == "/relay/address":
            host, port = self.server.server_address[:2]
            self._json(HTTPStatus.OK, {"ok": True, "urls": [f"http://{ip}:{port}" for ip in lan_addresses(host)]})
        elif path == "/relay/events":
            self._events()
        elif path == "/relay/send":
            self._json(HTTPStatus.METHOD_NOT_ALLOWED, {"ok": False, "error": "use POST"}, allow="POST")
        elif path.startswith("/relay/"):
            self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "unknown relay path"})
        elif path == "/speech/status":
            self._json(HTTPStatus.OK, {"ok": True, **self.server.speech.status()})
        elif path in ("/speech/stt", "/speech/tts"):
            self._json(HTTPStatus.METHOD_NOT_ALLOWED, {"ok": False, "error": "use POST"}, allow="POST")
        elif path.startswith("/speech/"):
            self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "unknown speech path"})
        else:
            super().do_GET()

    def do_POST(self):
        path = urlsplit(self.path).path
        if path == "/relay/send":
            self._send()
        elif path in ("/relay/ping", "/relay/events", "/relay/address"):
            self._discard_body()
            self._json(HTTPStatus.METHOD_NOT_ALLOWED, {"ok": False, "error": "use GET"}, allow="GET")
        elif path.startswith("/relay/"):
            self._discard_body()
            self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "unknown relay path"})
        elif path == "/speech/stt":
            self._speech(self._transcribe, MAX_AUDIO)
        elif path == "/speech/tts":
            self._speech(self._synthesize, MAX_SPEECH_TEXT)
        elif path == "/speech/status":
            self._discard_body()
            self._json(HTTPStatus.METHOD_NOT_ALLOWED, {"ok": False, "error": "use GET"}, allow="GET")
        elif path.startswith("/speech/"):
            self._discard_body()
            self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "unknown speech path"})
        else:
            self.send_error(HTTPStatus.NOT_IMPLEMENTED, "Unsupported method ('POST')")

    # speech endpoints (web/speech_service.py)

    def _speech(self, action, limit):
        try:
            length = self._content_length()
        except BadRequest as error:
            self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(error)})
            return
        if length > limit:
            self._skip_large(length)
            self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "too-long"})
            return
        body = self.rfile.read(length)
        try:
            action(body)
        except speech_service.SpeechError as error:
            self._json(error.status, {"ok": False, "error": error.code})
        except Exception as error:  # a broken model or an unexpected answer: say so, keep serving
            self.log_error("speech failed: %r", error)
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": "failed"})

    def _transcribe(self, body: bytes):
        self._json(HTTPStatus.OK, {"ok": True, "text": self.server.speech.transcribe(body)})

    def _synthesize(self, body: bytes):
        try:
            text = json.loads(body.decode("utf-8")).get("text")
        except (UnicodeDecodeError, ValueError, AttributeError):
            text = None
        if not isinstance(text, str):
            raise speech_service.SpeechError("bad-text", 400)
        audio = self.server.speech.synthesize(text)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "audio/mpeg")
        self.send_header("Content-Length", str(len(audio)))
        self.end_headers()
        self.wfile.write(audio)

    def _skip_large(self, length: int):
        """Drop a too-long recording (up to 16 MB) so the refusal still reaches the browser."""
        if length > 16 * 1024 * 1024:
            self.close_connection = True
            return
        while length > 0:
            chunk = self.rfile.read(min(256 * 1024, length))
            if not chunk:
                break
            length -= len(chunk)

    # relay endpoints

    def _room(self):
        values = parse_qs(urlsplit(self.path).query).get("room", [""])
        return check_room(values[0])

    def _send(self):
        try:
            body = self._read_body()  # first, so a refused request leaves nothing on the connection
            room = self._room()
            event = parse_event(body)
        except BadRequest as error:
            self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(error)})
            return
        event = self.server.relay.publish(room, event)
        self._json(HTTPStatus.OK, {"ok": True, "id": event["id"]})

    def _events(self):
        try:
            room = self._room()
        except BadRequest as error:
            self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(error)})
            return
        relay = self.server.relay
        history, listener = relay.subscribe(room)
        self.close_connection = True  # the stream has no length: it ends when the connection closes
        try:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Connection", "close")
            self.end_headers()
            self._write_events(history)
            last_write = time.monotonic()
            while True:
                try:
                    item = listener.get(timeout=self.poll_seconds)
                except queue.Empty:
                    item = None
                if item is not None:
                    batch = [item]
                    while True:  # several events at once go out in one write
                        try:
                            batch.append(listener.get_nowait())
                        except queue.Empty:
                            break
                    closing = _CLOSED in batch
                    self._write_events([event for event in batch if event is not _CLOSED])
                    last_write = time.monotonic()
                    if closing:
                        break
                elif time.monotonic() - last_write >= relay.ping_seconds:
                    self.wfile.write(b": ping\n\n")
                    last_write = time.monotonic()
                elif self._client_gone():
                    break
        except OSError:
            pass  # the browser closed the page or lost the network: nothing to report
        finally:
            relay.unsubscribe(room, listener)

    def _write_events(self, events):
        if not events:
            return
        chunks = []
        for event in events:
            data = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
            chunks.append(f"data: {data}\n\n")
        self.wfile.write("".join(chunks).encode("utf-8"))

    def _client_gone(self) -> bool:
        """True once the client has closed its side (EventSource never sends anything)."""
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            if not readable:
                return False
            return self.connection.recv(1, socket.MSG_PEEK) == b""
        except (OSError, ValueError):
            return True

    # helpers

    def _content_length(self) -> int:
        header = self.headers.get("Content-Length")
        if header is None:
            if self.headers.get("Transfer-Encoding"):
                self.close_connection = True  # the body is still on the wire
                raise BadRequest("Content-Length is required")
            return 0
        try:
            length = int(header)
        except ValueError:
            length = -1
        if length < 0:
            self.close_connection = True
            raise BadRequest("bad Content-Length")
        return length

    def _read_body(self) -> bytes:
        length = self._content_length()
        if length > MAX_BODY:
            self._skip(length)
            raise BadRequest(f"body too large: at most {MAX_BODY} bytes")
        return self.rfile.read(length)

    def _discard_body(self):
        try:
            self._skip(self._content_length())
        except BadRequest:
            pass

    def _skip(self, length: int):
        """Read and drop a body, so the answer can still reach the client."""
        if length > 1024 * 1024:
            self.close_connection = True
            return
        while length > 0:
            chunk = self.rfile.read(min(64 * 1024, length))
            if not chunk:
                break
            length -= len(chunk)

    def _json(self, status, payload, allow=None):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if allow:
            self.send_header("Allow", allow)
        self.end_headers()
        self.wfile.write(body)

    # static files

    def send_head(self):
        self._remaining = None
        path = self.translate_path(self.path)
        header = self.headers.get("Range")
        if header is None or not os.path.isfile(path):
            return super().send_head()
        try:
            source = open(path, "rb")
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return None
        stat = os.fstat(source.fileno())
        wanted = parse_range(header, stat.st_size)
        if wanted is None:
            source.close()
            return super().send_head()
        if wanted is UNSATISFIABLE:
            source.close()
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{stat.st_size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        start, end = wanted
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{stat.st_size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Last-Modified", self.date_time_string(stat.st_mtime))
        self.end_headers()
        source.seek(start)
        self._remaining = end - start + 1
        return source

    def copyfile(self, source, outputfile):
        remaining = getattr(self, "_remaining", None)
        if remaining is None:
            return super().copyfile(source, outputfile)
        while remaining > 0:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True  # the browser dropped a video request; nothing to do


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, *args, speech=None, **kwargs):
        self.relay = Relay()
        self.speech = speech if speech is not None else speech_service.from_environment()
        super().__init__(*args, **kwargs)

    def server_close(self):
        self.relay.close()
        super().server_close()


def make_server(host: str = HOST, port: int = DEFAULT_PORT, directory=ROOT, speech=None) -> Server:
    """A server for web/ with a fresh relay; port 0 picks a free port.

    `speech` answers /speech/ (web/speech_service.py); by default it is read from
    the environment and the repository's .env file."""
    return Server((host, port), partial(Handler, directory=str(directory)), speech=speech)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Serve the SignBridge web app and its conversation relay.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port to listen on (default {DEFAULT_PORT})")
    parser.add_argument(
        "--host",
        default=HOST,
        help=f"address to listen on (default {HOST}, this computer only; 0.0.0.0 for the local network)",
    )
    parser.add_argument("--stt-model", default=None,
                        help="folder of the Tunisian dictation model (default data/tunisian_stt)")
    return parser.parse_args(argv)


def describe_speech(speech) -> list:
    """One line each on Tunisian dictation and voice, for the console."""
    status = speech.status()
    lines = [
        "Tunisian dictation: on this computer (Vosk)." if status["stt"] else
        "Tunisian dictation: model not found, the browser's dictation is used "
        "(python3 web/speech_service.py download, and pip install vosk).",
        "Tunisian voice: ElevenLabs (the patient's sentences are sent to ElevenLabs)." if status["tts"] else
        "Tunisian voice: no ELEVENLABS_API_KEY, the browser's voice is used.",
    ]
    return lines


def main() -> None:
    args = parse_args()
    speech = speech_service.from_environment(stt_model=args.stt_model)
    server = make_server(args.host, args.port, speech=speech)
    host, port = server.server_address[:2]
    print(f"Serving {ROOT} at http://{host}:{port}/  (Ctrl+C to stop)", flush=True)
    for line in describe_speech(speech):
        print(line, flush=True)
    if speech.transcriber.available:
        threading.Thread(target=speech.transcriber.warm_up, daemon=True).start()
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(
            "Other devices on this network can open this computer's address on that port. "
            "The patient screen still needs http://localhost or https for the camera.",
            flush=True,
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
