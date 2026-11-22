#!/usr/bin/env python3
"""Serve the web/ folder for local use.

    python web/serve.py [--port 8520]

Listens on 127.0.0.1 only. Sends the MIME types the browser needs for ES modules,
WebAssembly and video, answers HTTP range requests (so videos can seek) and
tells the browser not to cache anything. Standard library only.
"""
import argparse
import os
import re
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

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


class Handler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "SignBridgeServe/1"

    def guess_type(self, path):
        return MIME_TYPES.get(Path(path).suffix.lower(), "application/octet-stream")

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the SignBridge web app on this computer only.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port to listen on (default {DEFAULT_PORT})")
    args = parser.parse_args()

    server = Server((HOST, args.port), partial(Handler, directory=str(ROOT)))
    print(f"Serving {ROOT} at http://{HOST}:{args.port}/  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
