"""Read single members of a large remote zip with HTTP range requests.

SWL-LSE ships its landmarks as one 3.5 GB archive. We need about one clip in
eight, so we read the archive's index and then fetch each needed member with
one ranged request instead of downloading the whole file.
"""
from __future__ import annotations

import io
import struct
import threading
import time
import urllib.error
import urllib.request
import zipfile
import zlib

_LOCAL_HEADER = 30
_EXTRA_GUESS = 256


class HttpRangeReader:
    """Byte ranges of one URL, with polite pacing and retries."""

    def __init__(self, url: str, min_interval: float = 0.25, timeout: float = 120.0):
        self.url = url
        self.timeout = timeout
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._next_slot = 0.0
        self._final = url
        self.size = 0
        self._resolve()

    def _resolve(self) -> None:
        request = urllib.request.Request(self.url, method="HEAD", headers={"User-Agent": "curl"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            self.size = int(response.headers["Content-Length"])
            self._final = response.geturl()

    def _wait_turn(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_slot - now
            self._next_slot = max(now, self._next_slot) + self.min_interval
        if wait > 0:
            time.sleep(wait)

    def read_range(self, start: int, end: int) -> bytes:
        """Bytes [start, end) of the file."""
        last_error: Exception | None = None
        for attempt in range(6):
            self._wait_turn()
            request = urllib.request.Request(
                self._final, headers={"Range": f"bytes={start}-{end - 1}", "User-Agent": "curl"})
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = response.read()
                if len(data) == end - start:
                    return data
                last_error = IOError(f"short read: {len(data)} of {end - start} bytes")
            except urllib.error.HTTPError as error:
                last_error = error
                if error.code == 429:  # asked to slow down
                    self.min_interval = max(self.min_interval, 1.3)
                    time.sleep(float(error.headers.get("Retry-After", 60)))
                elif error.code in (401, 403):  # the redirect target expired
                    self._resolve()
            except OSError as error:
                last_error = error
            time.sleep(min(2 ** attempt, 30))
        raise IOError(f"could not read bytes {start}-{end} of {self.url}: {last_error}")


class _ReaderFile(io.RawIOBase):
    """A seekable file over a range reader, cached in blocks, for zipfile's index scan."""

    def __init__(self, reader, block: int = 1 << 20):
        self._reader, self._block, self._pos, self._cache = reader, block, 0, {}

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, offset, whence=0):
        base = 0 if whence == 0 else self._pos if whence == 1 else self._reader.size
        self._pos = base + offset
        return self._pos

    def read(self, n=-1):
        size = self._reader.size
        if n < 0:
            n = size - self._pos
        out = bytearray()
        while n > 0 and self._pos < size:
            index, offset = divmod(self._pos, self._block)
            if index not in self._cache:
                start = index * self._block
                self._cache[index] = self._reader.read_range(start, min(size, start + self._block))
            chunk = self._cache[index][offset:offset + n]
            if not chunk:
                break
            out += chunk
            self._pos += len(chunk)
            n -= len(chunk)
        return bytes(out)


class RemoteZip:
    """`reader` needs a `size` attribute and `read_range(start, end) -> bytes`."""

    def __init__(self, reader):
        self._reader = reader
        with zipfile.ZipFile(_ReaderFile(reader)) as index:
            self._info = {i.filename: i for i in index.infolist() if not i.is_dir()}

    def names(self) -> list[str]:
        return list(self._info)

    def read(self, name: str) -> bytes:
        info = self._info[name]
        start = info.header_offset
        want = _LOCAL_HEADER + len(info.filename.encode()) + _EXTRA_GUESS + info.compress_size
        raw = self._reader.read_range(start, min(self._reader.size, start + want))
        name_len, extra_len = struct.unpack("<HH", raw[26:30])
        data_start = _LOCAL_HEADER + name_len + extra_len
        data_end = data_start + info.compress_size
        if data_end > len(raw):  # unusually large extra field
            raw += self._reader.read_range(start + len(raw), start + data_end)
        body = raw[data_start:data_end]
        if info.compress_type == zipfile.ZIP_STORED:
            return body
        if info.compress_type == zipfile.ZIP_DEFLATED:
            return zlib.decompress(body, -15)
        raise ValueError(f"{name}: unsupported compression {info.compress_type}")
