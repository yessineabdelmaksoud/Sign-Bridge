#!/usr/bin/env python3
"""Tunisian speech for the doctor screen, served by web/serve.py.

    dictation   the Tunisian Vosk (Kaldi) model from the earlier demo, run on this
                computer: the doctor's voice never leaves it.
    voice       the Tunisian ElevenLabs voice from the earlier demo: the patient's
                sentence is sent to ElevenLabs, which returns MP3 audio.

Both are optional. Without them the doctor screen falls back to the browser's
own speech services.

Settings (environment first, then a .env file at the repository root):

    ELEVENLABS_API_KEY     needed for the voice
    ELEVENLABS_VOICE_ID    default InB4o9iYj3MQ4HtGs2KV (the earlier demo's voice)
    ELEVENLABS_MODEL       default eleven_multilingual_v2
    SIGNBRIDGE_STT_MODEL   default data/tunisian_stt

Get the dictation model (about 540 MB) once:

    python3 web/speech_service.py download

Needs `pip install vosk` for dictation; the voice uses the standard library only.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import sys
import threading
import urllib.error
import urllib.request
import wave
import zipfile
from collections import OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_STT_DIR = Path("data") / "tunisian_stt"
STT_MODEL_URL = "https://huggingface.co/Sali7a8603/Tunisian_STT/resolve/main/STT_Tun_Model.zip"
DEFAULT_VOICE = "InB4o9iYj3MQ4HtGs2KV"
DEFAULT_TTS_MODEL = "eleven_multilingual_v2"
ELEVENLABS_URL = "https://api.elevenlabs.io"
MAX_TEXT = 600
SETTINGS = ("ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID", "ELEVENLABS_MODEL", "SIGNBRIDGE_STT_MODEL")


class SpeechError(Exception):
    """A short code the doctor screen turns into a sentence, and the HTTP status to answer with."""

    def __init__(self, code: str, status: int):
        super().__init__(code)
        self.code = code
        self.status = status


# --- dictation -------------------------------------------------------------------


class Transcriber:
    """The Tunisian Vosk model, loaded on first use (or by `warm_up`) and kept."""

    def __init__(self, model_dir, engine=None):
        self.model_dir = Path(model_dir)
        self._engine = engine
        self._model = None
        self._lock = threading.Lock()

    def _vosk(self):
        if self._engine is None:
            try:
                import vosk
            except ImportError:
                return None
            vosk.SetLogLevel(-1)
            self._engine = vosk
        return self._engine

    @property
    def available(self) -> bool:
        return self.model_dir.is_dir() and self._vosk() is not None

    def warm_up(self):
        """Load the model now, so the first dictation does not wait for it."""
        if not self.available:
            return None
        with self._lock:
            if self._model is None:
                self._model = self._vosk().Model(str(self.model_dir))
        return self._model

    def transcribe(self, audio: bytes) -> str:
        if not self.available:
            raise SpeechError("no-model", 503)
        try:
            with wave.open(io.BytesIO(audio), "rb") as wav:
                if wav.getcomptype() != "NONE" or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                    raise SpeechError("bad-audio", 400)
                rate = wav.getframerate()
                frames = wav.readframes(wav.getnframes())
        except (wave.Error, EOFError):
            raise SpeechError("bad-audio", 400) from None
        model = self.warm_up()
        recognizer = self._vosk().KaldiRecognizer(model, float(rate))
        recognizer.SetWords(True)
        parts = []
        step = 8000  # bytes: 0.25 s at 16 kHz
        for start in range(0, len(frames), step):
            if recognizer.AcceptWaveform(frames[start:start + step]):
                parts.append(json.loads(recognizer.Result()).get("text", ""))
        parts.append(json.loads(recognizer.FinalResult()).get("text", ""))
        return " ".join(part.strip() for part in parts if part.strip())


# --- voice -----------------------------------------------------------------------


class Voice:
    """ElevenLabs text to speech; each sentence is asked for once, then served from memory."""

    def __init__(self, api_key="", voice_id=DEFAULT_VOICE, model_id=DEFAULT_TTS_MODEL,
                 base_url=ELEVENLABS_URL, opener=None, cache_size=64):
        self.api_key = api_key or ""
        self.voice_id = voice_id or DEFAULT_VOICE
        self.model_id = model_id or DEFAULT_TTS_MODEL
        self.base_url = base_url.rstrip("/")
        self._open = opener or urllib.request.urlopen
        self._cache: OrderedDict[str, bytes] = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def synthesize(self, text) -> bytes:
        text = str(text or "").strip()
        if not text or len(text) > MAX_TEXT:
            raise SpeechError("bad-text", 400)
        if not self.available:
            raise SpeechError("no-key", 503)
        with self._lock:
            if text in self._cache:
                self._cache.move_to_end(text)
                return self._cache[text]
        request = urllib.request.Request(
            f"{self.base_url}/v1/text-to-speech/{self.voice_id}?output_format=mp3_44100_128",
            data=json.dumps({"text": text, "model_id": self.model_id}, ensure_ascii=False).encode("utf-8"),
            headers={"xi-api-key": self.api_key, "Content-Type": "application/json", "Accept": "audio/mpeg"},
            method="POST",
        )
        try:
            with self._open(request, timeout=30) as response:
                audio = response.read()
        except urllib.error.HTTPError as error:
            code = {401: "key-refused", 402: "paid-plan", 429: "busy"}.get(error.code, "upstream")
            raise SpeechError(code, 502) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise SpeechError("offline", 502) from None
        if not audio:
            raise SpeechError("upstream", 502)
        with self._lock:
            self._cache[text] = audio
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return audio


# --- both, as the server holds them ------------------------------------------------


class Speech:
    def __init__(self, transcriber: Transcriber, voice: Voice):
        self.transcriber = transcriber
        self.voice = voice

    def status(self) -> dict:
        return {"stt": self.transcriber.available, "tts": self.voice.available}

    def transcribe(self, audio: bytes) -> str:
        return self.transcriber.transcribe(audio)

    def synthesize(self, text) -> bytes:
        return self.voice.synthesize(text)


def read_env_file(path: Path) -> dict:
    """KEY=VALUE lines of a .env file, for the settings above only."""
    values = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip().removeprefix("export ").strip(), value.strip().strip("'\"")
        if key in SETTINGS:
            values[key] = value
    return values


def from_environment(env=None, root=REPO, stt_model=None) -> Speech:
    settings = read_env_file(Path(root) / ".env")
    settings.update({k: v for k, v in (os.environ if env is None else env).items() if k in SETTINGS and v})
    model_dir = Path(stt_model or settings.get("SIGNBRIDGE_STT_MODEL") or Path(root) / DEFAULT_STT_DIR)
    voice = Voice(settings.get("ELEVENLABS_API_KEY", ""), settings.get("ELEVENLABS_VOICE_ID", DEFAULT_VOICE),
                  settings.get("ELEVENLABS_MODEL", DEFAULT_TTS_MODEL))
    return Speech(Transcriber(model_dir), voice)


# --- getting the model -----------------------------------------------------------


def _model_folder(base: Path):
    """The extracted folder that holds the Kaldi model (it has conf/ and am/ or final.mdl)."""
    for folder in [base, *sorted(p for p in base.rglob("*") if p.is_dir())]:
        if (folder / "conf").is_dir() and ((folder / "am").is_dir() or (folder / "final.mdl").exists()):
            return folder
    return None


def install_model(archive: Path, target: Path) -> Path:
    """Unpack the downloaded zip and move its model folder to `target`."""
    work = target.parent / (target.name + ".unpacking")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            path = (work / member.filename).resolve()
            if not path.is_relative_to(work.resolve()):
                raise ValueError(f"unsafe path in the archive: {member.filename}")
        zf.extractall(work)
    folder = _model_folder(work)
    if folder is None:
        raise ValueError("no Kaldi model (conf/ and am/) found in the archive")
    shutil.rmtree(target, ignore_errors=True)
    shutil.move(str(folder), str(target))
    shutil.rmtree(work, ignore_errors=True)
    return target


def download_model(target: Path = REPO / DEFAULT_STT_DIR) -> Path:
    archive = target.parent / "STT_Tun_Model.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {STT_MODEL_URL} (about 540 MB)...", flush=True)
    with urllib.request.urlopen(STT_MODEL_URL, timeout=60) as response, open(archive, "wb") as out:
        shutil.copyfileobj(response, out, length=8 * 1024 * 1024)
    install_model(archive, target)
    archive.unlink(missing_ok=True)
    print(f"Tunisian dictation model ready in {target}", flush=True)
    return target


if __name__ == "__main__":
    if sys.argv[1:] == ["download"]:
        download_model()
    elif len(sys.argv) == 3 and sys.argv[1] == "install":
        print(install_model(Path(sys.argv[2]), REPO / DEFAULT_STT_DIR))
    else:
        print(__doc__)
