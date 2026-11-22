"""Tunisian speech on the local server: Vosk dictation and the ElevenLabs voice (web/speech_service.py)."""
import http.client
import importlib.util
import io
import json
import threading
import urllib.error
import wave
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1] / "web"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, WEB / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


serve = _load("signbridge_serve_for_speech", "serve.py")
speech = serve.speech_service  # the same module the server uses, so its errors are the server's


def wav_bytes(seconds=0.5, rate=16000, channels=1, width=2):
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(b"\x00" * int(seconds * rate) * channels * width)
    return out.getvalue()


class FakeEngine:
    """Stands in for the vosk module: hears one word per chunk of audio."""

    def __init__(self, words=("عسلامة", "لاباس")):
        self.words = list(words)
        self.models = []
        self.rates = []

    def Model(self, path):  # noqa: N802 - same name as vosk
        self.models.append(path)
        return object()

    def KaldiRecognizer(self, model, rate):  # noqa: N802
        self.rates.append(rate)
        engine = self

        class Recognizer:
            def __init__(self):
                self.heard = list(engine.words)

            def SetWords(self, value):  # noqa: N802
                pass

            def AcceptWaveform(self, data):  # noqa: N802
                return bool(self.heard)

            def Result(self):  # noqa: N802
                return json.dumps({"text": self.heard.pop(0) if self.heard else ""})

            def FinalResult(self):  # noqa: N802
                return json.dumps({"text": ""})

        return Recognizer()


# --- dictation ---------------------------------------------------------------------


def test_dictation_is_unavailable_without_the_model(tmp_path):
    transcriber = speech.Transcriber(tmp_path / "missing", engine=FakeEngine())
    assert transcriber.available is False
    with pytest.raises(speech.SpeechError) as caught:
        transcriber.transcribe(wav_bytes())
    assert caught.value.code == "no-model"
    assert caught.value.status == 503


def test_dictation_joins_what_the_model_hears(tmp_path):
    engine = FakeEngine()
    transcriber = speech.Transcriber(tmp_path, engine=engine)
    assert transcriber.available is True
    assert transcriber.transcribe(wav_bytes()) == "عسلامة لاباس"
    assert engine.rates == [16000.0]


def test_dictation_loads_the_model_once(tmp_path):
    engine = FakeEngine()
    transcriber = speech.Transcriber(tmp_path, engine=engine)
    transcriber.transcribe(wav_bytes())
    transcriber.transcribe(wav_bytes())
    assert engine.models == [str(tmp_path)]


@pytest.mark.parametrize("audio", [b"not a wav file", wav_bytes(channels=2), wav_bytes(width=1), b""])
def test_dictation_refuses_audio_it_cannot_read(tmp_path, audio):
    transcriber = speech.Transcriber(tmp_path, engine=FakeEngine())
    with pytest.raises(speech.SpeechError) as caught:
        transcriber.transcribe(audio)
    assert caught.value.code == "bad-audio"
    assert caught.value.status == 400


# --- voice -------------------------------------------------------------------------


class FakeOpener:
    def __init__(self, body=b"ID3fake-mp3", error=None):
        self.body, self.error, self.requests = body, error, []

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        if self.error:
            raise self.error

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return Response(self.body)


def test_voice_is_unavailable_without_a_key():
    voice = speech.Voice(api_key="", opener=FakeOpener())
    assert voice.available is False
    with pytest.raises(speech.SpeechError) as caught:
        voice.synthesize("أنا مريض.")
    assert caught.value.code == "no-key"
    assert caught.value.status == 503


def test_voice_asks_elevenlabs_with_the_tunisian_voice():
    opener = FakeOpener()
    voice = speech.Voice(api_key="k-123", opener=opener)
    assert voice.synthesize("أنا مريض. عندي وجيعة.") == b"ID3fake-mp3"
    request = opener.requests[0]
    assert request.full_url == (
        "https://api.elevenlabs.io/v1/text-to-speech/InB4o9iYj3MQ4HtGs2KV?output_format=mp3_44100_128"
    )
    assert request.get_header("Xi-api-key") == "k-123"
    body = json.loads(request.data.decode("utf-8"))
    assert body == {"text": "أنا مريض. عندي وجيعة.", "model_id": "eleven_multilingual_v2"}


def test_voice_says_a_sentence_once_and_then_reuses_it():
    opener = FakeOpener()
    voice = speech.Voice(api_key="k", opener=opener)
    voice.synthesize("لاباس.")
    voice.synthesize("لاباس.")
    voice.synthesize("إيه.")
    assert len(opener.requests) == 2


@pytest.mark.parametrize("text", ["", "   ", "ا" * 601])
def test_voice_refuses_empty_or_long_text(text):
    with pytest.raises(speech.SpeechError) as caught:
        speech.Voice(api_key="k", opener=FakeOpener()).synthesize(text)
    assert caught.value.code == "bad-text"


@pytest.mark.parametrize("status,code", [(401, "key-refused"), (402, "paid-plan"), (429, "busy"), (500, "upstream")])
def test_voice_names_elevenlabs_errors(status, code):
    error = urllib.error.HTTPError("https://api.elevenlabs.io", status, "x", {}, io.BytesIO(b"{}"))
    with pytest.raises(speech.SpeechError) as caught:
        speech.Voice(api_key="k", opener=FakeOpener(error=error)).synthesize("لا.")
    assert caught.value.code == code
    assert caught.value.status == 502


def test_voice_without_network_says_so():
    opener = FakeOpener(error=urllib.error.URLError("no route"))
    with pytest.raises(speech.SpeechError) as caught:
        speech.Voice(api_key="k", opener=opener).synthesize("لا.")
    assert caught.value.code == "offline"


# --- settings ----------------------------------------------------------------------


def test_settings_come_from_the_environment_then_the_env_file(tmp_path):
    (tmp_path / ".env").write_text(
        "# local settings\nELEVENLABS_API_KEY=from-file\nELEVENLABS_VOICE_ID=voice-file\nOTHER=x\n", encoding="utf-8"
    )
    service = speech.from_environment({"ELEVENLABS_VOICE_ID": "voice-env"}, root=tmp_path)
    assert service.voice.api_key == "from-file"
    assert service.voice.voice_id == "voice-env"
    assert service.transcriber.model_dir == tmp_path / "data" / "tunisian_stt"


def test_status_says_what_is_ready(tmp_path):
    service = speech.Speech(speech.Transcriber(tmp_path, engine=FakeEngine()), speech.Voice(api_key=""))
    assert service.status() == {"stt": True, "tts": False}


# --- the server's /speech/ endpoints -------------------------------------------------


class FakeSpeech:
    def __init__(self, stt=True, tts=True):
        self.stt, self.tts, self.audio = stt, tts, []

    def status(self):
        return {"stt": self.stt, "tts": self.tts}

    def transcribe(self, audio):
        if not self.stt:
            raise speech.SpeechError("no-model", 503)
        self.audio.append(audio)
        return "إمشي للسبيطار اليوم"

    def synthesize(self, text):
        if not self.tts:
            raise speech.SpeechError("no-key", 503)
        return b"ID3" + text.encode("utf-8")


@pytest.fixture
def start():
    servers = []

    def run(fake):
        server = serve.make_server("127.0.0.1", 0, speech=fake)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return server.server_address[1]

    yield run
    for server in servers:
        server.shutdown()
        server.server_close()


def call(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request(method, path, body=body, headers=headers or {})
    response = conn.getresponse()
    data = response.read()
    conn.close()
    return response.status, response.getheader("Content-Type"), data


def test_status_endpoint(start):
    port = start(FakeSpeech(stt=True, tts=False))
    status, kind, data = call(port, "GET", "/speech/status")
    assert status == 200 and kind == "application/json"
    assert json.loads(data) == {"ok": True, "stt": True, "tts": False}


def test_dictation_endpoint_returns_the_text(start):
    fake = FakeSpeech()
    port = start(fake)
    status, _, data = call(port, "POST", "/speech/stt", wav_bytes(), {"Content-Type": "audio/wav"})
    assert status == 200
    assert json.loads(data) == {"ok": True, "text": "إمشي للسبيطار اليوم"}
    assert fake.audio == [wav_bytes()]


def test_dictation_endpoint_accepts_a_minute_of_audio_but_not_more(start):
    port = start(FakeSpeech())
    status, _, _ = call(port, "POST", "/speech/stt", wav_bytes(seconds=60))
    assert status == 200
    status, _, data = call(port, "POST", "/speech/stt", wav_bytes(seconds=70))
    assert status == 400
    assert json.loads(data)["error"] == "too-long"


def test_dictation_endpoint_without_the_model(start):
    port = start(FakeSpeech(stt=False))
    status, _, data = call(port, "POST", "/speech/stt", wav_bytes())
    assert status == 503
    assert json.loads(data) == {"ok": False, "error": "no-model"}


def test_voice_endpoint_returns_mp3(start):
    port = start(FakeSpeech())
    body = json.dumps({"text": "أنا مريض."}).encode("utf-8")
    status, kind, data = call(port, "POST", "/speech/tts", body, {"Content-Type": "application/json"})
    assert status == 200 and kind == "audio/mpeg"
    assert data == "ID3أنا مريض.".encode("utf-8")


def test_voice_endpoint_without_a_key(start):
    port = start(FakeSpeech(tts=False))
    status, _, data = call(port, "POST", "/speech/tts", json.dumps({"text": "لا."}).encode("utf-8"))
    assert status == 503
    assert json.loads(data) == {"ok": False, "error": "no-key"}


def test_voice_endpoint_refuses_a_body_without_text(start):
    port = start(FakeSpeech())
    status, _, data = call(port, "POST", "/speech/tts", b'{"words": 1}')
    assert status == 400
    assert json.loads(data)["error"] == "bad-text"


def test_speech_paths_need_the_right_method(start):
    port = start(FakeSpeech())
    assert call(port, "GET", "/speech/stt")[0] == 405
    assert call(port, "POST", "/speech/status", b"")[0] == 405
    assert call(port, "GET", "/speech/nothing")[0] == 404
