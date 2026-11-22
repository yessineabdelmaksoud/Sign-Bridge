// The browser's voice helpers: reading text aloud and dictation (web/CONVERSATION.md).
// Every function looks the browser objects up when called and does nothing
// harmful when they are missing (another browser, or Node in the tests).

/** True when this browser can read text aloud. */
export function canSpeak() {
  return Boolean(globalThis.speechSynthesis) && typeof globalThis.SpeechSynthesisUtterance === "function";
}

/** "fr_FR" or "FR-fr" -> ["fr", "fr-fr"] (language, full tag). */
function tag(lang) {
  const full = String(lang ?? "").trim().toLowerCase().replace(/_/g, "-");
  return [full.split("-")[0], full];
}

/**
 * The voice to read `lang` with: a voice on this computer first (nothing leaves
 * the machine), then any voice of the language; inside each, the same region
 * first ("fr-FR" before "fr-CA"). Null when no voice speaks the language.
 * Pure: does not touch the browser.
 */
export function pickVoice(voices, lang = "fr-FR") {
  const [language, full] = tag(lang);
  if (!language) return null;
  let best = null;
  let bestScore = 0;
  for (const voice of voices ?? []) {
    const [voiceLanguage, voiceFull] = tag(voice?.lang);
    if (!voiceLanguage || voiceLanguage !== language) continue;
    const score = (voice.localService ? 2 : 0) + (voiceFull === full ? 1 : 0) + 1;
    if (score > bestScore) {
      best = voice;
      bestScore = score;
    }
  }
  return best;
}

/**
 * Reads `text` aloud, cutting off whatever was being said. Returns true when
 * the text was handed to the speech engine; does nothing without speech.
 */
export function speak(text, { lang = "fr-FR" } = {}) {
  if (!canSpeak()) return false;
  try {
    const synth = globalThis.speechSynthesis;
    synth.cancel();
    const words = String(text ?? "").trim();
    if (!words) return false;
    const utterance = new globalThis.SpeechSynthesisUtterance(words);
    const voice = pickVoice(synth.getVoices?.() ?? [], lang); // empty until the browser has loaded them
    if (voice) utterance.voice = voice;
    utterance.lang = lang;
    synth.speak(utterance);
    return true;
  } catch {
    return false;
  }
}

/**
 * Dictation with the browser's speech recognition (Chrome sends the audio to
 * its own speech service).
 *
 *   onText(text, isFinal): everything heard since start(), interim words included;
 *                          isFinal is true when none of it is interim any more.
 *   onState("listening" | "idle" | "error", detail): after an error ("not-allowed",
 *                          "no-speech", "network", ...) "idle" follows.
 *
 * Returns { supported, start(), stop() }. start() while listening does nothing.
 */
export function dictation({ lang = "fr-FR", onText, onState, continuous = true } = {}) {
  const Recognition = globalThis.SpeechRecognition || globalThis.webkitSpeechRecognition;
  const state = (name, detail) => onState?.(name, detail);
  if (typeof Recognition !== "function") {
    return {
      supported: false,
      start() {
        state("error", "not-supported");
      },
      stop() {},
    };
  }

  let active = null;

  function start() {
    if (active) return;
    let recognition;
    try {
      recognition = new Recognition();
      recognition.lang = lang;
      recognition.interimResults = true;
      recognition.continuous = continuous;
      recognition.maxAlternatives = 1;
    } catch (error) {
      state("error", error?.message ?? String(error));
      return;
    }
    recognition.onstart = () => state("listening");
    recognition.onresult = (event) => {
      if (recognition !== active) return;
      const parts = [];
      let isFinal = true;
      for (const result of Array.from(event.results ?? [])) {
        parts.push(String(result?.[0]?.transcript ?? "").trim());
        if (!result?.isFinal) isFinal = false;
      }
      onText?.(parts.filter(Boolean).join(" "), isFinal);
    };
    recognition.onerror = (event) => state("error", event?.error ?? "error");
    recognition.onend = () => {
      if (recognition === active) active = null;
      state("idle");
    };
    active = recognition;
    try {
      recognition.start();
    } catch (error) {
      active = null;
      state("error", error?.message ?? String(error));
    }
  }

  function stop() {
    try {
      active?.stop();
    } catch {
      // already stopped
    }
  }

  return { supported: true, start, stop };
}

/** True when this browser has a voice for `lang` (for example "ar-TN": any Arabic voice). */
export function hasVoice(lang) {
  if (!canSpeak()) return false;
  try {
    return Boolean(pickVoice(globalThis.speechSynthesis.getVoices?.() ?? [], lang));
  } catch {
    return false;
  }
}

// --- Tunisian speech through the local server (web/serve.py, /speech/) ---------------
//
// Dictation: the doctor's voice is recorded here, sent to the server as a 16 kHz WAV
// and transcribed there by the Tunisian Vosk model. Voice: the server asks ElevenLabs
// for the Tunisian voice of a sentence and sends back MP3 audio.

/** `samples` at `fromRate` resampled to `toRate` by averaging (only ever down). */
export function downsample(samples, fromRate, toRate = 16000) {
  if (!(fromRate > toRate)) return samples;
  const ratio = fromRate / toRate;
  const out = new Float32Array(Math.floor(samples.length / ratio));
  for (let i = 0; i < out.length; i += 1) {
    const start = Math.floor(i * ratio);
    const end = Math.min(samples.length, Math.floor((i + 1) * ratio));
    let sum = 0;
    for (let k = start; k < end; k += 1) sum += samples[k];
    out[i] = sum / Math.max(1, end - start);
  }
  return out;
}

/** Mono samples in [-1, 1] as a 16-bit PCM WAV file (an ArrayBuffer). */
export function encodeWav(samples, rate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const text = (offset, value) => [...value].forEach((ch, k) => view.setUint8(offset + k, ch.charCodeAt(0)));
  text(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  text(8, "WAVEfmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);          // PCM
  view.setUint16(22, 1, true);          // mono
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);   // bytes per second
  view.setUint16(32, 2, true);          // bytes per frame
  view.setUint16(34, 16, true);
  text(36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i += 1) {
    const value = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, Math.round(value < 0 ? value * 0x8000 : value * 0x7fff), true);
  }
  return buffer;
}

/**
 * The microphone, recorded into a 16 kHz WAV: start() asks for the microphone,
 * stop() gives the recording as a Blob, cancel() drops it. After `maxSeconds`
 * onLimit() is called (the caller stops).
 */
export function recorder({ maxSeconds = 45, onLimit } = {}) {
  let stream = null;
  let context = null;
  let source = null;
  let node = null;
  let chunks = [];
  let rate = 48000;
  let timer = 0;

  function release() {
    clearTimeout(timer);
    try {
      source?.disconnect();
      node?.disconnect();
    } catch { /* already disconnected */ }
    for (const track of stream?.getTracks?.() ?? []) track.stop();
    context?.close?.().catch(() => {});
    stream = context = source = node = null;
  }

  async function start() {
    const media = globalThis.navigator?.mediaDevices;
    if (typeof media?.getUserMedia !== "function") {
      throw Object.assign(new Error("no microphone here"), { name: "NotSupportedError" });
    }
    stream = await media.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
    const Context = globalThis.AudioContext || globalThis.webkitAudioContext;
    context = new Context();
    if (context.state === "suspended") await context.resume();
    rate = context.sampleRate;
    source = context.createMediaStreamSource(stream);
    node = context.createScriptProcessor(4096, 1, 1);
    node.onaudioprocess = (event) => chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
    source.connect(node);
    node.connect(context.destination);   // needed for the node to run; it outputs silence
    timer = setTimeout(() => onLimit?.(), maxSeconds * 1000);
  }

  async function stop() {
    const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
    const samples = new Float32Array(total);
    let offset = 0;
    for (const chunk of chunks) {
      samples.set(chunk, offset);
      offset += chunk.length;
    }
    chunks = [];
    const from = rate;
    release();
    return new Blob([encodeWav(downsample(samples, from, 16000), 16000)], { type: "audio/wav" });
  }

  function cancel() {
    chunks = [];
    release();
  }

  return { start, stop, cancel };
}

function microphoneError(error) {
  switch (error?.name) {
    case "NotAllowedError":
    case "SecurityError":
      return "not-allowed";
    case "NotSupportedError":
      return "not-supported";
    default:
      return "audio-capture";
  }
}

/**
 * Dictation through the server, with the same shape as dictation(): the doctor
 * presses once to talk and again to stop; the recording is then transcribed.
 *
 *   onState("listening" | "working" | "idle" | "error", detail): "working" while
 *          the server transcribes; errors: "not-allowed", "audio-capture",
 *          "no-speech", "network", or the server's code ("no-model", ...).
 *   onText(text, true) once, with what was heard.
 */
export function serverDictation({ transcribe, onText, onState, record, maxSeconds = 45 } = {}) {
  const state = (name, detail) => onState?.(name, detail);
  const makeRecorder = record ?? recorder;
  let active = null;
  let busy = false;

  async function start() {
    if (active || busy) return;
    const recording = makeRecorder({ maxSeconds, onLimit: () => stop() });
    active = recording;
    try {
      await recording.start();
    } catch (error) {
      active = null;
      recording.cancel?.();
      state("error", microphoneError(error));
      state("idle");
      return;
    }
    if (active === recording) state("listening");
  }

  async function stop() {
    const recording = active;
    if (!recording || busy) return;
    active = null;
    busy = true;
    state("working");
    try {
      const wav = await recording.stop();
      const text = String((await transcribe(wav)) ?? "").trim();
      if (text) onText?.(text, true);
      else state("error", "no-speech");
    } catch (error) {
      state("error", error?.code ?? "network");
    } finally {
      busy = false;
      state("idle");
    }
  }

  return { supported: typeof transcribe === "function", start, stop };
}

/**
 * The local server's speech endpoints. Errors carry the server's code
 * (error.code: "no-model", "no-key", "paid-plan", ...) or "network".
 */
export function speechServer(fetchImpl) {
  const go = fetchImpl ?? ((...args) => globalThis.fetch(...args));

  async function call(url, options) {
    let response;
    try {
      response = await go(url, options);
    } catch {
      throw Object.assign(new Error("network"), { code: "network" });
    }
    if (!response.ok) {
      let code = "failed";
      try {
        code = (await response.json())?.error || code;
      } catch { /* not JSON */ }
      throw Object.assign(new Error(code), { code });
    }
    return response;
  }

  return {
    /** { stt, tts }: what the server can do; both false without a server. */
    async status() {
      try {
        const response = await go("speech/status", { cache: "no-store" });
        if (!response.ok) return { stt: false, tts: false };
        const body = await response.json();
        return { stt: Boolean(body?.stt), tts: Boolean(body?.tts) };
      } catch {
        return { stt: false, tts: false };
      }
    },
    async transcribe(wav) {
      const response = await call("speech/stt", { method: "POST", body: wav, headers: { "Content-Type": "audio/wav" } });
      return String((await response.json())?.text ?? "");
    },
    async synthesize(text) {
      const response = await call("speech/tts", {
        method: "POST", body: JSON.stringify({ text }), headers: { "Content-Type": "application/json" },
      });
      return response.blob();
    },
  };
}

let playing = null;

/** Plays audio (the server's MP3), cutting off what was playing. Resolves true once it plays. */
export async function playAudio(blob) {
  stopAudio();
  if (typeof globalThis.Audio !== "function") return false;
  const url = URL.createObjectURL(blob);
  const audio = new globalThis.Audio(url);
  playing = { audio, url };
  audio.addEventListener("ended", () => stopAudio(audio));
  try {
    await audio.play();
    return true;
  } catch {
    stopAudio(audio);
    return false;
  }
}

/** Stops the audio playAudio() started (only `which`, when given). */
export function stopAudio(which) {
  if (!playing || (which && playing.audio !== which)) return;
  try {
    playing.audio.pause();
  } catch { /* not playing */ }
  URL.revokeObjectURL(playing.url);
  playing = null;
}
