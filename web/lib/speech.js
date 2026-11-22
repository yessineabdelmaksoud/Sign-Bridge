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
