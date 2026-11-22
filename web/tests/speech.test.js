import assert from "node:assert/strict";
import test from "node:test";

import { canSpeak, dictation, pickVoice, speak } from "../lib/speech.js";

const voice = (name, lang, localService) => ({ name, lang, localService, default: false });

/** Puts fake browser objects on globalThis for the length of one test. */
function withGlobals(t, values) {
  const saved = {};
  for (const [key, value] of Object.entries(values)) {
    saved[key] = Object.getOwnPropertyDescriptor(globalThis, key);
    Object.defineProperty(globalThis, key, { value, configurable: true, writable: true });
  }
  t.after(() => {
    for (const [key, descriptor] of Object.entries(saved)) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
  });
}

// --- pickVoice -----------------------------------------------------------------

test("pickVoice prefers a local voice of the language", () => {
  const voices = [
    voice("Google français", "fr-FR", false),
    voice("English", "en-US", true),
    voice("Amélie", "fr-FR", true),
  ];
  assert.equal(pickVoice(voices).name, "Amélie");
  assert.equal(pickVoice(voices, "fr-FR").name, "Amélie");
});

test("pickVoice takes any voice of the language when none is local", () => {
  const voices = [voice("English", "en-US", true), voice("Google français", "fr-FR", false)];
  assert.equal(pickVoice(voices, "fr-FR").name, "Google français");
});

test("pickVoice gives null when no voice speaks the language", () => {
  assert.equal(pickVoice([voice("English", "en-US", true), voice("Deutsch", "de-DE", true)], "fr-FR"), null);
  assert.equal(pickVoice([], "fr-FR"), null);
  assert.equal(pickVoice(undefined), null);
  assert.equal(pickVoice(null), null);
});

test("pickVoice prefers the same region, but a local voice comes first", () => {
  const canadian = voice("Canadien", "fr-CA", true);
  const french = voice("Français", "fr-FR", true);
  const remote = voice("Google français", "fr-FR", false);
  assert.equal(pickVoice([canadian, french], "fr-FR").name, "Français");
  assert.equal(pickVoice([remote, canadian], "fr-FR").name, "Canadien");
  assert.equal(pickVoice([voice("Remote CA", "fr-CA", false), remote], "fr-FR").name, "Google français");
});

test("pickVoice reads language tags loosely", () => {
  assert.equal(pickVoice([voice("Android", "fr_FR", true)], "fr-FR").name, "Android");
  assert.equal(pickVoice([voice("Upper", "FR-fr", true)], "fr-FR").name, "Upper");
  assert.equal(pickVoice([voice("Plain", "fr", true)], "fr-FR").name, "Plain");
  assert.equal(pickVoice([voice("Saudi", "ar-SA", true)], "ar-TN").name, "Saudi");
  assert.equal(pickVoice([voice("No tag", undefined, true)], "fr-FR"), null);
});

test("pickVoice does not change the list", () => {
  const voices = [voice("B", "fr-FR", false), voice("A", "fr-FR", true)];
  pickVoice(voices);
  assert.deepEqual(voices.map((v) => v.name), ["B", "A"]);
});

// --- speak -------------------------------------------------------------------------

test("without speech synthesis nothing happens and nothing throws", () => {
  assert.equal(canSpeak(), false);
  assert.equal(speak("Bonjour"), false);
  assert.equal(speak("Bonjour", { lang: "fr-FR" }), false);
});

function fakeSynthesis(voices) {
  const calls = [];
  class Utterance {
    constructor(text) {
      this.text = text;
    }
  }
  const synth = {
    getVoices: () => voices,
    cancel: () => calls.push(["cancel"]),
    speak: (utterance) => calls.push(["speak", utterance]),
  };
  return { synth, Utterance, calls };
}

test("speak cancels what is being said, then says the text with the chosen voice", (t) => {
  const amelie = voice("Amélie", "fr-FR", true);
  const { synth, Utterance, calls } = fakeSynthesis([voice("Google français", "fr-FR", false), amelie]);
  withGlobals(t, { speechSynthesis: synth, SpeechSynthesisUtterance: Utterance });
  assert.equal(canSpeak(), true);
  assert.equal(speak("J'ai de la fièvre."), true);
  assert.deepEqual(calls.map(([name]) => name), ["cancel", "speak"]);
  const utterance = calls[1][1];
  assert.equal(utterance.text, "J'ai de la fièvre.");
  assert.equal(utterance.voice, amelie);
  assert.equal(utterance.lang, "fr-FR");
});

test("speak still speaks, in the right language, before the voices are loaded", (t) => {
  const { synth, Utterance, calls } = fakeSynthesis([]);
  withGlobals(t, { speechSynthesis: synth, SpeechSynthesisUtterance: Utterance });
  assert.equal(speak("Bonjour"), true);
  assert.equal(calls[1][1].lang, "fr-FR");
  assert.equal(calls[1][1].voice, undefined);
});

test("speak with empty text only stops the current speech", (t) => {
  const { synth, Utterance, calls } = fakeSynthesis([]);
  withGlobals(t, { speechSynthesis: synth, SpeechSynthesisUtterance: Utterance });
  assert.equal(speak("   "), false);
  assert.deepEqual(calls.map(([name]) => name), ["cancel"]);
});

test("speak survives a speech engine that throws", (t) => {
  const synth = {
    getVoices: () => {
      throw new Error("boom");
    },
    cancel: () => {
      throw new Error("boom");
    },
    speak: () => {
      throw new Error("boom");
    },
  };
  withGlobals(t, { speechSynthesis: synth, SpeechSynthesisUtterance: class {} });
  assert.equal(speak("Bonjour"), false);
});

// --- dictation ------------------------------------------------------------------

test("without speech recognition, dictation is unsupported and harmless", () => {
  const states = [];
  const d = dictation({ onText: () => assert.fail("no text"), onState: (...args) => states.push(args) });
  assert.equal(d.supported, false);
  d.start();
  d.stop();
  assert.deepEqual(states, [["error", "not-supported"]]);
  const bare = dictation({});
  bare.start();
  bare.stop();
  dictation().start();
});

function fakeRecognition() {
  const made = [];
  class Recognition {
    constructor() {
      this.started = false;
      this.stopped = false;
      made.push(this);
    }
    start() {
      if (this.started) throw new Error("InvalidStateError");
      this.started = true;
    }
    stop() {
      this.stopped = true;
    }
    abort() {
      this.stopped = true;
    }
  }
  return { Recognition, made };
}

const result = (transcript, isFinal) => Object.assign([{ transcript, confidence: 0.9 }], { isFinal });

test("dictation passes on interim and final text, and its state", (t) => {
  const { Recognition, made } = fakeRecognition();
  withGlobals(t, { webkitSpeechRecognition: Recognition });
  const texts = [];
  const states = [];
  const d = dictation({ lang: "fr-FR", onText: (...args) => texts.push(args), onState: (...args) => states.push(args) });
  assert.equal(d.supported, true);
  d.start();
  assert.equal(made.length, 1);
  const rec = made[0];
  assert.equal(rec.started, true);
  assert.equal(rec.lang, "fr-FR");
  assert.equal(rec.interimResults, true);
  rec.onstart?.({});
  rec.onresult({ resultIndex: 0, results: [result("avez-vous", false)] });
  rec.onresult({ resultIndex: 0, results: [result("avez-vous de la fièvre", true)] });
  rec.onresult({ resultIndex: 1, results: [result("avez-vous de la fièvre", true), result(" depuis", false)] });
  d.stop();
  assert.equal(rec.stopped, true);
  rec.onend({});
  assert.deepEqual(texts, [
    ["avez-vous", false],
    ["avez-vous de la fièvre", true],
    ["avez-vous de la fièvre depuis", false],
  ]);
  assert.deepEqual(states, [["listening", undefined], ["idle", undefined]]);
});

test("dictation reports an error, then goes idle", (t) => {
  const { Recognition, made } = fakeRecognition();
  withGlobals(t, { SpeechRecognition: Recognition });
  const states = [];
  const d = dictation({ onState: (...args) => states.push(args) });
  d.start();
  made[0].onstart({});
  made[0].onerror({ error: "not-allowed" });
  made[0].onend({});
  assert.deepEqual(states, [["listening", undefined], ["error", "not-allowed"], ["idle", undefined]]);
});

test("starting twice keeps one recognition; a start that throws is reported", (t) => {
  const { Recognition, made } = fakeRecognition();
  withGlobals(t, { SpeechRecognition: Recognition });
  const states = [];
  const d = dictation({ onState: (...args) => states.push(args) });
  d.start();
  d.start();
  assert.equal(made.length, 1);
  made[0].onend({});
  Recognition.prototype.start = () => {
    throw new Error("InvalidStateError");
  };
  d.start();
  assert.deepEqual(states.at(-1)[0], "error");
});

test("dictation without callbacks does not throw", (t) => {
  const { Recognition, made } = fakeRecognition();
  withGlobals(t, { SpeechRecognition: Recognition });
  const d = dictation({});
  d.start();
  made[0].onstart({});
  made[0].onresult({ resultIndex: 0, results: [result("bonjour", true)] });
  made[0].onerror({ error: "no-speech" });
  made[0].onend({});
  d.stop();
});

// --- Tunisian speech through the local server (web/serve.py /speech/) ---------------

import { downsample, encodeWav, serverDictation, speechServer } from "../lib/speech.js";

test("downsample averages 48 kHz audio down to 16 kHz", () => {
  const input = Float32Array.from({ length: 48 }, (_, i) => (i % 3 === 0 ? 0.3 : 0.6));
  const output = downsample(input, 48000, 16000);
  assert.equal(output.length, 16);
  for (const value of output) assert.ok(Math.abs(value - 0.5) < 1e-6);
  assert.equal(downsample(input, 16000, 16000), input);
});

test("encodeWav writes 16-bit mono PCM with a WAV header", () => {
  const buffer = encodeWav(Float32Array.from([0, 1, -1, 0.5, 2]), 16000);
  const view = new DataView(buffer);
  const text = (offset, length) => String.fromCharCode(...new Uint8Array(buffer, offset, length));
  assert.equal(buffer.byteLength, 44 + 5 * 2);
  assert.equal(text(0, 4), "RIFF");
  assert.equal(text(8, 8), "WAVEfmt ");
  assert.equal(view.getUint16(20, true), 1);       // PCM
  assert.equal(view.getUint16(22, true), 1);       // mono
  assert.equal(view.getUint32(24, true), 16000);   // sample rate
  assert.equal(view.getUint16(34, true), 16);      // bits
  assert.equal(text(36, 4), "data");
  assert.equal(view.getUint32(40, true), 10);
  assert.deepEqual([0, 1, 2, 3, 4].map((k) => view.getInt16(44 + 2 * k, true)), [0, 32767, -32768, 16384, 32767]);
});

function fakeFetch(routes) {
  const calls = [];
  const fetch = async (url, options = {}) => {
    calls.push({ url, options });
    const answer = routes[url];
    if (!answer) throw new TypeError("network down");
    return {
      ok: answer.status === undefined || answer.status < 400,
      status: answer.status ?? 200,
      json: async () => answer.json,
      blob: async () => answer.blob,
    };
  };
  return { fetch, calls };
}

test("speechServer reads what the server can do, and nothing when there is no server", async () => {
  const { fetch } = fakeFetch({ "speech/status": { json: { ok: true, stt: true, tts: false } } });
  assert.deepEqual(await speechServer(fetch).status(), { stt: true, tts: false });
  assert.deepEqual(await speechServer(fakeFetch({}).fetch).status(), { stt: false, tts: false });
});

test("speechServer sends a recording and gets the text back", async () => {
  const { fetch, calls } = fakeFetch({ "speech/stt": { json: { ok: true, text: "عسلامة لاباس" } } });
  const wav = new Blob(["RIFF"], { type: "audio/wav" });
  assert.equal(await speechServer(fetch).transcribe(wav), "عسلامة لاباس");
  assert.equal(calls[0].options.method, "POST");
  assert.equal(calls[0].options.body, wav);
  assert.equal(calls[0].options.headers["Content-Type"], "audio/wav");
});

test("speechServer turns the server's refusals into error codes", async () => {
  const server = speechServer(fakeFetch({
    "speech/stt": { status: 503, json: { ok: false, error: "no-model" } },
    "speech/tts": { status: 502, json: { ok: false, error: "paid-plan" } },
  }).fetch);
  await assert.rejects(server.transcribe(new Blob([])), { code: "no-model" });
  await assert.rejects(server.synthesize("لا."), { code: "paid-plan" });
  await assert.rejects(speechServer(fakeFetch({}).fetch).synthesize("لا."), { code: "network" });
});

test("speechServer asks for the voice of a sentence", async () => {
  const mp3 = new Blob(["ID3"], { type: "audio/mpeg" });
  const { fetch, calls } = fakeFetch({ "speech/tts": { blob: mp3 } });
  assert.equal(await speechServer(fetch).synthesize("أنا مريض."), mp3);
  assert.deepEqual(JSON.parse(calls[0].options.body), { text: "أنا مريض." });
});

/** A recorder that "hears" for as long as it runs. */
function fakeRecorder({ failWith } = {}) {
  const log = [];
  return {
    log,
    create: () => ({
      async start() {
        log.push("start");
        if (failWith) throw Object.assign(new Error(failWith), { name: failWith });
      },
      async stop() {
        log.push("stop");
        return new Blob(["RIFF"], { type: "audio/wav" });
      },
      cancel() {
        log.push("cancel");
      },
    }),
  };
}

test("serverDictation: listen, then transcribe when stopped", async () => {
  const states = [];
  const texts = [];
  const recorder = fakeRecorder();
  const dictation = serverDictation({
    record: recorder.create,
    transcribe: async () => "إمشي للسبيطار اليوم",
    onText: (text, isFinal) => texts.push([text, isFinal]),
    onState: (value, detail) => states.push(detail ? `${value}:${detail}` : value),
  });
  assert.equal(dictation.supported, true);
  await dictation.start();
  assert.deepEqual(states, ["listening"]);
  await dictation.stop();
  assert.deepEqual(states, ["listening", "working", "idle"]);
  assert.deepEqual(texts, [["إمشي للسبيطار اليوم", true]]);
  assert.deepEqual(recorder.log, ["start", "stop"]);
});

test("serverDictation says when nothing was heard, or the microphone is blocked", async () => {
  const states = [];
  const onState = (value, detail) => states.push(detail ? `${value}:${detail}` : value);
  const quiet = serverDictation({ record: fakeRecorder().create, transcribe: async () => "  ", onState });
  await quiet.start();
  await quiet.stop();
  assert.deepEqual(states, ["listening", "working", "error:no-speech", "idle"]);

  states.length = 0;
  const blocked = serverDictation({ record: fakeRecorder({ failWith: "NotAllowedError" }).create, transcribe: async () => "", onState });
  await blocked.start();
  assert.deepEqual(states, ["error:not-allowed", "idle"]);

  states.length = 0;
  const broken = serverDictation({
    record: fakeRecorder().create,
    transcribe: async () => { throw Object.assign(new Error("x"), { code: "no-model" }); },
    onState,
  });
  await broken.start();
  await broken.stop();
  assert.deepEqual(states, ["listening", "working", "error:no-model", "idle"]);
});
