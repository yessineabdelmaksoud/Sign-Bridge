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
