import assert from "node:assert/strict";
import test from "node:test";

import { SignCapture } from "../lib/capture.js";
import { clipFeatures } from "../lib/features.js";
import { emptyFrame } from "../lib/landmarks.js";
import { DenseModel } from "../lib/model.js";
import { OTHER, Recognizer, narrow } from "../lib/recognizer.js";
import { fixtures, readBuffer, readJson, syntheticClip } from "./helpers.js";

const L = ["DOLOR", "FIEBRE2", "TOSER", "CITA", "OTHER"];
const NO_PREDICTION = { label: null, confidence: 0, top3: [] };

const manifest = readJson("models/health_signs.json");
const weights = readBuffer("models/health_signs.bin");
const recognizer = new Recognizer(DenseModel.fromBuffer(manifest, weights));

/** Stands in for the model, as FakeModel does in tests/test_recognizer.py. */
function fakeRecognizer(probs, { weakLabels = [] } = {}) {
  const model = {
    labels: L, threshold: 0.5, weakLabels, seen: null,
    predict(features) {
      this.seen = features;
      return Float32Array.from(probs);
    },
  };
  return { model, recognizer: new Recognizer(model) };
}

/** Same label, same top 3 in the same order, numbers within `tolerance`. */
function assertSamePrediction(actual, expected, tolerance, message) {
  assert.deepEqual(Object.keys(actual), ["label", "confidence", "top3"], message);
  assert.equal(actual.label, expected.label, `${message}: label`);
  assert.ok(Math.abs(actual.confidence - expected.confidence) <= tolerance,
    `${message}: confidence ${actual.confidence}, Python gives ${expected.confidence}`);
  assert.deepEqual(actual.top3.map(([label]) => label), expected.top3.map(([label]) => label), `${message}: top 3`);
  actual.top3.forEach(([, p], i) => {
    assert.ok(Math.abs(p - expected.top3[i][1]) <= tolerance, `${message}: top 3 probability ${i}`);
  });
}

// --- narrow -----------------------------------------------------------------------

test("OTHER is the label for none of these signs", () => {
  assert.equal(OTHER, "OTHER");
});

test("narrow equals the Python narrow on every fixture case", () => {
  const { index } = fixtures();
  assert.ok(index.narrow_cases.length >= 12);
  for (const item of index.narrow_cases) {
    const actual = narrow(item.probs, item.labels, item.allowed, item.threshold);
    assertSamePrediction(actual, item.expected, 1e-12, item.name);
  }
});

// The three narrowing tests of tests/test_recognizer.py.

test("narrowing ignores labels outside the question", () => {
  const p = narrow([0.30, 0.10, 0.05, 0.50, 0.05], L, ["DOLOR", "FIEBRE2", "TOSER"], 0.5);
  assert.equal(p.label, "DOLOR");
  assert.ok(Math.abs(p.confidence - 0.60) < 1e-6);
  assert.deepEqual(p.top3.map(([label]) => label), ["DOLOR", "FIEBRE2", "TOSER"]);
});

test("below the threshold gives no label but keeps the top 3", () => {
  const p = narrow([0.25, 0.20, 0.20, 0.0, 0.35], L, ["DOLOR", "FIEBRE2", "TOSER"], 0.6);
  assert.equal(p.label, null);
  assert.equal(p.top3.length, 3);
});

test("OTHER winning gives no label", () => {
  assert.equal(narrow([0.1, 0.1, 0.1, 0.0, 0.7], L, ["DOLOR", "FIEBRE2"], 0.3).label, null);
});

test("narrow takes the probabilities as a Float32Array too", () => {
  const p = narrow(Float32Array.of(0.5, 0.25, 0, 0, 0.25), L, ["DOLOR", "FIEBRE2"], 0.5);
  assert.deepEqual(p, { label: "DOLOR", confidence: 0.5, top3: [["DOLOR", 0.5], ["FIEBRE2", 0.25]] });
});

test("narrow gives no prediction when there is nothing to choose from", () => {
  assert.deepEqual(narrow([0.3, 0.1, 0.05, 0.5, 0.05], L, [], 0.5), NO_PREDICTION);
  assert.deepEqual(narrow([0, 0, 0, 1, 0], L, ["DOLOR", "FIEBRE2"], 0.5), NO_PREDICTION);
});

// --- Recognizer with a stand-in model: the rest of tests/test_recognizer.py -----------

test("predict feeds the model one clip of 32 x 184 features", () => {
  const { model, recognizer: r } = fakeRecognizer([0.7, 0.1, 0.1, 0.05, 0.05]);
  const clip = syntheticClip(20);
  const p = r.predict(clip, ["DOLOR", "FIEBRE2"]);
  assert.equal(p.label, "DOLOR");
  assert.deepEqual(model.seen, clipFeatures(clip));
  assert.equal(model.seen.length, 32 * 184);
});

test("weak labels are never proposed", () => {
  const { recognizer: r } = fakeRecognizer([0.7, 0.2, 0.05, 0.0, 0.05], { weakLabels: ["DOLOR"] });
  const p = r.predict(syntheticClip(20), ["DOLOR", "FIEBRE2"]);
  assert.equal(p.label, "FIEBRE2");
  assert.deepEqual(p.top3.map(([label]) => label), ["FIEBRE2"]);
});

test("a clip without a hand gives no prediction, and the model is not asked", () => {
  const { model, recognizer: r } = fakeRecognizer([0.7, 0.1, 0.1, 0.05, 0.05]);
  assert.deepEqual(r.predict(syntheticClip(20, { left: false }), ["DOLOR"]), NO_PREDICTION);
  assert.deepEqual(r.predict([], ["DOLOR"]), NO_PREDICTION);
  assert.equal(model.seen, null);
});

test("only a missing hand is turned into no prediction: other errors are not hidden", () => {
  const { recognizer: r } = fakeRecognizer([0.7, 0.1, 0.1, 0.05, 0.05]);
  assert.throws(() => r.predict(null, ["DOLOR"]), TypeError);
});

test("allowed defaults to every sign except OTHER and the weak labels", () => {
  const { recognizer: r } = fakeRecognizer([0.05, 0.1, 0.5, 0.05, 0.3], { weakLabels: ["TOSER"] });
  const p = r.predict(syntheticClip(20));
  // TOSER is dropped; of the rest, OTHER has 0.3 of 0.5 and wins, so no label.
  assert.equal(p.label, null);
  assert.deepEqual(p.top3.map(([label]) => label), ["FIEBRE2", "DOLOR", "CITA"]);
  assert.ok(Math.abs(p.confidence - 0.2) < 1e-6);
});

test("the recognizer lists every label, OTHER included", () => {
  assert.deepEqual(recognizer.labels, manifest.labels);
  assert.equal(recognizer.labels.length, 23);
  assert.ok(recognizer.labels.includes(OTHER));
});

// --- Recognizer with the real model, against Python -----------------------------------

test("predict on raw untrimmed clips gives the label Python SignRecognizer.predict gives", () => {
  const { index, clip } = fixtures();
  const signs = index.labels.filter((label) => label !== OTHER);
  const sets = { ...index.sign_sets, all: signs };
  let proposed = 0;
  for (const item of index.clips) {
    const raw = clip(item.offset, item.frames);
    for (const [name, allowed] of Object.entries(sets)) {
      const actual = recognizer.predict(raw, allowed);
      assertSamePrediction(actual, item.predictions[name], 1e-4, `${item.label} ${item.id} in ${name}`);
      if (actual.label !== null) proposed++;
    }
    assertSamePrediction(recognizer.predict(raw), item.predictions.all, 1e-4, `${item.label} ${item.id}, default`);
  }
  assert.ok(proposed > 56, "most clips get a proposal in at least one group");
});

test("a real clip with its hands blanked out gives no prediction", () => {
  const { index, clip } = fixtures();
  const item = index.clips[0];
  const raw = clip(item.offset, item.frames).map((frame) => Float32Array.from(frame).fill(NaN, 33 * 3));
  assert.deepEqual(recognizer.predict(raw), NO_PREDICTION);
  assert.deepEqual(recognizer.predict(raw, index.sign_sets.symptom), NO_PREDICTION);
});

test("a raw clip replayed through SignCapture is cut and named as in Python", () => {
  const { index, clip } = fixtures();
  for (const item of index.clips) {
    const capture = new SignCapture();
    const frames = [...clip(item.offset, item.frames), ...Array.from({ length: 12 }, emptyFrame)];
    let captured = null;
    frames.forEach((frame, i) => {
      captured = capture.push(frame, i / 30) ?? captured;
    });
    if (item.replay === null) {
      assert.equal(captured, null, item.id);
      continue;
    }
    assert.equal(captured.length, item.replay.frames, `${item.id}: frames captured`);
    assertSamePrediction(recognizer.predict(captured), item.replay.prediction, 1e-4, `${item.label} ${item.id}`);
  }
});

test("the JavaScript pipeline reproduces the test-split scores in the manifest", () => {
  const { index, clip } = fixtures();
  const model = DenseModel.fromBuffer(manifest, weights);
  const round = (value) => Math.round(value * 1e4) / 1e4;
  let top1 = 0;
  let top3 = 0;
  const narrowed = Object.fromEntries(Object.keys(index.sign_sets).map((name) => [name, { hits: 0, n: 0 }]));
  for (const item of index.clips) {
    const probs = model.predict(clipFeatures(clip(item.offset, item.frames)));
    const ranked = [...probs.keys()].sort((a, b) => probs[b] - probs[a]).map((i) => index.labels[i]);
    if (ranked[0] === item.label) top1++;
    if (ranked.slice(0, 3).includes(item.label)) top3++;
    for (const [name, allowed] of Object.entries(index.sign_sets)) {
      if (!allowed.includes(item.label)) continue;
      narrowed[name].n++;
      if (narrow(probs, index.labels, allowed, 0).label === item.label) narrowed[name].hits++;
    }
  }
  const official = manifest.metrics.official_test;
  assert.equal(index.clips.length, official.n);
  assert.equal(round(top1 / official.n), official.top1);
  assert.equal(round(top3 / official.n), official.top3);
  for (const [name, { hits, n }] of Object.entries(narrowed)) {
    assert.equal(n, official.narrowed[name].n, `${name}: clips`);
    assert.equal(round(hits / n), official.narrowed[name].top1, `${name}: accuracy`);
  }
});

// --- loading ------------------------------------------------------------------------

test("Recognizer.load reads models/health_signs.json unless told otherwise", async (t) => {
  const requested = [];
  t.mock.method(globalThis, "fetch", async (url) => {
    requested.push(String(url));
    return new Response(String(url).endsWith(".json") ? JSON.stringify(manifest) : weights);
  });
  const loaded = await Recognizer.load();
  assert.deepEqual(requested, ["models/health_signs.json", "models/health_signs.bin"]);
  assert.deepEqual(loaded.labels, manifest.labels);

  const { index, clip } = fixtures();
  const item = index.clips[1];
  const raw = clip(item.offset, item.frames);
  assert.deepEqual(loaded.predict(raw), recognizer.predict(raw));

  requested.length = 0;
  await Recognizer.load("elsewhere/health_signs.json");
  assert.deepEqual(requested, ["elsewhere/health_signs.json", "elsewhere/health_signs.bin"]);
});
