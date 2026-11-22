import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";

import { clipFeatures } from "../lib/features.js";
import { DenseModel } from "../lib/model.js";
import { fixtures, maxDifference, readBuffer, readJson } from "./helpers.js";

// The matrix products are summed in a different order than NumPy's linear
// algebra library sums them, so the last digits of a probability differ.
const TOLERANCE = 1e-5;

const manifest = readJson("models/health_signs.json");
const weights = readBuffer("models/health_signs.bin");
const model = DenseModel.fromBuffer(manifest, weights);

/** Replace fetch for one test. `files` maps a URL to its body; any other URL is a 404. */
function stubFetch(t, files) {
  const requested = [];
  t.mock.method(globalThis, "fetch", async (url) => {
    requested.push(String(url));
    return url in files ? new Response(files[url]) : new Response("not found", { status: 404 });
  });
  return requested;
}

test("the exported weights are the weights of the Python model", () => {
  const sha256 = createHash("sha256").update(new Uint8Array(weights)).digest("hex");
  assert.equal(sha256, fixtures().index.weights_sha256, "re-run training/export_web.py and make_fixtures.py");
});

test("the model carries the labels, the threshold and the weak labels", () => {
  assert.deepEqual(model.labels, fixtures().index.labels);
  assert.equal(model.labels.at(-1), "OTHER");
  assert.equal(model.threshold, fixtures().index.threshold);
  assert.deepEqual(model.weakLabels, fixtures().index.weak_labels);
});

test("predict gives one probability per label, and they sum to 1", () => {
  const { index, floats } = fixtures();
  const item = index.clips.find((entry) => entry.features_offset !== null);
  const probs = model.predict(floats(item.features_offset, 32 * 184));
  assert.ok(probs instanceof Float32Array);
  assert.equal(probs.length, model.labels.length);
  assert.ok(probs.every((p) => p >= 0 && p <= 1));
  assert.ok(Math.abs(probs.reduce((a, b) => a + b, 0) - 1) < 1e-5);
});

test("predict equals the Python model on the Python features of a dozen clips", () => {
  const { index, floats } = fixtures();
  const stored = index.clips.filter((item) => item.features_offset !== null);
  assert.equal(stored.length, 12);
  for (const item of stored) {
    const probs = model.predict(floats(item.features_offset, 32 * 184));
    const worst = maxDifference(probs, item.probs);
    assert.ok(worst <= TOLERANCE, `${item.label} ${item.id}: largest difference ${worst}`);
  }
});

test("features and model together equal Python on every raw test-split clip", () => {
  const { index, clip } = fixtures();
  assert.equal(index.clips.length, 56);
  for (const item of index.clips) {
    const probs = model.predict(clipFeatures(clip(item.offset, item.frames)));
    const worst = maxDifference(probs, item.probs);
    assert.ok(worst <= TOLERANCE, `${item.label} ${item.id}: largest difference ${worst}`);
  }
});

test("predict does not change the features it is given", () => {
  const { index, floats } = fixtures();
  const item = index.clips.find((entry) => entry.features_offset !== null);
  const features = Float32Array.from(floats(item.features_offset, 32 * 184));
  const before = Float32Array.from(features);
  model.predict(features);
  assert.deepEqual(features, before);
});

test("predict refuses anything but 32 x 184 features", () => {
  assert.throws(() => model.predict(new Float32Array(184)), RangeError);
  assert.throws(() => model.predict(new Float32Array(32 * 184 + 1)), RangeError);
});

test("fromBuffer also takes the bytes as a typed array, wherever it starts in its buffer", () => {
  const { index, floats } = fixtures();
  const item = index.clips.find((entry) => entry.features_offset !== null);
  const features = floats(item.features_offset, 32 * 184);
  // 8 can be read in place; 3 is not a multiple of four, so the values must be copied out.
  for (const start of [8, 3]) {
    const padded = new Uint8Array(weights.byteLength + start);
    padded.set(new Uint8Array(weights), start);
    const again = DenseModel.fromBuffer(manifest, padded.subarray(start));
    assert.deepEqual(again.predict(features), model.predict(features), `starting at byte ${start}`);
  }
});

test("fromBuffer refuses a .bin that is not the size the manifest gives", () => {
  assert.throws(() => DenseModel.fromBuffer(manifest, weights.slice(0, weights.byteLength - 4)), /health_signs\.bin/);
  assert.throws(() => DenseModel.fromBuffer(manifest, new ArrayBuffer(0)), new RegExp(String(manifest.byte_length)));
});

test("load fetches the manifest and the .bin beside it", async (t) => {
  const requested = stubFetch(t, {
    "models/health_signs.json": JSON.stringify(manifest),
    "models/health_signs.bin": weights,
  });
  const loaded = await DenseModel.load("models/health_signs.json");
  assert.deepEqual(requested, ["models/health_signs.json", "models/health_signs.bin"]);
  assert.deepEqual(loaded.labels, manifest.labels);
  assert.equal(loaded.threshold, manifest.threshold);

  const { index, floats } = fixtures();
  const item = index.clips.find((entry) => entry.features_offset !== null);
  const features = floats(item.features_offset, 32 * 184);
  assert.deepEqual(loaded.predict(features), model.predict(features));
});

test("load finds the .bin beside the manifest wherever the manifest is", async (t) => {
  const cases = {
    "health_signs.json": "health_signs.bin",
    "./models/health_signs.json": "./models/health_signs.bin",
    "/app/models/health_signs.json?v=3": "/app/models/health_signs.bin",
    "models/health_signs.json?from=/some/where#/else": "models/health_signs.bin",
    "https://signbridge.test/web/models/health_signs.json": "https://signbridge.test/web/models/health_signs.bin",
  };
  for (const [manifestUrl, weightsUrl] of Object.entries(cases)) {
    const requested = stubFetch(t, { [manifestUrl]: JSON.stringify(manifest), [weightsUrl]: weights });
    await DenseModel.load(manifestUrl);
    assert.deepEqual(requested, [manifestUrl, weightsUrl]);
    globalThis.fetch.mock.restore();
  }
});

test("load says which file could not be fetched", async (t) => {
  stubFetch(t, {});
  await assert.rejects(DenseModel.load("models/health_signs.json"), /models\/health_signs\.json.*404/);
  globalThis.fetch.mock.restore();
  stubFetch(t, { "models/health_signs.json": JSON.stringify(manifest) });
  await assert.rejects(DenseModel.load("models/health_signs.json"), /models\/health_signs\.bin.*404/);
});
