import assert from "node:assert/strict";
import test from "node:test";

import {
  FRAMES, NoHandError, WIDTH, clipFeatures, frameFeatures, resample, trimToHands,
} from "../lib/features.js";
import { LEFT_START, RIGHT_START } from "../lib/landmarks.js";
import { assertIdentical, fixtures, maxDifference, syntheticClip, syntheticFrame } from "./helpers.js";

// The port rounds to float32 wherever NumPy does, so its features are not just
// close to Python's: every value is the same number, and the tests ask for
// that. Should fixtures made on another machine differ in a last digit, that
// NumPy adds up the shoulder width differently (np.linalg.norm of a vector
// goes through its linear algebra library); compare with a margin there.

test("a clip becomes 32 frames of 184 features", () => {
  assert.deepEqual([FRAMES, WIDTH], [32, 184]);
});

// --- frameFeatures ----------------------------------------------------------------

test("frameFeatures equals Python frame_features on every fixture frame", () => {
  const { index, floats } = fixtures();
  const names = index.frame_cases.map((item) => item.name);
  assert.ok(names.includes("left hand only") && names.includes("both hands"));
  for (const item of index.frame_cases) {
    const actual = frameFeatures(floats(item.frame_offset, 225));
    assert.ok(actual instanceof Float32Array, item.name);
    assertIdentical(actual, floats(item.features_offset, WIDTH), item.name);
  }
});

test("a missing hand is all zeros with its presence flag at 0", () => {
  const x = frameFeatures(syntheticFrame({ left: true, right: false }));
  assert.equal(x.length, 184);
  assert.equal(x[182], 1);
  assert.equal(x[183], 0);
  assert.ok(x.subarray(56, 98).every((v) => v === 0), "right hand, body frame");
  assert.ok(x.subarray(140, 182).every((v) => v === 0), "right hand, local frame");
  assert.ok(x.subarray(14, 56).some((v) => v !== 0), "left hand, body frame");
  assert.ok(x.subarray(98, 140).some((v) => v !== 0), "left hand, local frame");
});

test("both hands present set both flags and fill all four hand blocks", () => {
  const x = frameFeatures(syntheticFrame({ left: true, right: true }));
  assert.deepEqual([x[182], x[183]], [1, 1]);
  for (const [from, to] of [[14, 56], [56, 98], [98, 140], [140, 182]]) {
    assert.ok(x.subarray(from, to).some((v) => v !== 0), `block ${from}..${to}`);
  }
  assert.ok(x.every(Number.isFinite));
});

test("features do not change when the signer moves or stands closer", () => {
  const frame = syntheticFrame({ left: true, right: true });
  const moved = Float32Array.from(frame);
  for (let point = 0; point < 75; point++) {
    moved[point * 3] = frame[point * 3] * 0.5 + 0.1;
    moved[point * 3 + 1] = frame[point * 3 + 1] * 0.5 + 0.1;
  }
  assert.ok(maxDifference(frameFeatures(moved), frameFeatures(frame)) < 1e-4);
});

test("frameFeatures does not change the frame it is given", () => {
  const frame = syntheticFrame({ left: true, right: false });
  const before = Float32Array.from(frame);
  frameFeatures(frame);
  assert.deepEqual(frame, before);
});

// --- trimToHands ------------------------------------------------------------------

test("trimToHands drops the rest frames at both ends and rejects a clip with no hand", () => {
  const clip = [...syntheticClip(4, { left: false }), ...syntheticClip(10), ...syntheticClip(6, { left: false })];
  const trimmed = trimToHands(clip);
  assert.equal(trimmed.length, 10);
  assert.equal(trimmed[0], clip[4]);
  assert.equal(trimmed[9], clip[13]);
  assert.throws(() => trimToHands(syntheticClip(5, { left: false })), NoHandError);
  assert.throws(() => trimToHands([]), NoHandError);
});

test("trimToHands keeps frames without a hand that lie inside the sign", () => {
  const clip = [...syntheticClip(3), ...syntheticClip(2, { left: false }), ...syntheticClip(3)];
  assert.equal(trimToHands(clip).length, 8);
});

test("NoHandError is an Error that can be told apart", () => {
  const error = new NoHandError("no hand");
  assert.ok(error instanceof Error && error instanceof NoHandError);
  assert.equal(error.name, "NoHandError");
});

test("trimToHands cuts every real clip where Python trim_to_hands cuts it", () => {
  const { index, clip } = fixtures();
  let trimmedSomething = 0;
  for (const item of index.clips) {
    const raw = clip(item.offset, item.frames);
    const trimmed = trimToHands(raw);
    const [first, stop] = item.trim;
    assert.equal(trimmed.length, stop - first, item.id);
    assert.equal(trimmed[0], raw[first], item.id);
    assert.equal(trimmed.at(-1), raw[stop - 1], item.id);
    if (trimmed.length < raw.length) trimmedSomething++;
  }
  assert.ok(trimmedSomething > 10, "the fixtures include clips with rest frames to remove");
});

// --- resample ---------------------------------------------------------------------

test("resample picks the frames NumPy picks, for clips shorter and longer than 32", () => {
  const { index } = fixtures();
  assert.ok(index.resample_cases.length > 80);
  for (const item of index.resample_cases) {
    const clip = Array.from({ length: item.length }, (_, i) => Float32Array.of(i));
    const picked = resample(clip, item.frames).map((frame) => frame[0]);
    assert.deepEqual(picked, item.index, `length ${item.length} to ${item.frames} frames`);
  }
});

test("resample gives 32 frames when the count is left out", () => {
  const clip = Array.from({ length: 50 }, (_, i) => Float32Array.of(i));
  const out = resample(clip);
  assert.equal(out.length, 32);
  assert.equal(out[0], clip[0]);
  assert.equal(out[31], clip[49]);
});

test("resample picks from every real clip the frames Python picks", () => {
  const { index, clip } = fixtures();
  for (const item of index.clips) {
    const raw = clip(item.offset, item.frames);
    const span = raw.slice(item.trim[0], item.trim[1]);
    assert.deepEqual(resample(span).map((frame) => span.indexOf(frame)), item.resample, item.id);
  }
});

// --- clipFeatures -----------------------------------------------------------------

test("clipFeatures equals Python clip_features on a dozen real test-split clips", () => {
  const { index, floats, clip } = fixtures();
  const stored = index.clips.filter((item) => item.features_offset !== null);
  assert.equal(stored.length, 12);
  const kinds = new Set(stored.map((item) => item.hands));
  for (const kind of ["no hand in some frames", "one hand comes and goes", "left hand only", "right hand only",
    "both hands throughout"]) {
    assert.ok(kinds.has(kind), `a clip with: ${kind}`);
  }
  for (const item of stored) {
    const actual = clipFeatures(clip(item.offset, item.frames));
    assert.ok(actual instanceof Float32Array, item.id);
    assertIdentical(actual, floats(item.features_offset, FRAMES * WIDTH), `${item.label} ${item.id}`);
  }
});

test("clipFeatures gives the same result for a clip that is already trimmed and resampled", () => {
  const { index, floats, clip } = fixtures();
  const item = index.clips.find((entry) => entry.features_offset !== null);
  const prepared = resample(trimToHands(clip(item.offset, item.frames)));
  assertIdentical(clipFeatures(prepared), floats(item.features_offset, FRAMES * WIDTH));
});

test("clipFeatures has no NaN when a hand drops out for a few frames", () => {
  const clip = syntheticClip(20, { left: true });
  for (let i = 5; i < 9; i++) clip[i].fill(NaN, LEFT_START * 3, RIGHT_START * 3);
  const x = clipFeatures(clip);
  assert.equal(x.length, 32 * 184);
  assert.ok(x.every(Number.isFinite));
});

test("clipFeatures is laid out frame by frame", () => {
  const clip = syntheticClip(32, { left: true, right: true });
  const x = clipFeatures(clip);
  for (const frame of [0, 13, 31]) {
    assert.deepEqual(x.subarray(frame * WIDTH, (frame + 1) * WIDTH), frameFeatures(clip[frame]), `frame ${frame}`);
  }
});

test("clipFeatures throws NoHandError when no frame shows a hand", () => {
  assert.throws(() => clipFeatures(syntheticClip(20, { left: false })), NoHandError);
  assert.throws(() => clipFeatures([]), NoHandError);
});
