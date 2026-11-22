import assert from "node:assert/strict";
import test from "node:test";

import { SignCapture } from "../lib/capture.js";
import { fixtures, syntheticFrame } from "./helpers.js";

/** Push one frame per pattern entry (1: a hand is visible), as tests/test_capture.py does. */
function feed(capture, pattern, dt = 1 / 30) {
  const clips = [];
  let t = 0;
  pattern.forEach((hasHand, index) => {
    const frame = syntheticFrame({ left: Boolean(hasHand) });
    frame[2] = index; // depth of the nose: tells us afterwards which frames came back
    const clip = capture.push(frame, t);
    t += dt;
    if (clip !== null) clips.push({ returnedAt: index, clip });
  });
  return clips;
}

const repeat = (value, count) => Array(count).fill(value);

test("the limits are the Python ones", () => {
  assert.deepEqual(
    [SignCapture.START_FRAMES, SignCapture.END_GAP_FRAMES, SignCapture.MAX_SECONDS, SignCapture.MIN_FRAMES],
    [3, 8, 4, 8],
  );
});

test("SignCapture cuts every pattern exactly where the Python SignCapture cuts it", () => {
  const { index } = fixtures();
  assert.ok(index.capture_cases.length >= 10);
  for (const item of index.capture_cases) {
    const capture = new SignCapture();
    const clips = feed(capture, item.pattern, item.dt);
    const summary = clips.map(({ returnedAt, clip }) => ({
      returned_at: returnedAt, first: clip[0][2], frames: clip.length,
    }));
    assert.deepEqual(summary, item.clips, item.name);
    assert.equal(capture.recording, item.recording, `${item.name}: recording`);
    for (const { clip } of clips) {
      clip.forEach((frame, i) => assert.equal(frame[2], clip[0][2] + i, `${item.name}: frames in order`));
    }
  }
});

// The five tests of tests/test_capture.py.

test("returns one clip without the trailing gap", () => {
  const clips = feed(new SignCapture(), [...repeat(0, 5), ...repeat(1, 20), ...repeat(0, 10)]);
  assert.equal(clips.length, 1);
  assert.equal(clips[0].clip.length, 20);
});

test("no hand never returns a clip", () => {
  const capture = new SignCapture();
  assert.deepEqual(feed(capture, repeat(0, 300)), []);
  assert.equal(capture.recording, false);
});

test("a short blip is discarded", () => {
  assert.deepEqual(feed(new SignCapture(), [...repeat(1, 5), ...repeat(0, 10)]), []);
});

test("a long sign is cut at four seconds", () => {
  const clips = feed(new SignCapture(), repeat(1, 200));
  assert.ok(clips.length >= 1);
  assert.ok(clips[0].clip.length >= 110 && clips[0].clip.length <= 122);
});

test("the recording flag follows the hand", () => {
  const capture = new SignCapture();
  assert.equal(capture.recording, false);
  feed(capture, repeat(1, 3));
  assert.equal(capture.recording, true);
  feed(capture, repeat(0, 8));
  assert.equal(capture.recording, false);
});

// What the browser relies on beyond the Python tests.

test("a finished clip is an array of Float32Array frames, returned once", () => {
  const capture = new SignCapture();
  const [{ clip }] = feed(capture, [...repeat(1, 12), ...repeat(0, 8)]);
  assert.ok(Array.isArray(clip));
  assert.ok(clip.every((frame) => frame instanceof Float32Array && frame.length === 225));
  assert.equal(capture.push(syntheticFrame({ left: false }), 10), null);
});

test("the clip keeps its own copy of each frame, so the caller may reuse a buffer", () => {
  const capture = new SignCapture();
  const buffer = new Float32Array(225);
  let clip = null;
  for (let i = 0; i < 20 && clip === null; i++) {
    buffer.set(syntheticFrame({ left: i < 10 }));
    buffer[2] = i;
    clip = capture.push(buffer, i / 30);
  }
  assert.deepEqual(clip.map((frame) => frame[2]), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
});

test("a subclass can change the limits, for a camera with another frame rate", () => {
  class PatientCapture extends SignCapture {
    static END_GAP_FRAMES = 16;
  }
  const pattern = [...repeat(1, 10), ...repeat(0, 12), ...repeat(1, 10), ...repeat(0, 16)];
  assert.deepEqual(feed(new SignCapture(), pattern).map(({ clip }) => clip.length), [10, 10]);
  assert.deepEqual(feed(new PatientCapture(), pattern).map(({ clip }) => clip.length), [32]);
});

test("reset drops a sign in progress", () => {
  const capture = new SignCapture();
  feed(capture, repeat(1, 10));
  assert.equal(capture.recording, true);
  capture.reset();
  assert.equal(capture.recording, false);
  assert.deepEqual(feed(capture, repeat(0, 10)), []);
});
