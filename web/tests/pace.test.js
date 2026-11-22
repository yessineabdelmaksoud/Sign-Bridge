import assert from "node:assert/strict";
import test from "node:test";

import { SignCapture } from "../lib/capture.js";
import { pacedCapture } from "../lib/pace.js";
import { syntheticFrame } from "./helpers.js";

const repeat = (value, count) => Array(count).fill(value);

function feed(capture, pattern, fps) {
  const clips = [];
  pattern.forEach((hasHand, index) => {
    const clip = capture.push(syntheticFrame({ left: Boolean(hasHand) }), index / fps);
    if (clip !== null) clips.push(clip);
  });
  return clips;
}

test("near ten frames a second it behaves like the reference capture", () => {
  const pattern = [...repeat(0, 5), ...repeat(1, 20), ...repeat(0, 10)];
  const paced = feed(pacedCapture(), pattern, 10);
  const reference = feed(new SignCapture(), pattern, 10);
  assert.equal(paced.length, 1);
  assert.equal(paced[0].length, reference[0].length);
});

test("at thirty frames a second a short tracking dropout does not cut the sign in two", () => {
  const pattern = [...repeat(1, 30), ...repeat(0, 9), ...repeat(1, 30), ...repeat(0, 30)];
  assert.equal(feed(new SignCapture(), pattern, 30).length, 2, "the frame-counting reference splits it");
  const clips = feed(pacedCapture(), pattern, 30);
  assert.equal(clips.length, 1);
  assert.equal(clips[0].length, 69);
});

test("at thirty frames a second a fifth of a second of hand is still a blip", () => {
  assert.deepEqual(feed(pacedCapture(), [...repeat(1, 6), ...repeat(0, 40)], 30), []);
});

test("the measured rate follows the camera", () => {
  const capture = pacedCapture();
  feed(capture, repeat(0, 60), 30);
  assert.ok(Math.abs(capture.rate - 30) < 2, `rate ${capture.rate}`);
});

test("finish closes a sign still being recorded when the video ends", () => {
  const capture = pacedCapture();
  assert.deepEqual(feed(capture, repeat(1, 28), 20), [], "nothing is returned while the hands are still up");
  const clip = capture.finish(28 / 20);
  assert.equal(clip.length, 28);
  assert.equal(capture.recording, false);
  assert.equal(capture.finish(2), null, "a second call has nothing left to return");
});

test("finish drops a blip, like push does", () => {
  const capture = pacedCapture();
  feed(capture, repeat(1, 5), 20);
  assert.equal(capture.finish(0.25), null);
});
