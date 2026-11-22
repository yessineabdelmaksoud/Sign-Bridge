import assert from "node:assert/strict";
import test from "node:test";

import { LEFT_START, POINTS, RIGHT_START, emptyFrame, fromHolistic, handPresent } from "../lib/landmarks.js";
import { fixtures, syntheticFrame } from "./helpers.js";

const points = (count, base) =>
  Array.from({ length: count }, (_, i) => ({ x: base + i, y: base + i + 0.25, z: base + i + 0.5 }));

test("the layout is 75 points: pose, then left hand, then right hand", () => {
  assert.deepEqual([POINTS, LEFT_START, RIGHT_START], [75, 33, 54]);
});

test("an empty frame is 225 NaN in a Float32Array", () => {
  const frame = emptyFrame();
  assert.ok(frame instanceof Float32Array);
  assert.equal(frame.length, 225);
  assert.ok(frame.every(Number.isNaN));
  assert.notEqual(emptyFrame(), frame, "each call gives a new array");
});

test("handPresent is true when either hand has any finite value", () => {
  assert.equal(handPresent(syntheticFrame({ left: true, right: false })), true);
  assert.equal(handPresent(syntheticFrame({ left: false, right: true })), true);
  assert.equal(handPresent(syntheticFrame({ left: false, right: false })), false);
  assert.equal(handPresent(emptyFrame()), false);

  const lastValue = emptyFrame();
  lastValue[224] = 0.5;
  assert.equal(handPresent(lastValue), true, "the last coordinate of the right hand counts");
  const poseOnly = emptyFrame();
  poseOnly.fill(0.5, 0, LEFT_START * 3);
  assert.equal(handPresent(poseOnly), false, "the body alone is not a hand");
  const infinite = emptyFrame();
  infinite[LEFT_START * 3] = Infinity;
  assert.equal(handPresent(infinite), false, "infinity is not a finite value");
});

test("handPresent agrees with Python on every fixture frame", () => {
  const { index, floats } = fixtures();
  for (const item of index.frame_cases) {
    assert.equal(handPresent(floats(item.frame_offset, 225)), item.hand_present, item.name);
  }
});

test("fromHolistic puts the body and each hand in its own block", () => {
  const frame = fromHolistic({ pose: points(33, 0), left: points(21, 100), right: points(21, 200) });
  assert.ok(frame instanceof Float32Array);
  assert.equal(frame.length, 225);
  assert.deepEqual([...frame.subarray(0, 3)], [0, 0.25, 0.5]);
  assert.deepEqual([...frame.subarray(32 * 3, 33 * 3)], [32, 32.25, 32.5]);
  assert.deepEqual([...frame.subarray(LEFT_START * 3, LEFT_START * 3 + 3)], [100, 100.25, 100.5]);
  assert.deepEqual([...frame.subarray(53 * 3, 54 * 3)], [120, 120.25, 120.5]);
  assert.deepEqual([...frame.subarray(RIGHT_START * 3, RIGHT_START * 3 + 3)], [200, 200.25, 200.5]);
  assert.deepEqual([...frame.subarray(74 * 3)], [220, 220.25, 220.5]);
});

test("fromHolistic leaves a part that was not detected as NaN", () => {
  const frame = fromHolistic({ pose: points(33, 0), left: null, right: points(21, 200) });
  assert.ok(frame.subarray(LEFT_START * 3, RIGHT_START * 3).every(Number.isNaN));
  assert.ok(frame.subarray(RIGHT_START * 3).every(Number.isFinite));
  assert.equal(handPresent(frame), true);

  const nothing = fromHolistic({ pose: null, left: null, right: null });
  assert.ok(nothing.every(Number.isNaN));
  assert.ok(fromHolistic({}).every(Number.isNaN), "a missing key is the same as null");
});

test("fromHolistic ignores points beyond the block and accepts a short list", () => {
  const long = fromHolistic({ pose: points(40, 0), left: null, right: null });
  assert.ok(long.subarray(LEFT_START * 3).every(Number.isNaN), "extra pose points do not spill into the hands");
  const short = fromHolistic({ pose: null, left: points(5, 1), right: null });
  assert.deepEqual([...short.subarray((LEFT_START + 4) * 3, (LEFT_START + 5) * 3)], [5, 5.25, 5.5]);
  assert.ok(short.subarray((LEFT_START + 5) * 3, RIGHT_START * 3).every(Number.isNaN));
});
