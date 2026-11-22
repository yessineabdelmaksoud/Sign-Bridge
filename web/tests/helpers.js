// Shared by the Node tests: reading files under web/, the Python fixtures, and
// comparing arrays of numbers.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const WEB = new URL("../", import.meta.url);

export function readJson(path) {
  return JSON.parse(readFileSync(new URL(path, WEB), "utf8"));
}

/** A file under web/ as an ArrayBuffer of exactly its own bytes. */
export function readBuffer(path) {
  const bytes = readFileSync(new URL(path, WEB));
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

let cached = null;

/**
 * The fixtures written by make_fixtures.py: `index` is fixtures/parity.json,
 * and the big arrays live in fixtures/parity.bin as little-endian float32.
 * Offsets and lengths in the index count float32 values, not bytes.
 */
export function fixtures() {
  if (cached) return cached;
  const index = readJson("tests/fixtures/parity.json");
  const all = new Float32Array(readBuffer("tests/fixtures/parity.bin"));
  const floats = (offset, length) => all.subarray(offset, offset + length);
  const clip = (offset, frames) =>
    Array.from({ length: frames }, (_, i) => floats(offset + i * 225, 225));
  cached = { index, floats, clip };
  return cached;
}

/** Largest absolute difference between two equally long arrays. */
export function maxDifference(actual, expected) {
  assert.equal(actual.length, expected.length, "length");
  let worst = 0;
  for (let i = 0; i < expected.length; i++) {
    const difference = Math.abs(actual[i] - expected[i]);
    if (!(difference <= worst)) worst = difference; // also catches NaN
  }
  return worst;
}

/** Every value is the very same number, not merely a close one. */
export function assertIdentical(actual, expected, message = "values") {
  assert.equal(actual.length, expected.length, `${message}: length`);
  for (let i = 0; i < expected.length; i++) {
    if (actual[i] !== expected[i]) {
      const difference = actual[i] - expected[i];
      assert.fail(`${message}: index ${i} is ${actual[i]}, Python gives ${expected[i]} (difference ${difference})`);
    }
  }
}

function mulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * A made-up frame like `frame()` in tests/test_features.py: every point
 * random in 0.2..0.8, the shoulders 0.2 apart, and each hand there or not.
 */
export function syntheticFrame({ left = true, right = false, seed = 0 } = {}) {
  const random = mulberry32(seed + 1);
  const frame = new Float32Array(225);
  for (let i = 0; i < 225; i++) frame[i] = 0.2 + 0.6 * random();
  frame.set([0.6, 0.4], 11 * 3);
  frame.set([0.4, 0.4], 12 * 3);
  if (!left) frame.fill(NaN, 33 * 3, 54 * 3);
  if (!right) frame.fill(NaN, 54 * 3, 75 * 3);
  return frame;
}

export function syntheticClip(frames = 20, options = {}) {
  return Array.from({ length: frames }, (_, seed) => syntheticFrame({ ...options, seed }));
}
