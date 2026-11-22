/**
 * Turn canonical landmark clips into fixed-size model features.
 * Port of Demo/consultation/features.py.
 *
 * Per frame (184 values, in this order):
 *   pose             14  points 0, 11-16 (x, y) relative to the shoulder
 *                        midpoint, in shoulder widths
 *   left hand        42  21 points (x, y), same origin and scale
 *   right hand       42
 *   left hand local  42  relative to its own wrist, in hand sizes
 *   right hand local 42
 *   presence          2  left present, right present
 * A missing hand is all zeros with presence 0, so features never contain NaN.
 *
 * The model was trained on what NumPy computes in float32, so every step here
 * is rounded to float32 (Math.fround) exactly where NumPy rounds it.
 */
import { LEFT_START, RIGHT_START, handPresent } from "./landmarks.js";

export const FRAMES = 32;
export const WIDTH = 184;

const POSE_POINTS = [0, 11, 12, 13, 14, 15, 16];
const HAND_POINTS = 21;
const FLOOR = 1e-3;
const FLOAT32_MAX = 3.4028234663852886e38;
const f32 = Math.fround;

// Where each block starts inside the 184 values.
const LEFT_BODY = 14;
const RIGHT_BODY = 56;
const LEFT_LOCAL = 98;
const RIGHT_LOCAL = 140;
const PRESENCE = 182;

/** Thrown when a clip has no frame with a visible hand. */
export class NoHandError extends Error {
  constructor(message = "no hand visible in any frame") {
    super(message);
    this.name = "NoHandError";
  }
}

/** Keep the span from the first to the last frame that shows a hand. */
export function trimToHands(clip) {
  let first = 0;
  while (first < clip.length && !handPresent(clip[first])) first++;
  if (first === clip.length) throw new NoHandError();
  let last = clip.length - 1;
  while (!handPresent(clip[last])) last--;
  return clip.slice(first, last + 1);
}

/** Round to the nearest integer, a tie going to the even one, as NumPy does. */
function roundHalfEven(value) {
  const below = Math.floor(value);
  const rest = value - below;
  if (rest !== 0.5) return rest < 0.5 ? below : below + 1;
  return below % 2 === 0 ? below : below + 1;
}

/** Pick `frames` frames spread evenly over the clip (nearest index). */
export function resample(clip, frames = FRAMES) {
  if (clip.length === 0) throw new RangeError("cannot resample an empty clip");
  const last = clip.length - 1;
  const step = frames > 1 ? last / (frames - 1) : 0;
  const out = new Array(frames);
  for (let i = 0; i < frames; i++) {
    // np.linspace sets its final value to the end point instead of computing it.
    const position = frames > 1 && i === frames - 1 ? last : i * step;
    out[i] = clip[roundHalfEven(position)];
  }
  return out;
}

/** The length of (dx, dy) as NumPy computes it for float32 input. */
function norm(dx, dy) {
  return f32(Math.sqrt(f32(f32(dx * dx) + f32(dy * dy))));
}

/** NaN and both infinities become 0. */
function finiteOrZero(value) {
  return Number.isFinite(value) ? value : 0;
}

/**
 * Write one hand's two blocks into `out` and return its presence flag.
 * A hand counts only if the x and y of all 21 points are finite.
 */
function writeHand(xyz, start, originX, originY, scale, out, at, bodyBlock, localBlock) {
  const base = start * 3;
  for (let i = 0; i < HAND_POINTS; i++) {
    if (!Number.isFinite(xyz[base + i * 3]) || !Number.isFinite(xyz[base + i * 3 + 1])) return 0;
  }
  const wristX = xyz[base];
  const wristY = xyz[base + 1];
  let size = 0;
  for (let i = 0; i < HAND_POINTS; i++) {
    const length = norm(f32(xyz[base + i * 3] - wristX), f32(xyz[base + i * 3 + 1] - wristY));
    if (length > size) size = length;
  }
  size = f32(FLOOR > size ? FLOOR : size);
  for (let i = 0; i < HAND_POINTS; i++) {
    const x = xyz[base + i * 3];
    const y = xyz[base + i * 3 + 1];
    out[at + bodyBlock + i * 2] = finiteOrZero(f32(f32(x - originX) / scale));
    out[at + bodyBlock + i * 2 + 1] = finiteOrZero(f32(f32(y - originY) / scale));
    out[at + localBlock + i * 2] = finiteOrZero(f32(f32(x - wristX) / size));
    out[at + localBlock + i * 2 + 1] = finiteOrZero(f32(f32(y - wristY) / size));
  }
  return 1;
}

/** Write the 184 features of one frame into `out`, which must be zero there. */
function writeFrame(frame, out, at) {
  const xyz = frame instanceof Float32Array ? frame : Float32Array.from(frame);
  const leftX = xyz[11 * 3], leftY = xyz[11 * 3 + 1];
  const rightX = xyz[12 * 3], rightY = xyz[12 * 3 + 1];
  const originX = f32(f32(leftX + rightX) / 2);
  const originY = f32(f32(leftY + rightY) / 2);

  // Shoulder width, with np.nan_to_num applied: no shoulders means the floor.
  let width = norm(f32(leftX - rightX), f32(leftY - rightY));
  if (Number.isNaN(width)) width = 0;
  else if (width === Infinity) width = FLOAT32_MAX;
  const scale = f32(FLOOR > width ? FLOOR : width);

  POSE_POINTS.forEach((point, i) => {
    out[at + i * 2] = finiteOrZero(f32(f32(xyz[point * 3] - originX) / scale));
    out[at + i * 2 + 1] = finiteOrZero(f32(f32(xyz[point * 3 + 1] - originY) / scale));
  });
  out[at + PRESENCE] = writeHand(xyz, LEFT_START, originX, originY, scale, out, at, LEFT_BODY, LEFT_LOCAL);
  out[at + PRESENCE + 1] = writeHand(xyz, RIGHT_START, originX, originY, scale, out, at, RIGHT_BODY, RIGHT_LOCAL);
}

/** The 184 features of one canonical frame. */
export function frameFeatures(frame) {
  const out = new Float32Array(WIDTH);
  writeFrame(frame, out, 0);
  return out;
}

/**
 * Trim to the signed span, resample to 32 frames, and featurise each frame.
 * Returns Float32Array(32 * 184), row-major [frame][feature].
 */
export function clipFeatures(clip) {
  const frames = resample(trimToHands(clip));
  const out = new Float32Array(FRAMES * WIDTH);
  frames.forEach((frame, i) => writeFrame(frame, out, i * WIDTH));
  return out;
}
