// A sign capture whose limits follow the camera's frame rate.
//
// SignCapture counts frames, and its counts were tuned near ten frames a
// second (the tracker on a laptop CPU). On a device that tracks at thirty,
// "8 frames without a hand" is a quarter of a second, and a brief tracking
// dropout would cut one sign into two. Here the same limits are kept in
// seconds, and never go below the reference counts.
import { SignCapture } from "./capture.js";
import { emptyFrame } from "./landmarks.js";

const END_GAP_SECONDS = 0.7;   // hands gone this long: the sign is over
const MIN_SECONDS = 0.4;       // shorter than this is a blip, not a sign
const SMOOTHING = 0.2;         // how fast the measured rate follows the camera

/** A new capture that measures the frame rate from the times it is given. */
export function pacedCapture() {
  let rate = 10;
  let last = null;

  class PacedCapture extends SignCapture {
    static get END_GAP_FRAMES() {
      return Math.max(SignCapture.END_GAP_FRAMES, Math.round(rate * END_GAP_SECONDS));
    }

    static get MIN_FRAMES() {
      return Math.max(SignCapture.MIN_FRAMES, Math.round(rate * MIN_SECONDS));
    }

    /** Frames per second, as measured so far. */
    get rate() {
      return rate;
    }

    push(frame, tSeconds) {
      if (last !== null) {
        const gap = tSeconds - last;
        if (gap > 0.005 && gap < 1) rate += SMOOTHING * (1 / gap - rate);
      }
      last = tSeconds;
      return super.push(frame, tSeconds);
    }

    /**
     * The source has ended (a video ran out): close the sign in progress and
     * return it, or null if nothing long enough was being recorded.
     */
    finish(tSeconds) {
      let clip = null;
      for (let i = 1; this.recording && i <= 1000; i++) {
        clip = super.push(emptyFrame(), tSeconds + i / rate);
      }
      return clip;
    }

    reset() {
      last = null;   // the next source has its own clock; the measured rate is kept
      super.reset();
    }
  }

  return new PacedCapture();
}

/** Shoulder widths below the shoulders; 99.8% of signing hand sightings in training are above it. */
export const REST_LINE = 1.5;

/**
 * A copy of the frame without the hands that hang below the signing space.
 *
 * In the training recordings a hand is seen only when raised: hands at rest are out
 * of the picture. At a desk, resting hands stay visible; left in, they keep the
 * capture recording and show the model positions it never learned (measured on
 * held-out recordings: 9% right instead of 77%). Same rule as
 * drop_resting_hands in training/tools.py.
 */
export function dropRestingHands(frame) {
  const out = Float32Array.from(frame);
  const [lx, ly, rx, ry] = [frame[33], frame[34], frame[36], frame[37]];   // pose points 11 and 12
  if (![lx, ly, rx, ry].every(Number.isFinite)) return out;
  const middleY = (ly + ry) / 2;
  const width = Math.hypot(lx - rx, ly - ry);
  if (width < 1e-6) return out;
  for (const start of [33, 54]) {
    const wristY = frame[start * 3 + 1];
    if (Number.isFinite(wristY) && (wristY - middleY) / width > REST_LINE) out.fill(NaN, start * 3, (start + 21) * 3);
  }
  return out;
}
