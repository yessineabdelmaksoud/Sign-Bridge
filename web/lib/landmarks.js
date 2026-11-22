/**
 * The canonical landmark frame shared by the tracker, the capture and the
 * features. Port of Demo/consultation/landmarks.py.
 *
 * A frame is a Float32Array(225): 75 points × (x, y, z), row-major. Points
 * 0–32 are MediaPipe pose landmarks, 33–53 the left hand, 54–74 the right
 * hand. x and y are image-normalised (0..1); a missing hand is all NaN.
 */

export const POINTS = 75;
export const LEFT_START = 33;
export const RIGHT_START = 54;

export function emptyFrame() {
  return new Float32Array(POINTS * 3).fill(NaN);
}

/** True if either hand has at least one finite coordinate. */
export function handPresent(frame) {
  for (let i = LEFT_START * 3; i < POINTS * 3; i++) {
    if (Number.isFinite(frame[i])) return true;
  }
  return false;
}

function fill(frame, start, stop, points) {
  if (!points) return;
  const count = Math.min(points.length, stop - start);
  for (let i = 0; i < count; i++) {
    const at = (start + i) * 3;
    frame[at] = points[i].x;
    frame[at + 1] = points[i].y;
    frame[at + 2] = points[i].z;
  }
}

/**
 * One MediaPipe Holistic result as a canonical frame. `pose` is 33 points
 * {x, y, z} or null; `left` and `right` are 21 points each or null.
 */
export function fromHolistic({ pose = null, left = null, right = null } = {}) {
  const frame = emptyFrame();
  fill(frame, 0, LEFT_START, pose);
  fill(frame, LEFT_START, RIGHT_START, left);
  fill(frame, RIGHT_START, POINTS, right);
  return frame;
}
