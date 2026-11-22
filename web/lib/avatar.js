// Signs drawn as a stick figure: lines for the body, arms and fingers.
//
// The patient never sees the dataset's videos of real people. The movement of
// each reference sign was extracted once (training/export_poses.py) into
// data/sign_poses.json: per frame, 7 body points and 21 points per hand, in
// shoulder widths from the shoulder midpoint, y pointing down, as seen facing
// the signer. This file only draws them.

// Two colours: the still body is light, the parts that sign (arms and hands) are dark,
// so the eye goes to the sign.
const BODY = "#C5D0DC";
const SIGN = "#10233A";

// Order of the 7 body points in the data: nose, then left/right shoulder,
// elbow, wrist (MediaPipe pose 0, 11, 12, 13, 14, 15, 16).
const NOSE = 0, L_SHOULDER = 1, R_SHOULDER = 2, L_ELBOW = 3, R_ELBOW = 4, L_WRIST = 5, R_WRIST = 6;

const HAND_BONES = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];

const HEAD_RADIUS = 0.34;
const HIP_Y = 1.75;
const HIP_HALF = 0.36;
const GAP_SECONDS = 0.3;       // between two signs: the body glides from one to the next
const END_HOLD_SECONDS = 0.5;  // the last pose stays briefly before returning to rest

// Hands are drawn larger than life around the wrist, so the handshape reads on a small screen.
const HAND_SCALE = 1.6;

/** The area, in body units, that is always visible (measured over all 22 signs, with a margin). */
export const BOX = { left: -1.8, right: 1.8, top: -1.25, bottom: 2.05 };

/** Standing still, arms down. */
export const REST_FRAME = {
  p: [0, -0.85, 0.5, 0, -0.5, 0, 0.62, 0.85, -0.62, 0.85, 0.66, 1.65, -0.66, 1.65],
  l: null,
  r: null,
};

/** Index the raw JSON: { fps, signs: Map(code -> { frames }) }. */
export function poseFromJson(json) {
  return { fps: json.fps || 20, signs: new Map(Object.entries(json.signs || {})) };
}

export async function loadSignPoses(url = new URL("../data/sign_poses.json", import.meta.url)) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`sign poses: HTTP ${response.status}`);
  return poseFromJson(await response.json());
}

/** Map body units to canvas pixels so that BOX fits, centred. */
export function layout(width, height, box = BOX) {
  const scale = Math.min(width / (box.right - box.left), height / (box.bottom - box.top));
  const ox = width / 2 - ((box.left + box.right) / 2) * scale;
  const oy = height / 2 - ((box.top + box.bottom) / 2) * scale;
  return { scale, toPx: ([x, y]) => [ox + x * scale, oy + y * scale] };
}

/** The frame of `sign` shown at `t` seconds. */
export function frameAt(sign, t, fps) {
  const index = Math.min(sign.frames.length - 1, Math.max(0, Math.floor(t * fps + 1e-9)));
  return sign.frames[index];
}

function point(list, i) {
  return [list[2 * i], list[2 * i + 1]];
}

function line(ctx, lay, a, b) {
  const [x0, y0] = lay.toPx(a);
  const [x1, y1] = lay.toPx(b);
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
}

/** Draw one frame. `frame` is { p: [14 numbers], l: [42] | null, r: [42] | null }. */
export function drawFigure(ctx, frame, lay) {
  const p = frame.p;
  const ls = point(p, L_SHOULDER), rs = point(p, R_SHOULDER);
  const neck = [(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2];
  const nose = point(p, NOSE);
  const width = lay.scale;

  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  // The still body, light: head, neck, shoulders, torso outline.
  ctx.strokeStyle = BODY;
  ctx.lineWidth = Math.max(3, width * 0.09);
  ctx.beginPath();
  const [hx, hy] = lay.toPx(nose);
  ctx.moveTo(hx + HEAD_RADIUS * width, hy);
  ctx.arc(hx, hy, HEAD_RADIUS * width, 0, Math.PI * 2);
  line(ctx, lay, [nose[0], nose[1] + HEAD_RADIUS], neck);
  line(ctx, lay, ls, rs);
  line(ctx, lay, ls, [neck[0] + HIP_HALF, neck[1] + HIP_Y]);
  line(ctx, lay, rs, [neck[0] - HIP_HALF, neck[1] + HIP_Y]);
  line(ctx, lay, [neck[0] + HIP_HALF, neck[1] + HIP_Y], [neck[0] - HIP_HALF, neck[1] + HIP_Y]);
  ctx.stroke();

  // The sign, dark: both arms, then every finger bone (hands drawn larger around the wrist).
  ctx.strokeStyle = SIGN;
  ctx.lineWidth = Math.max(3, width * 0.1);
  ctx.beginPath();
  for (const [shoulder, elbow, wrist, hand] of [
    [ls, point(p, L_ELBOW), point(p, L_WRIST), frame.l],
    [rs, point(p, R_ELBOW), point(p, R_WRIST), frame.r],
  ]) {
    line(ctx, lay, shoulder, elbow);
    line(ctx, lay, elbow, hand ? point(hand, 0) : wrist);   // the forearm ends where the hand is
  }
  ctx.stroke();

  ctx.lineWidth = Math.max(2, width * 0.055);
  ctx.beginPath();
  for (const hand of [frame.l, frame.r]) {
    if (!hand) continue;
    const [wx, wy] = point(hand, 0);
    const at = (i) => {
      const [x, y] = point(hand, i);
      return [wx + (x - wx) * HAND_SCALE, wy + (y - wy) * HAND_SCALE];
    };
    for (const [a, b] of HAND_BONES) line(ctx, lay, at(a), at(b));
  }
  ctx.stroke();
  ctx.restore();
}

function blend(a, b, w) {
  const mix = (x, y) => x.map((v, i) => v + (y[i] - v) * w);
  return { p: mix(a.p, b.p), l: a.l && b.l ? mix(a.l, b.l) : w < 0.5 ? a.l : b.l, r: a.r && b.r ? mix(a.r, b.r) : w < 0.5 ? a.r : b.r };
}

const defaultTime = () => ({
  now: () => performance.now(),
  schedule: (fn) => requestAnimationFrame(fn),
  cancel: (id) => cancelAnimationFrame(id),
});

/** Plays signs on a canvas, one after another. */
export class SignAvatar {
  #canvas;
  #poses;
  #time;
  #tick = 0;
  #finish = null;

  constructor(canvas, poses, time = defaultTime()) {
    this.#canvas = canvas;
    this.#poses = poses;
    this.#time = time;
    this.playing = false;
    this.drawRest();
  }

  has(code) {
    return this.#poses.signs.has(code);
  }

  /** Show `codes` in order (unknown ones are skipped). Resolves when done or stopped. */
  play(codes, { onSign } = {}) {
    this.stop();
    const fps = this.#poses.fps;
    const segments = [];
    let t = 0;
    for (const code of codes) {
      const sign = this.#poses.signs.get(code);
      if (!sign || !sign.frames.length) continue;
      if (segments.length) t += GAP_SECONDS;
      const rate = sign.fps || fps;   // a sign may carry its own frame rate
      const duration = sign.frames.length / rate;
      segments.push({ code, sign, rate, start: t, end: t + duration, index: segments.length });
      t += duration;
    }
    if (!segments.length) {
      this.drawRest();
      return Promise.resolve();
    }
    const total = t + END_HOLD_SECONDS;
    const start = this.#time.now();
    let announced = -1;
    this.playing = true;

    return new Promise((resolve) => {
      this.#finish = resolve;
      const step = () => {
        const elapsed = (this.#time.now() - start) / 1000;
        if (elapsed >= total) {
          this.#end();
          return;
        }
        let frame;
        const current = segments.findIndex((s) => elapsed < s.end + (s.index === segments.length - 1 ? END_HOLD_SECONDS : 0));
        const segment = segments[current];
        if (elapsed < segment.start) {   // gliding from the previous sign into this one
          const previous = segments[current - 1];
          const from = previous.sign.frames[previous.sign.frames.length - 1];
          frame = blend(from, segment.sign.frames[0], (elapsed - previous.end) / GAP_SECONDS);
        } else {
          if (segment.index !== announced) {
            announced = segment.index;
            onSign?.(segment.code, segment.index);
          }
          frame = frameAt(segment.sign, elapsed - segment.start, segment.rate);
        }
        this.#draw(frame);
        this.#tick = this.#time.schedule(step);
      };
      step();
    });
  }

  /** Stop at once and stand still. */
  stop() {
    if (this.#tick) this.#time.cancel(this.#tick);
    this.#tick = 0;
    if (this.playing) this.#end();
  }

  drawRest() {
    this.#draw(REST_FRAME);
  }

  #end() {
    this.playing = false;
    this.#tick = 0;
    this.drawRest();
    const finish = this.#finish;
    this.#finish = null;
    finish?.();
  }

  #draw(frame) {
    const canvas = this.#canvas;
    const ratio = globalThis.devicePixelRatio || 1;
    const w = canvas.clientWidth || canvas.width;
    const h = canvas.clientHeight || canvas.height;
    if (canvas.width !== Math.round(w * ratio) || canvas.height !== Math.round(h * ratio)) {
      canvas.width = Math.round(w * ratio);
      canvas.height = Math.round(h * ratio);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, w, h);
    drawFigure(ctx, frame, layout(w, h));
  }
}
