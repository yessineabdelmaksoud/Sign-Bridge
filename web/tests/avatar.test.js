import assert from "node:assert/strict";
import test from "node:test";

import { BOX, REST_FRAME, SignAvatar, drawFigure, frameAt, layout, poseFromJson } from "../lib/avatar.js";
import { readJson } from "./helpers.js";

const poses = poseFromJson(readJson("data/sign_poses.json"));
const signs = readJson("data/signs.json").signs;

/** A 2D context that records every coordinate it is given. */
function fakeContext() {
  const points = [];
  const record = (x, y) => points.push([x, y]);
  return {
    points,
    calls: 0,
    canvas: { width: 300, height: 400 },
    save() {}, restore() {}, beginPath() { this.calls += 1; }, stroke() {}, fill() {},
    clearRect() {}, setTransform() {},
    moveTo: record, lineTo: record,
    arc(x, y) { record(x, y); },
    set strokeStyle(v) {}, set fillStyle(v) {}, set lineWidth(v) {}, set lineCap(v) {}, set lineJoin(v) {},
  };
}

function fakeCanvas() {
  const ctx = fakeContext();
  return { ctx, width: 300, height: 400, clientWidth: 300, clientHeight: 400, getContext: () => ctx };
}

/** A clock and a frame scheduler under the test's control. */
function fakeTime() {
  let now = 0;
  let queue = [];
  return {
    now: () => now,
    schedule: (fn) => { queue.push(fn); return queue.length; },
    cancel: () => { queue = []; },
    advance(ms, step = 16) {
      for (let t = 0; t < ms; t += step) {
        now += step;
        const due = queue;
        queue = [];
        due.forEach((fn) => fn(now));
      }
    },
  };
}

test("every sign has stick-figure data with body and hands", () => {
  for (const sign of signs) {
    const data = poses.signs.get(sign.code);
    assert.ok(data, `${sign.code} has no pose data`);
    assert.ok(data.frames.length >= 10, `${sign.code}: ${data.frames.length} frames`);
    const withHand = data.frames.filter((f) => f.l || f.r).length;
    assert.ok(withHand >= data.frames.length / 2, `${sign.code}: hands in ${withHand} frames`);
    for (const f of data.frames) {
      assert.equal(f.p.length, 14);
      if (f.l) assert.equal(f.l.length, 42);
      if (f.r) assert.equal(f.r.length, 42);
    }
  }
});

test("the figure's box fits inside the canvas, centred", () => {
  for (const [w, h] of [[300, 400], [800, 300], [200, 200]]) {
    const lay = layout(w, h);
    const [x0, y0] = lay.toPx([BOX.left, BOX.top]);
    const [x1, y1] = lay.toPx([BOX.right, BOX.bottom]);
    assert.ok(x0 >= -0.01 && y0 >= -0.01 && x1 <= w + 0.01 && y1 <= h + 0.01, `${w}x${h}`);
    assert.ok(Math.abs((x0 + x1) / 2 - w / 2) < 0.5);
  }
});

test("frameAt picks the frame for a time and stays inside the sign", () => {
  const sign = { frames: [{ p: [1] }, { p: [2] }, { p: [3] }] };
  assert.equal(frameAt(sign, 0, 20).p[0], 1);
  assert.equal(frameAt(sign, 0.051, 20).p[0], 2);
  assert.equal(frameAt(sign, 99, 20).p[0], 3);
  assert.equal(frameAt(sign, -1, 20).p[0], 1);
});

test("drawing never passes a missing coordinate to the canvas", () => {
  const ctx = fakeContext();
  const lay = layout(300, 400);
  drawFigure(ctx, REST_FRAME, lay);
  for (const f of poses.signs.get("DOLOR").frames) drawFigure(ctx, f, lay);
  assert.ok(ctx.points.length > 100);
  assert.ok(ctx.points.every(([x, y]) => Number.isFinite(x) && Number.isFinite(y)));
});

test("play shows the known signs in order, skips unknown ones, and finishes", async () => {
  const time = fakeTime();
  const canvas = fakeCanvas();
  const avatar = new SignAvatar(canvas, poses, time);
  const seen = [];
  let done = false;
  const finished = avatar.play(["FIEBRE2", "NOT_A_SIGN", "TOSER"], { onSign: (code, i) => seen.push([code, i]) })
    .then(() => { done = true; });
  const total = (poses.signs.get("FIEBRE2").frames.length + poses.signs.get("TOSER").frames.length) / poses.fps;
  time.advance(total * 1000 * 0.5);
  await Promise.resolve();
  assert.equal(done, false);
  time.advance(total * 1000 + 3000);
  await finished;
  assert.deepEqual(seen, [["FIEBRE2", 0], ["TOSER", 1]]);
  assert.equal(avatar.has("TOSER"), true);
  assert.equal(avatar.has("NOT_A_SIGN"), false);
});

test("stop ends a sign early and the pending play resolves", async () => {
  const time = fakeTime();
  const avatar = new SignAvatar(fakeCanvas(), poses, time);
  const finished = avatar.play(["DOLOR", "DIABETES"]);
  time.advance(200);
  avatar.stop();
  await finished;
  assert.equal(avatar.playing, false);
});

test("a new play replaces the one in progress", async () => {
  const time = fakeTime();
  const avatar = new SignAvatar(fakeCanvas(), poses, time);
  const seen = [];
  const first = avatar.play(["DOLOR"], { onSign: (c) => seen.push(c) });
  time.advance(100);
  const second = avatar.play(["ASMA"], { onSign: (c) => seen.push(c) });
  await first;
  time.advance(10000);
  await second;
  assert.deepEqual(seen, ["DOLOR", "ASMA"]);
});
