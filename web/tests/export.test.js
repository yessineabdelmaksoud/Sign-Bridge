// The files written by training/export_web.py have the shape the contract gives.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";

import { readBuffer, readJson } from "./helpers.js";

const GROUPS = { reason: 4, symptom: 14, conditions: 4 };   // the SWL-LSE health signs
const SIGN_KEYS = ["code", "group", "fr", "ar", "clip", "weak", "language"];

test("signs.json has no icons, the 22 health signs and the Tunisian everyday words", () => {
  const data = readJson("data/signs.json");
  assert.deepEqual(Object.keys(data), ["other", "groups", "signs"]);
  assert.equal(data.other, "OTHER");
  assert.doesNotMatch(JSON.stringify(data), /icon/);

  for (const group of data.groups) {
    assert.deepEqual(Object.keys(group), ["id", "fr", "ar"]);
    assert.ok(group.fr && group.ar, group.id);
  }
  assert.deepEqual(data.groups.map((g) => g.id).slice(0, 3), Object.keys(GROUPS));

  assert.equal(new Set(data.signs.map((sign) => sign.code)).size, data.signs.length, "codes are unique");
  for (const sign of data.signs) {
    assert.deepEqual(Object.keys(sign), SIGN_KEYS);
    assert.equal(typeof sign.weak, "boolean");
    assert.ok(sign.fr && sign.ar, sign.code);
    if (sign.group === "everyday") {
      assert.equal(sign.language, "tunisian");
      assert.equal(sign.clip, null, `${sign.code} has no video`);
    } else {
      assert.equal(sign.language, "lse");
      assert.equal(sign.clip, `clips/${sign.code}.mp4`);
    }
  }
  for (const [id, count] of Object.entries(GROUPS)) {
    assert.equal(data.signs.filter((sign) => sign.group === id).length, count, id);
  }
  assert.ok(data.signs.filter((sign) => sign.group === "everyday").length >= 10, "Tunisian words");
});

test("signs.json agrees with the model manifest", () => {
  const data = readJson("data/signs.json");
  const manifest = readJson("models/health_signs.json");
  assert.deepEqual([...data.signs.map((sign) => sign.code), data.other], manifest.labels);
  assert.deepEqual(data.signs.filter((sign) => sign.weak).map((sign) => sign.code), manifest.weak_labels);
});

test("one sign from the contract's example comes out as written there", () => {
  const dolor = readJson("data/signs.json").signs.find((sign) => sign.code === "DOLOR");
  assert.deepEqual(dolor, {
    code: "DOLOR", group: "symptom", fr: "Douleur", ar: "وجيعة",
    clip: "clips/DOLOR.mp4", weak: false, language: "lse",
  });
});

test("the manifest describes the model and where each array sits in the .bin", () => {
  const manifest = readJson("models/health_signs.json");
  const signs = readJson("data/signs.json").signs;
  const n = manifest.labels.length;
  assert.equal(n, signs.length + 1);
  assert.equal(manifest.labels.at(-1), "OTHER");
  assert.ok(manifest.threshold >= 0.4 && manifest.threshold <= 0.9, `threshold ${manifest.threshold}`);
  assert.ok(manifest.weak_labels.every((label) => manifest.labels.includes(label)));
  assert.equal(manifest.frames, 32);
  assert.equal(manifest.features, 184);
  assert.equal(manifest.weights, "health_signs.bin");
  assert.equal(manifest.dtype, "float32");
  assert.equal(manifest.byte_order, "little");
  const shapes = Object.fromEntries(manifest.arrays.map((array) => [array.name, array.shape]));
  assert.deepEqual(shapes.mean, [5888]);
  assert.equal(shapes.w0[0], 5888);
  assert.equal(shapes[`w${manifest.layers - 1}`][1], n, "one output per label");
  let offset = 0;
  for (const array of manifest.arrays) {
    assert.equal(array.byte_offset, offset, `${array.name} offset`);
    assert.equal(array.length, array.shape.reduce((a, b) => a * b, 1), `${array.name} length`);
    offset += array.length * 4;
  }
  assert.equal(manifest.byte_length, offset);
});

test("the .bin is the size and the content the manifest says", () => {
  const manifest = readJson("models/health_signs.json");
  const bytes = new Uint8Array(readBuffer("models/health_signs.bin"));
  assert.equal(bytes.byteLength, manifest.byte_length);
  assert.equal(createHash("sha256").update(bytes).digest("hex"), manifest.sha256);
});

test("the manifest carries the headline metrics", () => {
  const { metrics, labels } = readJson("models/health_signs.json");
  assert.equal(metrics.official_test.n, 56);
  assert.ok(metrics.official_test.top1 > 0.5);
  assert.deepEqual(Object.keys(metrics.official_test.narrowed), Object.keys(GROUPS));
  assert.ok(metrics.note.includes("Spanish Sign Language"));
  if (labels.some((label) => label.startsWith("TN_"))) {
    assert.ok(metrics.tunisian_leave_one_signer_out.n > 0);
  }
});
