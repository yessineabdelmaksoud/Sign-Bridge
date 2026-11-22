// The files written by training/export_web.py have the shape the contract gives.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";

import { readBuffer, readJson } from "./helpers.js";

const GROUPS = { reason: 4, symptom: 14, conditions: 4 };

test("signs.json lists 22 signs in three groups, in the contract's shape", () => {
  const data = readJson("data/signs.json");
  assert.deepEqual(Object.keys(data), ["other", "groups", "signs"]);
  assert.equal(data.other, "OTHER");

  assert.deepEqual(data.groups.map((group) => group.id), Object.keys(GROUPS));
  for (const group of data.groups) {
    assert.deepEqual(Object.keys(group), ["id", "icon"]);
    assert.ok(typeof group.icon === "string" && group.icon.length > 0, `${group.id} icon`);
  }
  assert.equal(new Set(data.groups.map((group) => group.icon)).size, 3, "group icons differ");

  assert.equal(data.signs.length, 22);
  assert.equal(new Set(data.signs.map((sign) => sign.code)).size, 22, "codes are unique");
  for (const sign of data.signs) {
    assert.deepEqual(Object.keys(sign), ["code", "group", "icon", "fr", "ar", "clip", "weak"]);
    assert.equal(sign.clip, `clips/${sign.code}.mp4`);
    assert.equal(typeof sign.weak, "boolean");
    for (const key of ["icon", "fr", "ar"]) {
      assert.ok(typeof sign[key] === "string" && sign[key].length > 0, `${sign.code} ${key}`);
    }
  }
  for (const [id, count] of Object.entries(GROUPS)) {
    assert.equal(data.signs.filter((sign) => sign.group === id).length, count, id);
  }
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
    code: "DOLOR", group: "symptom", icon: "😣", fr: "Douleur", ar: "وجيعة",
    clip: "clips/DOLOR.mp4", weak: false,
  });
});

test("the manifest describes the model and where each array sits in the .bin", () => {
  const manifest = readJson("models/health_signs.json");
  assert.equal(manifest.labels.length, 23);
  assert.equal(manifest.labels.at(-1), "OTHER");
  assert.equal(manifest.threshold, 0.4);
  assert.deepEqual(manifest.weak_labels, ["CORAZON"]);
  assert.equal(manifest.frames, 32);
  assert.equal(manifest.features, 184);
  assert.equal(manifest.layers, 3);
  assert.equal(manifest.weights, "health_signs.bin");
  assert.equal(manifest.dtype, "float32");
  assert.equal(manifest.byte_order, "little");

  assert.deepEqual(
    manifest.arrays.map((array) => [array.name, array.shape]),
    [
      ["mean", [5888]], ["std", [5888]],
      ["w0", [5888, 256]], ["b0", [256]],
      ["w1", [256, 128]], ["b1", [128]],
      ["w2", [128, 23]], ["b2", [23]],
    ],
  );
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
  const { metrics } = readJson("models/health_signs.json");
  assert.equal(metrics.official_test.n, 56);
  assert.equal(metrics.official_test.top1, 0.75);
  assert.equal(metrics.official_test.top3, 0.8929);
  assert.equal(metrics.cv_folds, 5);
  assert.equal(metrics.cv_top1.mean, 0.8069);
  assert.equal(metrics.cv_top3.mean, 0.9328);
  assert.deepEqual(Object.keys(metrics.cv_narrowed_top1), Object.keys(GROUPS));
  assert.ok(metrics.note.includes("Spanish Sign Language"));
});
