import assert from "node:assert/strict";
import test from "node:test";

import { qrMatrix, qrSvg } from "../lib/qr.js";

// tests/test_qr.py decodes these codes with OpenCV; here: the structure the standard asks for.

const LINK = "http://192.168.1.20:8520/patient.html?room=4821";
const LEVELS = ["L", "M", "Q", "H"];
const FORMAT_LEVEL_BITS = { L: 1, M: 0, Q: 3, H: 2 };

/** Bytes a version holds in byte mode, versions 1 to 10 (the capacity table of ISO/IEC 18004). */
const CAPACITY = {
  L: [17, 32, 53, 78, 106, 134, 154, 192, 230, 271],
  M: [14, 26, 42, 62, 84, 106, 122, 152, 180, 213],
  Q: [11, 20, 32, 46, 60, 74, 86, 108, 130, 151],
  H: [7, 14, 24, 34, 44, 58, 64, 84, 98, 119],
};
const CAPACITY_40 = { L: 2953, M: 2331, Q: 1663, H: 1273 };

/** Version information words of the standard's table. */
const VERSION_WORDS = { 7: 0x07c94, 8: 0x085bc, 10: 0x0a4d3, 20: 0x149a6, 40: 0x28c69 };

/** "SignBridge room 4821. " repeated to `length` characters. */
const text = (length) => "SignBridge room 4821. ".repeat(Math.ceil(length / 22) + 1).slice(0, length);

/** Reads bits (bit 0 first) from the given [row, column] cells. */
const readBits = (modules, cells) => cells.reduce((bits, [r, c], i) => bits | ((modules[r][c] ? 1 : 0) << i), 0);

/** The two places of the 15 format bits, bit 0 first. */
const formatAroundTopLeft = () => [
  [0, 8], [1, 8], [2, 8], [3, 8], [4, 8], [5, 8], [7, 8], [8, 8],
  [8, 7], [8, 5], [8, 4], [8, 3], [8, 2], [8, 1], [8, 0],
];
const formatSplit = (size) => Array.from({ length: 15 }, (_, i) => (i < 8 ? [8, size - 1 - i] : [size - 15 + i, 8]));

/** The two places of the 18 version bits, bit 0 first. */
const versionTopRight = (size) => Array.from({ length: 18 }, (_, i) => [Math.floor(i / 3), size - 11 + (i % 3)]);
const versionBottomLeft = (size) => Array.from({ length: 18 }, (_, i) => [size - 11 + (i % 3), Math.floor(i / 3)]);

/** Remainder of `value` divided by `poly`, both read as polynomials over GF(2). */
function gf2Remainder(value, poly) {
  const degree = 31 - Math.clz32(poly);
  for (let bit = 31 - Math.clz32(value); bit >= degree; bit--) {
    if ((value >>> bit) & 1) value ^= poly << (bit - degree);
  }
  return value;
}

/** True when the 7x7 finder pattern has its top-left module at (row, col). */
function isFinder(modules, row, col) {
  for (let r = 0; r < 7; r++) {
    for (let c = 0; c < 7; c++) {
      const ring = Math.max(Math.abs(r - 3), Math.abs(c - 3));
      if (modules[row + r][col + c] !== (ring !== 2)) return false;
    }
  }
  return true;
}

// --- qrMatrix: shape and patterns ----------------------------------------------

test("the matrix is size x size booleans, with size 17 + 4 x version", () => {
  for (const [input, ecl] of [["HELLO", "M"], [LINK, "L"], [LINK, "H"], [text(180), "M"], [text(700), "Q"]]) {
    const { version, size, modules } = qrMatrix(input, { ecl });
    assert.equal(size, 17 + 4 * version);
    assert.equal(modules.length, size);
    for (const row of modules) {
      assert.equal(row.length, size);
      assert.ok(row.every((m) => typeof m === "boolean"));
    }
  }
});

test("finder patterns sit in three corners, with light separators", () => {
  for (const input of ["HELLO", LINK, text(300)]) {
    const { size, modules } = qrMatrix(input);
    assert.ok(isFinder(modules, 0, 0), "top left");
    assert.ok(isFinder(modules, 0, size - 7), "top right");
    assert.ok(isFinder(modules, size - 7, 0), "bottom left");
    for (let i = 0; i < 8; i++) {
      for (const [r, c] of [[7, i], [i, 7], [7, size - 1 - i], [i, size - 8], [size - 8, i], [size - 1 - i, 7]]) {
        assert.equal(modules[r][c], false, `separator at ${r},${c}`);
      }
    }
  }
});

test("timing patterns alternate and the dark module is dark", () => {
  for (const input of ["HELLO", text(500)]) {
    const { size, modules } = qrMatrix(input);
    for (let i = 8; i < size - 8; i++) {
      assert.equal(modules[6][i], i % 2 === 0, `row 6, column ${i}`);
      assert.equal(modules[i][6], i % 2 === 0, `column 6, row ${i}`);
    }
    assert.equal(modules[size - 8][8], true);
  }
});

test("alignment patterns are at the standard's places", () => {
  const isAlignment = (modules, row, col) => {
    for (let dr = -2; dr <= 2; dr++) {
      for (let dc = -2; dc <= 2; dc++) {
        if (modules[row + dr][col + dc] !== (Math.max(Math.abs(dr), Math.abs(dc)) !== 1)) return false;
      }
    }
    return true;
  };
  const v2 = qrMatrix(text(20));
  assert.equal(v2.version, 2);
  assert.ok(isAlignment(v2.modules, 18, 18));
  const v7 = qrMatrix(text(120)); // centres 6, 22, 38
  assert.equal(v7.version, 7);
  for (const [r, c] of [[6, 22], [22, 6], [22, 22], [22, 38], [38, 22], [38, 38]]) {
    assert.ok(isAlignment(v7.modules, r, c), `alignment at ${r},${c}`);
  }
});

// --- qrMatrix: version choice ----------------------------------------------------

test("the smallest version that holds the text is chosen", () => {
  assert.equal(qrMatrix("HELLO").version, 1);
  assert.equal(qrMatrix("HELLO", { ecl: "H" }).version, 1);
  assert.equal(qrMatrix("").version, 1);
  assert.equal(qrMatrix(LINK, { ecl: "L" }).version, 3);
  assert.equal(qrMatrix(LINK, { ecl: "M" }).version, 4);
  assert.equal(qrMatrix(LINK, { ecl: "Q" }).version, 5);
  assert.equal(qrMatrix(LINK, { ecl: "H" }).version, 6);
});

test("versions 1 to 10 end exactly at the standard's byte capacity", () => {
  for (const ecl of LEVELS) {
    CAPACITY[ecl].forEach((bytes, i) => {
      assert.equal(qrMatrix(text(bytes), { ecl }).version, i + 1, `${ecl}: ${bytes} bytes`);
      assert.equal(qrMatrix(text(bytes + 1), { ecl }).version, i + 2, `${ecl}: ${bytes + 1} bytes`);
    });
  }
});

test("the text is counted in UTF-8 bytes", () => {
  assert.equal(qrMatrix("é".repeat(7)).version, 1); // 14 bytes, the capacity of 1-M
  assert.equal(qrMatrix("é".repeat(8)).version, 2);
});

test("version 40 holds the standard's capacity, one byte more raises a RangeError", () => {
  for (const ecl of LEVELS) {
    const bytes = CAPACITY_40[ecl];
    assert.equal(qrMatrix("x".repeat(bytes), { ecl }).version, 40);
    assert.throws(() => qrMatrix("x".repeat(bytes + 1), { ecl }), RangeError);
  }
  assert.throws(() => qrMatrix("x".repeat(5000)), RangeError);
});

test("an unknown error correction level raises a RangeError", () => {
  for (const ecl of ["m", "X", "", "toString", null]) assert.throws(() => qrMatrix("HELLO", { ecl }), RangeError);
});

// --- qrMatrix: format and version information -------------------------------------

test("both copies of the format bits are equal, a BCH codeword, and carry the level", () => {
  for (const ecl of LEVELS) {
    for (const input of ["HELLO", LINK, text(100), text(180), text(400)]) {
      const { size, modules } = qrMatrix(input, { ecl });
      const first = readBits(modules, formatAroundTopLeft());
      const second = readBits(modules, formatSplit(size));
      assert.equal(first, second, `${ecl} ${input.length}`);
      const unmasked = first ^ 0x5412;
      assert.equal(gf2Remainder(unmasked, 0x537), 0, "BCH(15,5) remainder");
      assert.equal(unmasked >>> 13, FORMAT_LEVEL_BITS[ecl]);
    }
  }
});

test("versions 7 and up carry both copies of the version information", () => {
  const cases = [
    [text(120), "M", 7],
    [text(150), "M", 8],
    [text(200), "M", 10],
    [text(650), "M", 20],
    ["x".repeat(2953), "L", 40],
  ];
  for (const [input, ecl, expected] of cases) {
    const { version, size, modules } = qrMatrix(input, { ecl });
    assert.equal(version, expected);
    const topRight = readBits(modules, versionTopRight(size));
    assert.equal(readBits(modules, versionBottomLeft(size)), topRight);
    assert.equal(topRight, VERSION_WORDS[version]);
    assert.equal(gf2Remainder(topRight, 0x1f25), 0, "BCH(18,6) remainder");
    assert.equal(topRight >>> 12, version);
  }
});

test("the same text gives the same code", () => {
  assert.deepEqual(qrMatrix(LINK), qrMatrix(LINK));
  assert.notDeepEqual(qrMatrix(LINK).modules, qrMatrix(LINK.replace("4821", "4822")).modules);
});

// --- qrSvg -------------------------------------------------------------------------

/** Redraws the grid from the path's runs ("M x y h n v1 h-n z"). */
function gridFromPath(d, size) {
  const grid = Array.from({ length: size }, () => new Array(size).fill(false));
  const runs = [...d.matchAll(/M(\d+) (\d+)h(\d+)v1h-(\d+)z/g)];
  assert.equal(runs.map((m) => m[0]).join(""), d, "the path is only runs");
  for (const [, x, y, n, back] of runs.map((m) => m.map(Number))) {
    assert.equal(n, back);
    for (let i = 0; i < n; i++) grid[y][x + i] = true;
  }
  return grid;
}

test("qrSvg is an svg whose viewBox is the code plus the quiet zone", () => {
  for (const quiet of [4, 2, 0]) {
    const { size } = qrMatrix(LINK);
    const svg = qrSvg(LINK, { quiet });
    const n = size + 2 * quiet;
    assert.match(svg, /^<svg [^>]*>.*<\/svg>$/s);
    const open = svg.match(/^<svg [^>]*>/)[0];
    assert.ok(open.includes('xmlns="http://www.w3.org/2000/svg"'));
    assert.ok(open.includes(`viewBox="0 0 ${n} ${n}"`));
    assert.ok(open.includes('shape-rendering="crispEdges"'));
    assert.doesNotMatch(open, /\s(width|height)=/);
    assert.ok(svg.includes(`<rect width="${n}" height="${n}" fill="#FFFFFF"/>`));
    assert.equal((svg.match(/<path /g) ?? []).length, 1);
  }
});

test("qrSvg's path draws exactly the dark modules, shifted by the quiet zone", () => {
  for (const [input, ecl, quiet] of [["HELLO", "M", 4], [LINK, "Q", 4], [text(300), "L", 1]]) {
    const { size, modules } = qrMatrix(input, { ecl });
    const svg = qrSvg(input, { ecl, quiet });
    const d = svg.match(/<path fill="#10233A" d="([^"]*)"\/>/)[1];
    const grid = gridFromPath(d, size + 2 * quiet);
    for (let r = 0; r < size + 2 * quiet; r++) {
      for (let c = 0; c < size + 2 * quiet; c++) {
        const inside = r >= quiet && c >= quiet && r < size + quiet && c < size + quiet;
        assert.equal(grid[r][c], inside && modules[r - quiet][c - quiet], `${r},${c}`);
      }
    }
  }
});

test("qrSvg takes the colours and escapes them", () => {
  const svg = qrSvg("HELLO", { dark: "#000", light: "transparent" });
  assert.ok(svg.includes('fill="#000"'));
  assert.ok(svg.includes('fill="transparent"'));
  const odd = qrSvg("HELLO", { dark: '"><script>x</script>' });
  assert.ok(odd.includes('fill="&quot;>&lt;script>x&lt;/script>"'));
  assert.throws(() => qrSvg("x".repeat(3000)), RangeError);
});
