// QR codes drawn in the page, without any library (ISO/IEC 18004, model 2):
// byte mode with the text in UTF-8, versions 1 to 40, error correction L, M, Q or H.
// Rows and columns count from the top-left corner; true is a dark module.

/** Per level: the two format bits, and the row of the level in the tables below. */
const LEVELS = { L: { bits: 1, row: 0 }, M: { bits: 0, row: 1 }, Q: { bits: 3, row: 2 }, H: { bits: 2, row: 3 } };

/** Error correction codewords in each block, by level (L, M, Q, H), versions 1 to 40. */
const EC_PER_BLOCK = [
  [7, 10, 15, 20, 26, 18, 20, 24, 30, 18, 20, 24, 26, 30, 22, 24, 28, 30, 28, 28,
    28, 28, 30, 30, 26, 28, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
  [10, 16, 26, 18, 24, 16, 18, 22, 22, 26, 30, 22, 22, 24, 24, 28, 28, 26, 26, 26,
    26, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28],
  [13, 22, 18, 26, 18, 24, 18, 22, 20, 24, 28, 26, 24, 20, 30, 24, 28, 28, 26, 30,
    28, 30, 30, 30, 30, 28, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
  [17, 28, 22, 16, 22, 28, 26, 26, 24, 28, 24, 28, 22, 24, 24, 30, 28, 28, 26, 28,
    30, 24, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
];

/** Number of error correction blocks, by level (L, M, Q, H), versions 1 to 40. */
const BLOCKS = [
  [1, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 4, 6, 6, 6, 6, 7, 8,
    8, 9, 9, 10, 12, 12, 12, 13, 14, 15, 16, 17, 18, 19, 19, 20, 21, 22, 24, 25],
  [1, 1, 1, 2, 2, 4, 4, 4, 5, 5, 5, 8, 9, 9, 10, 10, 11, 13, 14, 16,
    17, 17, 18, 20, 21, 23, 25, 26, 28, 29, 31, 33, 35, 37, 38, 40, 43, 45, 47, 49],
  [1, 1, 2, 2, 4, 4, 6, 6, 8, 8, 8, 10, 12, 16, 12, 17, 16, 18, 21, 20,
    23, 23, 25, 27, 29, 34, 34, 35, 38, 40, 43, 45, 48, 51, 53, 56, 59, 62, 65, 68],
  [1, 1, 2, 4, 4, 4, 5, 6, 8, 8, 11, 11, 16, 16, 18, 16, 19, 21, 25, 25,
    25, 34, 30, 32, 35, 37, 40, 42, 45, 48, 51, 54, 57, 60, 63, 66, 70, 74, 77, 81],
];

const MAX_VERSION = 40;

/** [codewords per block, blocks] of error correction for a version and level. */
const ecLayout = (version, level) => [EC_PER_BLOCK[level.row][version - 1], BLOCKS[level.row][version - 1]];

// --- Reed-Solomon over GF(256), polynomial x^8 + x^4 + x^3 + x^2 + 1 ----------------

const EXP = new Uint8Array(512);
const LOG = new Uint8Array(256);
for (let i = 0, x = 1; i < 255; i++) {
  EXP[i] = x;
  LOG[x] = i;
  x <<= 1;
  if (x & 0x100) x ^= 0x11d;
}
for (let i = 255; i < 512; i++) EXP[i] = EXP[i - 255];

const multiply = (a, b) => (a && b ? EXP[LOG[a] + LOG[b]] : 0);

/** Generator polynomial (x - a^0)(x - a^1)...(x - a^(degree-1)), highest power first. */
function generator(degree) {
  let poly = [1];
  for (let i = 0; i < degree; i++) {
    const next = new Array(poly.length + 1).fill(0);
    poly.forEach((coef, j) => {
      next[j] ^= coef;
      next[j + 1] ^= multiply(coef, EXP[i]);
    });
    poly = next;
  }
  return poly;
}

/** The `degree` error correction codewords of `data`: the remainder of data(x)·x^degree. */
function errorCorrection(data, degree) {
  const gen = generator(degree);
  const rest = new Array(degree).fill(0);
  for (const byte of data) {
    const factor = byte ^ rest.shift();
    rest.push(0);
    for (let i = 0; i < degree; i++) rest[i] ^= multiply(gen[i + 1], factor);
  }
  return rest;
}

// --- Capacity and codewords -------------------------------------------------------

/** Modules left for data and error correction once the function patterns are drawn. */
function rawModules(version) {
  let modules = (16 * version + 128) * version + 64;
  if (version >= 2) {
    const align = Math.floor(version / 7) + 2;
    modules -= (25 * align - 10) * align - 55;
    if (version >= 7) modules -= 36;
  }
  return modules;
}

/** Data codewords (bytes) a version holds at a level. */
function dataCodewords(version, level) {
  const [ecLength, blocks] = ecLayout(version, level);
  return Math.floor(rawModules(version) / 8) - ecLength * blocks;
}

/** Bits of the byte-mode character count. */
const countBits = (version) => (version <= 9 ? 8 : 16);

/** Smallest version that holds `length` bytes, or 0 when none does. */
function smallestVersion(length, level) {
  for (let version = 1; version <= MAX_VERSION; version++) {
    if (4 + countBits(version) + 8 * length <= 8 * dataCodewords(version, level)) return version;
  }
  return 0;
}

/** Mode, count, bytes, terminator and padding, as the version's data codewords. */
function encodeData(bytes, version, level) {
  const capacity = 8 * dataCodewords(version, level);
  const bits = [];
  const push = (value, length) => {
    for (let i = length - 1; i >= 0; i--) bits.push((value >>> i) & 1);
  };
  push(0b0100, 4);
  push(bytes.length, countBits(version));
  for (const byte of bytes) push(byte, 8);
  push(0, Math.min(4, capacity - bits.length));
  push(0, (8 - (bits.length % 8)) % 8);
  const codewords = [];
  for (let i = 0; i < bits.length; i += 8) codewords.push(bits.slice(i, i + 8).reduce((byte, bit) => (byte << 1) | bit, 0));
  for (let pad = 0xec; codewords.length < capacity / 8; pad ^= 0xec ^ 0x11) codewords.push(pad);
  return codewords;
}

/** Splits the data into blocks, adds each block's error correction and interleaves them. */
function interleave(data, version, level) {
  const [ecLength, blocks] = ecLayout(version, level);
  const raw = Math.floor(rawModules(version) / 8);
  const shortBlocks = blocks - (raw % blocks);
  const shortData = Math.floor(raw / blocks) - ecLength;
  const dataBlocks = [];
  const ecBlocks = [];
  for (let b = 0, start = 0; b < blocks; b++) {
    const end = start + shortData + (b < shortBlocks ? 0 : 1);
    dataBlocks.push(data.slice(start, end));
    ecBlocks.push(errorCorrection(data.slice(start, end), ecLength));
    start = end;
  }
  const out = [];
  for (let i = 0; i <= shortData; i++) for (const block of dataBlocks) if (i < block.length) out.push(block[i]);
  for (let i = 0; i < ecLength; i++) for (const block of ecBlocks) out.push(block[i]);
  return out;
}

// --- Drawing ----------------------------------------------------------------------

/** Centres of the alignment patterns along one axis. */
function alignmentCentres(version) {
  if (version === 1) return [];
  const size = 17 + 4 * version;
  const count = Math.floor(version / 7) + 2;
  const step = Math.floor((version * 8 + count * 3 + 5) / (count * 4 - 4)) * 2;
  const centres = [6];
  for (let pos = size - 7; centres.length < count; pos -= step) centres.splice(1, 0, pos);
  return centres;
}

/** 15 format bits: level and mask, BCH(15,5), then masked with 101010000010010. */
function formatBits(level, mask) {
  const data = (level.bits << 3) | mask;
  let rest = data;
  for (let i = 0; i < 10; i++) rest = (rest << 1) ^ ((rest >>> 9) * 0x537);
  return ((data << 10) | rest) ^ 0x5412;
}

/** 18 version bits: the version, then BCH(18,6). */
function versionBits(version) {
  let rest = version;
  for (let i = 0; i < 12; i++) rest = (rest << 1) ^ ((rest >>> 11) * 0x1f25);
  return (version << 12) | rest;
}

/** An empty symbol: `dark` holds the colours, `fixed` marks the function modules. */
function blankSymbol(size) {
  const grid = () => Array.from({ length: size }, () => new Array(size).fill(false));
  const symbol = { size, dark: grid(), fixed: grid() };
  symbol.set = (row, col, value) => {
    symbol.dark[row][col] = Boolean(value);
    symbol.fixed[row][col] = true;
  };
  return symbol;
}

/** Writes the 15 format bits in both of their places (bit 0 first), with the dark module. */
function drawFormat(symbol, bits) {
  const { size, set } = symbol;
  const bit = (i) => (bits >>> i) & 1;
  for (let i = 0; i <= 5; i++) set(i, 8, bit(i));
  set(7, 8, bit(6));
  set(8, 8, bit(7));
  set(8, 7, bit(8));
  for (let i = 9; i < 15; i++) set(8, 14 - i, bit(i));
  for (let i = 0; i < 8; i++) set(8, size - 1 - i, bit(i));
  for (let i = 8; i < 15; i++) set(size - 15 + i, 8, bit(i));
  set(size - 8, 8, true);
}

/** Finder, separator, timing and alignment patterns, version information, and room for the format. */
function drawFunctionPatterns(symbol, version) {
  const { size, set } = symbol;
  for (let i = 0; i < size; i++) {
    set(6, i, i % 2 === 0);
    set(i, 6, i % 2 === 0);
  }
  for (const [row, col] of [[3, 3], [3, size - 4], [size - 4, 3]]) {
    for (let dr = -4; dr <= 4; dr++) {
      for (let dc = -4; dc <= 4; dc++) {
        const r = row + dr;
        const c = col + dc;
        const ring = Math.max(Math.abs(dr), Math.abs(dc));
        if (r >= 0 && r < size && c >= 0 && c < size) set(r, c, ring !== 2 && ring !== 4);
      }
    }
  }
  const centres = alignmentCentres(version);
  const last = centres.length - 1;
  centres.forEach((row, i) => {
    centres.forEach((col, j) => {
      if ((i === 0 && j === 0) || (i === 0 && j === last) || (i === last && j === 0)) return;
      for (let dr = -2; dr <= 2; dr++) {
        for (let dc = -2; dc <= 2; dc++) set(row + dr, col + dc, Math.max(Math.abs(dr), Math.abs(dc)) !== 1);
      }
    });
  });
  drawFormat(symbol, 0);
  if (version >= 7) {
    const bits = versionBits(version);
    for (let i = 0; i < 18; i++) {
      const value = (bits >>> i) & 1;
      set(Math.floor(i / 3), size - 11 + (i % 3), value);
      set(size - 11 + (i % 3), Math.floor(i / 3), value);
    }
  }
}

/** Places the codewords in the free modules, two columns at a time, zigzagging up and down. */
function drawCodewords(symbol, codewords) {
  const { size, dark, fixed } = symbol;
  const total = codewords.length * 8;
  let i = 0;
  for (let right = size - 1; right >= 1; right -= 2) {
    if (right === 6) right = 5; // the vertical timing pattern
    const upward = ((right + 1) & 2) === 0;
    for (let step = 0; step < size; step++) {
      const row = upward ? size - 1 - step : step;
      for (const col of [right, right - 1]) {
        if (fixed[row][col]) continue;
        // Remainder bits after the last codeword stay light.
        dark[row][col] = i < total && ((codewords[i >>> 3] >>> (7 - (i & 7))) & 1) === 1;
        i++;
      }
    }
  }
}

/** The eight data masks: true where the module is flipped. */
const MASKS = [
  (r, c) => (r + c) % 2 === 0,
  (r) => r % 2 === 0,
  (r, c) => c % 3 === 0,
  (r, c) => (r + c) % 3 === 0,
  (r, c) => (Math.floor(r / 2) + Math.floor(c / 3)) % 2 === 0,
  (r, c) => ((r * c) % 2) + ((r * c) % 3) === 0,
  (r, c) => (((r * c) % 2) + ((r * c) % 3)) % 2 === 0,
  (r, c) => (((r + c) % 2) + ((r * c) % 3)) % 2 === 0,
];

/** True when no module of `line` from `from` to `to` (excluded) is dark; outside the symbol is light. */
function lightBetween(line, from, to) {
  for (let i = Math.max(from, 0); i < Math.min(to, line.length); i++) if (line[i]) return false;
  return true;
}

/** Penalty score of a finished symbol under the four rules of the standard. */
function penalty(modules) {
  const size = modules.length;
  let score = 0;
  const lines = [];
  for (let i = 0; i < size; i++) lines.push(modules[i], modules.map((row) => row[i]));
  for (const line of lines) {
    // Rule 1: five or more modules of one colour in a row: 3 points, plus 1 per extra module.
    let run = 1;
    for (let i = 1; i <= size; i++) {
      if (i < size && line[i] === line[i - 1]) {
        run++;
        continue;
      }
      if (run >= 5) score += run - 2;
      run = 1;
    }
    // Rule 3: dark-light-dark-dark-dark-light-dark with four light modules on either side: 40 points.
    for (let i = 0; i + 7 <= size; i++) {
      if (line[i] && !line[i + 1] && line[i + 2] && line[i + 3] && line[i + 4] && !line[i + 5] && line[i + 6]) {
        if (lightBetween(line, i - 4, i) || lightBetween(line, i + 7, i + 11)) score += 40;
      }
    }
  }
  // Rule 2: each 2x2 block of one colour: 3 points.
  let dark = 0;
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      if (modules[r][c]) dark++;
      if (r + 1 < size && c + 1 < size) {
        const colour = modules[r][c];
        if (modules[r][c + 1] === colour && modules[r + 1][c] === colour && modules[r + 1][c + 1] === colour) score += 3;
      }
    }
  }
  // Rule 4: 10 points per full 5% the dark share is away from 50%.
  const total = size * size;
  score += 10 * Math.floor(Math.abs(dark * 20 - total * 10) / total);
  return score;
}

/**
 * The QR code of `text` (byte mode, UTF-8) at error correction level `ecl`
 * ("L", "M", "Q" or "H"), in the smallest version that holds it.
 * Returns { version, size, modules }: modules[row][column] is true for dark.
 * Throws a RangeError for an unknown level or a text too long for version 40.
 */
export function qrMatrix(text, { ecl = "M" } = {}) {
  const level = Object.hasOwn(LEVELS, ecl) ? LEVELS[ecl] : null;
  if (!level) throw new RangeError(`QR error correction level must be L, M, Q or H, not ${ecl}`);
  const bytes = new TextEncoder().encode(String(text ?? ""));
  const version = smallestVersion(bytes.length, level);
  if (!version) throw new RangeError(`text too long for a QR code at level ${ecl} (${bytes.length} bytes)`);
  const size = 17 + 4 * version;
  const symbol = blankSymbol(size);
  drawFunctionPatterns(symbol, version);
  drawCodewords(symbol, interleave(encodeData(bytes, version, level), version, level));

  let best = null;
  let bestScore = Infinity;
  MASKS.forEach((flip, mask) => {
    const candidate = blankSymbol(size);
    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        candidate.dark[r][c] = symbol.fixed[r][c] ? symbol.dark[r][c] : symbol.dark[r][c] !== flip(r, c);
      }
    }
    drawFormat(candidate, formatBits(level, mask));
    const score = penalty(candidate.dark);
    if (score < bestScore) {
      best = candidate.dark;
      bestScore = score;
    }
  });
  return { version, size, modules: best };
}

/** Escapes a value for a double-quoted XML attribute. */
const attribute = (value) =>
  String(value).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");

/**
 * The QR code of `text` as an SVG string, in module units with a `quiet`-module
 * light border; no width or height, so CSS sizes it. Dark modules are one path
 * of horizontal runs. Same errors as qrMatrix.
 */
export function qrSvg(text, { ecl = "M", quiet = 4, dark = "#10233A", light = "#FFFFFF" } = {}) {
  const { size, modules } = qrMatrix(text, { ecl });
  const margin = Math.max(0, Math.floor(Number(quiet) || 0));
  const full = size + 2 * margin;
  let path = "";
  modules.forEach((row, r) => {
    for (let c = 0; c < size; c++) {
      if (!row[c]) continue;
      const start = c;
      while (c + 1 < size && row[c + 1]) c++;
      const run = c - start + 1;
      path += `M${start + margin} ${r + margin}h${run}v1h-${run}z`;
    }
  });
  return (
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${full} ${full}" shape-rendering="crispEdges">` +
    `<rect width="${full}" height="${full}" fill="${attribute(light)}"/>` +
    `<path fill="${attribute(dark)}" d="${path}"/></svg>`
  );
}
