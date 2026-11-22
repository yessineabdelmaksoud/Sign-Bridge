/**
 * A small fully connected classifier: standardise, then Dense+ReLU layers,
 * then softmax. Inference only. Port of Demo/consultation/numpy_model.py.
 *
 * The model comes as two files written by training/export_web.py: a JSON
 * manifest and, beside it, a .bin holding every array as little-endian
 * float32. A weight matrix is stored row-major as (inputs, outputs).
 */

const LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

/** The URL of the file called `name` in the same directory as `url`. */
function beside(url, name) {
  const path = String(url).replace(/[?#].*$/, "");
  return path.slice(0, path.lastIndexOf("/") + 1) + name;
}

async function fetchOrThrow(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`could not load ${url}: HTTP ${response.status}`);
  return response;
}

/** `length` little-endian float32 values that start `byteOffset` bytes into `bytes`. */
function readFloats(bytes, byteOffset, length) {
  const start = bytes.byteOffset + byteOffset;
  if (LITTLE_ENDIAN && start % 4 === 0) return new Float32Array(bytes.buffer, start, length);
  const view = new DataView(bytes.buffer, start, length * 4);
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) out[i] = view.getFloat32(i * 4, true);
  return out;
}

export class DenseModel {
  labels;
  threshold;
  weakLabels;
  #mean;
  #std;
  #layers;

  /**
   * `layers` is a list of { weights, biases }: `weights` is a Float32Array of
   * inputs × outputs values, row-major, and `biases` has one value per output.
   */
  constructor({ labels, threshold, weakLabels = [], mean, std, layers }) {
    let width = mean.length;
    if (std.length !== width) throw new Error("mean and std differ in length");
    for (const { weights, biases } of layers) {
      if (weights.length !== width * biases.length) throw new Error("a layer does not fit the one before it");
      width = biases.length;
    }
    if (width !== labels.length) throw new Error("the last layer does not have one output per label");
    this.labels = labels;
    this.threshold = threshold;
    this.weakLabels = weakLabels;
    this.#mean = mean;
    this.#std = std;
    this.#layers = layers;
  }

  /** Read the manifest at `manifestUrl` and the .bin beside it. */
  static async load(manifestUrl) {
    const manifest = await (await fetchOrThrow(manifestUrl)).json();
    const weights = await (await fetchOrThrow(beside(manifestUrl, manifest.weights))).arrayBuffer();
    return DenseModel.fromBuffer(manifest, weights);
  }

  /**
   * Build the model from a manifest that is already parsed and the content of
   * its .bin, as an ArrayBuffer or a typed array. This is what `load` uses,
   * and what the tests use, since Node cannot fetch a local file.
   */
  static fromBuffer(manifest, buffer) {
    const bytes = ArrayBuffer.isView(buffer)
      ? new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength)
      : new Uint8Array(buffer);
    if (bytes.byteLength !== manifest.byte_length) {
      throw new Error(`${manifest.weights} is ${bytes.byteLength} bytes, the manifest says ${manifest.byte_length}`);
    }
    const arrays = {};
    for (const { name, byte_offset: byteOffset, length } of manifest.arrays) {
      arrays[name] = readFloats(bytes, byteOffset, length);
    }
    const layers = Array.from({ length: manifest.layers }, (_, i) => ({
      weights: arrays[`w${i}`],
      biases: arrays[`b${i}`],
    }));
    return new DenseModel({
      labels: manifest.labels,
      threshold: manifest.threshold,
      weakLabels: manifest.weak_labels ?? [],
      mean: arrays.mean,
      std: arrays.std,
      layers,
    });
  }

  /**
   * The probability of each label for one clip. `features` is what
   * clipFeatures returns: 32 × 184 values. Returns Float32Array(labels.length).
   */
  predict(features) {
    const mean = this.#mean;
    const std = this.#std;
    if (features.length !== mean.length) {
      throw new RangeError(`expected ${mean.length} features, got ${features.length}`);
    }
    let activations = new Float64Array(mean.length);
    for (let i = 0; i < mean.length; i++) activations[i] = (features[i] - mean[i]) / std[i];

    const last = this.#layers.length - 1;
    this.#layers.forEach(({ weights, biases }, layer) => {
      const outputs = biases.length;
      const next = Float64Array.from(biases);
      for (let i = 0; i < activations.length; i++) {
        const value = activations[i];
        if (value === 0) continue;
        const row = i * outputs;
        for (let j = 0; j < outputs; j++) next[j] += value * weights[row + j];
      }
      if (layer < last) {
        for (let j = 0; j < outputs; j++) if (next[j] < 0) next[j] = 0;
      }
      activations = next;
    });

    // Softmax, shifted by the largest value so that exp cannot overflow.
    let largest = -Infinity;
    for (const value of activations) if (value > largest) largest = value;
    let total = 0;
    for (let j = 0; j < activations.length; j++) {
      activations[j] = Math.exp(activations[j] - largest);
      total += activations[j];
    }
    const probs = new Float32Array(activations.length);
    for (let j = 0; j < probs.length; j++) probs[j] = activations[j] / total;
    return probs;
  }
}
