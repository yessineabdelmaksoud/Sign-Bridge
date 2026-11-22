/**
 * Sign recognition, narrowed to the answers that make sense right now.
 * Port of `narrow` and `SignRecognizer` in Demo/consultation/recognizer.py.
 *
 * Asking the model "which of these four?" is far more reliable than "which of
 * all 22?", so every prediction is renormalised over the allowed labels plus
 * OTHER.
 */
import { NoHandError, clipFeatures } from "./features.js";
import { DenseModel } from "./model.js";

export const OTHER = "OTHER";

/** A new object each time, so a caller that changes one does not change the next. */
const noPrediction = () => ({ label: null, confidence: 0, top3: [] });

/**
 * Renormalise `probs` over `allowed` plus OTHER and pick the winner.
 *
 * Returns { label, confidence, top3 }. `label` is null when OTHER wins or the
 * best sign is below `threshold`; `confidence` is the probability of the best
 * allowed sign after narrowing; `top3` is [[label, probability], ...], allowed
 * signs only, best first.
 */
export function narrow(probs, labels, allowed, threshold) {
  const wanted = new Set(allowed);
  const kept = [];
  let total = 0;
  labels.forEach((label, i) => {
    if (label !== OTHER && !wanted.has(label)) return;
    kept.push([label, probs[i]]);
    total += probs[i];
  });
  // Python tests `total <= 0`. Written this way a NaN total is no prediction too.
  if (!(total > 0) || kept.every(([label]) => label === OTHER)) return noPrediction();

  // Array.prototype.sort is stable, like Python's sorted: ties keep label order.
  const ranked = kept.map(([label, p]) => [label, p / total]).sort((a, b) => b[1] - a[1]);
  const top3 = ranked.filter(([label]) => label !== OTHER).slice(0, 3);
  const [bestLabel, best] = ranked[0];
  if (bestLabel === OTHER || best < threshold) return { label: null, confidence: top3[0][1], top3 };
  return { label: bestLabel, confidence: best, top3 };
}

/** Moving signs: one clip in, one prediction out. */
export class Recognizer {
  labels;      // every label, OTHER included
  threshold;
  weakLabels;  // signs the model is not trusted on: never proposed
  #model;

  /** `model` is a DenseModel, or anything with its labels, threshold, weakLabels and predict. */
  constructor(model) {
    this.#model = model;
    this.labels = model.labels;
    this.threshold = model.threshold;
    this.weakLabels = model.weakLabels ?? [];
  }

  static async load(manifestUrl = "models/health_signs.json") {
    return new Recognizer(await DenseModel.load(manifestUrl));
  }

  /**
   * Name the sign in `clip` (an array of canonical frames, rest frames
   * allowed), choosing among `allowed`. Without `allowed` every sign is a
   * candidate. Weak labels are dropped either way. A clip with no hand
   * gives { label: null, confidence: 0, top3: [] }.
   */
  predict(clip, allowed = null) {
    const candidates = [...(allowed ?? this.labels)].filter(
      (label) => label !== OTHER && !this.weakLabels.includes(label),
    );
    let features;
    try {
      features = clipFeatures(clip);
    } catch (error) {
      if (error instanceof NoHandError) return noPrediction();
      throw error;
    }
    return narrow(this.#model.predict(features), this.labels, candidates, this.threshold);
  }
}
