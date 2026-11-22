/**
 * Cut one sign out of a stream of landmark frames.
 * Port of Demo/consultation/capture.py.
 *
 * A sign starts when a hand has been visible for a few frames in a row and
 * ends when the hands have been gone for a moment, or after a hard time limit.
 */
import { handPresent } from "./landmarks.js";

export class SignCapture {
  static START_FRAMES = 3;    // consecutive frames with a hand before recording starts
  static END_GAP_FRAMES = 8;  // consecutive frames without a hand that end the sign
  static MAX_SECONDS = 4;     // hard limit, measured from the first recorded frame
  static MIN_FRAMES = 8;      // anything shorter is a blip, not a sign

  recording = false;
  #lead = [];      // frames seen before recording starts
  #leadStart = 0;  // time of the first of them
  #frames = [];
  #gap = 0;
  #start = 0;

  reset() {
    this.recording = false;
    this.#lead = [];
    this.#frames = [];
    this.#gap = 0;
    this.#start = 0;
  }

  /**
   * Add one frame, seen at `tSeconds`. Returns the finished clip (an array of
   * frames) once, otherwise null. The frame is copied, so the caller may
   * reuse its buffer for the next one.
   */
  push(frame, tSeconds) {
    const limits = this.constructor;
    const present = handPresent(frame);
    if (!this.recording) {
      if (!present) {
        this.#lead = [];
        return null;
      }
      if (this.#lead.length === 0) this.#leadStart = tSeconds;
      this.#lead.push(Float32Array.from(frame));
      if (this.#lead.length >= limits.START_FRAMES) {
        this.recording = true;
        this.#start = this.#leadStart;
        this.#frames = this.#lead;
        this.#lead = [];
        this.#gap = 0;
      }
      return null;
    }

    this.#frames.push(Float32Array.from(frame));
    this.#gap = present ? 0 : this.#gap + 1;
    if (this.#gap >= limits.END_GAP_FRAMES || tSeconds - this.#start >= limits.MAX_SECONDS) {
      return this.#finish();
    }
    return null;
  }

  #finish() {
    const frames = this.#frames.slice(0, this.#frames.length - this.#gap);
    this.reset();
    return frames.length < this.constructor.MIN_FRAMES ? null : frames;
  }
}
