// Body and hand landmark tracker: a thin wrapper around MediaPipe HolisticLandmarker.
//
// Everything is loaded from web/vendor/mediapipe/, so it works with no network.
// The video frame handed to detect() must be the unmirrored camera image: "left"
// then means the signer's anatomical left hand, as in the training data.

import { FilesetResolver, HolisticLandmarker } from "../vendor/mediapipe/vision_bundle.mjs";
import { fromHolistic } from "./landmarks.js";

// Resolved against this file, so the paths hold whichever page imports the tracker.
const WASM_PATH = new URL("../vendor/mediapipe/wasm", import.meta.url).href;
const MODEL_PATH = new URL("../vendor/mediapipe/holistic_landmarker.task", import.meta.url).href;

const pairs = (connections) => connections.map(({ start, end }) => [start, end]);

function firstOrNull(lists, length) {
  const points = lists && lists[0];
  return points && points.length === length ? points : null;
}

function blankCanvas(width, height) {
  return typeof OffscreenCanvas === "undefined"
    ? Object.assign(document.createElement("canvas"), { width, height })
    : new OffscreenCanvas(width, height);
}

// MediaPipe hands every video frame to its graph as a WebGL texture, with the CPU
// delegate too, so without WebGL it cannot run at all. When WebGL is drawn in
// software (no usable graphics card), the GPU delegate still works but is several
// times slower than the CPU delegate, so the CPU one should be tried first.
function probeWebGL() {
  try {
    const canvas = blankCanvas(1, 1);
    const gl = canvas.getContext("webgl2") || canvas.getContext("webgl");
    if (!gl) return { available: false, software: false };
    const info = gl.getExtension("WEBGL_debug_renderer_info");
    const renderer = String(gl.getParameter(info ? info.UNMASKED_RENDERER_WEBGL : gl.RENDERER));
    gl.getExtension("WEBGL_lose_context")?.loseContext();
    return { available: true, software: /swiftshader|llvmpipe|softpipe|software|basic render/i.test(renderer) };
  } catch {
    return { available: false, software: false };
  }
}

export class Tracker {
  /** Index pairs for drawing the skeleton from `pose` (33 points) and a hand (21 points). */
  static POSE_CONNECTIONS = pairs(HolisticLandmarker.POSE_CONNECTIONS);
  static HAND_CONNECTIONS = pairs(HolisticLandmarker.HAND_CONNECTIONS);

  /**
   * @param {object} [options]
   * @param {string} [options.wasmPath]   folder holding vision_wasm_internal.{js,wasm}, no trailing slash
   * @param {string} [options.modelPath]  holistic_landmarker.task
   * @param {"GPU"|"CPU"} [options.delegate]  force one delegate; default tries GPU, then CPU
   *                                          (CPU first when WebGL is software-rendered)
   * @param {number} [options.minPoseDetectionConfidence]
   * @param {number} [options.minPosePresenceConfidence]
   * @param {number} [options.minHandLandmarksConfidence]
   */
  static async create(options = {}) {
    const { wasmPath = WASM_PATH, modelPath = MODEL_PATH, delegate, ...thresholds } = options;
    const webgl = probeWebGL();
    if (!webgl.available) {
      throw new Error("Tracker could not start: this browser has no WebGL, which MediaPipe needs to read video frames (with the CPU delegate too).");
    }
    const fileset = await FilesetResolver.forVisionTasks(wasmPath.replace(/\/+$/, ""));
    const delegates = delegate ? [delegate] : webgl.software ? ["CPU", "GPU"] : ["GPU", "CPU"];
    const errors = [];
    for (const choice of delegates) {
      let landmarker = null;
      try {
        landmarker = await HolisticLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: modelPath, delegate: choice },
          runningMode: "VIDEO",
          ...thresholds,
        });
        const tracker = new Tracker(landmarker, choice);
        tracker._warmUp(); // a broken GPU delegate usually fails on the first frame, not on creation
        return tracker;
      } catch (error) {
        errors.push(`${choice}: ${error?.message ?? error}`);
        try { landmarker?.close(); } catch { /* already dead */ }
      }
    }
    throw new Error(`Tracker could not start (${errors.join("; ")})`);
  }

  constructor(landmarker, delegate) {
    this._landmarker = landmarker;
    this._lastStamp = -1;
    this._offset = 0;
    /** "GPU" or "CPU": the delegate that is actually running. */
    this.delegate = delegate;
  }

  /**
   * Run the tracker on the current frame of a video (or any canvas image source).
   * Synchronous: it blocks until the landmarks are ready.
   *
   * @param {HTMLVideoElement|HTMLCanvasElement|ImageBitmap|VideoFrame} video
   * @param {number} timestampMs  frame time in milliseconds, e.g. performance.now()
   * @returns {{frame: Float32Array, pose: object[]|null, left: object[]|null, right: object[]|null}}
   */
  detect(video, timestampMs) {
    if (!this._landmarker) throw new Error("Tracker is closed");
    if (video.videoWidth === 0 || video.readyState < 2) {
      return { frame: fromHolistic({ pose: null, left: null, right: null }), pose: null, left: null, right: null };
    }
    const result = this._landmarker.detectForVideo(video, this._stamp(timestampMs));
    const pose = firstOrNull(result.poseLandmarks, 33);
    const left = firstOrNull(result.leftHandLandmarks, 21);
    const right = firstOrNull(result.rightHandLandmarks, 21);
    return { frame: fromHolistic({ pose, left, right }), pose, left, right };
  }

  close() {
    this._landmarker?.close();
    this._landmarker = null;
  }

  // MediaPipe rejects a timestamp that is not greater than the previous one. When the
  // clock restarts (a new reference clip after the camera, a replay), shift the new
  // timeline to continue after the old one; spacing between frames is kept.
  _stamp(timestampMs) {
    const t = Number.isFinite(timestampMs) ? timestampMs : performance.now();
    if (t + this._offset <= this._lastStamp) this._offset = this._lastStamp + 1 - t;
    this._lastStamp = t + this._offset;
    return this._lastStamp;
  }

  _warmUp() {
    const blank = blankCanvas(64, 64);
    blank.getContext("2d").fillRect(0, 0, 64, 64);
    this._landmarker.detectForVideo(blank, this._stamp(0));
  }
}
