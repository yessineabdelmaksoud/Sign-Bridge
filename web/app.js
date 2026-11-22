// SignBridge interface.
//
// Plain ES module, no dependency, no static import: every other module is
// loaded with import() so that a missing or broken file becomes a visible state
// in the stage instead of a blank page.
//
// Pipeline: video frame -> Tracker.detect -> skeleton on the canvas ->
// SignCapture.push -> (finished clip) -> Recognizer.predict -> result panel.
// The camera and the reference videos go through exactly the same loop.

window.__signbridgeBooted = true;

const MOCK = new URLSearchParams(location.search).get("mock") === "1";

const SIGNS_URL = "data/signs.json";
const MODEL_URL = "models/health_signs.json";
const ALL = "all";
const HISTORY_SIZE = 6;
const FLUSH_FRAMES = 12;        // empty frames pushed after a reference video ends
const FLUSH_STEP = 1 / 30;      // seconds between two of them
const CLIP_HOLD_MS = 1200;      // the last image of a reference video stays this long
const CLIP_FRAME_MS = 50;        // the reference videos run at 20 frames a second
const CLIP_RATE_MIN = 0.1;       // slowest playback of a reference video
const CLIP_RATE_START = 0.5;     // first guess, before the tracker's speed is known
const DEFAULT_THRESHOLD = 0.4;
const DETECT_FAILURES_MAX = 5;

const WHITE = "#FFFFFF";
const INK = "#10233A";
const BLUE = "#12508F";
const GREEN = "#0F8A5F";
const AMBER = "#E8A100";
const WALL = "#EEF2F6";

// Upper body: shoulders 11 and 12, elbows 13 and 14, wrists 15 and 16.
const POSE_LINES = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16]];
const POSE_JOINTS = [11, 12, 13, 14, 15, 16];
// The standard 21-point hand.
const HAND_LINES = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

// signs.json gives each group an id and a pictogram; the names live here.
const GROUP_NAMES = {
  [ALL]: { ar: "الكل", fr: "Tous" },
  reason: { ar: "السبب", fr: "Motif" },
  symptom: { ar: "الأعراض", fr: "Symptômes" },
  conditions: { ar: "أمراض معروفة", fr: "Antécédents" },
};

const TEXT = {
  idle: { ar: "في انتظار إشارة", fr: "En attente d'un signe" },
  again: { ar: "أعد الإشارة", fr: "Refaites le signe" },
};

const HAND_LABELS = {
  none: "Aucune main visible",
  seen: "Main visible",
  recording: "Signe en cours",
};

// One short sentence each: what failed, and what to do.
const ERRORS = {
  signs: "La liste des signes n'a pas pu être chargée : rechargez la page.",
  module: "Un fichier de l'application est introuvable : rechargez la page.",
  model: "Le modèle de reconnaissance n'a pas pu être chargé : rechargez la page.",
  tracker: "Le suivi des mains n'a pas démarré : rechargez la page ou changez de navigateur.",
  "tracker-run": "Le suivi des mains s'est arrêté : rechargez la page.",
  camera: "La caméra n'a pas démarré : réessayez.",
  "camera-denied": "La caméra est bloquée : autorisez la caméra dans le navigateur, puis réessayez.",
  "camera-missing": "Aucune caméra trouvée : branchez une caméra, ou touchez un signe ci-dessous.",
  "camera-busy": "La caméra est prise par une autre application : fermez cette application, puis réessayez.",
  "camera-unavailable": "La caméra n'est pas accessible ici : ouvrez la page en http://localhost ou en https.",
  "camera-lost": "La caméra s'est arrêtée : rebranchez la caméra, puis réessayez.",
  clip: "La vidéo de ce signe n'a pas pu être lue : essayez un autre signe.",
};

class StageError extends Error {
  constructor(kind, cause) {
    super(kind);
    this.name = "StageError";
    this.kind = kind;
    this.cause = cause;
  }
}

const SVG_NS = "http://www.w3.org/2000/svg";
const $ = (selector) => document.querySelector(selector);
const clamp01 = (value) => (Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 0);
const percent = (p) => Math.round(clamp01(p) * 100);

const dom = {
  app: $("#app"),
  stage: $("#stage"),
  cam: $("#cam"),
  clip: $("#clip"),
  overlay: $("#overlay"),
  status: $("#status"),
  clipTag: $("#clip-tag"),
  clipEmoji: $("#clip-emoji"),
  clipName: $("#clip-name"),
  mockTag: $("#mock-tag"),
  steps: $("#steps"),
  errorText: $("#error-text"),
  retry: $("#btn-retry"),
  retryGlyph: $("#retry-glyph"),
  start: $("#btn-start"),
  camera: $("#btn-camera"),
  sound: $("#btn-sound"),
  result: $("#result"),
  resEmoji: $("#res-emoji"),
  resGlyph: $("#res-glyph"),
  resAr: $("#res-ar"),
  resFr: $("#res-fr"),
  resMeter: $("#res-meter"),
  resNum: $("#res-num"),
  others: $("#others"),
  history: $("#history"),
  announce: $("#announce"),
  filters: $("#filters"),
  gallery: $("#gallery"),
};

const state = {
  signs: [],
  groups: [],
  byCode: new Map(),
  tiles: new Map(),          // code -> gallery button
  filter: ALL,

  enginePromise: null,
  engine: null,              // { tracker, recognizer, SignCapture, emptyFrame, mock }
  capture: null,
  threshold: DEFAULT_THRESHOLD,

  source: "none",            // "none" | "camera" | "clip"
  cameraOn: false,
  stream: null,
  cameraRequest: 0,
  loopToken: 0,
  detectFailures: 0,
  clipRun: null,             // the reference video being played, if any

  soundOn: true,
  result: null,              // { label, confidence, top3: [{code, p}], corrected }
  history: [],               // codes, newest first
  retry: null,
};

/* ------------------------------------------------------------------ */
/* Small DOM helpers                                                   */
/* ------------------------------------------------------------------ */

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attrs)) {
    if (value === false || value == null) continue;
    if (name === "class") node.className = value;
    else node.setAttribute(name, value === true ? "" : String(value));
  }
  for (const child of children) {
    if (child != null) node.append(child);
  }
  return node;
}

function icon(name) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("class", "icon");
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS(SVG_NS, "use");
  use.setAttribute("href", `#i-${name}`);
  svg.append(use);
  return svg;
}

const emoji = (char) => el("span", { class: "emoji", "aria-hidden": "true" }, char);

/** Arabic above French. */
function bilingual(ar, fr) {
  return el("span", { class: "bi" },
    el("span", { class: "bi__ar ar", lang: "ar", dir: "rtl" }, ar),
    el("span", { class: "bi__fr" }, fr));
}

function groupName(group) {
  const known = GROUP_NAMES[group.id] ?? {};
  return { ar: group.ar ?? known.ar ?? group.id, fr: group.fr ?? known.fr ?? group.id };
}

/* ------------------------------------------------------------------ */
/* Stage states                                                        */
/* ------------------------------------------------------------------ */

function setView(view) {
  dom.stage.dataset.view = view;
}

function setHand(hand) {
  if (dom.stage.dataset.hand === hand) return;
  dom.stage.dataset.hand = hand;
  dom.status.setAttribute("aria-label", HAND_LABELS[hand]);
}

function markStep(index) {
  dom.steps.children[index]?.classList.add("is-done");
}

/**
 * Show a failure in the stage: a warning pictogram, one French sentence and one
 * button. The technical detail goes to the console.
 */
function showError(kind, detail, retry = null) {
  if (detail !== undefined) console.error(`[SignBridge] ${kind}:`, detail);
  abandonClip();   // a failure takes the stage: no reference video keeps running behind it
  stopLoop();
  clearOverlay();
  setHand("none");
  dom.errorText.textContent = ERRORS[kind] ?? ERRORS.module;
  state.retry = retry?.run ?? (() => location.reload());
  dom.retryGlyph.setAttribute("href", `#i-${retry?.glyph ?? "again"}`);
  dom.retry.setAttribute("aria-label", retry?.label ?? "Recharger la page");
  setView("error");
}

/* ------------------------------------------------------------------ */
/* Loading: signs, modules, model, tracker                             */
/* ------------------------------------------------------------------ */

async function loadSigns() {
  let data;
  try {
    const response = await fetch(SIGNS_URL, { cache: "no-cache" });
    if (!response.ok) throw new Error(`${SIGNS_URL}: HTTP ${response.status}`);
    data = await response.json();
    if (!Array.isArray(data?.signs) || data.signs.length === 0) throw new Error(`${SIGNS_URL}: no signs in the file`);
  } catch (error) {
    if (!MOCK) throw error;
    console.info("[SignBridge mock] data/signs.json is not available; using the built-in list.", error);
    data = MOCK_SIGNS;
  }
  state.signs = data.signs.filter((sign) => sign && typeof sign.code === "string");
  state.byCode = new Map(state.signs.map((sign) => [sign.code, sign]));
  const used = [...new Set(state.signs.map((sign) => sign.group))];
  const listed = (Array.isArray(data.groups) ? data.groups : []).filter((group) => group && used.includes(group.id));
  const unlisted = used.filter((id) => !listed.some((group) => group.id === id)).map((id) => ({ id, icon: "" }));
  state.groups = [...listed, ...unlisted];
}

async function importModule(path) {
  try {
    return await import(path);
  } catch (error) {
    throw new StageError("module", new Error(`${path}: ${error?.message ?? error}`, { cause: error }));
  }
}

async function loadEngine() {
  if (MOCK) return loadMockEngine();

  const [captureModule, paceModule, landmarksModule, trackerModule, recognizerModule] = await Promise.all([
    importModule("./lib/capture.js"),
    importModule("./lib/pace.js"),
    importModule("./lib/landmarks.js"),
    importModule("./lib/tracker.js"),
    importModule("./lib/recognizer.js"),
  ]);
  const { SignCapture } = captureModule;
  const { pacedCapture } = paceModule;
  const { emptyFrame } = landmarksModule;
  const { Tracker } = trackerModule;
  const { Recognizer } = recognizerModule;
  if (typeof SignCapture !== "function" || typeof emptyFrame !== "function"
    || typeof Tracker?.create !== "function" || typeof Recognizer?.load !== "function") {
    throw new StageError("module", new Error("a module does not export what CONTRACT.md lists"));
  }
  markStep(0);

  const step = (index, kind, start) => Promise.resolve().then(start).then(
    (value) => { markStep(index); return value; },
    (error) => { throw new StageError(kind, error); });
  const [recognizer, tracker] = await Promise.all([
    step(1, "model", () => Recognizer.load(MODEL_URL)),
    step(2, "tracker", () => Tracker.create({})),
  ]);
  // The capture follows the camera's frame rate, so a fast device does not cut signs short.
  return { tracker, recognizer, SignCapture, newCapture: pacedCapture, emptyFrame, mock: false };
}

/** The engine, or null after showing why it is not there. */
async function ensureEngine() {
  try {
    return await state.enginePromise;
  } catch (error) {
    showError(error?.kind ?? "module");   // already logged by boot()
    return null;
  }
}

/* ------------------------------------------------------------------ */
/* Filter: which signs the model may answer with                       */
/* ------------------------------------------------------------------ */

function allowedCodes() {
  return state.signs
    .filter((sign) => !sign.weak && (state.filter === ALL || sign.group === state.filter))
    .map((sign) => sign.code);
}

/** The `allowed` argument of Recognizer.predict: nothing means every sign. */
function allowedArgument() {
  return state.filter === ALL ? undefined : allowedCodes();
}

function setFilter(id) {
  state.filter = id;
  for (const input of dom.filters.querySelectorAll("input")) input.checked = input.value === id;
  for (const section of dom.gallery.querySelectorAll(".group")) {
    section.toggleAttribute("data-out", id !== ALL && section.dataset.group !== id);
  }
}

function renderFilters() {
  const options = [{ id: ALL, icon: "" }, ...state.groups];
  const chips = options.map((group) => {
    const name = groupName(group);
    const input = el("input", { type: "radio", name: "filter", value: group.id, "aria-label": name.fr, checked: group.id === state.filter });
    input.checked = group.id === state.filter;
    return el("label", { class: "chip" },
      input,
      el("span", { class: "chip__face" },
        el("span", { class: "chip__picto" }, group.icon ? emoji(group.icon) : icon("grid")),
        bilingual(name.ar, name.fr)));
  });
  dom.filters.append(...chips);
  dom.filters.addEventListener("change", (event) => {
    if (event.target?.name === "filter") setFilter(event.target.value);
  });
}

/* ------------------------------------------------------------------ */
/* Gallery                                                             */
/* ------------------------------------------------------------------ */

function tileLabel(sign, verdict) {
  const outcome = verdict === "ok" ? ", reconnu par l'IA" : verdict === "no" ? ", non reconnu par l'IA" : "";
  return `Montrer le signe : ${sign.fr}${outcome}`;
}

function renderGallery() {
  const columns = [];
  const sections = state.groups.map((group) => {
    const signs = state.signs.filter((sign) => sign.group === group.id);
    const name = groupName(group);
    const cols = Math.max(2, Math.ceil(signs.length / 2));
    columns.push(`minmax(0, ${cols}fr)`);
    const tiles = signs.map((sign) => {
      const tile = el("button", {
        type: "button", class: "tile", "data-code": sign.code, "data-weak": Boolean(sign.weak),
        "aria-pressed": "false", "aria-label": tileLabel(sign),
      },
        emoji(sign.icon),
        el("span", { class: "tile__ar ar", lang: "ar", dir: "rtl" }, sign.ar),
        el("span", { class: "tile__fr" }, sign.fr),
        el("span", { class: "tile__mark tile__mark--play", "aria-hidden": "true" }, icon("play")),
        el("span", { class: "tile__mark tile__mark--ok", "aria-hidden": "true" }, icon("check")),
        el("span", { class: "tile__mark tile__mark--no", "aria-hidden": "true" }, icon("cross")));
      state.tiles.set(sign.code, tile);
      return tile;
    });
    const section = el("section", { class: "group", "data-group": group.id },
      el("h2", { class: "group__head" },
        group.icon ? emoji(group.icon) : icon("grid"),
        bilingual(name.ar, name.fr)),
      el("div", { class: "group__tiles" }, ...tiles));
    section.style.setProperty("--cols", cols);
    return section;
  });
  dom.gallery.style.setProperty("--gallery-cols", columns.join(" "));
  dom.gallery.replaceChildren(...sections);
  dom.gallery.addEventListener("click", (event) => {
    const tile = event.target.closest?.(".tile");
    if (tile) playClip(tile.dataset.code);
  });
}

function setVerdict(code, verdict) {
  const tile = state.tiles.get(code);
  const sign = state.byCode.get(code);
  if (!tile || !sign) return;
  if (verdict) tile.dataset.verdict = verdict;
  else delete tile.dataset.verdict;
  tile.setAttribute("aria-label", tileLabel(sign, verdict));
}

/* ------------------------------------------------------------------ */
/* Result panel                                                        */
/* ------------------------------------------------------------------ */

function normalisePrediction(prediction) {
  const top3 = (Array.isArray(prediction?.top3) ? prediction.top3 : [])
    .map((entry) => ({ code: entry?.[0], p: clamp01(entry?.[1]) }))
    .filter((candidate) => state.byCode.has(candidate.code));
  const label = state.byCode.has(prediction?.label) ? prediction.label : null;
  return { label, confidence: clamp01(prediction?.confidence), top3, corrected: false };
}

function renderResult() {
  const result = state.result;
  const sign = result?.label ? state.byCode.get(result.label) : null;
  const view = !result ? "idle" : !sign ? "unsure" : result.corrected ? "corrected" : "recognised";
  dom.result.dataset.state = view;

  dom.resEmoji.hidden = !sign;
  dom.resEmoji.textContent = sign ? sign.icon : "";
  dom.resGlyph.setAttribute("href", view === "unsure" ? "#i-sign-again" : "#i-hand");
  const words = sign ?? (view === "unsure" ? TEXT.again : TEXT.idle);
  dom.resAr.textContent = words.ar;
  dom.resFr.textContent = words.fr;

  const value = result ? percent(result.confidence) : 0;
  dom.resMeter.style.setProperty("--value", value);
  dom.resMeter.style.setProperty("--threshold", percent(state.threshold));
  dom.resMeter.setAttribute("aria-valuenow", value);
  dom.resNum.textContent = result && (sign || result.top3.length) ? `${value} %` : "";

  const candidates = (result?.top3 ?? [])
    .filter((candidate) => candidate.code !== result.label)
    .slice(0, 2);
  const buttons = candidates.map((candidate) => {
    const other = state.byCode.get(candidate.code);
    const bar = el("span", { class: "alt__bar", "aria-hidden": "true" });
    bar.style.setProperty("--value", percent(candidate.p));
    return el("button", { type: "button", class: "alt", "data-code": other.code, "aria-label": `Corriger : ${other.fr}` },
      emoji(other.icon), bilingual(other.ar, other.fr), bar);
  });
  for (const old of dom.others.querySelectorAll(".alt")) old.remove();
  dom.others.append(...buttons);
  dom.others.classList.toggle("is-empty", buttons.length === 0);
}

function renderHistory() {
  const slots = [el("li", { class: "history__lead", "aria-hidden": "true" }, icon("history"))];
  for (let i = 0; i < HISTORY_SIZE; i++) {
    const sign = state.byCode.get(state.history[i]);
    if (sign) {
      slots.push(el("li", { class: `history__slot${i === 0 ? " history__slot--latest" : ""}`, title: sign.fr },
        emoji(sign.icon), el("span", { class: "vh" }, sign.fr)));
    } else {
      slots.push(el("li", { class: "history__slot history__slot--empty", "aria-hidden": "true" }));
    }
  }
  dom.history.replaceChildren(...slots);
}

function announce(message) {
  dom.announce.textContent = message;
}

/** Say the French name aloud. Never in the way: no voice, or any failure, means silence. */
function speak(text) {
  if (!state.soundOn) return;
  try {
    const synth = window.speechSynthesis;
    if (!synth || typeof window.SpeechSynthesisUtterance !== "function") return;
    const voices = synth.getVoices();
    if (!voices.length) return;   // no voice on this device: stay silent
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "fr-FR";
    const french = voices.find((voice) => /^fr\b/i.test(voice.lang));
    if (french) utterance.voice = french;
    synth.cancel();
    synth.speak(utterance);
  } catch (error) {
    console.warn("[SignBridge] speech is not available:", error);
  }
}

/** A prediction from the model. `run` is the reference video it belongs to, if any. */
function showPrediction(prediction, run = null) {
  const result = normalisePrediction(prediction);
  state.result = result;
  const sign = result.label ? state.byCode.get(result.label) : null;
  if (sign) state.history = [sign.code, ...state.history].slice(0, HISTORY_SIZE);
  if (run) {
    run.predictions += 1;
    if (result.label === run.code) run.matched = true;
  }
  renderResult();
  renderHistory();
  if (sign) {
    announce(`Signe reconnu : ${sign.fr}, ${percent(result.confidence)} %.`);
    speak(sign.fr);
  } else {
    announce("Signe non reconnu. Refaites le signe.");
  }
}

/** The user taps another candidate: their choice replaces the model's. */
function correct(code) {
  const previous = state.result;
  const sign = state.byCode.get(code);
  if (!previous || !sign) return;
  const candidate = previous.top3.find((entry) => entry.code === code);
  state.result = { ...previous, label: code, confidence: candidate?.p ?? 0, corrected: true };
  // A result that already had a sign owns the newest history entry: replace it.
  state.history = previous.label ? [code, ...state.history.slice(1)] : [code, ...state.history].slice(0, HISTORY_SIZE);
  renderResult();
  renderHistory();
  announce(`Corrigé : ${sign.fr}.`);
  speak(sign.fr);
  dom.others.querySelector(".alt")?.focus({ preventScroll: true });
}

/* ------------------------------------------------------------------ */
/* Skeleton                                                            */
/* ------------------------------------------------------------------ */

const overlayContext = dom.overlay.getContext("2d");

function clearOverlay() {
  overlayContext.clearRect(0, 0, dom.overlay.width, dom.overlay.height);
}

const usable = (point) => point && Number.isFinite(point.x) && Number.isFinite(point.y);

function strokeLines(g, points, lines, w, h, width, colour) {
  g.lineWidth = width;
  g.strokeStyle = colour;
  g.beginPath();
  for (const [a, b] of lines) {
    const from = points[a];
    const to = points[b];
    if (!usable(from) || !usable(to)) continue;
    g.moveTo(from.x * w, from.y * h);
    g.lineTo(to.x * w, to.y * h);
  }
  g.stroke();
}

function fillJoints(g, points, indexes, w, h, radius, colour) {
  g.fillStyle = colour;
  g.strokeStyle = INK;
  g.lineWidth = Math.max(1, radius * 0.45);
  for (const index of indexes) {
    const point = points[index];
    if (!usable(point)) continue;
    g.beginPath();
    g.arc(point.x * w, point.y * h, radius, 0, Math.PI * 2);
    g.fill();
    g.stroke();
  }
}

const HAND_JOINTS = Array.from({ length: 21 }, (_, index) => index);

/**
 * Landmarks are in unmirrored image coordinates, so they are drawn as they
 * come: the canvas is mirrored in CSS together with the video.
 */
function drawSkeleton({ pose, left, right }, video) {
  const w = video.videoWidth;
  const h = video.videoHeight;
  if (!w || !h) return;
  const canvas = dom.overlay;
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  const g = overlayContext;
  g.clearRect(0, 0, w, h);
  g.lineCap = "round";
  g.lineJoin = "round";
  const unit = Math.max(w, h) / 320;   // 2 px on a 640 px image

  if (pose) {
    strokeLines(g, pose, POSE_LINES, w, h, unit * 4, INK);
    strokeLines(g, pose, POSE_LINES, w, h, unit * 2, WHITE);
    fillJoints(g, pose, POSE_JOINTS, w, h, unit * 2.4, WHITE);
  }
  // Hands take the state colour: green when seen, amber while a sign is recorded.
  const handColour = state.capture?.recording ? AMBER : GREEN;
  for (const hand of [left, right]) {
    if (!hand) continue;
    strokeLines(g, hand, HAND_LINES, w, h, unit * 3.2, INK);
    strokeLines(g, hand, HAND_LINES, w, h, unit * 1.6, handColour);
    fillJoints(g, hand, HAND_JOINTS, w, h, unit * 1.7, handColour);
  }
}

/* ------------------------------------------------------------------ */
/* Frame loop                                                          */
/* ------------------------------------------------------------------ */

function frameHasHand(frame) {
  for (let i = 33 * 3; i < frame.length; i++) {
    if (Number.isFinite(frame[i])) return true;
  }
  return false;
}

/** One landmark frame into the capture; a finished clip goes to the model. */
function feed(frame, seconds, run) {
  const clip = state.capture.push(frame, seconds);
  setHand(state.capture.recording ? "recording" : frameHasHand(frame) ? "seen" : "none");
  if (clip) recognise(clip, run);
}

function recognise(clip, run) {
  let pending;
  try {
    pending = Promise.resolve(state.engine.recognizer.predict(clip, allowedArgument()));
  } catch (error) {
    pending = Promise.reject(error);
  }
  const shown = pending
    .catch((error) => {
      console.error("[SignBridge] the model could not read this clip:", error);
      return { label: null, confidence: 0, top3: [] };
    })
    .then((prediction) => {
      if (run && state.clipRun !== run) return;   // that reference video was dismissed
      showPrediction(prediction, run);
    })
    .catch((error) => console.error("[SignBridge] the result could not be shown:", error));
  run?.pending.push(shown);
}

function stopLoop() {
  state.loopToken += 1;
}

/**
 * Slow a reference video down to the speed the tracker can follow, so that it
 * sees every frame. Played at full speed on a slow device, most frames would be
 * skipped and the sign would reach the model with holes in it.
 */
function paceClip(video, run, tookMs) {
  run.detectMs = run.detectMs ? 0.7 * run.detectMs + 0.3 * tookMs : tookMs;
  const rate = Math.min(1, Math.max(CLIP_RATE_MIN, CLIP_FRAME_MS / (run.detectMs * 1.25)));
  state.clipRate = rate;
  if (Math.abs(video.playbackRate - rate) > 0.05) video.playbackRate = rate;
}

/**
 * Track `video` frame by frame until the loop is stopped or replaced.
 * `source` is "camera" (clock time) or "clip" (the video's own time).
 */
function startLoop(video, source) {
  const token = ++state.loopToken;
  const run = source === "clip" ? state.clipRun : null;
  const byVideoFrame = typeof video.requestVideoFrameCallback === "function";
  let lastMediaTime = -1;

  const schedule = () => {
    if (token !== state.loopToken) return;
    if (byVideoFrame) video.requestVideoFrameCallback((_now, metadata) => tick(metadata.mediaTime));
    else requestAnimationFrame(() => tick(video.currentTime));
  };

  const tick = async (mediaTime) => {
    if (token !== state.loopToken) return;
    try {
      // Never the same video time twice: the tracker would see one image two times.
      if (video.readyState >= 2 && video.videoWidth > 0 && mediaTime !== lastMediaTime) {
        lastMediaTime = mediaTime;
        const now = performance.now();
        const detection = await state.engine.tracker.detect(video, now);
        if (token !== state.loopToken) return;
        if (run) paceClip(video, run, performance.now() - now);
        state.detectFailures = 0;
        if (detection?.frame) {
          drawSkeleton(detection, video);
          const seconds = source === "clip" ? mediaTime : now / 1000;
          if (run) run.lastSeconds = seconds;
          feed(detection.frame, seconds, run);
        }
      }
    } catch (error) {
      if (token !== state.loopToken) return;
      state.detectFailures += 1;
      console.warn("[SignBridge] the tracker failed on a frame:", error);
      if (state.detectFailures >= DETECT_FAILURES_MAX) {
        showError("tracker-run", error);
        return;
      }
    }
    schedule();
  };

  schedule();
}

/* ------------------------------------------------------------------ */
/* What the stage shows: nothing, the camera, or a reference video     */
/* ------------------------------------------------------------------ */

function showSource(source) {
  state.source = source;
  stopLoop();
  state.capture?.reset();
  clearOverlay();
  setHand("none");
  dom.cam.hidden = source !== "camera";
  dom.clip.hidden = source !== "clip";
  dom.stage.classList.toggle("is-mirrored", source === "camera");   // a mirror for the user; videos stay as filmed
  if (source === "none") {
    setView("idle");
    return;
  }
  setView("live");
  state.engine.tracker.mockScene?.(source);   // MOCK only: restarts the stand-in's script
  startLoop(source === "camera" ? dom.cam : dom.clip, source);
}

function bringStageIntoView() {
  const calm = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  dom.stage.scrollIntoView({ block: "nearest", behavior: calm ? "auto" : "smooth" });
}

function reflectCamera() {
  dom.camera.setAttribute("aria-pressed", String(state.cameraOn));
}

function releaseStream() {
  const stream = state.stream;
  state.stream = null;
  if (!stream) return;
  stream.stopMock?.();   // MOCK only
  for (const track of stream.getTracks()) {
    track.onended = null;
    track.stop();
  }
  dom.cam.srcObject = null;
}

function cameraErrorKind(error) {
  if (error instanceof StageError) return error.kind;
  switch (error?.name) {
    case "NotAllowedError":
    case "SecurityError":
      return "camera-denied";
    case "NotFoundError":
    case "OverconstrainedError":
      return "camera-missing";
    case "NotReadableError":
    case "AbortError":
      return "camera-busy";
    default:
      return "camera";
  }
}

const retryCamera = { run: () => startCamera(), glyph: "again", label: "Réessayer la caméra" };

async function startCamera() {
  const request = ++state.cameraRequest;
  const engine = await ensureEngine();
  if (!engine || request !== state.cameraRequest) return;
  abandonClip();
  stopLoop();
  setView("loading");
  bringStageIntoView();

  let stream;
  try {
    if (engine.mock) {
      stream = mockCameraStream();
    } else {
      if (!navigator.mediaDevices?.getUserMedia) throw new StageError("camera-unavailable");
      stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      });
    }
  } catch (error) {
    if (request !== state.cameraRequest) return;
    state.cameraOn = false;
    reflectCamera();
    showError(cameraErrorKind(error), error, retryCamera);
    return;
  }
  if (request !== state.cameraRequest) {   // switched off, or asked again, while the browser was asking
    stream.stopMock?.();
    for (const track of stream.getTracks()) track.stop();
    return;
  }

  releaseStream();
  state.stream = stream;
  const [track] = stream.getVideoTracks();
  if (track) {
    track.onended = () => {
      if (state.stream !== stream) return;
      stopCamera();
      showError("camera-lost", new Error("the camera track ended"), retryCamera);
    };
  }
  dom.cam.srcObject = stream;
  try {
    await dom.cam.play();
  } catch (error) {
    console.warn("[SignBridge] the camera image did not start playing by itself:", error);
  }
  if (request !== state.cameraRequest) return;
  state.cameraOn = true;
  reflectCamera();
  // A reference video started while the browser was asking: let it finish, the camera comes after.
  if (!state.clipRun) showSource("camera");
}

function stopCamera() {
  state.cameraRequest += 1;
  state.cameraOn = false;
  reflectCamera();
  releaseStream();
  if (state.source === "camera" || dom.stage.dataset.view !== "live") showSource("none");
}

function toggleCamera() {
  if (state.cameraOn) stopCamera();
  else startCamera();
}

/* ------------------------------------------------------------------ */
/* Reference videos                                                    */
/* ------------------------------------------------------------------ */

const dismissClip = { run: () => showSource(state.cameraOn ? "camera" : "none"), glyph: "cross", label: "Fermer" };

/** Stop the reference video without a verdict (the user moved on). */
function abandonClip() {
  const run = state.clipRun;
  if (!run) return;
  state.clipRun = null;
  clearTimeout(run.holdTimer);
  clearTimeout(run.watchdog);
  clearTimeout(run.mockTimer);
  state.tiles.get(run.code)?.setAttribute("aria-pressed", "false");
  dom.clipTag.hidden = true;
  if (state.engine?.mock) state.engine.recognizer.expect = null;
  const video = dom.clip;
  video.pause();
  if (video.srcObject) {
    video.srcObject.stopMock?.();
    video.srcObject = null;
  }
  video.removeAttribute("src");
  video.load();
}

/** Play one sign's reference video in the stage, through the same pipeline as the camera. */
async function playClip(code) {
  const sign = state.byCode.get(code);
  if (!sign) return;
  if (state.clipRun?.code === code) {   // a second tap on the same tile stops it
    abandonClip();
    showSource(state.cameraOn ? "camera" : "none");
    return;
  }
  abandonClip();
  const run = {
    code, matched: false, predictions: 0, pending: [], finished: false,
    lastSeconds: 0, holdTimer: 0, watchdog: 0, mockTimer: 0,
  };
  state.clipRun = run;
  state.tiles.get(code)?.setAttribute("aria-pressed", "true");
  setVerdict(code, null);

  const engine = await ensureEngine();
  if (state.clipRun !== run) return;
  if (!engine) {
    abandonClip();
    return;
  }
  // The test must be fair: a sign outside the current filter could never be answered.
  if (!sign.weak && !allowedCodes().includes(code)) setFilter(ALL);
  if (engine.mock) engine.recognizer.expect = code;

  dom.clipEmoji.textContent = sign.icon;
  dom.clipName.textContent = sign.fr;
  dom.clipTag.hidden = false;
  bringStageIntoView();   // the gallery can be far below the stage, on a phone above all

  const video = dom.clip;
  video.src = sign.clip;
  video.defaultPlaybackRate = state.clipRate ?? CLIP_RATE_START;   // see paceClip
  video.playbackRate = video.defaultPlaybackRate;
  showSource("clip");
  try {
    await video.play();
  } catch (error) {
    if (state.clipRun !== run) return;   // replaced by another tap: not a failure
    if (engine.mock) {
      playMockClip(run);
      return;
    }
    showError("clip", error, dismissClip);
    return;
  }
  if (state.clipRun !== run) return;
  const duration = Number.isFinite(video.duration) ? video.duration : 8;
  run.watchdog = setTimeout(() => finishClip(run), (duration * 1000) / CLIP_RATE_MIN + 4000);   // if "ended" never comes
}

/** The reference video is over: close the sign, give the verdict, then hand the stage back. */
async function finishClip(run) {
  if (state.clipRun !== run || run.finished) return;
  run.finished = true;
  clearTimeout(run.watchdog);
  stopLoop();
  // The video ran out with the hands possibly still up: close the sign in progress.
  if (typeof state.capture.finish === "function") {
    const clip = state.capture.finish(run.lastSeconds);
    if (clip) recognise(clip, run);
  } else {   // the stand-in capture of ?mock=1: empty frames until it lets go
    let seconds = run.lastSeconds;
    for (let i = 0; i < FLUSH_FRAMES; i++) {
      seconds += FLUSH_STEP;
      feed(state.engine.emptyFrame(), seconds, run);
    }
  }
  setHand("none");
  await Promise.all(run.pending);
  if (state.clipRun !== run) return;
  if (run.predictions === 0) showPrediction({ label: null, confidence: 0, top3: [] }, run);   // no sign was seen at all
  setVerdict(run.code, run.matched ? "ok" : "no");
  run.holdTimer = setTimeout(() => {
    if (state.clipRun !== run) return;
    abandonClip();
    showSource(state.cameraOn ? "camera" : "none");
  }, CLIP_HOLD_MS);
}

/* ------------------------------------------------------------------ */
/* Controls                                                            */
/* ------------------------------------------------------------------ */

function bindControls() {
  dom.retry.onclick = null;   // the fallback handler of index.html, if it was installed
  dom.retry.addEventListener("click", () => (state.retry ?? (() => location.reload()))());
  dom.start.addEventListener("click", () => startCamera());
  dom.camera.addEventListener("click", toggleCamera);
  dom.sound.addEventListener("click", () => {
    state.soundOn = !state.soundOn;
    dom.sound.setAttribute("aria-pressed", String(state.soundOn));
    if (!state.soundOn) window.speechSynthesis?.cancel();
  });
  dom.others.addEventListener("click", (event) => {
    const button = event.target.closest?.(".alt");
    if (button) correct(button.dataset.code);
  });

  // The stage takes the camera's shape (within reason), so the image fills it.
  const fitStage = () => {
    const { videoWidth: w, videoHeight: h } = dom.cam;
    if (w && h) dom.stage.style.setProperty("--ratio", Math.min(16 / 9, Math.max(3 / 4, w / h)).toFixed(4));
  };
  dom.cam.addEventListener("loadedmetadata", fitStage);
  dom.cam.addEventListener("resize", fitStage);

  dom.clip.addEventListener("ended", () => {
    if (state.clipRun) finishClip(state.clipRun);
  });
  dom.clip.addEventListener("error", () => {
    const run = state.clipRun;
    if (!run || run.finished || !dom.clip.getAttribute("src")) return;
    if (state.engine?.mock) {
      playMockClip(run);
      return;
    }
    const detail = dom.clip.error ? `${dom.clip.getAttribute("src")}: media error ${dom.clip.error.code} ${dom.clip.error.message}` : "media error";
    showError("clip", new Error(detail), dismissClip);
  });

  // Some browsers fill the voice list late; asking early makes it ready in time.
  try {
    window.speechSynthesis?.getVoices();
  } catch { /* no speech on this device */ }

  // A hidden tab gets no video frames: drop a half-recorded sign rather than
  // finish it later with frames from before and after the pause.
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && state.source === "camera") {
      state.capture?.reset();
      setHand("none");
    }
  });

  window.addEventListener("pagehide", () => {
    stopLoop();
    releaseStream();
    try {
      state.engine?.tracker.close?.();
    } catch { /* closing anyway */ }
  });
}

async function boot() {
  bindControls();
  if (MOCK) dom.mockTag.hidden = false;
  setView("loading");

  try {
    await loadSigns();
  } catch (error) {
    showError("signs", error);
    return;
  }
  renderFilters();
  renderGallery();
  renderResult();
  renderHistory();

  state.enginePromise = loadEngine().then((engine) => {
    state.engine = engine;
    state.capture = engine.newCapture ? engine.newCapture() : new engine.SignCapture();
    state.threshold = Number(engine.recognizer.threshold) || DEFAULT_THRESHOLD;
    renderResult();
    return engine;
  });
  try {
    await state.enginePromise;
  } catch (error) {
    showError(error?.kind ?? "module", error?.cause ?? error);
    return;
  }
  if (state.source === "none" && dom.stage.dataset.view === "loading" && state.cameraRequest === 0) setView("idle");
}

/* ==================================================================== */
/* MOCK STAND-INS: used only with ?mock=1.                              */
/*                                                                      */
/* They replace the tracker, the recogniser and the camera so the       */
/* interface can be exercised with no model, no MediaPipe and no        */
/* camera. Nothing below runs in the real application. The stage shows  */
/* a "Simulation" tag whenever they are in use.                         */
/* ==================================================================== */

// Only used when data/signs.json itself is missing.
const MOCK_SIGNS = {
  other: "OTHER",
  groups: [{ id: "reason", icon: "🏥" }, { id: "symptom", icon: "🩺" }, { id: "conditions", icon: "📋" }],
  signs: [
    ["ENFERMO", "reason", "🤒", "Malade", "مريض"],
    ["CITA", "reason", "📅", "Rendez-vous", "موعد"],
    ["FARMACIA", "reason", "💊", "Médicament", "دواء"],
    ["URGENTE", "reason", "🚨", "Urgent", "حالة مستعجلة"],
    ["DOLOR", "symptom", "😣", "Douleur", "وجيعة"],
    ["DOLOR-DE-CABEZA", "symptom", "🤕", "Mal de tête", "وجيعة راس"],
    ["FIEBRE2", "symptom", "🌡️", "Fièvre", "سخانة"],
    ["TOSER", "symptom", "😷", "Toux", "كحة"],
    ["MAREO", "symptom", "😵", "Vertige", "دوخة"],
    ["VOMITAR", "symptom", "🤮", "Vomissements", "تقيؤ"],
    ["DIARREA", "symptom", "🚽", "Diarrhée", "إسهال"],
    ["RESPIRAR", "symptom", "🫁", "Essoufflement", "ضيق نفس"],
    ["CANSADO2", "symptom", "😴", "Fatigue", "تعب"],
    ["HERIDA", "symptom", "🩹", "Blessure", "جرح"],
    ["ALERGIA2", "symptom", "🤧", "Allergie", "حساسية"],
    ["GRIPE", "symptom", "🦠", "Grippe", "نزلة برد"],
    ["INSOMNIO", "symptom", "🌙", "Insomnie", "قلة نوم"],
    ["ANSIEDAD", "symptom", "😰", "Anxiété", "قلق"],
    ["DIABETES", "conditions", "🩸", "Diabète", "سكري"],
    ["ASMA", "conditions", "🌬️", "Asthme", "ربو"],
    ["CORAZON", "conditions", "❤️", "Maladie du cœur", "مرض القلب"],
    ["EMBARAZO", "conditions", "🤰", "Grossesse", "حمل"],
  ].map(([code, group, icon, fr, ar]) => ({ code, group, icon, fr, ar, clip: `clips/${code}.mp4`, weak: code === "CORAZON" })),
};

// The stand-in signer: hands down, then a sign, then hands down again.
const mockScene = { startMs: performance.now(), mode: "camera" };
const MOCK_PERIOD = 6.5;        // seconds per cycle in front of the "camera"
const MOCK_CLIP_SECONDS = 2.6;  // length of a stand-in reference video

function mockBody(nowMs) {
  const t = Math.max(0, (nowMs - mockScene.startMs) / 1000);
  const live = mockScene.mode !== "clip";
  const phase = live ? t % MOCK_PERIOD : t;
  const [from, to] = live ? [1.4, 4.0] : [0.35, 1e6];   // a reference video: hands stay up until it ends
  const ramp = 0.3;
  const raise = clamp01((phase - from) / ramp) * clamp01((to + ramp - phase) / ramp);
  const leftRaise = live && Math.floor(t / MOCK_PERIOD) % 2 === 1 ? raise : 0;   // every other sign uses both hands
  const sway = (shift) => Math.sin(t * 5 + shift) * 0.035;
  const mix = (a, b, u) => a + (b - a) * u;
  return {
    raise,
    leftRaise,
    nose: { x: 0.5, y: 0.3 + Math.sin(t * 2) * 0.004 },
    rightShoulder: { x: 0.37, y: 0.58 },
    leftShoulder: { x: 0.63, y: 0.58 },
    rightElbow: { x: mix(0.3, 0.25, raise), y: mix(0.8, 0.74, raise) },
    leftElbow: { x: mix(0.7, 0.75, leftRaise), y: mix(0.8, 0.74, leftRaise) },
    rightWrist: { x: mix(0.31, 0.36 + sway(0), raise), y: mix(0.99, 0.5 + sway(1.6), raise) },
    leftWrist: { x: mix(0.69, 0.64 + sway(3.1), leftRaise), y: mix(0.99, 0.5 + sway(4.7), leftRaise) },
  };
}

// An open hand, fingers up, wrist at the origin.
const MOCK_HAND = [
  [0, 0],
  [-0.25, -0.15], [-0.45, -0.35], [-0.58, -0.55], [-0.68, -0.72],
  [-0.22, -0.62], [-0.26, -0.92], [-0.28, -1.1], [-0.29, -1.26],
  [0, -0.66], [0, -1], [0, -1.2], [0, -1.37],
  [0.2, -0.62], [0.23, -0.92], [0.25, -1.1], [0.26, -1.25],
  [0.38, -0.52], [0.44, -0.75], [0.48, -0.9], [0.5, -1.02],
];

const mockHand = (wrist, side) => MOCK_HAND.map(([x, y]) => ({ x: wrist.x + side * x * 0.075, y: wrist.y + y * 0.1, z: 0 }));

function mockEmptyFrame() {
  return new Float32Array(225).fill(NaN);
}

function mockFrame(pose, left, right) {
  const frame = mockEmptyFrame();
  const write = (points, start) => points?.forEach((point, i) => frame.set([point.x, point.y, point.z], (start + i) * 3));
  write(pose, 0);
  write(left, 33);
  write(right, 54);
  return frame;
}

/** Stand-in for lib/tracker.js: same detect() result, drawn from a script instead of the image. */
class MockTracker {
  mockScene(mode) {
    mockScene.mode = mode;
    mockScene.startMs = performance.now();
  }

  detect(_video, timestampMs) {
    const body = mockBody(timestampMs);
    const pose = Array.from({ length: 33 }, () => ({ x: body.nose.x, y: body.nose.y, z: 0 }));
    Object.assign(pose[11], body.leftShoulder);
    Object.assign(pose[12], body.rightShoulder);
    Object.assign(pose[13], body.leftElbow);
    Object.assign(pose[14], body.rightElbow);
    Object.assign(pose[15], body.leftWrist);
    Object.assign(pose[16], body.rightWrist);
    const right = body.raise > 0.55 ? mockHand(body.rightWrist, 1) : null;
    const left = body.leftRaise > 0.55 ? mockHand(body.leftWrist, -1) : null;
    return { frame: mockFrame(pose, left, right), pose, left, right };
  }

  close() {}
}

/**
 * Stand-in for lib/recognizer.js. Answers follow a fixed rota so every state of
 * the result panel comes up: sure, sure, not sure, sure-but-wrong.
 */
class MockRecognizer {
  constructor(signs) {
    this.codes = signs.filter((sign) => !sign.weak).map((sign) => sign.code);
    this.labels = [...signs.map((sign) => sign.code), "OTHER"];
    this.threshold = DEFAULT_THRESHOLD;
    this.expect = null;   // set by the app while a reference video plays
    this.calls = 0;
  }

  predict(clip, allowed) {
    const pool = (allowed ?? this.codes).filter((code) => this.codes.includes(code));
    if (!clip?.length || !pool.length) return { label: null, confidence: 0, top3: [] };
    const turn = this.calls++;
    const pick = (step) => pool[(turn * 5 + step * 3) % pool.length];
    const expected = this.expect && pool.includes(this.expect) ? this.expect : pick(0);
    const mode = turn % 4;
    const best = mode === 3 ? pool.find((code) => code !== expected) ?? expected : expected;
    const confidence = [0.86, 0.64, 0.33, 0.58][mode];
    const others = [...new Set([pick(1), pick(2), expected, ...pool])].filter((code) => code !== best).slice(0, 2);
    const top3 = [[best, confidence]];
    if (others[0]) top3.push([others[0], (1 - confidence) * 0.6]);
    if (others[1]) top3.push([others[1], (1 - confidence) * 0.25]);
    return { label: mode === 2 ? null : best, confidence, top3 };
  }
}

/** Stand-in for lib/capture.js, used only if the real file is not there yet. Same rules. */
class MockSignCapture {
  constructor() {
    this.reset();
  }

  reset() {
    this.recording = false;
    this.lead = [];
    this.frames = [];
    this.gap = 0;
    this.startSeconds = 0;
  }

  push(frame, seconds) {
    const present = frameHasHand(frame);
    if (!this.recording) {
      if (!present) {
        this.lead = [];
        return null;
      }
      this.lead.push([Float32Array.from(frame), seconds]);
      if (this.lead.length >= 3) {
        this.recording = true;
        this.startSeconds = this.lead[0][1];
        this.frames = this.lead.map(([kept]) => kept);
        this.lead = [];
        this.gap = 0;
      }
      return null;
    }
    this.frames.push(Float32Array.from(frame));
    this.gap = present ? 0 : this.gap + 1;
    if (this.gap < 8 && seconds - this.startSeconds < 4) return null;
    const frames = this.frames.slice(0, this.frames.length - this.gap);
    this.reset();
    return frames.length < 8 ? null : frames;
  }
}

/** Stand-in for the camera: a flat drawn figure that follows the same script as MockTracker. */
function mockCameraStream() {
  const canvas = document.createElement("canvas");
  canvas.width = 640;
  canvas.height = 480;
  const g = canvas.getContext("2d");
  const { width: w, height: h } = canvas;
  const paint = () => {
    const body = mockBody(performance.now());
    const at = (point) => [point.x * w, point.y * h];
    g.fillStyle = WALL;
    g.fillRect(0, 0, w, h);
    g.fillStyle = BLUE;
    g.strokeStyle = BLUE;
    g.lineCap = "round";
    g.lineJoin = "round";
    const [rx, ry] = at(body.rightShoulder);
    const [lx, ly] = at(body.leftShoulder);
    g.beginPath();
    g.moveTo(rx - 30, h);
    g.lineTo(rx - 20, ry + 8);
    g.quadraticCurveTo(w / 2, ry - 44, lx + 20, ly + 8);
    g.lineTo(lx + 30, h);
    g.closePath();
    g.fill();
    g.beginPath();
    g.arc(...at(body.nose), 48, 0, Math.PI * 2);
    g.fill();
    g.lineWidth = 30;
    for (const [shoulder, elbow, wrist, raise] of [
      [body.rightShoulder, body.rightElbow, body.rightWrist, body.raise],
      [body.leftShoulder, body.leftElbow, body.leftWrist, body.leftRaise],
    ]) {
      g.beginPath();
      g.moveTo(...at(shoulder));
      g.lineTo(...at(elbow));
      g.lineTo(...at(wrist));
      g.stroke();
      const [x, y] = at(wrist);
      g.beginPath();
      g.arc(x, y - 30 * raise, 26, 0, Math.PI * 2);
      g.fill();
    }
  };
  paint();
  const timer = setInterval(paint, 66);
  const stream = canvas.captureStream(15);
  stream.stopMock = () => clearInterval(timer);
  return stream;
}

/** A reference video that cannot be loaded is replaced by the drawn figure for a few seconds. */
function playMockClip(run) {
  if (state.clipRun !== run || run.mockTimer) return;
  console.info(`[SignBridge mock] no video for ${run.code}; playing the stand-in figure.`);
  const video = dom.clip;
  video.removeAttribute("src");
  video.srcObject = mockCameraStream();
  video.play().catch(() => {});
  showSource("clip");
  run.mockTimer = setTimeout(() => finishClip(run), MOCK_CLIP_SECONDS * 1000);
}

async function loadMockEngine() {
  const optional = (path) => import(path).catch(() => null);
  const [captureModule, landmarksModule] = await Promise.all([optional("./lib/capture.js"), optional("./lib/landmarks.js")]);
  markStep(0);
  markStep(1);
  markStep(2);
  console.info("[SignBridge mock] the tracker, the recogniser and the camera are stand-ins (?mock=1).");
  return {
    tracker: new MockTracker(),
    recognizer: new MockRecognizer(state.signs),
    SignCapture: captureModule?.SignCapture ?? MockSignCapture,
    emptyFrame: landmarksModule?.emptyFrame ?? mockEmptyFrame,
    mock: true,
  };
}

boot();
