// SignBridge, patient screen (web/CONVERSATION.md).
//
// The patient signs one sign, lowers the hands, signs the next: each recognised
// sign becomes a word of the sentence. The patient fixes or removes words, then
// sends; the doctor's answers arrive as large text and as sign videos.
//
// Recognition is the pipeline of the earlier single-screen demo, unchanged:
// video frame -> Tracker.detect -> skeleton -> pacedCapture().push -> finished
// clip -> Recognizer.predict. The camera and the demonstration videos go through
// the same loop; videos are slowed to the tracker's speed (paceClip) and the
// sign in progress is closed with capture.finish() when a video ends.
//
// Every module is loaded with import(), so a missing or broken file becomes one
// plain sentence on screen instead of a blank page.

window.__signbridgeBooted = true;

const SIGNS_URL = "data/signs.json";
const MODEL_URL = "models/health_signs.json";
const DEFAULT_ROOM = "demo";
const ROOM_PATTERN = /^[A-Za-z0-9_-]{1,32}$/;

const HISTORY_SIZE = 4;          // messages kept in the conversation strip
const NOTICE_MS = 2000;          // "Signe non reconnu" and "Envoyé" stay this long
const NOTICE_LONG_MS = 6000;     // a failure to send stays longer
const NEW_WORD_MS = 2500;        // the newest word keeps its green border this long
const LIVE_WINDOW_MS = 15000;    // older events are history: shown, never played
const ACTIVITY_TIMEOUT_MS = 30000;   // a "typing" the doctor never ended fades out

const CLIP_HOLD_MS = 1200;       // the last image of a demonstration video stays this long
const CLIP_FRAME_MS = 50;        // the reference videos run at 20 frames a second
const CLIP_RATE_MIN = 0.1;       // slowest playback of a reference video
const CLIP_RATE_START = 0.5;     // first guess, before the tracker's speed is known
const DETECT_FAILURES_MAX = 5;

const WHITE = "#FFFFFF";
const INK = "#10233A";
const AMBER = "#E8A100";

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
const HAND_JOINTS = Array.from({ length: 21 }, (_, index) => index);

// Everything the patient reads: Arabic, then French.
const TEXT = {
  loading: { ar: "جارٍ التحميل", fr: "Chargement" },
  cameraStarting: { ar: "جارٍ تشغيل الكاميرا", fr: "Démarrage de la caméra" },
  handNone: { ar: "ارفع يديك للإشارة", fr: "Levez les mains pour signer" },
  handSeen: { ar: "اليد ظاهرة", fr: "Main visible" },
  handRecording: { ar: "إشارة جارية", fr: "Signe en cours" },
  emptySentence: { ar: "قم بإشارة، ثم أنزل يديك", fr: "Faites un signe, puis baissez les mains" },
  emptyPreview: { ar: "الجملة تظهر هنا", fr: "La phrase apparaîtra ici" },
  unknown: { ar: "لم يتم التعرف على الإشارة، أعدها", fr: "Signe non reconnu, recommencez" },
  sent: { ar: "أُرسلت إلى الطبيب", fr: "Envoyé au médecin" },
  sendFailed: { ar: "لم تُرسل الرسالة: تحقّق من الاتصال، ثم اضغط إرسال", fr: "Message non envoyé : vérifiez la connexion, puis appuyez sur Envoyer" },
  noLink: { ar: "لا يوجد اتصال بالطبيب: أعد تحميل الصفحة", fr: "Pas de connexion avec le médecin : rechargez la page" },
  or: { ar: "أو", fr: "ou" },
  doctorTyping: { ar: "الطبيب يكتب…", fr: "Le médecin écrit…" },
  doctorTalking: { ar: "الطبيب يتكلم…", fr: "Le médecin parle…" },
  me: { ar: "أنا", fr: "Moi" },
  doctor: { ar: "الطبيب", fr: "Médecin" },
  noHistory: { ar: "لا توجد رسائل بعد", fr: "Aucun message pour l'instant" },
  demoTag: { ar: "فيديو توضيحي (مخفي)", fr: "Vidéo de démonstration (masquée)" },
  retry: { ar: "أعد المحاولة", fr: "Réessayer" },
  reload: { ar: "أعد تحميل الصفحة", fr: "Recharger la page" },
  close: { ar: "أغلق", fr: "Fermer" },
  linkConnecting: { ar: "جارٍ الاتصال", fr: "Connexion…" },
  linkConnected: { ar: "متصل بالطبيب", fr: "Connecté" },
  linkLocal: { ar: "متصل، في هذا المتصفح فقط", fr: "Connecté, dans ce navigateur seulement" },
  linkReconnecting: { ar: "جارٍ إعادة الاتصال", fr: "Reconnexion…" },
  linkOffline: { ar: "غير متصل: الرسائل لا تصل", fr: "Hors ligne : les messages ne partent pas" },
  linkError: { ar: "الاتصال لم يبدأ: أعد تحميل الصفحة", fr: "La connexion n'a pas démarré : rechargez la page" },
};

// One short sentence each: what failed, and what to do.
const ERRORS = {
  signs: { ar: "تعذّر تحميل قائمة الإشارات: أعد تحميل الصفحة.", fr: "La liste des signes n'a pas pu être chargée : rechargez la page." },
  module: { ar: "ملف من التطبيق غير موجود: أعد تحميل الصفحة.", fr: "Un fichier de l'application est introuvable : rechargez la page." },
  model: { ar: "تعذّر تحميل نموذج التعرف: أعد تحميل الصفحة.", fr: "Le modèle de reconnaissance n'a pas pu être chargé : rechargez la page." },
  tracker: { ar: "تتبّع اليدين لم يشتغل: أعد تحميل الصفحة أو بدّل المتصفح.", fr: "Le suivi des mains n'a pas démarré : rechargez la page ou changez de navigateur." },
  "tracker-run": { ar: "تتبّع اليدين توقّف: أعد تحميل الصفحة.", fr: "Le suivi des mains s'est arrêté : rechargez la page." },
  camera: { ar: "الكاميرا لم تشتغل: أعد المحاولة.", fr: "La caméra n'a pas démarré : réessayez." },
  "camera-denied": { ar: "الكاميرا ممنوعة: اسمح بالكاميرا في المتصفح، ثم أعد المحاولة.", fr: "La caméra est bloquée : autorisez la caméra dans le navigateur, puis réessayez." },
  "camera-missing": { ar: "لا توجد كاميرا: وصّل كاميرا، ثم أعد المحاولة.", fr: "Aucune caméra trouvée : branchez une caméra, puis réessayez." },
  "camera-busy": { ar: "الكاميرا مستعملة في تطبيق آخر: أغلقه، ثم أعد المحاولة.", fr: "La caméra est prise par une autre application : fermez-la, puis réessayez." },
  "camera-unavailable": { ar: "الكاميرا غير متاحة هنا: افتح الصفحة عبر localhost أو https.", fr: "La caméra n'est pas accessible ici : ouvrez la page en http://localhost ou en https." },
  "camera-lost": { ar: "الكاميرا توقّفت: أعد توصيلها، ثم أعد المحاولة.", fr: "La caméra s'est arrêtée : rebranchez-la, puis réessayez." },
  clip: { ar: "تعذّر تشغيل فيديو هذه الإشارة: جرّب إشارة أخرى.", fr: "La vidéo de ce signe n'a pas pu être lue : essayez un autre signe." },
};

class StageError extends Error {
  constructor(kind, cause) {
    super(kind);
    this.name = "StageError";
    this.kind = kind;
    this.cause = cause;
  }
}

const $ = (selector) => document.querySelector(selector);
const clamp01 = (value) => (Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 0);

const dom = {
  roomName: $("#room-name"),
  link: $("#link-status"),
  stage: $("#stage"),
  cam: $("#cam"),
  clip: $("#clip"),
  overlay: $("#overlay"),
  handPill: $("#hand-pill"),
  clipTag: $("#clip-tag"),
  loadingText: $("#loading-text"),
  start: $("#btn-start"),
  errorAr: $("#error-ar"),
  errorFr: $("#error-fr"),
  retry: $("#btn-retry"),
  retryAr: $("#retry-ar"),
  retryFr: $("#retry-fr"),
  words: $("#words"),
  preview: $(".preview"),
  previewAr: $("#preview-ar"),
  previewFr: $("#preview-fr"),
  clear: $("#btn-clear"),
  send: $("#btn-send"),
  reply: $("#reply"),
  activity: $("#doctor-activity"),
  replyText: $("#reply-text"),
  replyFigure: $("#reply-figure"),
  replyName: $("#reply-name"),
  replyCount: $("#reply-count"),
  replay: $("#btn-replay"),
  history: $("#history"),
  demoSign: $("#demo-sign"),
  demo: $("#btn-demo"),
  announce: $("#announce"),
};

const state = {
  room: DEFAULT_ROOM,
  signs: [],
  byCode: new Map(),
  lexicon: null,             // lib/lexicon.js, or null while (or if not) loaded

  // Recognition, as in the earlier single-screen demo.
  enginePromise: null,
  engine: null,              // { tracker, recognizer, newCapture }
  capture: null,
  source: "none",            // "none" | "camera" | "clip"
  cameraOn: false,
  stream: null,
  cameraRequest: 0,
  loopToken: 0,
  detectFailures: 0,
  clipRun: null,             // the demonstration video being played, if any
  clipRate: null,
  retry: null,

  // Conversation.
  channel: null,             // { transport, send, close }
  link: "connecting",        // "connecting" | "connected" | "reconnecting" | "offline" | "error"
  sentStatus: null,          // last status sent, so only changes go out
  signing: false,

  words: [],                 // [{ code, top3: [{ code, p }], key }]
  wordKey: 0,
  newKey: 0,
  newTimer: 0,
  notice: null,              // { kind, text }
  noticeTimer: 0,
  sending: false,

  seen: new Set(),           // message ids already shown
  history: [],               // message events, oldest first
  reply: null,               // { text, codes, index, playing, token, wanted }
  avatar: null,              // SignAvatar from lib/avatar.js, or null (text only)
  activityTimer: 0,
};

/* ------------------------------------------------------------------ */
/* Small helpers                                                       */
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

const arabic = (text, className = "ar") => el("span", { class: className, lang: "ar", dir: "rtl" }, text);

/** A two-line label, Arabic above French. */
function bilingual(words, className = "bi") {
  return el("span", { class: className }, arabic(words.ar), el("span", {}, words.fr));
}

/** Fill an element made of two spans (Arabic, French). */
function setBilingual(node, words) {
  const [ar, fr] = node.children;
  ar.textContent = words?.ar ?? "";
  fr.textContent = words?.fr ?? "";
}

function readRoom() {
  const asked = new URLSearchParams(location.search).get("room");
  if (asked && ROOM_PATTERN.test(asked)) return asked;
  if (asked) console.warn(`[SignBridge] room name not allowed, using "${DEFAULT_ROOM}":`, asked);
  return DEFAULT_ROOM;
}

function newId() {
  try {
    return crypto.randomUUID();
  } catch {
    return `p-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
  }
}

/** Live means "just happened": a stored event replayed on connect is only shown. */
function isLive(event) {
  return !Number.isFinite(event?.ts) || Math.abs(Date.now() - event.ts) < LIVE_WINDOW_MS;
}

function announce(message) {
  dom.announce.textContent = message;
}

/** French typography: "?", "!", ":" and ";" never start a line on their own. */
const frenchSpaces = (text) => String(text ?? "").replace(/ ([?!:;»])/g, " $1").replace(/« /g, "« ");

/* ------------------------------------------------------------------ */
/* Words: lexicon, with a plain fallback if lib/lexicon.js is missing   */
/* ------------------------------------------------------------------ */

function nameOf(code, lang) {
  if (state.lexicon) {
    try {
      const name = state.lexicon.signName(code, state.signs, lang);
      if (name) return name;
    } catch (error) {
      console.warn("[SignBridge] signName failed:", error);
    }
  }
  const sign = state.byCode.get(code);
  return sign ? sign[lang] ?? sign.fr : code;
}

function names(code) {
  return { ar: nameOf(code, "ar"), fr: nameOf(code, "fr") };
}

/** The sentence the doctor will read, from the signs in order. */
function sentenceOf(codes) {
  if (!codes.length) return { fr: "", ar: "" };
  if (state.lexicon) {
    try {
      const sentence = state.lexicon.signsToSentence(codes, state.signs);
      return { fr: String(sentence?.fr ?? ""), ar: String(sentence?.ar ?? "") };
    } catch (error) {
      console.warn("[SignBridge] signsToSentence failed; the sign names are used instead:", error);
    }
  }
  return {
    fr: `${codes.map((code) => nameOf(code, "fr")).join(", ")}.`,
    ar: codes.map((code) => nameOf(code, "ar")).join("، "),
  };
}

async function loadLexicon() {
  try {
    const module = await import("./lib/lexicon.js");
    if (typeof module.signsToSentence !== "function" || typeof module.signName !== "function") {
      throw new Error("lib/lexicon.js does not export signsToSentence and signName");
    }
    state.lexicon = module;
  } catch (error) {
    console.error("[SignBridge] lib/lexicon.js could not be loaded; the sentence is the list of sign names:", error);
  }
  renderWords();
  renderPreview();
  renderHistory();
  renderReply();
}

/* ------------------------------------------------------------------ */
/* Stage states                                                        */
/* ------------------------------------------------------------------ */

function setView(view) {
  dom.stage.dataset.view = view;
}

const HAND_TEXT = { none: TEXT.handNone, seen: TEXT.handSeen, recording: TEXT.handRecording };

function setHand(hand) {
  if (dom.stage.dataset.hand === hand) return;
  dom.stage.dataset.hand = hand;
  setBilingual(dom.handPill, HAND_TEXT[hand]);
}

/**
 * Show a failure in the stage: one sentence (Arabic, French) and one button.
 * The technical detail goes to the console.
 */
function showError(kind, detail, retry = null) {
  if (detail !== undefined) console.error(`[SignBridge] ${kind}:`, detail);
  abandonClip();   // a failure takes the stage: no demonstration video keeps running behind it
  stopLoop();
  clearOverlay();
  setHand("none");
  const message = ERRORS[kind] ?? ERRORS.module;
  dom.errorAr.textContent = message.ar;
  dom.errorFr.textContent = message.fr;
  state.retry = retry?.run ?? (() => location.reload());
  const label = retry?.label ?? TEXT.reload;
  dom.retryAr.textContent = label.ar;
  dom.retryFr.textContent = label.fr;
  setView("error");
  reflectSigning();
}

/* ------------------------------------------------------------------ */
/* Loading: signs, modules, model, tracker                             */
/* ------------------------------------------------------------------ */

async function loadSigns() {
  const response = await fetch(SIGNS_URL, { cache: "no-cache" });
  if (!response.ok) throw new Error(`${SIGNS_URL}: HTTP ${response.status}`);
  const data = await response.json();
  if (!Array.isArray(data?.signs) || data.signs.length === 0) throw new Error(`${SIGNS_URL}: no signs in the file`);
  state.signs = data.signs.filter((sign) => sign && typeof sign.code === "string");
  state.byCode = new Map(state.signs.map((sign) => [sign.code, sign]));
}

async function importModule(path) {
  try {
    return await import(path);
  } catch (error) {
    throw new StageError("module", new Error(`${path}: ${error?.message ?? error}`, { cause: error }));
  }
}

async function loadEngine() {
  const [paceModule, trackerModule, recognizerModule] = await Promise.all([
    importModule("./lib/pace.js"),
    importModule("./lib/tracker.js"),
    importModule("./lib/recognizer.js"),
  ]);
  const { pacedCapture } = paceModule;
  const { Tracker } = trackerModule;
  const { Recognizer } = recognizerModule;
  if (typeof pacedCapture !== "function" || typeof Tracker?.create !== "function" || typeof Recognizer?.load !== "function") {
    throw new StageError("module", new Error("a module does not export what CONTRACT.md lists"));
  }
  const step = (kind, start) => Promise.resolve().then(start).catch((error) => {
    throw new StageError(kind, error);
  });
  const [recognizer, tracker] = await Promise.all([
    step("model", () => Recognizer.load(MODEL_URL)),
    step("tracker", () => Tracker.create({})),
  ]);
  // The capture follows the camera's frame rate, so a fast device does not cut signs short.
  return { tracker, recognizer, newCapture: pacedCapture };
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

/** A head outline around the face points (nose 0, ears 7 and 8) and a neck to the shoulders. */
function drawHead(g, pose, w, h, unit) {
  const [nose, leftEar, rightEar, leftShoulder, rightShoulder] = [pose[0], pose[7], pose[8], pose[11], pose[12]];
  if (!usable(nose)) return;
  const span = (a, b) => (usable(a) && usable(b) ? Math.hypot((a.x - b.x) * w, (a.y - b.y) * h) : 0);
  const ears = usable(leftEar) && usable(rightEar);
  const radius = Math.max(span(leftEar, rightEar) * 0.7, span(leftShoulder, rightShoulder) * 0.26, unit * 10);
  const cx = ears ? ((leftEar.x + rightEar.x) / 2) * w : nose.x * w;
  const cy = nose.y * h - radius * 0.2;
  const outline = (width, colour) => {
    g.lineWidth = width;
    g.strokeStyle = colour;
    g.beginPath();
    if (usable(leftShoulder) && usable(rightShoulder)) {
      g.moveTo(cx, cy + radius);
      g.lineTo(((leftShoulder.x + rightShoulder.x) / 2) * w, ((leftShoulder.y + rightShoulder.y) / 2) * h);
    }
    g.moveTo(cx + radius, cy);
    g.arc(cx, cy, radius, 0, Math.PI * 2);
    g.stroke();
  };
  outline(unit * 4, INK);
  outline(unit * 2, WHITE);
}

/**
 * Landmarks are in unmirrored image coordinates, so they are drawn as they
 * come: the canvas is mirrored in CSS together with the camera image.
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
    // A demonstration video is hidden: give its skeleton a head and a neck so it reads as a person.
    if (state.source === "clip") drawHead(g, pose, w, h, unit);
    strokeLines(g, pose, POSE_LINES, w, h, unit * 4, INK);
    strokeLines(g, pose, POSE_LINES, w, h, unit * 2, WHITE);
    fillJoints(g, pose, POSE_JOINTS, w, h, unit * 2.4, WHITE);
  }
  // Hands: white when seen, amber while a sign is being recorded.
  const handColour = state.capture?.recording ? AMBER : WHITE;
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

/** "signing" while the capture records, "idle" otherwise; only changes are sent. */
function reflectSigning() {
  const signing = Boolean(state.capture?.recording);
  if (signing === state.signing) return;
  state.signing = signing;
  sendStatus(signing ? "signing" : "idle");
}

/** One landmark frame into the capture; a finished clip goes to the model. */
function feed(frame, seconds, run) {
  const clip = state.capture.push(frame, seconds);
  setHand(state.capture.recording ? "recording" : frameHasHand(frame) ? "seen" : "none");
  reflectSigning();
  if (clip) recognise(clip, run);
}

/**
 * Every sign of signs.json the model may answer with, whatever its group:
 * all of them except those marked weak (shown, never proposed). The model's
 * own weak labels are dropped by the recogniser as well.
 */
function proposable() {
  return state.signs.filter((sign) => !sign.weak).map((sign) => sign.code);
}

function recognise(clip, run) {
  let pending;
  try {
    // Across all the signs, in every group, at once.
    pending = Promise.resolve(state.engine.recognizer.predict(clip, proposable()));
  } catch (error) {
    pending = Promise.reject(error);
  }
  const shown = pending
    .catch((error) => {
      console.error("[SignBridge] the model could not read this clip:", error);
      return { label: null, confidence: 0, top3: [] };
    })
    .then((prediction) => {
      if (run && state.clipRun !== run) return;   // that demonstration video was dismissed
      takePrediction(prediction, run);
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
          if (run) {
            run.lastSeconds = seconds;
            run.frames += 1;
          }
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
/* What the stage shows: nothing, the camera, or a demonstration video */
/* ------------------------------------------------------------------ */

function showSource(source) {
  state.source = source;
  stopLoop();
  state.capture?.reset();
  reflectSigning();
  clearOverlay();
  setHand("none");
  dom.cam.hidden = source !== "camera";
  dom.clip.hidden = source !== "clip";
  dom.stage.classList.toggle("is-mirrored", source === "camera");   // a mirror for the patient; videos stay as filmed
  if (source === "none") {
    setView("idle");
    return;
  }
  setView("live");
  startLoop(source === "camera" ? dom.cam : dom.clip, source);
}

function releaseStream() {
  const stream = state.stream;
  state.stream = null;
  if (!stream) return;
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

const retryCamera = { run: () => startCamera(), label: TEXT.retry };

async function startCamera() {
  const request = ++state.cameraRequest;
  const engine = await ensureEngine();
  if (!engine || request !== state.cameraRequest) return;
  if (state.clipRun) return;   // a demonstration video holds the stage; the camera comes after it
  stopLoop();
  setBilingual(dom.loadingText, TEXT.cameraStarting);
  setView("loading");

  let stream;
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new StageError("camera-unavailable");
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
    });
  } catch (error) {
    if (request !== state.cameraRequest) return;
    state.cameraOn = false;
    if (state.clipRun) {   // a demonstration video started while the browser was asking
      console.warn("[SignBridge] the camera did not start:", error);
      return;
    }
    showError(cameraErrorKind(error), error, retryCamera);
    return;
  }
  if (request !== state.cameraRequest) {   // asked again while the browser was asking
    for (const track of stream.getTracks()) track.stop();
    return;
  }

  releaseStream();
  state.stream = stream;
  const [track] = stream.getVideoTracks();
  if (track) {
    track.onended = () => {
      if (state.stream !== stream) return;
      state.cameraOn = false;
      releaseStream();
      if (state.source === "camera") showError("camera-lost", new Error("the camera track ended"), retryCamera);
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
  // A demonstration video started while the browser was asking: let it finish, the camera comes after.
  if (!state.clipRun) showSource("camera");
}

/* ------------------------------------------------------------------ */
/* Demonstration videos                                                */
/* ------------------------------------------------------------------ */

const dismissClip = { run: () => showSource(state.cameraOn ? "camera" : "none"), label: TEXT.close };

function reflectDemo() {
  dom.demo.textContent = state.clipRun ? "Arrêter la vidéo" : "Faire signer la vidéo";
}

/** Stop the demonstration video without a result (the presenter moved on). */
function abandonClip() {
  const run = state.clipRun;
  if (!run) return;
  state.clipRun = null;
  clearTimeout(run.holdTimer);
  clearTimeout(run.watchdog);
  dom.clipTag.hidden = true;
  const video = dom.clip;
  video.pause();
  video.removeAttribute("src");
  video.load();
  reflectDemo();
}

/** Play one sign's reference video in the stage, through the same pipeline as the camera. */
async function playClip(code) {
  if (state.clipRun) {   // the button reads "Arrêter la vidéo" while one plays
    abandonClip();
    showSource(state.cameraOn ? "camera" : "none");
    return;
  }
  const sign = state.byCode.get(code);
  if (!hasClip(sign)) return;   // only signs with a reference video can be demonstrated
  const run = {
    code, predictions: 0, added: new Set(), pending: [], finished: false,
    lastSeconds: 0, holdTimer: 0, watchdog: 0, detectMs: 0, frames: 0,
  };
  state.clipRun = run;
  reflectDemo();

  const engine = await ensureEngine();
  if (state.clipRun !== run) return;
  if (!engine) {
    abandonClip();
    return;
  }

  setBilingual(dom.clipTag, { ar: `${TEXT.demoTag.ar}: ${names(code).ar}`, fr: `${TEXT.demoTag.fr} : ${names(code).fr}` });
  dom.clipTag.hidden = false;

  const video = dom.clip;
  video.src = sign.clip;
  video.defaultPlaybackRate = state.clipRate ?? CLIP_RATE_START;   // see paceClip
  video.playbackRate = video.defaultPlaybackRate;
  showSource("clip");
  try {
    await video.play();
  } catch (error) {
    if (state.clipRun !== run) return;   // replaced or stopped: not a failure
    showError("clip", error, dismissClip);
    return;
  }
  if (state.clipRun !== run) return;
  const duration = Number.isFinite(video.duration) ? video.duration : 8;
  run.watchdog = setTimeout(() => finishClip(run), (duration * 1000) / CLIP_RATE_MIN + 4000);   // if "ended" never comes
}

/** The video is over: close the sign in progress, wait for the model, then hand the stage back. */
async function finishClip(run) {
  if (state.clipRun !== run || run.finished) return;
  run.finished = true;
  clearTimeout(run.watchdog);
  stopLoop();
  const expected = Math.round((Number.isFinite(dom.clip.duration) ? dom.clip.duration : 0) * 1000 / CLIP_FRAME_MS);
  console.info(`[SignBridge] video ${run.code}: ${run.frames} of about ${expected} frames tracked, `
    + `${Math.round(run.detectMs)} ms per frame, played at ${(state.clipRate ?? 1).toFixed(2)}x.`);
  // The video ran out with the hands possibly still up: close the sign in progress.
  const clip = state.capture.finish(run.lastSeconds);
  if (clip) recognise(clip, run);
  reflectSigning();
  setHand("none");
  await Promise.all(run.pending);
  if (state.clipRun !== run) return;
  if (run.added.size === 0) showNotice("unknown");   // no sign was recognised in the whole video
  run.holdTimer = setTimeout(() => {
    if (state.clipRun !== run) return;
    abandonClip();
    showSource(state.cameraOn ? "camera" : "none");
  }, CLIP_HOLD_MS);
}

/* ------------------------------------------------------------------ */
/* The sentence: words added sign after sign                           */
/* ------------------------------------------------------------------ */

function normalisePrediction(prediction) {
  const top3 = (Array.isArray(prediction?.top3) ? prediction.top3 : [])
    .map((entry) => ({ code: entry?.[0], p: clamp01(entry?.[1]) }))
    .filter((candidate) => state.byCode.has(candidate.code));
  const label = state.byCode.has(prediction?.label) ? prediction.label : null;
  return { label, top3 };
}

/** A prediction from the model. `run` is the demonstration video it belongs to, if any. */
function takePrediction(prediction, run) {
  const result = normalisePrediction(prediction);
  console.info(`[SignBridge] sign${run ? ` (video ${run.code})` : ""}: ${result.label ?? "not recognised"};`,
    result.top3.map(({ code, p }) => `${code} ${p.toFixed(2)}`).join(", "));
  if (run) run.predictions += 1;
  if (!result.label) {
    // From the camera: say so now. From a video: decided once the video is over.
    if (!run) showNotice("unknown");
    return;
  }
  if (run) {
    if (run.added.has(result.label)) return;   // one video, one word
    run.added.add(result.label);
  }
  addWord(result.label, result.top3);
}

function addWord(code, top3) {
  const key = ++state.wordKey;
  state.words.push({ code, top3, key });
  markNew(key);
  hideNotice(["unknown", "sent"]);
  renderSentence();
  announce(`Mot ajouté : ${nameOf(code, "fr")}.`);
}

function markNew(key) {
  state.newKey = key;
  clearTimeout(state.newTimer);
  state.newTimer = setTimeout(() => {
    state.newKey = 0;
    renderWords();
  }, NEW_WORD_MS);
}

/** Tap on an alternative: it replaces the last word. */
function replaceLast(code) {
  const last = state.words.at(-1);
  if (!last || !state.byCode.has(code)) return;
  last.code = code;
  markNew(last.key);
  renderSentence();
  announce(`Mot corrigé : ${nameOf(code, "fr")}.`);
  dom.words.querySelector(".alt")?.focus({ preventScroll: true });
}

/** Tap on a word: it goes. */
function removeWord(index) {
  const [removed] = state.words.splice(index, 1);
  if (!removed) return;
  renderSentence();
  announce(`Mot retiré : ${nameOf(removed.code, "fr")}.`);
  const next = dom.words.querySelectorAll(".word")[Math.min(index, state.words.length - 1)];
  (next ?? dom.send).focus({ preventScroll: true });
}

function eraseLast() {
  if (!state.words.length) return;
  removeWord(state.words.length - 1);
  if (!state.words.length) dom.clear.focus({ preventScroll: true });
}

const NOTICES = {
  unknown: { text: TEXT.unknown, ms: NOTICE_MS },
  sent: { text: TEXT.sent, ms: NOTICE_MS },
  sendFailed: { text: TEXT.sendFailed, ms: NOTICE_LONG_MS },
  noLink: { text: TEXT.noLink, ms: NOTICE_LONG_MS },
};

function showNotice(kind) {
  const notice = NOTICES[kind];
  state.notice = { kind, text: notice.text };
  clearTimeout(state.noticeTimer);
  state.noticeTimer = setTimeout(() => {
    state.notice = null;
    renderWords();
  }, notice.ms);
  renderWords();
  announce(notice.text.fr);
}

function hideNotice(kinds) {
  if (!state.notice || !kinds.includes(state.notice.kind)) return;
  clearTimeout(state.noticeTimer);
  state.notice = null;
}

function renderWords() {
  const nodes = state.words.map((word, index) => {
    const label = names(word.code);
    return el("button", {
      type: "button",
      class: `word${word.key === state.newKey ? " is-new" : ""}`,
      "data-index": index,
      "aria-label": `${label.fr} : toucher pour retirer ce mot`,
      title: "Toucher pour retirer ce mot",
    }, arabic(label.ar, "word__ar ar"), el("span", { class: "word__fr" }, label.fr));
  });

  const last = state.words.at(-1);
  const alternatives = last ? last.top3.filter((candidate) => candidate.code !== last.code).slice(0, 2) : [];
  if (alternatives.length) {
    nodes.push(el("span", { class: "alts", role: "group", "aria-label": `Ou bien, à la place de ${nameOf(last.code, "fr")}` },
      bilingual(TEXT.or, "alts__or bi"),
      ...alternatives.map((candidate) => {
        const label = names(candidate.code);
        return el("button", {
          type: "button", class: "alt", "data-code": candidate.code,
          "aria-label": `Remplacer par ${label.fr}`,
        }, arabic(label.ar, "alt__ar ar"), el("span", { class: "alt__fr" }, label.fr));
      })));
  }

  if (state.notice) {
    nodes.push(bilingual(state.notice.text, `notice notice--${state.notice.kind}`));
  } else if (!state.words.length) {
    nodes.push(bilingual(TEXT.emptySentence, "words__empty bi"));
  }
  dom.words.replaceChildren(...nodes);
}

function renderPreview() {
  const sentence = sentenceOf(state.words.map((word) => word.code));
  const empty = !sentence.fr;
  dom.preview.classList.toggle("is-empty", empty);
  dom.previewAr.textContent = empty ? TEXT.emptyPreview.ar : sentence.ar;
  dom.previewFr.textContent = empty ? TEXT.emptyPreview.fr : sentence.fr;
}

function renderActions() {
  const none = state.words.length === 0;
  dom.clear.disabled = none;
  dom.send.disabled = none || state.sending;
}

function renderSentence() {
  renderWords();
  renderPreview();
  renderActions();
}

async function sendSentence() {
  if (!state.words.length || state.sending) return;
  if (!state.channel) {
    showNotice("noLink");
    return;
  }
  const codes = state.words.map((word) => word.code);
  const sentence = sentenceOf(codes);
  state.sending = true;
  renderActions();
  try {
    await state.channel.send({
      type: "message", id: newId(), ts: Date.now(),
      from: "patient", via: "sign", text: sentence.fr, signs: codes,
    });
    state.words = [];
    state.newKey = 0;
    showNotice("sent");
  } catch (error) {
    console.error("[SignBridge] the message could not be sent:", error);
    showNotice("sendFailed");
  } finally {
    state.sending = false;
    renderSentence();
  }
}

/* ------------------------------------------------------------------ */
/* The doctor's message: text, then a stick figure signs it            */
/* ------------------------------------------------------------------ */
//
// The signs are made by a drawn figure (lib/avatar.js), never by the dataset's
// videos of a real person. Without the figure, the doctor's text shows alone.

/** The figure at rest, redrawn (the avatar sizes its canvas to its box when it draws). */
function restFigure() {
  const avatar = state.avatar;
  if (!avatar || avatar.playing || typeof avatar.drawRest !== "function") return;
  try {
    avatar.drawRest();
  } catch (error) {
    console.warn("[SignBridge] the figure could not be drawn:", error);
  }
}

async function loadAvatar() {
  try {
    const module = await import("./lib/avatar.js");
    if (typeof module.loadSignPoses !== "function" || typeof module.SignAvatar !== "function") {
      throw new Error("lib/avatar.js does not export loadSignPoses and SignAvatar");
    }
    const poses = await module.loadSignPoses();
    state.avatar = new module.SignAvatar(dom.replyFigure, poses);
  } catch (error) {
    console.error("[SignBridge] the signing figure could not be loaded; the doctor's messages show as text only:", error);
    state.avatar = null;
  }
  renderReply();
  if (state.reply?.wanted) playReply();
}

/** The signs of the reply that the figure knows how to make. */
function figureCodes(reply) {
  if (!state.avatar || !reply) return [];
  return reply.codes.filter((code) => {
    try {
      return state.avatar.has(code);
    } catch {
      return false;
    }
  });
}

function showReply(event, { play }) {
  const codes = (Array.isArray(event.signs) ? event.signs : []).filter((code) => state.byCode.has(code));
  stopReply();
  state.reply = { text: String(event.text ?? ""), codes, index: 0, playing: false, token: 0, wanted: play };
  renderReply();
  if (play) playReply();
}

/** Back to the figure at rest; a play() still running is ignored when it resolves. */
function stopReply() {
  if (state.reply) {
    state.reply.token += 1;
    state.reply.playing = false;
  }
  try {
    state.avatar?.stop();
  } catch (error) {
    console.warn("[SignBridge] the figure could not stop:", error);
  }
}

/** Make the reply's signs from the first one ("Revoir" calls this again). */
async function playReply() {
  const reply = state.reply;
  const codes = figureCodes(reply);
  if (!codes.length) return;   // no figure (yet), or no sign it knows: the text alone
  stopReply();
  const token = reply.token;
  reply.wanted = false;
  reply.playing = true;
  reply.index = 0;
  renderReplyCaption();
  try {
    await state.avatar.play(codes, {
      onSign(_code, index) {
        if (state.reply !== reply || reply.token !== token) return;
        reply.index = index;
        renderReplyCaption();
      },
    });
  } catch (error) {
    console.warn("[SignBridge] the figure could not make these signs:", codes, error);
  }
  if (state.reply !== reply || reply.token !== token) return;   // replaced or replayed meanwhile
  reply.playing = false;
  renderReplyCaption();
}

function renderReplyCaption() {
  const codes = figureCodes(state.reply);
  if (!codes.length) return;
  const index = Math.min(Math.max(0, state.reply.index), codes.length - 1);
  setBilingual(dom.replyName, names(codes[index]));
  dom.replyCount.textContent = codes.length > 1 ? `${index + 1} / ${codes.length}` : "";
}

function renderReply() {
  const reply = state.reply;
  dom.reply.dataset.state = !reply ? "empty" : figureCodes(reply).length ? "signs" : "text";
  dom.replyText.textContent = frenchSpaces(reply?.text);
  renderReplyCaption();
  // The figure's box only gets a size once it is shown.
  if (dom.reply.dataset.state === "signs") restFigure();
}

const ACTIVITY = { typing: TEXT.doctorTyping, listening: TEXT.doctorTalking };

function setDoctorActivity(activity) {
  clearTimeout(state.activityTimer);
  const words = ACTIVITY[activity];
  dom.activity.hidden = !words;
  if (!words) return;
  setBilingual(dom.activity, words);
  state.activityTimer = setTimeout(() => {
    dom.activity.hidden = true;
  }, ACTIVITY_TIMEOUT_MS);
}

/* ------------------------------------------------------------------ */
/* Conversation strip                                                  */
/* ------------------------------------------------------------------ */

function renderHistory() {
  const items = state.history.slice(-HISTORY_SIZE).map((event) => {
    const mine = event.from === "patient";
    let text;
    if (mine) {
      const codes = Array.isArray(event.signs) ? event.signs.filter((code) => state.byCode.has(code)) : [];
      const sentence = codes.length ? sentenceOf(codes) : { ar: "", fr: "" };
      text = el("span", { class: "history__text" },
        sentence.ar ? arabic(sentence.ar) : null,
        el("span", {}, frenchSpaces(event.text ?? sentence.fr)));
    } else {
      text = el("span", { class: "history__text" }, el("span", {}, frenchSpaces(event.text)));
    }
    return el("li", { class: `history__item history__item--${mine ? "patient" : "doctor"}` },
      bilingual(mine ? TEXT.me : TEXT.doctor, "history__who bi"),
      text);
  });
  if (!items.length) items.push(el("li", {}, bilingual(TEXT.noHistory, "history__empty bi")));
  dom.history.replaceChildren(...items);
}

/* ------------------------------------------------------------------ */
/* Room: events in, status out                                         */
/* ------------------------------------------------------------------ */

const LINK_TEXT = {
  connecting: TEXT.linkConnecting,
  reconnecting: TEXT.linkReconnecting,
  offline: TEXT.linkOffline,
  error: TEXT.linkError,
};

function renderLink() {
  let words = LINK_TEXT[state.link] ?? TEXT.linkConnecting;
  if (state.link === "connected") words = state.channel?.transport === "local" ? TEXT.linkLocal : TEXT.linkConnected;
  dom.link.dataset.state = state.link;
  setBilingual(dom.link, words);
}

function sendStatus(value) {
  if (state.sentStatus === value) return;
  if (!state.channel) return;   // "online" is sent on joining; the rest only matters once joined
  state.sentStatus = value;
  state.channel.send({ type: "status", from: "patient", state: value })
    .catch((error) => console.warn("[SignBridge] status not sent:", error));
}

function onEvent(event) {
  if (!event || typeof event !== "object") return;
  try {
    if (event.type === "message") onMessage(event);
    else if (event.type === "status") onStatus(event);
    else if (event.type === "clear") onClear();
  } catch (error) {
    console.error("[SignBridge] an event could not be shown:", error, event);
  }
}

function onMessage(event) {
  if (event.from !== "patient" && event.from !== "doctor") return;
  if (event.id != null) {
    if (state.seen.has(event.id)) return;
    state.seen.add(event.id);
  }
  state.history.push(event);
  if (state.history.length > 50) state.history.splice(0, state.history.length - 50);
  renderHistory();
  if (event.from === "doctor") {
    const live = isLive(event);
    setDoctorActivity(null);
    showReply(event, { play: live });
    if (live) {
      announce(`Message du médecin : ${event.text ?? ""}`);
      // On a narrow screen the doctor's panel can be below the fold.
      const calm = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
      dom.reply.scrollIntoView({ block: "nearest", behavior: calm ? "auto" : "smooth" });
    }
  }
}

function onStatus(event) {
  if (event.from !== "doctor") return;   // our own status, echoed back
  setDoctorActivity(event.state);
}

/** New consultation: the conversation and the sentence in progress are emptied. */
function onClear() {
  state.history = [];
  state.seen.clear();
  stopReply();
  state.reply = null;
  state.words = [];
  state.newKey = 0;
  state.notice = null;
  setDoctorActivity(null);
  renderHistory();
  renderReply();
  renderSentence();
}

async function connect() {
  try {
    const { joinRoom } = await import("./lib/channel.js");
    if (typeof joinRoom !== "function") throw new Error("lib/channel.js does not export joinRoom");
    state.channel = await joinRoom(state.room, {
      onEvent,
      onState(value) {
        state.link = value;
        renderLink();
      },
    });
    if (state.link === "connecting") state.link = "connected";
    renderLink();
    console.info(`[SignBridge] joined room "${state.room}" (${state.channel.transport}).`);
    sendStatus("online");
  } catch (error) {
    console.error("[SignBridge] the room could not be joined:", error);
    state.channel = null;
    state.link = "error";
    renderLink();
  }
}

/* ------------------------------------------------------------------ */
/* Controls                                                            */
/* ------------------------------------------------------------------ */

const hasClip = (sign) => typeof sign?.clip === "string" && sign.clip !== "";

/**
 * The signs a reference video can demonstrate: those with a clip (not every
 * sign has one) that the model may propose (a weak sign would never come out).
 */
function renderDemoOptions() {
  const signs = state.signs.filter((sign) => hasClip(sign) && !sign.weak);
  dom.demoSign.replaceChildren(...signs.map((sign) => el("option", { value: sign.code }, nameOf(sign.code, "fr"))));
  return signs.length;
}

function enableDemo() {
  const usable = dom.demoSign.options.length > 0;
  dom.demoSign.disabled = !usable;
  dom.demo.disabled = !usable;
}

function bindControls() {
  dom.retry.onclick = null;   // the fallback handler of patient.html, if it was installed
  dom.retry.addEventListener("click", () => (state.retry ?? (() => location.reload()))());
  dom.start.addEventListener("click", () => startCamera());

  dom.words.addEventListener("click", (event) => {
    const alt = event.target.closest?.(".alt");
    if (alt) {
      replaceLast(alt.dataset.code);
      return;
    }
    const word = event.target.closest?.(".word");
    if (word) removeWord(Number(word.dataset.index));
  });
  dom.clear.addEventListener("click", eraseLast);
  dom.send.addEventListener("click", sendSentence);

  dom.replay.addEventListener("click", () => playReply());
  // A new size: the figure at rest is redrawn to fit (a playing figure redraws itself).
  if (typeof ResizeObserver === "function") new ResizeObserver(restFigure).observe(dom.replyFigure);

  dom.demo.addEventListener("click", () => playClip(dom.demoSign.value));

  dom.clip.addEventListener("ended", () => {
    if (state.clipRun) finishClip(state.clipRun);
  });
  dom.clip.addEventListener("error", () => {
    const run = state.clipRun;
    if (!run || run.finished || !dom.clip.getAttribute("src")) return;
    const detail = dom.clip.error ? `${dom.clip.getAttribute("src")}: media error ${dom.clip.error.code} ${dom.clip.error.message}` : "media error";
    showError("clip", new Error(detail), dismissClip);
  });

  // A hidden tab gets no video frames: drop a half-recorded sign rather than
  // finish it later with frames from before and after the pause.
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && state.source === "camera") {
      state.capture?.reset();
      setHand("none");
      reflectSigning();
    }
  });

  window.addEventListener("pagehide", () => {
    stopLoop();
    releaseStream();
    try {
      state.engine?.tracker.close?.();
    } catch { /* closing anyway */ }
    try {
      state.channel?.close();
    } catch { /* closing anyway */ }
  });
}

async function boot() {
  state.room = readRoom();
  dom.roomName.textContent = `Salle ${state.room}`;
  bindControls();
  setView("loading");
  setBilingual(dom.handPill, HAND_TEXT[dom.stage.dataset.hand] ?? TEXT.handNone);
  renderLink();
  renderSentence();
  renderReply();
  renderHistory();

  let signsLoaded = true;
  try {
    await loadSigns();
  } catch (error) {
    signsLoaded = false;
    showError("signs", error);
  }
  renderDemoOptions();
  loadLexicon();
  loadAvatar();
  connect();   // the conversation works even if recognition does not
  if (!signsLoaded) return;

  state.enginePromise = loadEngine().then((engine) => {
    state.engine = engine;
    state.capture = engine.newCapture();
    return engine;
  });
  try {
    await state.enginePromise;
  } catch (error) {
    showError(error?.kind ?? "module", error?.cause ?? error);
    return;
  }
  console.info(`[SignBridge] recognition ready (tracker: ${state.engine.tracker.delegate ?? "?"}).`);
  enableDemo();
  if (state.source === "none" && dom.stage.dataset.view === "loading" && state.cameraRequest === 0) {
    setView("idle");
    startCamera();
  }
}

boot();
