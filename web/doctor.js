// SignBridge, doctor screen (web/CONVERSATION.md).
//
// Patient messages arrive as a French sentence, read aloud. The doctor answers
// by typing, dictating or picking a ready-made question; the screen shows in
// advance which signs the patient will see.
//
// Modules are loaded with import(): a missing one becomes one plain sentence on
// screen (or, for the lexicon and speech, a reduced but working screen).

window.__signbridgeBooted = true;

const SIGNS_URL = "data/signs.json";
const DEFAULT_ROOM = "demo";
const ROOM_PATTERN = /^[A-Za-z0-9_-]{1,32}$/;
const LIVE_WINDOW_MS = 15000;        // older events are history: shown, never read aloud
const TYPING_IDLE_MS = 5000;         // no key for this long: "typing" becomes "idle"
const ACTIVITY_TIMEOUT_MS = 15000;   // a "signing" the patient never ended fades out
const MAX_MESSAGES = 200;

const QUESTIONS = [
  "Bonjour, comment allez-vous ?",
  "Bienvenue.",
  "Oui.",
  "Non.",
  "Avez-vous de la fièvre ?",
  "Avez-vous mal à la tête ?",
  "Toussez-vous ?",
  "Avez-vous des vertiges ?",
  "Avez-vous du mal à respirer ?",
  "Êtes-vous diabétique ?",
  "Êtes-vous enceinte ?",
  "Avez-vous une allergie ?",
  "Prenez ce médicament.",
  "Revenez pour un rendez-vous.",
  "Revenez jeudi.",
  "Allez à l'hôpital aujourd'hui.",
  "Allez au dispensaire.",
  "Votre famille peut vous aider.",
  "Voulez-vous que je vous l'écrive ?",
];

const LINK_TEXT = {
  connecting: "Connexion…",
  connected: "Connecté",
  local: "Connecté, dans ce navigateur seulement",
  reconnecting: "Reconnexion…",
  offline: "Hors ligne : les messages ne partent pas",
  error: "La connexion n'a pas démarré : rechargez la page",
};

const DICTATION_NOTE = "La dictée utilise le service vocal du navigateur.";
const DICTATION_ERRORS = {
  "not-supported": "La dictée n'est pas disponible dans ce navigateur : écrivez votre réponse.",
  "not-allowed": "Le micro est bloqué : autorisez le micro dans le navigateur, ou écrivez votre réponse.",
  "service-not-allowed": "Le service vocal du navigateur est bloqué : écrivez votre réponse.",
  "audio-capture": "Aucun micro trouvé : branchez un micro, ou écrivez votre réponse.",
  network: "Le service vocal du navigateur ne répond pas : vérifiez Internet, ou écrivez votre réponse.",
  "no-speech": "Aucune parole entendue : appuyez sur Dicter et parlez, ou écrivez votre réponse.",
};
const DICTATION_ERROR_DEFAULT = "La dictée s'est arrêtée : réessayez, ou écrivez votre réponse.";

const $ = (selector) => document.querySelector(selector);

const dom = {
  roomName: $("#room-name"),
  link: $("#link-status"),
  voice: $("#btn-voice"),
  newVisit: $("#btn-new"),
  confirm: $("#confirm"),
  confirmYes: $("#btn-confirm"),
  confirmNo: $("#btn-cancel"),
  talkError: $("#talk-error"),
  empty: $("#thread-empty"),
  thread: $("#thread"),
  activity: $("#patient-activity"),
  questions: $("#questions"),
  composer: $("#composer"),
  field: $("#field"),
  send: $("#btn-send"),
  signsPreview: $("#signs-preview"),
  dictate: $("#btn-dictate"),
  dictationNote: $("#dictation-note"),
};

const state = {
  room: DEFAULT_ROOM,
  signs: [],
  byCode: new Map(),
  lexicon: null,
  speech: null,
  figure: null,              // SignAvatar, only asked which signs it can make
  figureKnown: false,

  channel: null,
  link: "connecting",
  sentStatus: null,

  messages: [],
  seen: new Set(),
  voiceOn: true,
  activityTimer: 0,

  sending: false,
  dictated: false,           // part of the field came from dictation: the message is "dicté"
  dictation: null,           // { supported, start, stop }
  listening: false,
  dictationBase: "",         // what the field held when dictation started
  typingTimer: 0,
};

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
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
    return `d-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
  }
}

function isLive(event) {
  return !Number.isFinite(event?.ts) || Math.abs(Date.now() - event.ts) < LIVE_WINDOW_MS;
}

function clock(ts) {
  const date = new Date(Number.isFinite(ts) ? ts : Date.now());
  return date.toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" });
}

/** French typography: "?", "!", ":" and ";" never start a line on their own. */
const frenchSpaces = (text) => String(text ?? "").replace(/ ([?!:;»])/g, "\u00A0$1").replace(/« /g, "«\u00A0");

function showProblem(text) {
  dom.talkError.textContent = text;
  dom.talkError.hidden = !text;
}

/* ------------------------------------------------------------------ */
/* Signs and lexicon                                                   */
/* ------------------------------------------------------------------ */

async function loadSigns() {
  const response = await fetch(SIGNS_URL, { cache: "no-cache" });
  if (!response.ok) throw new Error(`${SIGNS_URL}: HTTP ${response.status}`);
  const data = await response.json();
  if (!Array.isArray(data?.signs) || data.signs.length === 0) throw new Error(`${SIGNS_URL}: no signs in the file`);
  state.signs = data.signs.filter((sign) => sign && typeof sign.code === "string");
  state.byCode = new Map(state.signs.map((sign) => [sign.code, sign]));
}

async function loadLexicon() {
  try {
    const module = await import("./lib/lexicon.js");
    if (typeof module.textToSigns !== "function" || typeof module.signName !== "function") {
      throw new Error("lib/lexicon.js does not export textToSigns and signName");
    }
    state.lexicon = module;
  } catch (error) {
    console.error("[SignBridge] lib/lexicon.js could not be loaded; no signs will be shown to the patient:", error);
  }
}

function nameOf(code) {
  if (state.lexicon) {
    try {
      return state.lexicon.signName(code, state.signs, "fr");
    } catch (error) {
      console.warn("[SignBridge] signName failed:", error);
    }
  }
  return state.byCode.get(code)?.fr ?? code;
}

/** The signs found in `text`: what the message carries to the patient's screen. */
function signsFor(text) {
  if (!state.lexicon || !text.trim()) return [];
  try {
    return state.lexicon.textToSigns(text, state.signs).filter((code) => state.byCode.has(code));
  } catch (error) {
    console.warn("[SignBridge] textToSigns failed:", error);
    return [];
  }
}

/**
 * The patient's screen shows a sign only if the stick figure can make it, so
 * the doctor is told exactly that: the same lib/avatar.js is asked here (a
 * figure on a canvas that is never shown). Until it has loaded, all are listed.
 */
async function loadFigure() {
  try {
    const module = await import("./lib/avatar.js");
    if (typeof module.loadSignPoses !== "function" || typeof module.SignAvatar !== "function") {
      throw new Error("lib/avatar.js does not export loadSignPoses and SignAvatar");
    }
    state.figure = new module.SignAvatar(document.createElement("canvas"), await module.loadSignPoses());
  } catch (error) {
    console.error("[SignBridge] lib/avatar.js could not be loaded; the patient will see the text only:", error);
    state.figure = null;
  }
  state.figureKnown = true;
  onFieldChanged();
  renderThread();
}

function shownToPatient(codes) {
  if (!state.figureKnown) return codes;
  return codes.filter((code) => {
    try {
      return Boolean(state.figure?.has(code));
    } catch {
      return false;
    }
  });
}

/* ------------------------------------------------------------------ */
/* Thread                                                              */
/* ------------------------------------------------------------------ */

function messageNode(event) {
  const codes = Array.isArray(event.signs) ? event.signs.filter((code) => state.byCode.has(code)) : [];
  const list = codes.map(nameOf).join(", ");
  if (event.from === "patient") {
    return el("li", { class: "msg msg--patient" },
      el("p", { class: "msg__who" }, el("span", {}, "Patient"), el("span", { class: "msg__time" }, clock(event.ts))),
      el("p", { class: "msg__text", lang: "fr" }, frenchSpaces(event.text)),
      el("p", { class: "msg__meta" }, list ? `Signé : ${list}` : "Signé"));
  }
  const how = event.via === "voice" ? "Dicté" : "Écrit";
  const shown = shownToPatient(codes).map(nameOf).join(", ");
  return el("li", { class: "msg msg--doctor" },
    el("p", { class: "msg__who" }, el("span", {}, "Vous"), el("span", { class: "msg__time" }, clock(event.ts))),
    el("p", { class: "msg__text", lang: "fr" }, frenchSpaces(event.text)),
    el("p", { class: "msg__meta" }, shown ? `${how}. Signes montrés : ${shown}` : `${how}. Texte seul, sans signe.`));
}

function renderThread() {
  dom.thread.replaceChildren(...state.messages.map(messageNode));
  dom.empty.hidden = state.messages.length > 0;
  dom.thread.scrollTop = dom.thread.scrollHeight;
}

function appendMessage(event) {
  state.messages.push(event);
  if (state.messages.length > MAX_MESSAGES) {
    state.messages.splice(0, state.messages.length - MAX_MESSAGES);
    renderThread();
    return;
  }
  dom.thread.append(messageNode(event));
  dom.empty.hidden = true;
  dom.thread.scrollTop = dom.thread.scrollHeight;
}

/* ------------------------------------------------------------------ */
/* Reading aloud                                                       */
/* ------------------------------------------------------------------ */

function canSpeak() {
  try {
    return Boolean(state.speech?.canSpeak());
  } catch {
    return false;
  }
}

function renderVoice() {
  const available = canSpeak();
  dom.voice.disabled = !available;
  dom.voice.setAttribute("aria-pressed", String(available && state.voiceOn));
  const stateText = !available ? "indisponible" : state.voiceOn ? "activée" : "désactivée";
  // The long label on a wide screen, "Voix" on a phone (style.css picks one).
  dom.voice.replaceChildren(
    Object.assign(document.createElement("span"), { className: "label-long", textContent: "Lecture à voix haute" }),
    Object.assign(document.createElement("span"), { className: "label-short", textContent: "Voix" }),
    document.createTextNode(` : ${stateText}`));
  dom.voice.setAttribute("aria-label", `Lecture à voix haute : ${stateText}`);
}

// Chrome lets a page speak only after someone has clicked on it. If a patient message
// arrives first (in split view, a click on the patient side does not count), say so.
const VOICE_NOTE = "La lecture à voix haute peut utiliser le service vocal du navigateur.";
const VOICE_BLOCKED = "Pour entendre les messages, cliquez une fois sur cet écran (le navigateur l'exige).";

function voiceAllowed() {
  return !globalThis.navigator?.userActivation || globalThis.navigator.userActivation.hasBeenActive;
}

function showVoiceNote() {
  const note = document.getElementById("voice-note");
  if (!note) return;
  const blocked = state.voiceOn && canSpeak() && !voiceAllowed();
  note.textContent = blocked ? VOICE_BLOCKED : VOICE_NOTE;
  note.classList.toggle("is-warning", blocked);
}

function readAloud(text) {
  if (!state.voiceOn || !canSpeak()) return;
  if (!voiceAllowed()) {
    showVoiceNote();
    state.unspoken = text;   // read once the doctor clicks
    return;
  }
  try {
    state.speech.speak(text, { lang: "fr-FR" });
  } catch (error) {
    console.warn("[SignBridge] the message could not be read aloud:", error);
  }
}

async function loadSpeech() {
  try {
    const module = await import("./lib/speech.js");
    if (typeof module.speak !== "function" || typeof module.dictation !== "function") {
      throw new Error("lib/speech.js does not export speak and dictation");
    }
    state.speech = module;
    // Some browsers fill the voice list late; asking early makes it ready in time.
    try {
      globalThis.speechSynthesis?.getVoices();
    } catch { /* no speech here */ }
  } catch (error) {
    console.error("[SignBridge] lib/speech.js could not be loaded; no reading aloud, no dictation:", error);
  }
  renderVoice();
  setupDictation();
}

/* ------------------------------------------------------------------ */
/* Composer                                                            */
/* ------------------------------------------------------------------ */

function renderSignsPreview() {
  const text = dom.field.value;
  const codes = shownToPatient(signsFor(text));
  const node = dom.signsPreview;
  if (!text.trim()) {
    node.className = "composer__signs is-quiet";
    node.textContent = "Les signes montrés au patient s'afficheront ici.";
  } else if (codes.length) {
    node.className = "composer__signs";
    node.replaceChildren("Signes montrés au patient : ", el("strong", {}, codes.map(nameOf).join(", ")));
  } else {
    node.className = "composer__signs is-quiet";
    node.textContent = "Aucun signe connu : le patient verra le texte seul.";
  }
}

function renderSend() {
  dom.send.disabled = !dom.field.value.trim() || state.sending;
}

function onFieldChanged() {
  if (!dom.field.value.trim()) state.dictated = false;
  renderSignsPreview();
  renderSend();
}

function sendStatus(value) {
  if (state.sentStatus === value || !state.channel) return;
  state.sentStatus = value;
  state.channel.send({ type: "status", from: "doctor", state: value })
    .catch((error) => console.warn("[SignBridge] status not sent:", error));
}

/** "typing" while keys come, "idle" a few seconds after the last one. */
function noteTyping() {
  clearTimeout(state.typingTimer);
  if (state.listening) return;
  if (!dom.field.value.trim()) {
    sendStatus("idle");
    return;
  }
  sendStatus("typing");
  state.typingTimer = setTimeout(() => {
    if (!state.listening) sendStatus("idle");
  }, TYPING_IDLE_MS);
}

async function sendReply() {
  const text = dom.field.value.trim();
  if (!text || state.sending) return;
  if (!state.channel) {
    showProblem(LINK_TEXT.error);
    return;
  }
  if (state.listening) state.dictation?.stop();
  state.sending = true;
  renderSend();
  try {
    await state.channel.send({
      type: "message", id: newId(), ts: Date.now(),
      from: "doctor", via: state.dictated ? "voice" : "text",
      text, signs: signsFor(text),
    });
    dom.field.value = "";
    state.dictated = false;
    showProblem("");
    clearTimeout(state.typingTimer);
    sendStatus("idle");
  } catch (error) {
    console.error("[SignBridge] the message could not be sent:", error);
    showProblem("Message non envoyé : vérifiez la connexion, puis envoyez de nouveau. Votre texte est gardé.");
  } finally {
    state.sending = false;
    onFieldChanged();
    dom.field.focus({ preventScroll: true });
  }
}

function useQuestion(text) {
  dom.field.value = text;
  state.dictated = false;
  onFieldChanged();
  noteTyping();
  dom.field.focus({ preventScroll: true });
  dom.field.setSelectionRange(text.length, text.length);
}

function renderQuestions() {
  dom.questions.replaceChildren(...QUESTIONS.map((text) =>
    el("button", { type: "button", class: "btn btn--quiet question", "data-text": text }, text)));
}

/* ------------------------------------------------------------------ */
/* Dictation                                                           */
/* ------------------------------------------------------------------ */

function renderDictation() {
  const supported = Boolean(state.dictation?.supported);
  dom.dictate.disabled = !supported;
  dom.dictate.setAttribute("aria-pressed", String(state.listening));
  dom.dictate.textContent = state.listening ? "Arrêter la dictée" : "Dicter";
}

function setupDictation() {
  if (!state.speech) {
    state.dictation = null;
    dom.dictationNote.textContent = DICTATION_ERRORS["not-supported"];
    renderDictation();
    return;
  }
  state.dictation = state.speech.dictation({
    lang: "fr-FR",
    onText(text, _isFinal) {
      // Everything heard since start(), after what the field held before.
      const base = state.dictationBase;
      dom.field.value = base && text ? `${base} ${text}` : base || text;
      if (text) state.dictated = true;
      onFieldChanged();
    },
    onState(value, detail) {
      if (value === "listening") {
        state.listening = true;
        dom.dictationNote.textContent = `À l'écoute. ${DICTATION_NOTE}`;
        clearTimeout(state.typingTimer);
        sendStatus("listening");
      } else if (value === "error") {
        console.warn("[SignBridge] dictation:", detail);
        dom.dictationNote.textContent = DICTATION_ERRORS[detail] ?? DICTATION_ERROR_DEFAULT;
        state.dictationFailed = true;
        if (detail === "not-supported") {
          state.listening = false;
          sendStatus("idle");
        }
      } else {
        state.listening = false;
        if (!state.dictationFailed) dom.dictationNote.textContent = DICTATION_NOTE;
        sendStatus(dom.field.value.trim() && document.activeElement === dom.field ? "typing" : "idle");
        if (state.sentStatus === "typing") noteTyping();
      }
      renderDictation();
    },
  });
  if (!state.dictation.supported) dom.dictationNote.textContent = DICTATION_ERRORS["not-supported"];
  renderDictation();
}

function toggleDictation() {
  if (!state.dictation?.supported) return;
  if (state.listening) {
    state.dictation.stop();
    return;
  }
  state.dictationFailed = false;
  state.dictationBase = dom.field.value.trim();
  dom.dictationNote.textContent = DICTATION_NOTE;
  state.dictation.start();
  dom.field.focus({ preventScroll: true });
}

/* ------------------------------------------------------------------ */
/* Room                                                                */
/* ------------------------------------------------------------------ */

function renderLink() {
  let text = LINK_TEXT[state.link] ?? LINK_TEXT.connecting;
  if (state.link === "connected" && state.channel?.transport === "local") text = LINK_TEXT.local;
  dom.link.textContent = text;
  dom.link.dataset.state = state.link;
}

function setPatientActivity(signing) {
  clearTimeout(state.activityTimer);
  dom.activity.hidden = !signing;
  if (signing) {
    state.activityTimer = setTimeout(() => {
      dom.activity.hidden = true;
    }, ACTIVITY_TIMEOUT_MS);
  }
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
  appendMessage(event);
  if (event.from === "patient") {
    setPatientActivity(false);
    if (isLive(event)) readAloud(String(event.text ?? ""));
  }
}

function onStatus(event) {
  if (event.from !== "patient") return;   // our own status, echoed back
  setPatientActivity(event.state === "signing");
}

function onClear() {
  state.messages = [];
  state.seen.clear();
  setPatientActivity(false);
  try {
    globalThis.speechSynthesis?.cancel();
  } catch { /* nothing being said */ }
  renderThread();
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
    showProblem("La connexion avec l'écran du patient n'a pas démarré : rechargez la page.");
  }
}

/* ------------------------------------------------------------------ */
/* New consultation                                                    */
/* ------------------------------------------------------------------ */

function openConfirm() {
  dom.confirm.hidden = false;
  dom.confirmNo.focus({ preventScroll: true });
}

function closeConfirm() {
  dom.confirm.hidden = true;
  dom.newVisit.focus({ preventScroll: true });
}

async function confirmClear() {
  dom.confirm.hidden = true;
  if (!state.channel) {
    showProblem(LINK_TEXT.error);
    return;
  }
  try {
    await state.channel.send({ type: "clear" });
    showProblem("");
  } catch (error) {
    console.error("[SignBridge] the conversation could not be cleared:", error);
    showProblem("La conversation n'a pas été effacée : vérifiez la connexion, puis réessayez.");
  }
  dom.field.focus({ preventScroll: true });
}

/* ------------------------------------------------------------------ */
/* Boot                                                                */
/* ------------------------------------------------------------------ */

function bindControls() {
  dom.composer.addEventListener("submit", (event) => {
    event.preventDefault();
    sendReply();
  });
  dom.field.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      sendReply();
    }
  });
  dom.field.addEventListener("input", () => {
    onFieldChanged();
    noteTyping();
  });
  dom.field.addEventListener("blur", () => {
    if (!state.listening && state.sentStatus === "typing") {
      clearTimeout(state.typingTimer);
      sendStatus("idle");
    }
  });
  dom.questions.addEventListener("click", (event) => {
    const button = event.target.closest?.(".question");
    if (button) useQuestion(button.dataset.text);
  });
  dom.dictate.addEventListener("click", toggleDictation);

  dom.voice.addEventListener("click", () => {
    state.voiceOn = !state.voiceOn;
    if (!state.voiceOn) {
      try {
        globalThis.speechSynthesis?.cancel();
      } catch { /* nothing being said */ }
    }
    renderVoice();
  });

  dom.newVisit.addEventListener("click", openConfirm);
  dom.confirmNo.addEventListener("click", closeConfirm);
  dom.confirmYes.addEventListener("click", confirmClear);
  dom.confirm.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeConfirm();
  });

  window.addEventListener("pagehide", () => {
    state.dictation?.stop();
    try {
      state.channel?.close();
    } catch { /* closing anyway */ }
  });
}

async function boot() {
  state.room = readRoom();
  dom.roomName.textContent = `Salle ${state.room}`;
  bindControls();
  renderQuestions();
  renderLink();
  renderVoice();
  renderDictation();
  renderThread();
  onFieldChanged();

  try {
    await loadSigns();
  } catch (error) {
    // Text still works: the patient sees the doctor's words without signs.
    console.error("[SignBridge] the list of signs could not be loaded:", error);
    showProblem("La liste des signes n'a pas pu être chargée : le patient verra le texte seul. Rechargez la page pour réessayer.");
  }
  await loadLexicon();
  onFieldChanged();
  renderThread();
  loadFigure();
  loadSpeech();
  connect();
}

boot();

// The first click on the doctor screen unlocks the voice: read the message that was waiting.
document.addEventListener("pointerdown", () => {
  if (!state.unspoken) return showVoiceNote();
  const text = state.unspoken;
  state.unspoken = null;
  setTimeout(() => { showVoiceNote(); readAloud(text); }, 0);
}, { capture: true });
