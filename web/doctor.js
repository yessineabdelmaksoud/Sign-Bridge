// SignBridge, doctor screen (web/CONVERSATION.md).
//
// Patient messages arrive as a sentence (French and Tunisian Arabic), read aloud
// in the doctor's language. The doctor answers by typing, dictating or picking a
// ready-made question, in Tounsi or French; the screen shows in advance which
// signs the patient will see.
//
// Tounsi uses the local server when it is set up (web/speech_service.py): the
// Tunisian Vosk model for dictation, the Tunisian ElevenLabs voice for reading
// aloud. Otherwise the browser's own Arabic dictation and voice are used.
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
const LANG_KEY = "signbridge.doctor.lang";
const ARABIC = /[ء-ي]/;

// [French, Tunisian Arabic]. The Tunisian side must be checked by a native speaker.
const QUESTIONS = [
  ["Bonjour, comment allez-vous ?", "عسلامة، لاباس؟"],
  ["Bienvenue.", "مرحبا بيك."],
  ["Oui.", "إيه."],
  ["Non.", "لا."],
  ["Avez-vous de la fièvre ?", "عندك سخانة؟"],
  ["Avez-vous mal à la tête ?", "يوجعك راسك؟"],
  ["Toussez-vous ?", "عندك الكحة؟"],
  ["Avez-vous des vertiges ?", "تحس بالدوخة؟"],
  ["Avez-vous du mal à respirer ?", "عندك ضيق في النفس؟"],
  ["Êtes-vous diabétique ?", "عندك السكر؟"],
  ["Êtes-vous enceinte ?", "إنتي حامل؟"],
  ["Avez-vous une allergie ?", "عندك حساسية؟"],
  ["Prenez ce médicament.", "خوذ الدواء هذا."],
  ["Revenez pour un rendez-vous.", "إرجع في موعد آخر."],
  ["Revenez jeudi.", "إرجع نهار الخميس."],
  ["Allez à l'hôpital aujourd'hui.", "إمشي للسبيطار اليوم."],
  ["Allez au dispensaire.", "إمشي للمستوصف."],
  ["Votre famille peut vous aider.", "عايلتك تنجم تعاونك."],
  ["Voulez-vous que je vous l'écrive ?", "تحب نكتبهولك؟"],
];

const LINK_TEXT = {
  connecting: "Connexion…",
  connected: "Connecté",
  local: "Connecté, dans ce navigateur seulement",
  reconnecting: "Reconnexion…",
  offline: "Hors ligne : les messages ne partent pas",
  error: "La connexion n'a pas démarré : rechargez la page",
};

const DICTATION_NOTES = {
  fr: "La dictée utilise le service vocal du navigateur.",
  "tn-server": "Dictée en tounsi, faite sur cet ordinateur : appuyez, parlez, puis appuyez de nouveau.",
  "tn-browser": "Dictée en tounsi par le service vocal du navigateur.",
};
const DICTATION_ERRORS = {
  "not-supported": "La dictée n'est pas disponible dans ce navigateur : écrivez votre réponse.",
  "not-allowed": "Le micro est bloqué : autorisez le micro dans le navigateur, ou écrivez votre réponse.",
  "service-not-allowed": "Le service vocal du navigateur est bloqué : écrivez votre réponse.",
  "audio-capture": "Aucun micro trouvé : branchez un micro, ou écrivez votre réponse.",
  network: "Le service vocal ne répond pas : vérifiez la connexion, ou écrivez votre réponse.",
  "no-speech": "Aucune parole entendue : appuyez sur Dicter et parlez, ou écrivez votre réponse.",
  "no-model": "Le modèle de dictée tounsi n'est pas installé sur le serveur : écrivez votre réponse.",
  "too-long": "Enregistrement trop long (une minute au plus) : dictez en plusieurs fois.",
};
const DICTATION_ERROR_DEFAULT = "La dictée s'est arrêtée : réessayez, ou écrivez votre réponse.";

const VOICE_ERRORS = {
  "no-key": "pas de clé ElevenLabs sur le serveur",
  "key-refused": "clé ElevenLabs refusée",
  "paid-plan": "cette voix demande un abonnement ElevenLabs payant",
  "no-voice": "la voix tounsi n'est pas dans votre compte ElevenLabs",
  busy: "ElevenLabs est saturé",
  offline: "ElevenLabs ne répond pas (Internet ?)",
  network: "le serveur ne répond pas",
};

const $ = (selector) => document.querySelector(selector);

const dom = {
  roomName: $("#room-name"),
  link: $("#link-status"),
  lang: $("#lang"),
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
  lang: "tn",                // the doctor's language: "tn" (Tounsi) or "fr"
  server: { stt: false, tts: false },   // what the local server's Tunisian speech can do
  serverApi: null,
  voiceProblem: "",          // why the Tunisian voice fell back to the browser's, once
  unspoken: null,            // a patient message waiting for the first click to be read

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

/** The patient's signs as { fr, ar } sentences ("ar" is Tunisian Arabic). */
function sentenceOf(codes) {
  if (!state.lexicon || typeof state.lexicon.signsToSentence !== "function" || !codes.length) return { fr: "", ar: "" };
  try {
    const sentence = state.lexicon.signsToSentence(codes, state.signs);
    return { fr: String(sentence?.fr ?? ""), ar: String(sentence?.ar ?? "") };
  } catch (error) {
    console.warn("[SignBridge] signsToSentence failed:", error);
    return { fr: "", ar: "" };
  }
}

function patientCodes(event) {
  return Array.isArray(event.signs) ? event.signs.filter((code) => state.byCode.has(code)) : [];
}

/** A paragraph of text, set right to left when it is Arabic. */
function textLine(text, className) {
  if (ARABIC.test(text)) return el("p", { class: `${className} ar`, lang: "ar", dir: "rtl" }, text);
  return el("p", { class: className, lang: "fr" }, frenchSpaces(text));
}

function messageNode(event) {
  const codes = patientCodes(event);
  const list = codes.map(nameOf).join(", ");
  if (event.from === "patient") {
    // Both languages: the doctor's language first, in large letters.
    const sentence = sentenceOf(codes);
    const fr = String(event.text ?? "") || sentence.fr;
    const [first, second] = state.lang === "tn" && sentence.ar ? [sentence.ar, fr] : [fr, sentence.ar];
    return el("li", { class: "msg msg--patient" },
      el("p", { class: "msg__who" }, el("span", {}, "Patient"), el("span", { class: "msg__time" }, clock(event.ts))),
      textLine(first, "msg__text"),
      second ? textLine(second, "msg__text msg__text--second") : null,
      el("p", { class: "msg__meta" }, list ? `Signé : ${list}` : "Signé"));
  }
  const how = event.via === "voice" ? "Dicté" : "Écrit";
  const shown = shownToPatient(codes).map(nameOf).join(", ");
  return el("li", { class: "msg msg--doctor" },
    el("p", { class: "msg__who" }, el("span", {}, "Vous"), el("span", { class: "msg__time" }, clock(event.ts))),
    textLine(String(event.text ?? ""), "msg__text"),
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

function browserCanSpeak() {
  try {
    return Boolean(state.speech?.canSpeak());
  } catch {
    return false;
  }
}

/** Something can read aloud: the server's Tunisian voice (in Tounsi), or the browser. */
function canSpeak() {
  return (state.lang === "tn" && state.server.tts && Boolean(state.speech)) || browserCanSpeak();
}

function browserHasVoice(lang) {
  try {
    return Boolean(state.speech?.hasVoice?.(lang));
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

/** Which voice reads the patient's messages, said in one line under the buttons. */
function voiceNote() {
  if (state.lang !== "tn") return VOICE_NOTE;
  if (state.server.tts && !state.voiceProblem) {
    return "Voix tounsi : ElevenLabs. Les phrases du patient y sont envoyées pour être lues.";
  }
  const why = state.voiceProblem ? ` (${state.voiceProblem})` : "";
  return browserHasVoice("ar-TN")
    ? `Voix tounsi : celle du navigateur${why}.`
    : `Pas de voix arabe dans ce navigateur${why} : les messages sont lus en français.`;
}

function showVoiceNote() {
  const note = document.getElementById("voice-note");
  if (!note) return;
  const blocked = Boolean(state.unspoken) && state.voiceOn && canSpeak() && !voiceAllowed();
  note.textContent = blocked ? VOICE_BLOCKED : voiceNote();
  note.classList.toggle("is-warning", blocked);
}

/** Reads `text` with the browser's voice; Tunisian text without an Arabic voice falls back to `french`. */
function speakWithBrowser(text, french) {
  if (!browserCanSpeak()) return;
  try {
    if (ARABIC.test(text) && !browserHasVoice("ar-TN")) state.speech.speak(french, { lang: "fr-FR" });
    else state.speech.speak(text, { lang: ARABIC.test(text) ? "ar-TN" : "fr-FR" });
  } catch (error) {
    console.warn("[SignBridge] the message could not be read aloud:", error);
  }
}

/** Reads a patient message aloud in the doctor's language. */
function readAloud(event) {
  if (!state.voiceOn || !canSpeak()) return;
  if (!voiceAllowed()) {
    state.unspoken = event;   // read once the doctor clicks
    showVoiceNote();
    return;
  }
  const sentence = sentenceOf(patientCodes(event));
  const french = String(event.text ?? "") || sentence.fr;
  if (state.lang !== "tn" || !sentence.ar) {
    speakWithBrowser(french, french);
    return;
  }
  if (!state.server.tts) {
    speakWithBrowser(sentence.ar, french);
    return;
  }
  state.serverApi.synthesize(sentence.ar)
    .then((audio) => state.speech.playAudio(audio))
    .then((played) => {
      if (!played) throw Object.assign(new Error("not played"), { code: "network" });
      if (state.voiceProblem) {
        state.voiceProblem = "";
        showVoiceNote();
      }
    })
    .catch((error) => {
      // The Tunisian voice did not come: read with the browser now, and say why.
      console.warn("[SignBridge] the Tunisian voice failed:", error);
      state.voiceProblem = VOICE_ERRORS[error?.code] ?? "voix tounsi indisponible";
      showVoiceNote();
      speakWithBrowser(sentence.ar, french);
    });
}

function stopVoices() {
  try {
    globalThis.speechSynthesis?.cancel();
  } catch { /* nothing being said */ }
  try {
    state.speech?.stopAudio?.();
  } catch { /* nothing playing */ }
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
      globalThis.speechSynthesis?.addEventListener?.("voiceschanged", showVoiceNote);
    } catch { /* no speech here */ }
    if (typeof module.speechServer === "function") {
      state.serverApi = module.speechServer();
      state.server = await state.serverApi.status();
      console.info("[SignBridge] Tunisian speech on the server:", state.server);
    }
  } catch (error) {
    console.error("[SignBridge] lib/speech.js could not be loaded; no reading aloud, no dictation:", error);
  }
  renderVoice();
  showVoiceNote();
  setupDictation();
}

/* ------------------------------------------------------------------ */
/* Language                                                            */
/* ------------------------------------------------------------------ */

function readLang() {
  try {
    const saved = globalThis.localStorage?.getItem(LANG_KEY);
    if (saved === "tn" || saved === "fr") return saved;
  } catch { /* storage blocked */ }
  return "tn";
}

function renderLang() {
  for (const button of dom.lang.querySelectorAll("[data-lang]")) {
    button.setAttribute("aria-pressed", String(button.dataset.lang === state.lang));
  }
  dom.field.placeholder = state.lang === "tn"
    ? "Écrivez en tounsi ou en français, puis Entrée"
    : "Écrivez votre réponse, puis Entrée";
}

function setLang(lang) {
  if (lang === state.lang || (lang !== "tn" && lang !== "fr")) return;
  state.lang = lang;
  try {
    globalThis.localStorage?.setItem(LANG_KEY, lang);
  } catch { /* storage blocked */ }
  if (state.listening) state.dictation?.stop();
  stopVoices();
  renderLang();
  renderQuestions();
  renderVoice();
  showVoiceNote();
  setupDictation();
  renderThread();
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
  const tounsi = state.lang === "tn";
  dom.questions.replaceChildren(...QUESTIONS.map(([french, tunisian]) => {
    const text = tounsi ? tunisian : french;
    return el("button", {
      type: "button", class: `btn btn--quiet question${tounsi ? " ar" : ""}`, "data-text": text,
      lang: tounsi ? "ar" : "fr", dir: tounsi ? "rtl" : null, title: tounsi ? french : null,
    }, text);
  }));
}

/* ------------------------------------------------------------------ */
/* Dictation                                                           */
/* ------------------------------------------------------------------ */

function renderDictation() {
  const supported = Boolean(state.dictation?.supported);
  dom.dictate.disabled = !supported || state.transcribing;
  dom.dictate.setAttribute("aria-pressed", String(state.listening));
  dom.dictate.textContent = state.transcribing ? "Transcription…" : state.listening ? "Arrêter la dictée" : "Dicter";
}

/** "fr", "tn-server" (the Tunisian model on the server) or "tn-browser". */
function dictationMode() {
  if (state.lang !== "tn") return "fr";
  return state.server.stt && typeof state.speech?.serverDictation === "function" ? "tn-server" : "tn-browser";
}

function setupDictation() {
  state.dictation?.stop?.();
  state.listening = false;
  state.transcribing = false;
  if (!state.speech) {
    state.dictation = null;
    dom.dictationNote.textContent = DICTATION_ERRORS["not-supported"];
    renderDictation();
    return;
  }
  const mode = dictationMode();
  const note = DICTATION_NOTES[mode];
  const handlers = {
    onText(text, _isFinal) {
      if (state.dictation !== engine) return;
      // Everything heard since start(), after what the field held before.
      const base = state.dictationBase;
      dom.field.value = base && text ? `${base} ${text}` : base || text;
      if (text) state.dictated = true;
      onFieldChanged();
    },
    onState(value, detail) {
      if (state.dictation !== engine) return;   // an engine replaced by a change of language
      if (value === "listening") {
        state.listening = true;
        dom.dictationNote.textContent = `À l'écoute. ${note}`;
        clearTimeout(state.typingTimer);
        sendStatus("listening");
      } else if (value === "working") {
        state.listening = false;
        state.transcribing = true;
        dom.dictationNote.textContent = "Transcription du tounsi en cours…";
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
        state.transcribing = false;
        if (!state.dictationFailed) dom.dictationNote.textContent = note;
        sendStatus(dom.field.value.trim() && document.activeElement === dom.field ? "typing" : "idle");
        if (state.sentStatus === "typing") noteTyping();
      }
      renderDictation();
    },
  };
  const engine = mode === "tn-server"
    ? state.speech.serverDictation({ ...handlers, transcribe: (wav) => state.serverApi.transcribe(wav) })
    : state.speech.dictation({ ...handlers, lang: mode === "fr" ? "fr-FR" : "ar-TN" });
  state.dictation = engine;
  dom.dictationNote.textContent = engine.supported ? note : DICTATION_ERRORS["not-supported"];
  renderDictation();
}

function toggleDictation() {
  if (!state.dictation?.supported || state.transcribing) return;
  if (state.listening) {
    state.dictation.stop();
    return;
  }
  state.dictationFailed = false;
  state.dictationBase = dom.field.value.trim();
  dom.dictationNote.textContent = DICTATION_NOTES[dictationMode()];
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
    if (isLive(event)) readAloud(event);
  }
}

function onStatus(event) {
  if (event.from !== "patient") return;   // our own status, echoed back
  setPatientActivity(event.state === "signing");
}

function onClear() {
  state.messages = [];
  state.seen.clear();
  state.unspoken = null;
  setPatientActivity(false);
  stopVoices();
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
  dom.lang.addEventListener("click", (event) => {
    const button = event.target.closest?.("[data-lang]");
    if (button) setLang(button.dataset.lang);
  });

  dom.voice.addEventListener("click", () => {
    state.voiceOn = !state.voiceOn;
    if (!state.voiceOn) stopVoices();
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
  state.lang = readLang();
  dom.roomName.textContent = `Salle ${state.room}`;
  bindControls();
  renderLang();
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
  const event = state.unspoken;
  state.unspoken = null;
  setTimeout(() => { showVoiceNote(); readAloud(event); }, 0);
}, { capture: true });
