# SignBridge conversation: shared contract

This extends `web/CONTRACT.md` (data shapes, the tracker, the recogniser and the capture stay as they are). It describes the two-screen conversation: a deaf patient and a hearing doctor talk in real time, each on their own screen.

## The real setup

- **Patient screen**: a laptop or tablet facing the patient, with a camera. The patient signs. Each recognised sign is added to a sentence being built on screen; the patient fixes or removes words, then sends. Messages from the doctor arrive as large text and as signs performed by a stick figure, one after another.
- **Doctor screen**: the doctor's computer, no camera. Patient messages arrive as a written French sentence, read aloud by the browser. The doctor answers by typing, by dictating, or with a ready-made question; the screen shows in advance which signs the patient will see.
- **One laptop for a demo**: `split.html` shows both screens side by side.

Both screens join a *room* (`?room=<name>`, default `demo`). Everything sent in a room reaches both screens at once.

## Signs shown as a stick figure (decided)

The patient never sees the dataset's videos of real people. Signs are drawn as a stick figure (lines for the body, arms and fingers) by `web/lib/avatar.js`, from `web/data/sign_poses.json`, which `training/export_poses.py` extracted once from the reference videos (movement only). The reference videos are used only as input to the tracker in the demonstration control, and are never displayed.

```js
import { loadSignPoses, SignAvatar } from "./lib/avatar.js";
const poses = await loadSignPoses();             // data/sign_poses.json
const avatar = new SignAvatar(canvas, poses);     // neutral figure at rest; fits any canvas size
avatar.has(code);                                 // pose data exists for that sign
await avatar.play(codes, { onSign(code, index) {} });   // signs one after another; resolves at the end
avatar.stop();                                    // back to rest; the pending play resolves
```

## Design rules (decided)

- **No icons, no emoji, no pictograms anywhere.** Words only, in buttons and labels.
- Palette, and nothing beyond it: white `#FFFFFF`, ink `#10233A`, blue `#12508F`, wall grey `#EEF2F6`; green `#0F8A5F` only for "recognised / sent", amber `#E8A100` only for "recording / someone is signing or talking".
- System fonts only (Latin: `"Inter", "Noto Sans", "Segoe UI", "Helvetica Neue", Arial, sans-serif`; Arabic: `"Noto Kufi Arabic", "Noto Sans Arabic", "Segoe UI", Tahoma, sans-serif`, right to left).
- Flat: no gradients, shadows or decorative motion. Tap targets at least 56 px. Visible keyboard focus.
- Patient screen: Arabic above French for every label the patient must read. Doctor screen: French.
- No network request leaves the machine, except the browser's own speech services if the doctor uses dictation (say so next to the dictation button).

## Files and owners

| Path | Owner |
|---|---|
| `web/serve.py` (add the relay), `tests/test_relay.py` | relay |
| `web/lib/channel.js`, `web/lib/lexicon.js`, `web/lib/speech.js`, `web/tests/channel.test.js`, `lexicon.test.js`, `speech.test.js` (and add them to `web/tests/index.js`) | relay |
| `web/index.html`, `web/split.html`, `web/patient.html`, `web/doctor.html`, `web/style.css`, `web/patient.js`, `web/doctor.js` | ui |
| `web/app.js`, `web/styles.css` (the old single-screen demo): may be deleted by ui | ui |

Existing files that stay unchanged: `web/lib/landmarks.js`, `features.js`, `capture.js`, `pace.js`, `model.js`, `recognizer.js`, `tracker.js`, `web/data/signs.json`, `web/models/`, `web/clips/`, `web/vendor/`.

## Events

Every event is a JSON object. Three types:

```js
{ type: "message", id, ts,              // id and ts are added by the relay if missing
  from: "patient" | "doctor",
  via: "sign" | "text" | "voice",
  text: "Je suis malade. J'ai de la fièvre.",   // what the other side reads (and the doctor hears)
  signs: ["ENFERMO", "FIEBRE2"] }               // patient: the signs made; doctor: the signs to show the patient

{ type: "status", from: "patient" | "doctor",
  state: "online" | "idle" | "signing" | "typing" | "listening" }   // not stored, only passed on

{ type: "clear" }                                 // new consultation: both screens empty their conversation
```

## The relay (in `web/serve.py`)

Python standard library only. `ThreadingHTTPServer` is already used.

- `GET /relay/ping` → `200 {"ok": true}`.
- `GET /relay/events?room=R` → `text/event-stream`. On connect it first sends every stored event of room R, in order, then each new event as it happens. Format: `data: <json>\n\n`. A comment line `: ping\n\n` every 15 seconds keeps the connection open.
- `POST /relay/send?room=R`, body one JSON event → `message`: add `id` (if missing) and `ts` (milliseconds), store it (the last 200 per room, in memory only), pass it on. `status`: pass it on, do not store. `clear`: empty the room's store, store and pass on the `clear`. Answer `200 {"ok": true, "id": ...}`.
- Validation: room matches `[A-Za-z0-9_-]{1,32}`, body at most 16 KB, `type` one of the three, `from` one of the two where required; otherwise `400` with a short reason.
- Nothing is written to disk. Listens on `127.0.0.1` by default; `--host 0.0.0.0` lets a doctor join from another device on the same network (the camera still needs the patient screen on `localhost` or `https`).

## `web/lib/channel.js`

```js
export async function joinRoom(room, { onEvent, onState }) // -> { transport: "relay" | "local", send(event): Promise<void>, close(): void }
```

- Uses the relay when `/relay/ping` answers; otherwise falls back to `BroadcastChannel("signbridge:" + room)` (two tabs of the same browser, no history).
- **Every event sent reaches `onEvent` of every participant, the sender included, exactly once.** (The relay echoes to the sender; in local mode `send` must call `onEvent` itself.)
- `onState("connected" | "reconnecting" | "offline")` reports the connection. Reconnects on its own after the relay restarts, without delivering stored events twice (keep the ids already seen).

## `web/lib/lexicon.js`

```js
export function signsToSentence(codes, signs)   // -> { fr: string, ar: string }
export function textToSigns(text, signs)        // -> string[] of sign codes
export function signName(code, signs, lang)     // -> the sign's name in "fr" or "ar"
```

`signs` is the `signs` array of `web/data/signs.json`.

**`signsToSentence`** turns the patient's signs, in the order they were made, into natural sentences, by rule (no text generation):

- Group by the sign's `group`, keeping the order in which groups first appear.
- `reason`: ENFERMO "Je suis malade.", CITA "J'ai un rendez-vous.", FARMACIA "J'ai besoin d'un médicament.", URGENTE "C'est urgent."
- `symptom`: one sentence, "J'ai " + the phrases joined with ", " and " et " before the last, + ".". Phrases: DOLOR "mal", DOLOR-DE-CABEZA "mal à la tête", FIEBRE2 "de la fièvre", TOSER "de la toux", MAREO "des vertiges", VOMITAR "des vomissements", DIARREA "la diarrhée", RESPIRAR "du mal à respirer", CANSADO2 "de la fatigue", HERIDA "une blessure", ALERGIA2 "une allergie", GRIPE "la grippe", INSOMNIO "des insomnies", ANSIEDAD "de l'anxiété". (So: "J'ai de la fièvre, de la toux et mal à la tête.")
- `conditions`: DIABETES "Je suis diabétique.", ASMA "J'ai de l'asthme.", CORAZON "J'ai une maladie du cœur.", EMBARAZO "Je suis enceinte."
- A sign repeated twice in a row counts once. Unknown codes are ignored. No signs gives `""`.
- The Arabic side (`ar`) does the same in Tunisian Arabic, written in Arabic script; keep it simple and mark in a comment that a native speaker must check it.

**`textToSigns`** finds the signs to show for a doctor's sentence: case- and accent-insensitive, whole words, longest match first ("mal à la tête" before "mal"), each sign at most once, in order of appearance. Match the French names in `signs.json` plus synonyms, at least: malade, maladie; rendez-vous; médicament, médicaments, pharmacie, ordonnance, traitement; urgent, urgence; mal, douleur, douleurs; mal de tête, mal à la tête, migraine; fièvre, température; toux, tousser, toussez; vertige, vertiges, étourdissement; vomir, vomissement, vomissements, vomi; diarrhée; respirer, respiration, souffle, essoufflé; fatigue, fatigué, fatiguée; blessure, plaie, blessé; allergie, allergique; grippe, rhume; insomnie, dormir, sommeil; anxiété, angoisse, stress, stressé; diabète, diabétique; asthme; cœur, coeur, cardiaque; enceinte, grossesse.

## `web/lib/speech.js`

```js
export function canSpeak()                                  // speechSynthesis exists
export function pickVoice(voices, lang = "fr-FR")           // pure: prefers a local voice of that language, then any of that language, then null
export function speak(text, { lang = "fr-FR" } = {})       // cancels what is being said; does nothing if speech is unavailable
export function dictation({ lang = "fr-FR", onText, onState }) // -> { supported, start(), stop() }
// onText(text, isFinal); onState("listening" | "idle" | "error", detail)
```

`dictation` wraps `SpeechRecognition` / `webkitSpeechRecognition` with interim results. Nothing throws when speech is not supported.

## Patient screen (`patient.html`, `patient.js`)

- Joins the room, sends `status online`.
- Recognition: the same pipeline as the current `web/app.js` (`Tracker`, `Recognizer`, `pacedCapture` from `lib/pace.js`, the frame loop, `paceClip` for videos, and `capture.finish()` when a video ends). Keep the capture recognising **across all 22 signs**.
- **Building a sentence from several signs.** The patient signs one sign, lowers the hands, signs the next. Each recognised sign is appended to the sentence as a word (French large, Arabic small). The last word shows up to two alternatives (the next candidates of `top3`) as small buttons: tapping one replaces that word. Tapping any word removes it. An unrecognised sign shows "Signe non reconnu, recommencez" for two seconds and adds nothing.
- Under the words, the sentence the doctor will receive (`signsToSentence(...).fr`, and `.ar`), updated live.
- Two large buttons: **Envoyer** (green) sends `{type: "message", from: "patient", via: "sign", text: <sentence fr>, signs: <codes>}` and empties the sentence; **Effacer** removes the last word.
- `status signing` while the capture is recording, `idle` otherwise (only on change).
- **Doctor messages**: a panel shows the doctor's text very large; beside it a video plays the reference clip of each sign in `signs`, one after another, with the sign's name under it; a **Revoir** button replays them. No signs: the text alone.
- When the doctor's status is `typing` or `listening`, show "Le médecin écrit…" or "Le médecin parle…".
- A short history of the conversation (last few messages, words only).
- **Demonstration without a signer**: a small, clearly secondary control (a select with the 22 sign names and a button "Faire signer la vidéo") plays that sign's reference clip in the camera area, through exactly the same pipeline, so a presenter can build and send a sentence from reference videos.

## Doctor screen (`doctor.html`, `doctor.js`)

- Joins the room, sends `status online`.
- The conversation as a thread: patient messages (with "signé" and the signs made, small), the doctor's own messages (with "écrit" or "dicté").
- Every new patient message is read aloud with `speak` (a toggle "Lecture à voix haute : activée / désactivée", on by default).
- A live line: "Le patient signe…" while the patient's status is `signing`.
- Composer: a text field and **Envoyer** (Enter sends). **Dicter** starts dictation; interim text appears in the field; the doctor reviews and sends. Next to it: "La dictée utilise le service vocal du navigateur." Sends `status typing` / `listening` / `idle` on change.
- Under the field, live: "Signes montrés au patient : Fièvre, Toux" from `textToSigns`, or "Aucun signe connu : le patient verra le texte seul."
- Ready-made questions as text buttons that fill the field: "Avez-vous de la fièvre ?", "Avez-vous mal à la tête ?", "Toussez-vous ?", "Avez-vous des vertiges ?", "Avez-vous du mal à respirer ?", "Êtes-vous diabétique ?", "Êtes-vous enceinte ?", "Avez-vous une allergie ?", "Prenez ce médicament.", "Revenez pour un rendez-vous."
- **Nouvelle consultation** sends `clear` after a confirmation.

## Landing and split view

- `index.html`: the name, one sentence on what it does, a room field (default `demo`), and three buttons: **Écran du patient**, **Écran du médecin**, **Les deux sur cet écran**.
- `split.html?room=R`: the two screens side by side in iframes (`allow="camera; microphone; autoplay"`), patient on the left.
