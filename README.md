# SignBridge

A deaf patient and a hearing doctor talk in real time, each on their own screen, without an interpreter.

- **The patient signs.** Each sign the camera recognises becomes a word on the patient's screen. The patient builds a sentence from several signs, fixes or removes words, and sends it.
- **The doctor reads and hears it** as a proper sentence, in Tounsi ("أنا مريض. عندي سخانة وكحة.") or French ("Je suis malade. J'ai de la fièvre et de la toux."), then answers by typing, dictating in Tounsi or French, or choosing a ready-made phrase.
- **The patient sees the answer** in large text, and the signs for it performed by a stick figure, one after another.

Everything runs in the browser on one computer: the camera image is turned into body and hand points, and the sign model reads those points. No camera image is stored or sent anywhere.

![Patient and doctor screens side by side](docs/screens.jpg)

A 10-second demo on two phones: [docs/demo-mobile.mp4](docs/demo-mobile.mp4).

## Start it

Python 3 only (standard library); no install for the app itself.

```bash
git clone https://github.com/MohamedYassineAbid/Sign-Bridge.git
```

```bash
cd Sign-Bridge
```

```bash
python3 web/serve.py
```

Open http://localhost:8520 in Chrome or Edge, choose a room name, then:

- **Les deux sur cet écran**: both screens side by side, for a demo on one laptop.
- **Écran du patient** on the laptop with the camera, and **Écran du médecin** on another window or another device, with the same room name.

To let the doctor join from another device on the same network, start the server with `--host 0.0.0.0` and open `http://<laptop address>:8520/doctor.html?room=<room>` there. The patient screen must stay on `localhost` (browsers only give camera access on `localhost` or `https`).

**Tunisian speech (optional).** The doctor screen speaks Tounsi by default (the Tounsi / Français switch is at the top). It uses the Tunisian speech of the earlier demo when it is set up:

- **Dictation in Tounsi**: the Tunisian Vosk model ([Sali7a8603/Tunisian_STT](https://huggingface.co/Sali7a8603/Tunisian_STT), Apache 2.0), run on this computer, so the doctor's voice never leaves it. About 540 MB to download, 1.4 GB on disk, 1.5 GB of memory; it loads in a few seconds when the server starts.

```bash
pip install vosk
```

```bash
python3 web/speech_service.py download
```

- **The Tunisian voice**: ElevenLabs, voice `InB4o9iYj3MQ4HtGs2KV`. Put your key in a `.env` file at the repository root (it is never committed), then restart the server:

```bash
echo "ELEVENLABS_API_KEY=your-key" >> .env
```

Without them, the browser's Arabic dictation (Chrome, online) and Arabic voice are used; if the browser has no Arabic voice, the patient's messages are read in French. When it starts, `python3 web/serve.py` says which dictation and voice are in use, and the doctor screen says it under its buttons. `ELEVENLABS_VOICE_ID`, `ELEVENLABS_MODEL` (default `eleven_multilingual_v2`) and `--stt-model` change the voice, the voice model and the dictation model.

**The signs page.** http://localhost:8520/signs.html shows every sign as a stick figure you can play, says whether the camera recognises it, and lists the doctor's words that show it, in French and in Tounsi.

**Diagnosing a sign.** Add `&debug=1` to the patient screen's address (`patient.html?room=demo&debug=1`): each attempt is listed with what was proposed, the next guesses with their percentages, and how many frames had a hand. If the shoulders are out of view, the screen asks the patient to step back.

**Demonstration without a signer.** The patient screen has a "Démonstration sans signeur" control: it plays a reference recording of a medical sign through exactly the same recognition path, without showing the person in it, so a presenter can build and send a sentence.

## How a conversation works

| Step | Patient screen | Doctor screen |
|---|---|---|
| The patient signs "malade", lowers the hands, signs "fièvre" | Each recognised sign appears as a word; the last word offers the next two guesses as a correction | "Le patient signe…" |
| The patient presses **Envoyer** | The sentence is sent | "أنا مريض. عندي سخانة." and "Je suis malade. J'ai de la fièvre." appear; the first is read aloud in Tounsi |
| The doctor dictates "إمشي للسبيطار اليوم" (or types "Allez à l'hôpital aujourd'hui.") | "Le médecin écrit…" | Below the field: the signs the patient will see (Hôpital, Aujourd'hui) |
| The doctor presses **Envoyer** | The text in large letters, and the stick figure signs Hôpital then Aujourd'hui | |

## Where AI is used, and where it is not

| Task | Method | Why |
|---|---|---|
| Find the body and hands in the camera image | MediaPipe Holistic, in the browser | No rule can find a hand in pixels |
| Recognise a sign | A small neural network on the body and hand points | Signs vary with the person, speed and angle |
| Turn the patient's signs into a sentence | Fixed rules (`web/lib/lexicon.js`) | The vocabulary is closed; nothing needs generating |
| Find the signs for the doctor's words | Word matching with synonyms, in French and Tounsi (Arabic spelling variants ignored) | Predictable, and the doctor sees the result before sending |
| Show a sign to the patient | A stick figure replaying recorded movement (`web/lib/avatar.js`) | No video of a real person is shown |
| Dictation in Tounsi | The Tunisian Vosk (Kaldi) speech model, on the local server (`web/speech_service.py`) | Tunisian Arabic is poorly served by general speech services |
| Reading the patient's sentence in Tounsi | The Tunisian ElevenLabs voice, through the local server | A natural Tunisian voice; optional |
| French dictation and voice, and the fallback | The browser's speech services | Optional; browser dictation needs its online speech service |

## The signs

- **22 medical signs** (pain, headache, fever, cough, dizziness, diabetes, pregnancy, appointment, medicine, urgent…) in **Spanish Sign Language**, from SWL-LSE: 1,061 recordings by 124 signers. No open Tunisian medical sign dataset exists yet.
- **27 everyday words in Tunisian Sign Language** (bonjour, ça va, oui, non, vouloir, aider, hôpital, dispensaire, maison, family members, the seven days), from the Tunisian Sign Language Dataset made with ATILS.

Every sign can be shown to the patient as a stick figure. Not every sign is recognised from the patient's camera: a sign is proposed only if it was recognised reliably on signers the model never saw (see Results).

## Results

One model recognises both sets: 22 medical signs, 27 Tunisian words and an "other" class. Each set is scored so that it cannot flatter itself.

**Medical signs** (SWL-LSE official test split, signers' identity not checked against training):

| Measure | Result | Clips |
|---|---|---|
| Right sign among all 50 classes | 71.4% | 56 |
| Right sign in the top 3 | 92.9% | 56 |
| As the patient screen works (every proposed sign allowed), replaying held-out recordings frame by frame through the capture | 76% right first time, 15% "sign again", 7% wrong | 125 |

When the model hesitates between a few signs (Toux or Asthme, for example), the patient is offered its three best guesses to tap instead of only "sign again": on the same held-out recordings, 105 of 125 (84%) then end up as the right word. A wrong proposal is not sent: the patient sees the word, with the next two guesses offered as corrections, and can remove it.

**Hands resting in view.** In the training recordings a hand is seen only when raised (99.8% of sightings are less than 1.5 shoulder widths below the shoulders). At a desk, resting hands stay visible; with them left in, the held-out recordings dropped to 9% right. The patient screen therefore ignores a hand below that line, which restores 77% (`training/desk_check.py`). An earlier model trained on the medical signs alone scored 80.7% ± 1.3 over 5-fold cross-validation.

**Tunisian everyday words** (each of the 7 signers held out in turn, so the model never saw the person it is scored on):

| Word | Right on unseen signers |
|---|---|
| Père | 5 / 5 |
| Frère, Vendredi | 3 / 3 |
| Non | 3 / 4 |
| Mère, Jeudi | 2 / 2 |
| Oui | 2 / 3 |
| All 27 words | 28 / 74 (38%) |

The camera proposes a sign only if it scored at least 70% this way. Six Tunisian words pass (père, frère, vendredi, non, mère, jeudi); "oui" misses by one recording. The other words, and those performed by a single signer in the dataset (bienvenue, ça va, nom, lundi, mardi, mercredi), cannot be recognised reliably yet: they are shown to the patient as stick figures but never proposed from the camera. Among the medical signs, "essoufflement" is also shown but not proposed.

Metrics: `Demo/consultation/models/health_signs.json`.


## Limits

- **The medical signs are Spanish Sign Language.** They are the closest open dataset of health signs. Recording Tunisian medical signs with ATILS and deaf signers is the next step.
- **The Tunisian data is small**: a few seconds of images per word from one to five signers, cropped at the chin. Several words cannot be recognised reliably yet and are only shown, not recognised.
- **One sign at a time.** The patient lowers the hands between signs; continuous signing is not recognised.
- **Lighting and framing matter.** The camera must see the shoulders and the hands; hand tracking degrades in poor light.
- **Similar signs get confused.** Toux and Asthme, or Malade and Allergie, move alike; the screen then offers both.
- **It is not a diagnostic tool.** The sentence rules and the Arabic wording need review by a clinician and a native speaker.
- **Not yet tested on a real webcam with a deaf signer.** Recognition was checked with recorded signs through the same path the camera uses.
- **Tounsi dictation not yet tested with a Tunisian speaker here.** The whole path was checked (microphone, recording, the real model on the server, the words to signs), but with a test tone, not a voice. The Tunisian ready phrases and the Tounsi words that call each sign (`web/lib/lexicon.js`) need checking by a native speaker.
- **The ElevenLabs voice was checked against a stand-in**, not the real service (no key here): the request (voice, model, Tunisian text) and the playback were verified; the voice itself depends on your ElevenLabs plan, and some voices need a paid plan.

## Privacy

- The camera image is turned into body and hand points in memory, then dropped. No image or video is written or sent.
- The relay between the two screens keeps the conversation in memory only, and forgets it when the server stops or the doctor starts a new consultation.
- No name or identity is asked.
- Tounsi dictation with the Vosk model stays on the computer: the recording is transcribed in memory by the local server and dropped.
- What can leave the computer, each said on the doctor's screen:
  - the patient's sentences, sent to ElevenLabs when the Tunisian voice is set up (the server keeps the audio of recent sentences in memory, so a sentence is sent once);
  - the doctor's voice, when the browser's dictation is used (French, or Tounsi without the model);
  - the text read aloud, when the browser's voice is an online one.

## Project layout

| Path | What it is |
|---|---|
| `web/` | The app: `patient.html`, `doctor.html`, `split.html`, `serve.py` (static server and relay) and `speech_service.py` (Tunisian dictation and voice) |
| `web/lib/` | Tracker, features, capture, model, recogniser, lexicon, avatar, channel, speech |
| `web/data/` | The sign list and the stick-figure movements |
| `web/models/` | The sign model exported for the browser |
| `training/` | Building the datasets, training, exporting |
| `Demo/consultation/` | The Python reference of the recognition code, used by training |
| `Demo/` (other pages) | The earlier Streamlit experiments (text to speech, speech to text, sign to text) |

## Rebuilding the model

Needs `pip install numpy mediapipe==0.10.13 opencv-python-headless` (Python 3.11 or 3.12).

```bash
PYTHONPATH=Demo python -m training.swl_convert
```

```bash
git clone --depth 1 https://github.com/W7CH/First-ever-Tunisian-Sign-Language-Dataset data/tunsl
```

```bash
PYTHONPATH=Demo python -m training.tunisian_signs extract
```

```bash
PYTHONPATH=Demo python -m training.train_combined
```

```bash
PYTHONPATH=Demo python -m training.export_web
```

The first command fetches only the SWL-LSE recordings it needs (about 1,060 of 8,000) and reads them without running any code from the files. The stick figures are rebuilt with `training.export_poses` and `training.tunisian_signs poses`.

## Tests

```bash
node --test web/tests/
```

```bash
PYTHONPATH=Demo python -m pytest
```

## Data and credits

See [ATTRIBUTION.md](ATTRIBUTION.md). SWL-LSE (Universidade de Vigo) and the Tunisian Sign Language Dataset (Chakroun and Jerbi, with ATILS) are both CC BY 4.0. MediaPipe is Apache 2.0.
