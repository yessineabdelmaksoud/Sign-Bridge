# SignBridge

SignBridge helps a deaf patient and a hearing doctor or pharmacist get through a short consultation without an interpreter. Both sit at one screen. The patient answers each question by signing or by tapping a picture, confirms what the app understood, and the doctor receives a written summary in French and Arabic.

The consultation page works with the network switched off, and no camera image is stored or sent anywhere.

![The consultation page](docs/consultation.png)

## What the consultation does

1. **Consent.** The patient is told that the camera reads hand positions only and agrees or refuses.
2. **Why are you here today?** Signed or tapped: sick, appointment, medicine, urgent.
3. **Main problem.** Signed or tapped: one of 14 symptoms.
4. **Where, how strong, since when.** Body map, 0–10 scale and duration, by tapping.
5. **Known conditions.** Signed or tapped: diabetes, asthma, heart disease, pregnancy, or none.
6. **Three yes/no questions.** Medicine allergy, current treatment, fever.
7. **Summary.** The doctor reads the note, sees any urgency flag, and can show the patient a reply such as "come back in 3 days".

After every answer the app shows what it understood and asks "correct?". Only a confirmed answer reaches the doctor's note. Every question can be answered by tapping, and a "request an interpreter" button is always on screen.

## Where AI is used, and where it is not

| Task | Method | Why |
|---|---|---|
| Find hands and body in the camera image | MediaPipe (learned model) | No rule can find a hand in pixels |
| Recognise a sign | Small neural network on landmarks | Signs vary with the person, speed and angle |
| Question order and skip logic | Rules in `Demo/consultation/flow.json` | Must be predictable and reviewable by a clinician |
| Urgency flags | Rules in the same file | Must be auditable; they are never a diagnosis |
| Summary text | Fixed templates | The vocabulary is closed, so nothing needs generating |

The fixed question also makes recognition easier: at each step the model chooses only among that question's valid answers (4, 14 or 4 signs) instead of all 22.

## Results

All numbers are for the 22 health signs plus an "other" class, on SWL-LSE (Spanish Sign Language): 1,061 clips. Whether the dataset's split keeps signers apart is not stated in what we checked, so treat these as accuracy on unseen clips, not proven accuracy on unseen people.

| Measure | Result | Clips |
|---|---|---|
| Right sign among all 23 classes (5-fold cross-validation) | 80.7% ± 1.3 | 1,061 |
| Right sign in the top 3 (5-fold cross-validation) | 93.3% ± 1.5 | 1,061 |
| "Why today?", choosing among its 4 answers | 92.4% ± 2.8 | 170 |
| "Main problem", choosing among its 14 answers | 90.6% ± 2.3 | 567 |
| "Known conditions", choosing among its 4 answers | 91.0% ± 3.4 | 180 |
| Official test split, all 23 classes | 75.0% | 56 |
| Official test split, per question (4 / 14 / 4 answers) | 87.5% / 85.7% / 87.5% | 8 / 28 / 8 |

The official test split has two clips per sign, which is why the cross-validation figures are the ones to quote.

**Through the camera path.** Replaying the 125 held-out clips (validation and test) frame by frame through the capture logic and the shipped model: 107 were proposed correctly (86%), 14 led to "please sign again", 2 were not captured, and 2 were proposed wrongly (1.6%), which the patient's "not correct" button is there to catch. The validation clips were used to choose the stopping point and the threshold, so this is slightly optimistic.

**What the app does with this.** A proposal is made only above a confidence threshold of 0.40, chosen so that wrong proposals stayed under 5% on validation. One sign, "heart disease", scored 3 of 6 on held-out clips and is therefore answered by tapping only. On a laptop CPU the full camera path runs at about 10 frames per second.

The model is a small fully connected network (two hidden layers) on hand and upper-body landmarks, trained and run with NumPy. Metrics file: `Demo/consultation/models/health_signs.json`. Confusion matrix: `docs/metrics/health_signs_confusion.png`.


## Limits

- **The medical signs are Spanish Sign Language, not Tunisian.** No open Tunisian medical sign dataset exists yet. The 22 health signs come from SWL-LSE. Recording Tunisian medical signs with ATILS is the next step.
- **Signed yes/no and days are not switched on.** The Tunisian model for them is trained only when its accuracy on a separate take reaches 90% (`training/train_tunisl10.py`). Until then those answers are tapped.
- **Lighting and framing matter.** Hand tracking degrades in poor light or when the hands leave the frame.
- **It is not a diagnostic tool.** It collects answers. The urgency rules and the wording of the questions are waiting for review by a clinician, and the Arabic labels for review by a native speaker.
- **The camera path has not been tested end to end on a real webcam** by the author of this change. The recogniser, the capture logic and the page (by tapping) are covered by automated tests.

## Privacy

- The camera image is turned into hand and body landmarks in memory and then dropped. No image or video is written to disk or sent anywhere.
- The consultation page makes no network calls. A test (`tests/test_offline.py`) fails if network or file-writing code is added to it.
- No name or identity is asked. Answers live in memory and are erased by "Terminer et effacer les réponses" or after 10 idle minutes. The note leaves the app only when the doctor clicks "Télécharger la note".
- The camera stays off until the patient switches it on.

## Quick start

Python 3.11 or 3.12.

```bash
git clone https://github.com/MohamedYassineAbid/Sign-Bridge.git
cd Sign-Bridge
python -m venv .venv && source .venv/bin/activate
```

To run the consultation by tapping (small install):

```bash
pip install streamlit "numpy<2"
```

To sign answers with the camera as well:

```bash
pip install streamlit-webrtc av mediapipe==0.10.13 opencv-python-headless
```

Then:

```bash
streamlit run Demo/app.py
```

and open **Consultation** in the sidebar. To install everything the other demo pages need (TensorFlow, Vosk, ElevenLabs client), use `pip install -r Demo/requirements.txt`.

## The other demo pages

| Page | What it really uses |
|---|---|
| Text to Speech | The ElevenLabs cloud API. Set `ELEVENLABS_API_KEY` in your environment. |
| Speech to Text | A Vosk model for Tunisian Arabic, downloaded on first use. |
| Sign to Text | The `tunisl_v4` Keras model: 55 everyday Tunisian signs, one hand. Its published accuracy came from a leaky split and should not be quoted. |
| Text to Sign | Word-by-word lookup of 58 signs, drawn as a skeleton. Unknown words are skipped. |
| Conversation Bridge | The four above combined. |

The notebooks in `Tun_STT/`, `Tun_TTS/` and `Tun_SiTT/` are experiments: a Whisper fine-tune that reached a 62% word error rate, an XTTS fine-tune that did not complete, and the training of `tunisl_v4`.

## Rebuilding the health sign model

```bash
PYTHONPATH=Demo python -m training.swl_convert --out data/swl22.npz
PYTHONPATH=Demo python -m training.train_health --data data/swl22.npz
```

The first command fetches only the clips it needs from SWL-LSE (about 1,060 of 8,000) with ranged downloads, and reads them without executing the pickle files they come in. The second trains the network, scores it, and writes the model and its metrics to `Demo/consultation/models/`.

## Tests

```bash
pip install pytest
pytest
```

## Roadmap

- Record Tunisian medical signs with ATILS and deaf signers, and retrain.
- Train and gate the Tunisian yes/no and day signs.
- Clinical review of the questions and urgency rules.
- A sequence model (BiLSTM or similar) in place of the small network, if it scores better on the same split.
- Pharmacy and emergency versions of the question flow.

## Data and credits

See [ATTRIBUTION.md](ATTRIBUTION.md). The Tunisian Sign Language Dataset (Chakroun and Jerbi, with ATILS) and SWL-LSE (Universidade de Vigo) are both CC BY 4.0. Tunisian speech resources: TunArTTS and the Linagora Tunisian datasets.
