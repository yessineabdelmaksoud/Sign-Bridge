# SignBridge Health — design

Date: 2026-11-22
Status: draft, waiting for approval
Base code: the existing Sign-Bridge repo (Streamlit demo in `Demo/`)

Lines marked **(my call)** are choices made by the designer, not yet confirmed by the team. Anything else was decided in conversation.

## 1. Intent

Sign-Bridge today is a general Tunisian sign ↔ speech demo. For the Tech4Health hackathon it becomes one focused product: **a guided consultation between a deaf patient and a hearing doctor or pharmacist, without an interpreter.**

Why this shape:

- The hackathon scores relevance to Tunisian healthcare, a justified use of AI versus rules, responsible data handling, and a convincing demo. A general translator is weak on the first three.
- Open translation is not achievable with the data available. A closed medical vocabulary inside a fixed question flow is.
- The team wants real, useful medical signs in the demo. The source chosen for them is the open SWL-LSE dataset (Spanish Sign Language, health domain).

## 2. Goals, non-goals, anti-goals

**Goals**

1. A deaf patient completes a short consultation (main symptom, location, intensity, duration, key yes/no history) by signing or tapping.
2. The patient confirms every answer before the doctor sees it.
3. The doctor receives a structured written summary in French and Arabic.
4. Sign recognition is trained and measured on SWL-LSE with a split that does not leak, and the real numbers are shown.
5. The consultation page works with the network switched off.
6. The README describes what the code actually does.

**Non-goals**

- Open-ended sign translation or continuous signing.
- Diagnosis. The app records answers and raises urgency flags only.
- A mobile app, cloud deployment, or the Azure architecture in the current README.
- Free speech from the doctor inside the consultation (the current speech model scored a 62% word error rate).
- Retraining the speech or voice models.

**Anti-goals (these make it a failure even if it runs)**

- Claiming it understands Tunisian medical signs. It recognises Spanish Sign Language health signs plus a few Tunisian everyday signs.
- Inventing signs.
- Putting an unconfirmed answer into the doctor's summary, or speaking it aloud. The "Correct?" prompt itself appears on the shared screen; that is expected **(my call)**.
- Any camera image leaving the device or being stored.
- Reporting accuracy from a leaky split.

## 3. Constraints and assumptions

- Hackathon rules: functional end to end, free tools, pitch in English, privacy and ethics scored.
- Assumed **(my call)**: 24–48 hour event, 3–4 people, a few days of preparation, existing code allowed. None of this is confirmed.
- The team's connection is slow. Large downloads happen inside Colab or Kaggle, never on a laptop.
- No confirmed access to a deaf signer, ATILS, or a clinician. The design works without them and improves with them.
- Tunisian Sign Language derives from Italian Sign Language. No open Tunisian or Italian medical sign dataset was found.

## 4. Approaches considered

| | Approach | Trade-off |
|---|---|---|
| A | No new signs: sign only for yes/no, days, family; everything else by tapping | Buildable and honest, but signing plays a small role |
| B | A plus a "learn my signs" teach step (few examples per signer) | Strong demo and no dataset needed, but the matching is unproven |
| C | Record a Tunisian medical dataset with ATILS and retrain | Correct long-term path; needs weeks and a partner |
| **D** | **A plus a medical sign recogniser trained on an open health dataset (SWL-LSE)** | **Real medical signs, many signers, honest metrics; the signs are Spanish, not Tunisian** |

**Picked: D, built around SWL-LSE.** The team asked to add useful signs from an open dataset and chose SWL-LSE over the Google ASL set because it is health-specific, ships as landmarks, and is CC BY 4.0. B's teach step stays as an optional stretch **(my call)**. C is the roadmap slide.

## 5. How it works for the two users

Setting: one laptop or tablet with a camera on the desk, screen visible to both.

```
0. Consent        "The camera reads hand positions only. No video is saved."   patient: YES / tap
1. Why today?     SIGN: sick / appointment / medicine / urgent        (4 signs)
2. Main problem   SIGN a symptom, or tap a picture                    (14 signs)
3. Where?         body map tap            (only for pain or wound)
4. How strong?    0-10 faces scale tap    (only for pain)
5. Since when?    tap, or sign a day
6. Conditions     SIGN: diabetes / asthma / heart / pregnant, or "none" (4 signs)
                  repeats with "anything else?" until the patient says no
7. Yes/No series  allergy to a medicine? taking medicine now? fever? (if not already given)
8. Summary        doctor reads the note; urgency flag if a rule fires
9. Doctor's reply doctor picks from fixed instructions; patient sees picture + text
```

Steps 1, 2 and 6 are the three places where a medical sign is the answer, so every trained sign is a valid answer somewhere.

After every answer the app shows what it understood (picture + Arabic and French text) and asks "Correct?". Only a confirmed answer is added to the summary. Every signed answer also has a tap alternative, and an "I need an interpreter" button is always visible.

If recognition confidence is low, the app asks for the sign once more, then offers its top three guesses as pictures to tap.

## 6. AI versus rules

| Task | Method | Why |
|---|---|---|
| Find hands and body in the image | MediaPipe (AI) | No rule finds hands in pixels |
| Recognise a sign | Learned sequence model | Signs vary with person, speed, and angle |
| Question order, skip logic | Rules (a data file) | Must be predictable and reviewable |
| Urgency flags | Rules | Must be auditable; never a diagnosis |
| Summary text | Templates | Fixed vocabulary; no generation needed |
| Voice for the doctor | Pre-recorded Derja clips, optional | Closed vocabulary; no speech model or cloud |

The fixed question also narrows recognition: at each step only the valid answers are candidates. The pitch should show accuracy with and without this narrowing.

## 7. Recognition engine

### Data

**SWL-LSE** (Zenodo record 13691887, CC BY 4.0): 300 health signs, 8,000 clips, 124 signers, shared as MediaPipe output only (`MEDIAPIPE.zip`, 3.5 GB, one `.pkl` per clip with keys `pose`, `hands`, `holistic_legacy`). Official split: 6,348 train / 1,052 validation / 600 test, exactly 2 test clips per class. Clips per class: 12 to 54, median 23.

Pickle files can run code when opened. Open them only in Colab or Kaggle and convert once to plain arrays.

Sign subset **(my call)**: 22 signs in three answer sets, one per signed question (total clips in brackets):

- Why today (4): ENFERMO sick (53), CITA appointment (36), FARMACIA pharmacy (45), URGENTE urgent (36)
- Main problem (14): DOLOR pain (54), DOLOR-DE-CABEZA headache (37), FIEBRE2 fever (41), TOSER cough (33), MAREO dizzy (53), VOMITAR vomit (23), DIARREA (30), RESPIRAR breathe (31), CANSADO2 tired (54), HERIDA wound (41), ALERGIA2 allergy (40), GRIPE flu (40), INSOMNIO (36), ANSIEDAD anxiety (54)
- Conditions (4): DIABETES (54), ASMA (34), CORAZON heart (38), EMBARAZO pregnant (54)

Plus one "other" class sampled from the remaining signs, so the model can say "none of these". Signs with no question (HOSPITAL, AMBULANCIA, FUMAR, ALCOHOL, COMER, DORMIR) are left out.

**Tunisian set** (Mendeley `fbjjgzgv7f`, CC BY 4.0, 7 signers) for `oui`, `non`, `lyoum`, and the seven days **(my call)**:

- The shipped `tunisl_v4.keras` has seen every frame, so it cannot be scored honestly. Retrain a small model on these ten classes only, from the full Mendeley release (the repo holds a subset), and ship the retrained weights.
- Hold out whole signers where different people can be told apart in a folder; otherwise hold out contiguous blocks.
- If held-out accuracy on yes/no is below 90%, or there is too little held-out data to tell, signed yes/no, consent and days become tap-only and the demo taps those answers. Signing then stays in steps 1, 2 and 6.

### Features and model

- Input: upper-body pose plus both hands (the SWL-LSE baseline uses 61 keypoints: 19 pose, 21 per hand). Origin at the mid-shoulder point, scaled by shoulder width, so hand position relative to the body is kept. A missing hand becomes zeros plus a presence flag.
- Each clip is resampled to a fixed length (32 frames) **(my call)**. No sliding windows across clips.
- Model: keep the current BiLSTM with temporal attention **(my call)**; the builder may swap it if another small model does better on validation.
- Augmentation: the existing jitter, scale, rotation and time-warp, plus horizontal mirroring for left-handed signers.

### Evaluation

- Report top-1, top-3, per-class accuracy and a confusion matrix on the subset.
- The official test split gives only 2 clips per class, so also run 5-fold cross-validation over the subset and report the spread.
- Find out from the SWL-LSE paper whether the split separates signers. If it does not, or it is unclear, say so next to the number.
- Report accuracy three ways: all 22 signs at once, and narrowed to each of the three answer sets (4, 14 and 4 signs).
- Publish the retrained Tunisian model's held-out number, whatever it is.

### Live use

- Capture starts when a hand is raised and ends when hands drop or after about 4 seconds, with a visible recording indicator.
- Inference by direct model call or TFLite, not `model.predict` (the notebook measured 103 ms per call with it).
- A confidence threshold on the narrowed probabilities, tuned on validation.
- Presenters learn the selected signs from the 300 SWL-LSE reference videos (`VIDEOS_REF.zip`, 24 MB). The same clips can serve as "how to sign this" hints in the app.

## 8. Privacy, safety, fairness

- Only landmarks are computed, in memory. No frame is written to disk or sent anywhere.
- The consultation page makes no network calls: no ElevenLabs, no remote fonts. Demo it with Wi-Fi off.
- No name or identity field. Session data is cleared on "End consultation" or after 10 idle minutes. Export happens only when the doctor clicks.
- Revoke the committed ElevenLabs key; read any remaining keys from the environment.
- Add attribution for both datasets (CC BY 4.0) and a licence for the code.
- State the limits in the README and the pitch: Spanish medical signs, small Tunisian set, lighting sensitivity, not a diagnostic tool.
- Urgency rules and the question list should be reviewed by a doctor or medical student before the pitch.

## 9. Repository changes

- New page `Demo/pages/6_Consultation.py`, with the question flow as a data file, a recogniser module, and a summary module **(my call on layout)**.
- Rewrite the README: one name, real clone URL, real numbers, a limitations section, a demo GIF, and remove the claims the code does not support (fine-tuned XTTS, Whisper in the demo, "100+ signs", mobile and Azure deployment).
- Add `.gitignore`, remove `__pycache__`, keep one copy of the image data.
- Leave the existing pages in place; fix or remove the bad-word filter, which censors ordinary words.

## 10. Demo in three minutes

1. Switch Wi-Fi off in front of the judges.
2. A teammate plays the patient: consents, signs "sick", signs "pain", taps the chest, picks 8 on the scale, signs "diabetes", then answers two yes/no questions (signed if the Tunisian model passed its check, tapped if not).
3. The urgency flag appears; the doctor's summary appears in French and Arabic.
4. One slide: accuracy numbers, the AI-versus-rules table, and the plain statement of limits.
5. Roadmap: Tunisian medical signs recorded with ATILS, then pharmacy and emergency use.

## 11. Left to the builder

- Exact keypoint selection and feature layout, as long as body-relative position is kept.
- Pictogram source (an openly licensed set) and visual design.
- Body map zones, the wording of questions and instructions, the urgency rules (pending clinical review).
- Whether to record Derja audio clips.
- The teach step, if time allows.

## 12. Open points for the team

1. Is pre-built code and a pre-trained model allowed at the event?
2. Event date, duration, and team size.
3. Can the team reach a deaf signer, ATILS, or a clinician before the event?
