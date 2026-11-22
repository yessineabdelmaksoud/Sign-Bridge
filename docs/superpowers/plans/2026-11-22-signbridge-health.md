# SignBridge Health Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a guided deaf-patient ↔ doctor consultation page to Sign-Bridge, with a health sign recogniser trained on SWL-LSE, honest metrics, and no network use.

**Architecture:** Pure-Python logic modules under `Demo/consultation/` (landmark format, features, capture, flow, session, summary, recogniser) are unit-tested without TensorFlow or MediaPipe. One Streamlit page wires them to the camera. Training scripts under `training/` run in Colab and write model files into `Demo/consultation/models/`.

**Tech Stack:** Python 3.11, Streamlit, streamlit-webrtc, MediaPipe 0.10.13 (legacy Holistic), TensorFlow/Keras, NumPy, pytest.

**Spec:** `docs/superpowers/specs/2026-11-22-signbridge-health-design.md`

**Where to work:** a real `git clone` of `MohamedYassineAbid/Sign-Bridge`. All paths below are relative to that repo root. Tasks 8, 9 and 11 run in Colab or Kaggle (3.5 GB download, GPU). Tasks 2–7 need only NumPy and pytest.

## Changes made while building

These decisions were taken during execution and override the text below where they differ.

- **Model:** the health sign model is a small fully connected network trained and run with NumPy (`training/mlp.py`, `Demo/consultation/numpy_model.py`), not the Keras BiLSTM. The connection could not download TensorFlow, and a NumPy model keeps the consultation page independent of it. The recogniser still loads a `.keras` file if one is supplied.
- **SWL-LSE access:** only the needed clips are fetched, with ranged downloads (`training/swl_remote.py`), and the pickles are read by a loader that executes nothing from them (`training/swl_convert.py`). This replaces the "open only in Colab or Kaggle" constraint and lets conversion and training run on a laptop.
- **Python:** 3.12 was used; 3.11 also works.
- **Page styling:** the consultation page has its own offline stylesheet (`Demo/consultation/ui.py`); `apply_tunisign_theme` was left unchanged.
- **Page tests:** the page is driven by Streamlit's `AppTest` in `tests/test_consultation_page.py`.
- **Task 11:** script, hold-out logic and gate are written and tested; the training run is still to do (it needs MediaPipe and the full Tunisian dataset). Signed yes/no and days stay tap-only until then.
- **Licence:** not added; the repo owner decides.

## Global Constraints

- The consultation page and `Demo/consultation/` make no network calls: no `elevenlabs`, `requests`, `urllib`, or remote URLs.
- No camera frame is written to disk or sent anywhere; only landmarks are held, in memory.
- An unconfirmed answer never enters the summary and is never spoken.
- Sign set: exactly 22 SWL-LSE signs plus one `OTHER` class (lists in Task 4).
- Every signed question also has a tap answer; an "I need an interpreter" control is always visible.
- Clip length 32 frames; feature width 184; idle timeout 600 seconds.
- Modules in `Demo/consultation/` import only the standard library and NumPy at module level; TensorFlow and MediaPipe are imported inside functions.
- SWL-LSE `.pkl` files are opened only in Colab or Kaggle, never on a laptop.
- Both datasets are CC BY 4.0 and must be attributed.
- Never claim Tunisian medical sign recognition; the medical signs are Spanish Sign Language.
- Report every metric from held-out data and name the split next to it.

## Review Focus

1. **No hand ever detected** (dark room, blocked camera): every question stays answerable by tap and nothing hangs. Test in Task 3 and manual check in Task 10.
2. **Hand leaves the frame mid-sign:** features contain no NaN. Test in Task 2.
3. **Left-handed signer:** a mirrored clip is handled as the same sign. Test in Task 2, augmentation in Task 9.
4. **Model file missing or unreadable:** the page still runs tap-only with a visible notice. Test in Task 7, manual check in Task 10.
5. **Abandoned session, next patient sits down:** no earlier answer is shown. Test in Task 5.

---

### Task 1: Repository safety and test scaffold

**Files:**
- Modify: `Demo/app.py:6`, `Demo/pages/5_Conversation_Bridge.py:52`, `Tun_STT/app.py` (the `filter_bad_words` default), `Tun_TTSi/README.md`
- Create: `.gitignore`, `pytest.ini`, `ATTRIBUTION.md`, `LICENSE`, `tests/test_repo_hygiene.py`
- Delete: `Demo/pages/__pycache__/`, `Tun_TTSi/data/`

**Interfaces:**
- Produces: `pytest` runs from the repo root with `Demo/` on the import path.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_repo_hygiene.py
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def test_no_hardcoded_api_key():
    for p in (ROOT / "Demo").rglob("*.py"):
        assert not re.search(r"sk_[0-9a-f]{20,}", p.read_text(encoding="utf-8")), p

def test_no_bytecode_tracked():
    assert not list((ROOT / "Demo").rglob("*.pyc"))

def test_single_copy_of_sign_images():
    assert not (ROOT / "Tun_TTSi" / "data").exists()
```

- [ ] **Step 2: Create `pytest.ini`** with `pythonpath = Demo .` and `testpaths = tests`, and an empty `tests/__init__.py`. Run `pytest tests/test_repo_hygiene.py -v`. Expected: 3 FAIL.
- [ ] **Step 3: Replace both hard-coded keys** with `API_KEY = os.getenv("ELEVENLABS_API_KEY", "")` (add `import os`).
- [ ] **Step 4: Add `.gitignore`** (`__pycache__/`, `*.pyc`, `.venv/`, `.env`, `Demo/model/`) and run `git rm -r --cached Demo/pages/__pycache__`.
- [ ] **Step 5: Remove the duplicate images.** Run `cmp Tun_TTSi/app.py Demo/TextToSign/app.py`. If identical, `git rm -r Tun_TTSi/data Tun_TTSi/app.py Tun_TTSi/requirements.txt` and point `Tun_TTSi/README.md` at `Demo/TextToSign/`. If they differ, remove only `Tun_TTSi/data` and set that app's data root to `../Demo/TextToSign/data`.
- [ ] **Step 6: Set `filter_bad_words: bool = False`** as the default in `Tun_STT/app.py`. The word list is built from whole abusive sentences, so it censors ordinary words.
- [ ] **Step 7: Write `ATTRIBUTION.md`** naming the Tunisian Sign Language Dataset (Chakroun and Jerbi, Mendeley Data, DOI 10.17632/fbjjgzgv7f.1, CC BY 4.0) and SWL-LSE (Zenodo record 13691887, CC BY 4.0; Vázquez-Enríquez et al., Technologies 2024, 12(10), 205).
- [ ] **Step 8: Add `LICENSE`** (MIT) only after the repo owner agrees to that licence.
- [ ] **Step 9: Run `pytest tests/test_repo_hygiene.py -v`.** Expected: 3 PASS.
- [ ] **Step 10: Commit** `chore: remove committed key, bytecode and duplicate data; add test scaffold`.

Human step, outside the repo: the owner revokes the old ElevenLabs key in the ElevenLabs dashboard. It remains in git history.

---

### Task 2: Landmark format and features

**Files:**
- Create: `Demo/consultation/__init__.py`, `Demo/consultation/landmarks.py`, `Demo/consultation/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Produces, in `landmarks.py`:
  - Canonical frame: `np.ndarray` shape `(75, 3)` float32. Rows 0–32 pose, 33–53 left hand, 54–74 right hand. `x`, `y` are image-normalised; a missing hand is all NaN.
  - `POSE = slice(0, 33)`, `LEFT = slice(33, 54)`, `RIGHT = slice(54, 75)`
  - `hand_present(frame: np.ndarray) -> bool` — true if rows 33–74 of a `(75, 3)` frame hold any finite value.
  - `mirror_clip(clip: np.ndarray) -> np.ndarray` — `x → 1 − x`, swap left and right hand blocks, swap MediaPipe pose pairs (1,4) (2,5) (3,6) (7,8) (9,10) (11,12) (13,14) (15,16) (17,18) (19,20) (21,22) (23,24) (25,26) (27,28) (29,30) (31,32).
  - `canonical_from_holistic(results) -> np.ndarray` — reads `.pose_landmarks`, `.left_hand_landmarks`, `.right_hand_landmarks`, each `None` or an object with `.landmark[i].x/.y/.z`. Missing pose gives an all-NaN frame.
- Produces, in `features.py`:
  - `FRAMES = 32`, `WIDTH = 184`, `class NoHandError(ValueError)`
  - `trim_to_hands(clip) -> np.ndarray` — keeps first to last frame with a hand; raises `NoHandError` if none.
  - `resample(clip, frames: int = FRAMES) -> np.ndarray` — nearest-index sampling with `np.round(np.linspace(0, T - 1, frames))`.
  - `frame_features(frame) -> np.ndarray` shape `(184,)`
  - `clip_features(clip) -> np.ndarray` shape `(32, 184)` — trim, resample, then per-frame features.

Feature layout per frame, in this order:

| Block | Width | Content |
|---|---|---|
| Pose | 14 | pose points 0, 11, 12, 13, 14, 15, 16 (`x`, `y`), origin = midpoint of 11 and 12, scale = distance 11–12 (floor `1e-3`) |
| Left hand, body-relative | 42 | 21 points (`x`, `y`), same origin and scale |
| Right hand, body-relative | 42 | same |
| Left hand, local | 42 | minus its wrist (point 0), divided by its largest distance from the wrist (floor `1e-3`) |
| Right hand, local | 42 | same |
| Presence | 2 | left present, right present (0.0 or 1.0) |

A missing hand gives zeros in both its blocks and 0.0 presence.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_features.py
import numpy as np, pytest
from consultation.landmarks import LEFT, RIGHT, hand_present, mirror_clip
from consultation.features import clip_features, frame_features, trim_to_hands, NoHandError

def frame(left=True, right=False, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.uniform(0.2, 0.8, (75, 3)).astype(np.float32)
    f[11, :2], f[12, :2] = (0.6, 0.4), (0.4, 0.4)
    if not left: f[LEFT] = np.nan
    if not right: f[RIGHT] = np.nan
    return f

def clip(n=20, **kw): return np.stack([frame(seed=i, **kw) for i in range(n)])

def test_shape_and_no_nan_when_hand_drops_out():
    c = clip(left=True); c[5:9, LEFT] = np.nan
    x = clip_features(c)
    assert x.shape == (32, 184) and np.isfinite(x).all()

def test_missing_hand_is_zero_with_flag():
    x = frame_features(frame(left=True, right=False))
    assert x[182] == 1.0 and x[183] == 0.0
    assert not x[56:98].any() and not x[140:182].any()

def test_translation_and_scale_invariant():
    f = frame(left=True, right=True); g = f.copy()
    g[:, :2] = g[:, :2] * 0.5 + 0.1
    assert np.allclose(frame_features(f), frame_features(g), atol=1e-4)

def test_trim_removes_rest_frames_and_rejects_empty():
    c = np.concatenate([clip(4, left=False), clip(10), clip(6, left=False)])
    assert len(trim_to_hands(c)) == 10
    with pytest.raises(NoHandError): trim_to_hands(clip(5, left=False))

def test_mirror_swaps_hands_and_is_its_own_inverse():
    c = clip(left=True, right=False); m = mirror_clip(c)
    assert np.isnan(m[0][LEFT]).all() and np.isfinite(m[0][RIGHT]).all()
    assert np.allclose(m[0][RIGHT][:, 0], 1.0 - c[0][LEFT][:, 0])
    assert np.allclose(mirror_clip(m), c, equal_nan=True)

def test_hand_present():
    assert hand_present(frame(left=True)) and not hand_present(frame(left=False))
```

- [ ] **Step 2: Run `pytest tests/test_features.py -v`.** Expected: FAIL with `ModuleNotFoundError: consultation`.
- [ ] **Step 3: Implement `landmarks.py` and `features.py`** to the interfaces and layout above.
- [ ] **Step 4: Run `pytest tests/test_features.py -v`.** Expected: 6 PASS.
- [ ] **Step 5: Commit** `feat: canonical landmark format and clip features`.

---

### Task 3: Sign capture

**Files:**
- Create: `Demo/consultation/capture.py`
- Test: `tests/test_capture.py`

**Interfaces:**
- Consumes: `hand_present`, canonical frames (Task 2).
- Produces: `class SignCapture` with `START_FRAMES = 3`, `END_GAP_FRAMES = 8`, `MAX_SECONDS = 4.0`, `MIN_FRAMES = 8`; `push(frame: np.ndarray, t: float) -> np.ndarray | None`; `recording: bool`; `reset() -> None`.

Behaviour: recording starts after 3 consecutive hand frames and includes them; `t_start` is the time of the first included frame. It ends after 8 consecutive frames without a hand, or when `t − t_start ≥ 4.0`. Trailing no-hand frames are dropped. A clip shorter than 8 frames is discarded. `push` returns the finished clip `(T, 75, 3)` exactly once, otherwise `None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_capture.py
import numpy as np
from consultation.capture import SignCapture
from tests.test_features import frame

def feed(cap, pattern, dt=1 / 30):
    out, t = [], 0.0
    for has_hand in pattern:
        r = cap.push(frame(left=bool(has_hand)), t); t += dt
        if r is not None: out.append(r)
    return out

def test_returns_one_clip_without_trailing_gap():
    clips = feed(SignCapture(), [0]*5 + [1]*20 + [0]*10)
    assert len(clips) == 1 and len(clips[0]) == 20

def test_no_hand_never_returns_a_clip():
    cap = SignCapture()
    assert feed(cap, [0]*300) == [] and not cap.recording

def test_short_blip_is_discarded():
    assert feed(SignCapture(), [1]*5 + [0]*10) == []

def test_cut_at_four_seconds():
    clips = feed(SignCapture(), [1]*200)
    assert len(clips) >= 1 and 110 <= len(clips[0]) <= 122
```

- [ ] **Step 2: Run `pytest tests/test_capture.py -v`.** Expected: FAIL (module missing).
- [ ] **Step 3: Implement `SignCapture`.**
- [ ] **Step 4: Run `pytest tests/test_capture.py -v`.** Expected: 4 PASS.
- [ ] **Step 5: Commit** `feat: segment one sign from a landmark stream`.

---

### Task 4: Question flow

**Files:**
- Create: `Demo/consultation/flow.py`, `Demo/consultation/flow.json`
- Test: `tests/test_flow.py`

**Interfaces:**
- Produces:
  - `class FlowError(ValueError)`
  - `@dataclass(frozen=True) class Question`: `id: str`, `kind: str` (`"yesno" | "sign" | "sign_multi" | "bodymap" | "scale" | "choice"`), `text: dict[str, str]`, `short: dict[str, str]`, `options: list[str]`, `show_if: dict[str, list[str]] | None`, `hide_if: dict[str, list[str]] | None`
  - `class Flow`: `Flow.load(path) -> Flow`; `questions: list[Question]`; `rules: list[dict]`; `instructions: list[str]`; `next_question(answers: dict) -> Question | None`; `label(code: str, lang: str) -> str`; `icon(code: str) -> str`
  - `SIGN_SETS: dict[str, list[str]]` with keys `"reason"`, `"symptom"`, `"conditions"`
- Answer types: `yesno` → `bool`; `sign`, `choice`, `bodymap` → `str`; `sign_multi` → `list[str]`; `scale` → `int`.

`next_question` returns the first unanswered question whose `show_if` matches (every key's answer is in its list) and whose `hide_if` does not. It returns `None` when `answers["consent"] is False`. `Flow.load` raises `FlowError` if an option has no label or a condition names an unknown or later question.

Questions, in order:

| id | kind | options | condition | fr | ar | short fr / ar |
|---|---|---|---|---|---|---|
| consent | yesno | | | La caméra lit seulement la position des mains. Aucune vidéo n'est enregistrée. D'accord ? | الكاميرا تقرأ موضع اليدين فقط. لا يتم تسجيل أي فيديو. موافق؟ | (not in summary) |
| reason | sign | ENFERMO, CITA, FARMACIA, URGENTE | | Pourquoi venez-vous aujourd'hui ? | علاش جيت اليوم؟ | Motif / السبب |
| symptom | sign | DOLOR, DOLOR-DE-CABEZA, FIEBRE2, TOSER, MAREO, VOMITAR, DIARREA, RESPIRAR, CANSADO2, HERIDA, ALERGIA2, GRIPE, INSOMNIO, ANSIEDAD | | Quel est le problème principal ? | شنوة المشكل الرئيسي؟ | Symptôme principal / العرض الرئيسي |
| location | bodymap | HEAD, THROAT, CHEST, ABDOMEN, BACK, ARM_L, ARM_R, LEG_L, LEG_R | show_if symptom ∈ DOLOR, HERIDA | Où ? | وين؟ | Localisation / المكان |
| intensity | scale (0–10) | | show_if symptom ∈ DOLOR, DOLOR-DE-CABEZA | Quelle intensité ? | قداش قوية؟ | Intensité / الشدة |
| since | choice | TODAY, YESTERDAY, DAYS_2_3, WEEK, LONGER, MON, TUE, WED, THU, FRI, SAT, SUN | | Depuis quand ? | من وقتاش؟ | Depuis / منذ |
| conditions | sign_multi | DIABETES, ASMA, CORAZON, EMBARAZO | | Avez-vous une maladie connue ? | عندك مرض معروف؟ | Antécédents / أمراض معروفة |
| allergy_medicine | yesno | | | Allergie à un médicament ? | عندك حساسية من دواء؟ | Allergie médicamenteuse / حساسية من دواء |
| taking_medicine | yesno | | | Prenez-vous un médicament en ce moment ? | تاخذ في دواء توا؟ | Traitement en cours / دواء حالي |
| fever | yesno | | hide_if symptom ∈ FIEBRE2 | Avez-vous de la fièvre ? | عندك سخانة؟ | Fièvre / سخانة |

Sign labels (`code`: fr / ar / icon):

- ENFERMO: Malade / مريض / 🤒 · CITA: Rendez-vous / موعد / 📅 · FARMACIA: Médicament / دواء / 💊 · URGENTE: Urgent / حالة مستعجلة / 🚨
- DOLOR: Douleur / وجيعة / 😣 · DOLOR-DE-CABEZA: Mal de tête / وجيعة راس / 🤕 · FIEBRE2: Fièvre / سخانة / 🌡️ · TOSER: Toux / كحة / 😷 · MAREO: Vertige / دوخة / 😵 · VOMITAR: Vomissements / تقيؤ / 🤮 · DIARREA: Diarrhée / إسهال / 🚽 · RESPIRAR: Difficulté à respirer / ضيق في التنفس / 🫁 · CANSADO2: Fatigue / تعب / 😴 · HERIDA: Blessure / جرح / 🩹 · ALERGIA2: Allergie / حساسية / 🤧 · GRIPE: Grippe / نزلة برد / 🦠 · INSOMNIO: Insomnie / قلة نوم / 🌙 · ANSIEDAD: Anxiété / قلق / 😰
- DIABETES: Diabète / سكري / 🩸 · ASMA: Asthme / ربو / 🌬️ · CORAZON: Maladie du cœur / مرض القلب / ❤️ · EMBARAZO: Grossesse / حمل / 🤰

Other labels: body zones (Tête / الراس, Gorge / الحلق, Poitrine / الصدر, Ventre / البطن, Dos / الظهر, Bras gauche / اليد اليسرى, Bras droit / اليد اليمنى, Jambe gauche / الساق اليسرى, Jambe droite / الساق اليمنى); durations (Aujourd'hui / اليوم, Hier / البارح, 2–3 jours / 2-3 أيام, Une semaine / جمعة, Plus longtemps / أكثر, then the seven weekdays).

Urgency rules. Each rule is a dict `{"id", "when": {question_id: [codes]}, "min": {question_id: int}, "fr", "ar", "clinical_review": "pending"}`. It fires when every `when` answer is in its list and every `min` answer is at least its value; a missing answer never matches.

| id | when / min | fr | ar |
|---|---|---|---|
| breathing | when symptom ∈ RESPIRAR | Difficulté respiratoire : à voir en priorité | صعوبة في التنفس: أولوية |
| chest_pain | when symptom ∈ DOLOR, location ∈ CHEST | Douleur thoracique : à voir en priorité | وجيعة في الصدر: أولوية |
| severe_pain | min intensity 8 | Douleur intense (≥ 8/10) | وجيعة قوية (8/10 أو أكثر) |
| patient_urgent | when reason ∈ URGENTE | Le patient signale une urgence | المريض يقول الحالة مستعجلة |

Doctor instructions: TAKE_MEDICINE (Prenez ce médicament / خوذ هذا الدواء / 💊), TWICE_A_DAY (2 fois par jour / مرتين في النهار / 🕑), COME_BACK_3_DAYS (Revenez dans 3 jours / ارجع بعد 3 أيام / 📅), BLOOD_TEST (Analyse de sang / تحليل دم / 🩸), GO_HOSPITAL_NOW (Allez à l'hôpital maintenant / امشي للسبيطار توا / 🏥), WAIT_HERE (Attendez ici / استنى هنا / ⏳).

A Tunisian speaker on the team proofreads the Arabic before the pitch.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_flow.py
import json, pytest
from pathlib import Path
from consultation.flow import Flow, FlowError, SIGN_SETS
FLOW = Path(__file__).resolve().parents[1] / "Demo" / "consultation" / "flow.json"

def ids(flow, answers):
    out = []
    while (q := flow.next_question(answers)) is not None:
        out.append(q.id); answers[q.id] = answers.get("_fill", {}).get(q.id, True)
    return out

def test_sign_sets_are_22_and_all_labelled():
    f = Flow.load(FLOW)
    codes = [c for s in SIGN_SETS.values() for c in s]
    assert [len(SIGN_SETS[k]) for k in ("reason", "symptom", "conditions")] == [4, 14, 4]
    assert len(set(codes)) == 22
    assert all(f.label(c, "fr") and f.label(c, "ar") and f.icon(c) for c in codes)

def test_pain_path_asks_location_and_intensity():
    f = Flow.load(FLOW)
    a = {"_fill": {"reason": "ENFERMO", "symptom": "DOLOR", "location": "CHEST",
                   "intensity": 8, "since": "TODAY", "conditions": []}}
    assert ids(f, a) == ["consent", "reason", "symptom", "location", "intensity", "since",
                         "conditions", "allergy_medicine", "taking_medicine", "fever"]

def test_fever_path_skips_location_intensity_and_fever_question():
    f = Flow.load(FLOW)
    a = {"_fill": {"reason": "ENFERMO", "symptom": "FIEBRE2", "since": "TODAY", "conditions": []}}
    assert ids(f, a) == ["consent", "reason", "symptom", "since", "conditions",
                         "allergy_medicine", "taking_medicine"]

def test_refused_consent_ends_the_flow():
    assert Flow.load(FLOW).next_question({"consent": False}) is None

def test_unlabelled_option_is_rejected(tmp_path):
    d = json.loads(FLOW.read_text(encoding="utf-8"))
    d["questions"][1]["options"].append("NOT_A_CODE")
    p = tmp_path / "bad.json"; p.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(FlowError): Flow.load(p)
```

- [ ] **Step 2: Run `pytest tests/test_flow.py -v`.** Expected: FAIL (module missing).
- [ ] **Step 3: Write `flow.json`** from the tables above, with top-level keys `questions`, `labels`, `rules`, `instructions`.
- [ ] **Step 4: Implement `flow.py`.** `SIGN_SETS` is read from the three sign questions' options.
- [ ] **Step 5: Run `pytest tests/test_flow.py -v`.** Expected: 5 PASS.
- [ ] **Step 6: Commit** `feat: data-driven consultation flow`.

---

### Task 5: Consultation session

**Files:**
- Create: `Demo/consultation/session.py`
- Test: `tests/test_session.py`

**Interfaces:**
- Consumes: `Flow`, `Question` (Task 4).
- Produces: `class Consultation` with `IDLE_SECONDS = 600`, `MAX_SIGN_ATTEMPTS = 2`;
  `__init__(flow: Flow, now: Callable[[], float] = time.monotonic)`; `current: Question | None`; `pending: object | None`; `answers: dict[str, object]` (confirmed only); `finished: bool`; `needs_tap_fallback: bool`;
  `propose(value) -> None`; `confirm() -> None` (raises `RuntimeError` if nothing is pending); `reject() -> None`; `finish_multi() -> None`; `expired() -> bool`; `clear() -> None`.

Behaviour: `reject` clears `pending` and counts one failed attempt on the current question; after 2, `needs_tap_fallback` is true until the question changes. For `sign_multi`, each `confirm` appends the code and stays on the question; `finish_multi` closes it (an empty list is valid). `propose`, `confirm`, `reject` and `finish_multi` refresh the idle clock; `expired()` is `now() − last_activity ≥ IDLE_SECONDS` and does not refresh it. `clear` empties answers and pending and returns to the first question.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_session.py
import pytest
from pathlib import Path
from consultation.flow import Flow
from consultation.session import Consultation
FLOW = Path(__file__).resolve().parents[1] / "Demo" / "consultation" / "flow.json"

class Clock:
    t = 0.0
    def __call__(self): return self.t

def new(clock=None): return Consultation(Flow.load(FLOW), now=clock or Clock())

def test_pending_answer_is_not_an_answer_until_confirmed():
    c = new(); c.propose(True)
    assert c.answers == {} and c.current.id == "consent"
    c.confirm()
    assert c.answers == {"consent": True} and c.current.id == "reason"

def test_confirm_without_pending_raises():
    with pytest.raises(RuntimeError): new().confirm()

def test_two_rejections_switch_to_tap_fallback():
    c = new(); c.propose(True); c.confirm()
    for _ in range(2): c.propose("CITA"); c.reject()
    assert c.needs_tap_fallback and "reason" not in c.answers
    c.propose("ENFERMO"); c.confirm()
    assert not c.needs_tap_fallback

def test_conditions_collects_several_then_closes():
    c = new()
    for v in (True, "ENFERMO", "FIEBRE2", "TODAY"): c.propose(v); c.confirm()
    assert c.current.id == "conditions"
    c.propose("DIABETES"); c.confirm(); c.propose("ASMA"); c.confirm(); c.finish_multi()
    assert c.answers["conditions"] == ["DIABETES", "ASMA"] and c.current.id == "allergy_medicine"

def test_idle_session_expires_and_clear_removes_everything():
    clock = Clock(); c = new(clock); c.propose(True); c.confirm()
    clock.t = 599; assert not c.expired()
    clock.t = 600; assert c.expired()
    c.clear()
    assert c.answers == {} and c.pending is None and c.current.id == "consent"
```

- [ ] **Step 2: Run `pytest tests/test_session.py -v`.** Expected: FAIL (module missing).
- [ ] **Step 3: Implement `Consultation`.**
- [ ] **Step 4: Run `pytest tests/test_session.py -v`.** Expected: 5 PASS.
- [ ] **Step 5: Commit** `feat: consultation session with confirm-before-record`.

---

### Task 6: Summary and urgency flags

**Files:**
- Create: `Demo/consultation/summary.py`
- Test: `tests/test_summary.py`

**Interfaces:**
- Consumes: `Flow` (Task 4), an answers dict (Task 5).
- Produces: `@dataclass(frozen=True) class Summary`: `lines: dict[str, list[str]]` (keys `"fr"`, `"ar"`), `flags: list[dict]` (each `{"id", "fr", "ar"}`); `build_summary(flow: Flow, answers: dict) -> Summary`.

Each answered question except `consent` gives one line `"{short} : {value}"` in flow order. Value rendering: code → its label; `bool` → `Oui`/`Non` and `نعم`/`لا`; scale → `"8/10"`; list → labels joined by `", "`, empty list → `Aucun` / `لا شيء`. Unanswered questions give no line.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_summary.py
from pathlib import Path
from consultation.flow import Flow
from consultation.summary import build_summary
F = Flow.load(Path(__file__).resolve().parents[1] / "Demo" / "consultation" / "flow.json")
A = {"consent": True, "reason": "ENFERMO", "symptom": "DOLOR", "location": "CHEST",
     "intensity": 8, "since": "TODAY", "conditions": ["DIABETES"],
     "allergy_medicine": False, "taking_medicine": True, "fever": False}

def test_french_summary_lines():
    assert build_summary(F, A).lines["fr"] == [
        "Motif : Malade", "Symptôme principal : Douleur", "Localisation : Poitrine",
        "Intensité : 8/10", "Depuis : Aujourd'hui", "Antécédents : Diabète",
        "Allergie médicamenteuse : Non", "Traitement en cours : Oui", "Fièvre : Non"]

def test_arabic_has_same_number_of_lines():
    s = build_summary(F, A); assert len(s.lines["ar"]) == len(s.lines["fr"]) == 9

def test_flags_for_chest_pain_and_severity():
    assert {f["id"] for f in build_summary(F, A).flags} == {"chest_pain", "severe_pain"}

def test_partial_answers_and_empty_conditions():
    s = build_summary(F, {"consent": True, "reason": "CITA", "conditions": []})
    assert s.lines["fr"] == ["Motif : Rendez-vous", "Antécédents : Aucun"] and s.flags == []
```

- [ ] **Step 2: Run `pytest tests/test_summary.py -v`.** Expected: FAIL (module missing).
- [ ] **Step 3: Implement `build_summary`.**
- [ ] **Step 4: Run `pytest tests/test_summary.py -v`.** Expected: 4 PASS.
- [ ] **Step 5: Commit** `feat: doctor summary and urgency flags`.

---

### Task 7: Recogniser and answer narrowing

**Files:**
- Create: `Demo/consultation/recognizer.py`
- Test: `tests/test_recognizer.py`

**Interfaces:**
- Consumes: `clip_features`, `frame_features`, `NoHandError` (Task 2).
- Produces:
  - `OTHER = "OTHER"`
  - `@dataclass(frozen=True) class Prediction`: `label: str | None`, `confidence: float`, `top3: list[tuple[str, float]]`
  - `narrow(probs: np.ndarray, labels: list[str], allowed: list[str], threshold: float) -> Prediction`
  - `class SignRecognizer`: `__init__(model_path: Path, meta_path: Path)`; `available: bool`; `labels: list[str]`; `threshold: float`; `predict(clip: np.ndarray, allowed: list[str]) -> Prediction`
  - `class StaticSignRecognizer` with the same constructor and `predict`; it averages per-frame probabilities over the clip's hand frames (used in Task 11); extra attribute `signing_enabled: bool` read from its meta file.
- Meta file (`*.json`): `{"labels": [...], "threshold": float, "frames": 32, "features": 184, "weak_labels": [...], "metrics": {...}}`.

`narrow` keeps the `allowed` labels plus `OTHER`, renormalises to sum 1, and returns `label=None` when `OTHER` wins or the best probability is below `threshold`. `top3` lists allowed labels only, best first. `SignRecognizer.predict` removes any label listed in `weak_labels` from `allowed` before narrowing, so those answers stay tap-only. The constructor never raises: a missing or unreadable file sets `available = False`, and `predict` then returns `Prediction(None, 0.0, [])`. A clip with no hand also returns that. Inference calls the model directly (`model(x, training=False)`), not `model.predict`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_recognizer.py
import numpy as np
from pathlib import Path
from consultation.recognizer import narrow, SignRecognizer, Prediction
L = ["DOLOR", "FIEBRE2", "TOSER", "CITA", "OTHER"]

def test_narrowing_ignores_labels_outside_the_question():
    p = narrow(np.array([0.30, 0.10, 0.05, 0.50, 0.05]), L, ["DOLOR", "FIEBRE2", "TOSER"], 0.5)
    assert p.label == "DOLOR" and abs(p.confidence - 0.60) < 1e-6
    assert [c for c, _ in p.top3] == ["DOLOR", "FIEBRE2", "TOSER"]

def test_below_threshold_gives_no_label_but_keeps_top3():
    p = narrow(np.array([0.25, 0.20, 0.20, 0.0, 0.35]), L, ["DOLOR", "FIEBRE2", "TOSER"], 0.6)
    assert p.label is None and len(p.top3) == 3

def test_other_winning_gives_no_label():
    assert narrow(np.array([0.1, 0.1, 0.1, 0.0, 0.7]), L, ["DOLOR", "FIEBRE2"], 0.3).label is None

def test_missing_model_file_is_tap_only_not_a_crash(tmp_path):
    r = SignRecognizer(tmp_path / "none.keras", tmp_path / "none.json")
    assert r.available is False
    assert r.predict(np.zeros((10, 75, 3), np.float32), ["DOLOR"]) == Prediction(None, 0.0, [])
```

- [ ] **Step 2: Run `pytest tests/test_recognizer.py -v`.** Expected: FAIL (module missing).
- [ ] **Step 3: Implement `recognizer.py`.** Register the `TemporalAttention` layer (copy it from `Demo/pages/3_Sign_to_Text.py:110`) as a custom object when loading.
- [ ] **Step 4: Run `pytest tests/test_recognizer.py -v`.** Expected: 4 PASS.
- [ ] **Step 5: Commit** `feat: sign recogniser with per-question narrowing`.

---

### Task 8: Convert SWL-LSE to canonical clips (Colab)

**Files:**
- Create: `training/__init__.py`, `training/swl_convert.py`
- Test: `tests/test_swl_convert.py`

**Interfaces:**
- Consumes: `canonical_from_holistic` (Task 2), `trim_to_hands`, `resample`, `SIGN_SETS` (Task 4).
- Produces:
  - `frame_to_canonical(frame: dict) -> np.ndarray` — `canonical_from_holistic` applied to `frame["holistic_legacy"]`, a dict with keys `pose_landmarks`, `left_hand_landmarks`, `right_hand_landmarks`; wrap the dict so its keys read as attributes.
  - `pickle_to_clip(path) -> np.ndarray` shape `(T, 75, 3)`
  - `build_dataset(mediapipe_dir, annotations_dir, glosses_csv, out_path, seed: int = 42) -> dict` (returns counts)
  - Output `swl22.npz`: `clips (N, 32, 75, 3) float32` (trimmed and resampled), `y (N,) int16`, `split (N,) int8` (0 train, 1 validation, 2 test), `labels` (23 strings, `OTHER` last).

Verified data layout: each `MEDIAPIPE/<id>.pkl` is a `list` of frames; each frame is a dict with keys `pose`, `hands`, `holistic_legacy`. `holistic_legacy` is a dict whose `left_hand_landmarks` / `right_hand_landmarks` are a MediaPipe `NormalizedLandmarkList` or `None`. Unpickling needs `mediapipe` installed. `ANNOTATIONS/{train,val,test}_labels.csv` hold `<id>,<class_id>` rows without a header. `videos_ref_annotations.csv` has the header `FILENAME,CLASS_ID,LABEL`.

`OTHER` is drawn from clips of the 278 unused classes with `seed`: 120 from train, 12 from validation, 12 from test. A clip with no hand in any frame is skipped and counted under `"skipped_no_hand"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_swl_convert.py
import numpy as np
from types import SimpleNamespace as NS
from training.swl_convert import frame_to_canonical
from consultation.landmarks import LEFT, RIGHT

def lms(n, v): return NS(landmark=[NS(x=v, y=v + 0.1, z=0.0) for _ in range(n)])

def test_frame_to_canonical_maps_holistic_fields():
    f = frame_to_canonical({"pose": None, "hands": None, "holistic_legacy": {
        "pose_landmarks": lms(33, 0.5), "left_hand_landmarks": lms(21, 0.3),
        "right_hand_landmarks": None}})
    assert f.shape == (75, 3) and f.dtype == np.float32
    assert np.allclose(f[0, :2], [0.5, 0.6]) and np.allclose(f[LEFT][0, :2], [0.3, 0.4])
    assert np.isnan(f[RIGHT]).all()
```

- [ ] **Step 2: Run `pytest tests/test_swl_convert.py -v`.** Expected: FAIL (module missing).
- [ ] **Step 3: Implement `swl_convert.py`** with a `__main__` CLI: `PYTHONPATH=Demo python -m training.swl_convert --mediapipe MEDIAPIPE --annotations ANNOTATIONS --glosses videos_ref_annotations.csv --out swl22.npz`.
- [ ] **Step 4: Run `pytest tests/test_swl_convert.py -v`.** Expected: 1 PASS. Commit `feat: SWL-LSE to canonical clip converter`.
- [ ] **Step 5: In Colab**, clone the repo, `pip install mediapipe==0.10.13`, download `MEDIAPIPE.zip`, `ANNOTATIONS.zip` and `videos_ref_annotations.csv` from Zenodo record 13691887, unzip, and load one pickle. Expected: a list of dicts with the three keys. If unpickling fails on a class mismatch, install the current MediaPipe release for this step only.
- [ ] **Step 6: Run the converter.** Expected printed counts: 22 signs between 23 and 54 clips each before skips (DOLOR 54, VOMITAR 23) and `OTHER` 144. If `skipped_no_hand` is more than 5% of clips, stop and inspect a skipped file before training. Save `swl22.npz` to Drive.

---

### Task 9: Train and evaluate the health sign model (Colab)

**Files:**
- Create: `training/train_health.py`
- Output (committed): `Demo/consultation/models/health_signs.keras`, `Demo/consultation/models/health_signs.json`, `docs/metrics/health_signs_confusion.png`

**Interfaces:**
- Consumes: `swl22.npz` (Task 8), `clip_features`, `mirror_clip` (Task 2), `narrow` (Task 7), `SIGN_SETS` (Task 4).
- Produces: the model file and meta file read by `SignRecognizer`. `metrics` in the meta file holds: `official_top1`, `official_top3`, `cv5_top1_mean`, `cv5_top1_std`, `cv5_top3_mean`, `narrowed_top1` (`{"reason", "symptom", "conditions"}`), `per_class_top1`, `n_test`, `split_signer_independent` (`true`, `false` or `"unknown"`).

Decisions:
- Model: input `(32, 184)` → BiLSTM(128, sequences) → BatchNorm → TemporalAttention, in parallel LSTM(64) → concatenate → Dropout 0.35 → Dense(256, GELU) → Dropout 0.3 → softmax(23). This is the architecture in `Tun_SiTT/Tun_SiTT.ipynb` with the new input shape.
- Training: Adam 8e-4 with cosine decay, batch 32, up to 120 epochs, early stopping on validation accuracy with patience 18, label smoothing 0.10, balanced class weights.
- Augmentation on canonical clips before `clip_features`, 8 copies per training clip: mirror with probability 0.5, Gaussian jitter σ = 0.01 on `x`, `y`, scale ±15% about the shoulder midpoint, in-plane rotation ±15°, time-warp ±20%.
- Run A (shipped): train on the official train split, stop on validation, score on test.
- Run B: 5-fold stratified cross-validation over all clips; report mean and standard deviation.
- Narrowed accuracy: for each answer set, over test clips whose true label is in the set, the share where `narrow(probs, labels, set, 0.0).label` is correct.
- Threshold: the smallest value in 0.40, 0.45 … 0.90 whose wrong-accept rate on validation (narrowed) is at most 5%; if none, 0.60.
- `split_signer_independent`: read the SWL-LSE paper (Technologies 2024, 12(10), 205); record what it says.

- [ ] **Step 1: Implement `train_health.py`** with CLI `PYTHONPATH=Demo python -m training.train_health --data swl22.npz --out Demo/consultation/models --plots docs/metrics [--epochs 120] [--folds 5]`.
- [ ] **Step 2: Smoke-run in Colab** with `--epochs 1 --folds 0`. Expected: both output files are written and `SignRecognizer(...).available` is `True`.
- [ ] **Step 3: Full run in Colab.** Expected: the script prints every metric above and saves the confusion matrix.
- [ ] **Step 4: Record weak signs.** The script writes every sign whose per-class test-plus-validation accuracy is below 0.70 into `weak_labels`; those signs stay tap-only. Whatever `narrowed_top1["symptom"]` turns out to be, that number goes into the README and the pitch unchanged.
- [ ] **Step 5: Commit** the two model files and the plot: `feat: health sign model trained on SWL-LSE with metrics`.

---

### Task 10: Consultation page

**Files:**
- Create: `Demo/pages/6_Consultation.py`, `Demo/consultation/assets/ref_signs/` (22 clips, see Step 5), `tests/test_offline.py`
- Modify: `Demo/ui_theme.py` (`apply_tunisign_theme`, sidebar entries)

**Interfaces:**
- Consumes: everything from Tasks 2–7 and the model files from Task 9.
- Produces: `apply_tunisign_theme(remote_fonts: bool = True)`; when `False` it emits no `@import url(...)`.

Page behaviour:
- Camera through `streamlit-webrtc`, following the processor-plus-polling pattern of `Demo/pages/3_Sign_to_Text.py:276`, with `rtc_configuration={"iceServers": []}`.
- The processor runs legacy `mp.solutions.holistic.Holistic(model_complexity=1)`, converts each result with `canonical_from_holistic`, feeds `SignCapture`, and hands a finished clip to the UI thread through a `queue.Queue`. It draws only a recording indicator on the preview.
- For `sign` and `sign_multi` questions: a finished clip goes to `SignRecognizer.predict(clip, question.options)`. A label becomes `session.propose(label)`. No label counts as a failed attempt; when `session.needs_tap_fallback`, show `top3` (or all options if empty) as large icon buttons.
- Every question always shows its tap answers as icon buttons. `bodymap`, `scale` and `choice` are tap-only.
- A pending answer shows icon + French + Arabic and two buttons: correct / not correct. Yes/no and confirmation are tap-only until Task 11.
- Next to each sign option, the matching clip from `assets/ref_signs/` plays as a "how to sign" hint.
- When the flow ends: the summary in French and Arabic, urgency flags above it, and the doctor's instruction buttons, which show the chosen icon and text full-width to the patient.
- The only export is a "Download summary" button the doctor clicks (text built in memory). Nothing is saved automatically.
- Always visible: "I need an interpreter" (ends the consultation with that single line in the summary) and "End consultation" (`session.clear()`). An expired session is cleared on the next rerun.
- If `SignRecognizer.available` is false, show "Sign recognition unavailable: answer by tapping" and continue.
- A footer states: Spanish Sign Language medical signs; not a diagnostic tool.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_offline.py
import re
from pathlib import Path
DEMO = Path(__file__).resolve().parents[1] / "Demo"
FILES = [DEMO / "pages" / "6_Consultation.py", *sorted((DEMO / "consultation").glob("*.py"))]

def test_no_network_code_in_consultation():
    for p in FILES:
        s = p.read_text(encoding="utf-8")
        assert not re.search(r"elevenlabs|\brequests\b|urllib|https?://", s), p

def test_no_frame_is_written_to_disk():
    for p in FILES:
        assert not re.search(r"imwrite|VideoWriter|\.save\(|tempfile", p.read_text(encoding="utf-8")), p

def test_theme_can_skip_remote_fonts():
    import inspect, pytest
    pytest.importorskip("streamlit")
    import ui_theme
    assert "remote_fonts" in inspect.signature(ui_theme.apply_tunisign_theme).parameters
```

- [ ] **Step 2: Run `pytest tests/test_offline.py -v`.** Expected: FAIL (page missing, parameter missing).
- [ ] **Step 3: Add the `remote_fonts` parameter** to `apply_tunisign_theme` and a sidebar link to the new page.
- [ ] **Step 4: Implement `6_Consultation.py`** to the behaviour above.
- [ ] **Step 5: Add the 22 reference clips** from SWL-LSE `VIDEOS_REF.zip` (24 MB), renamed `<LABEL>.mp4` using `videos_ref_annotations.csv`.
- [ ] **Step 6: Run `pytest -v`.** Expected: every test in the repo passes.
- [ ] **Step 7: Manual check** in a Python 3.11 environment with `Demo/requirements.txt` installed: `cd Demo && streamlit run app.py`, open Consultation, then confirm each line:
  - With Wi-Fi off, the page loads, the camera starts, and a full consultation completes.
  - Signing one learned sign at steps 1, 2 and 6 proposes the right answer.
  - "Not correct" twice shows the tap fallback.
  - With the camera covered, every question can still be answered by tap.
  - With `health_signs.keras` renamed, the page shows the tap-only notice and still completes.
  - "End consultation" then reload shows no earlier answer.
  - The summary matches the answers given, with the chest-pain flag for pain + chest.
- [ ] **Step 8: Commit** `feat: guided consultation page`.

---

### Task 11: Tunisian yes/no and day signs (Colab)

**Files:**
- Create: `training/train_tunisl10.py`, `training/tunisl_blocks.json`
- Output (committed): `Demo/consultation/models/tunisl10.keras`, `Demo/consultation/models/tunisl10.json`
- Modify: `Demo/pages/6_Consultation.py`
- Test: `tests/test_tunisl_gate.py`

**Interfaces:**
- Consumes: `frame_features` (Task 2), `StaticSignRecognizer`, `narrow` (Task 7).
- Produces: `passes_gate(acc_yes_no: float, weak_holdout: dict[str, bool]) -> bool`; meta file with `labels` (`oui`, `non`, `lyoum`, `thnin`, `thleth`, `erb3a`, `5mis`, `jom3a`, `sebt`, `a7ad`), `signing_enabled`, `metrics` (`heldout_top1`, `acc_yes_no`, `n_heldout`, `weak_holdout`), and `to_answer` (`oui` → `true`, `non` → `false`, `lyoum` → `TODAY`, then `MON` … `SUN`).

Decisions:
- Data: `github.com/W7CH/First-ever-Tunisian-Sign-Language-Dataset`, folders `Data/Demandes/{oui,non,lyoum}` and `Data/Jours/*`. The frames are images with no clip boundaries, so the model is per-frame: Holistic in static-image mode → `frame_features` → MLP 184 → 128 → 64 → 10.
- `tunisl_blocks.json` is made by hand from contact sheets: for each class, the frame-number ranges that belong to one person or take.
- Hold-out: the last block of each class when it has two or more blocks; otherwise the last 25% of frames, marked `weak_holdout = true`.
- Gate: `signing_enabled` is true only if yes/no held-out accuracy is at least 0.90 and neither `oui` nor `non` has a weak hold-out.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tunisl_gate.py
from training.train_tunisl10 import passes_gate

def test_gate():
    ok = {"oui": False, "non": False}
    assert passes_gate(0.90, ok) is True
    assert passes_gate(0.89, ok) is False
    assert passes_gate(0.99, {"oui": True, "non": False}) is False
```

- [ ] **Step 2: Run `pytest tests/test_tunisl_gate.py -v`.** Expected: FAIL (module missing).
- [ ] **Step 3: Implement `train_tunisl10.py`**, including a `--sheets` mode that writes one contact sheet per class.
- [ ] **Step 4: Run `pytest tests/test_tunisl_gate.py -v`.** Expected: PASS.
- [ ] **Step 5: In Colab**, produce the contact sheets, fill `tunisl_blocks.json`, train, and write the two output files.
- [ ] **Step 6: Wire it into the page.** When `StaticSignRecognizer.available and signing_enabled`, yes/no questions, confirmation and `since` also accept a signed answer, narrowed to that question's valid labels. Otherwise they stay tap-only and nothing else changes.
- [ ] **Step 7: Repeat Task 10's manual check** for a signed yes and a signed no if the gate passed.
- [ ] **Step 8: Commit** `feat: Tunisian yes/no and day signs behind an accuracy gate`.

---

### Task 12: README and documentation

**Files:**
- Modify: `README.md`, `Demo/README.md`, `Tun_STT/README.md`, `Tun_SiTT/README.md`, `Tun_TTS/README.md`
- Create: `docs/demo.gif`

- [ ] **Step 1: Rewrite `README.md`** around the consultation: one name (SignBridge), the real clone URL, a screenshot or GIF, quick start for Python 3.11, and the AI-versus-rules table from the spec.
- [ ] **Step 2: Add a "Results" section** copied from `health_signs.json` and `tunisl10.json`: every metric with its split and sample count, including the signer-independence note.
- [ ] **Step 3: Add a "Limits" section:** Spanish Sign Language medical signs, small Tunisian set, lighting sensitivity, not a diagnostic tool, urgency rules pending clinical review.
- [ ] **Step 4: Remove unsupported claims:** fine-tuned XTTS voice in the demo (it calls ElevenLabs), "Whisper" in the demo (it loads Vosk), "100+ signs", "high accuracy" for speech (measured word error rate 62%), on-device mobile and Azure deployment. Move real future work under "Roadmap".
- [ ] **Step 5: Add "Privacy"** (landmarks only, nothing stored, consultation works offline) and link `ATTRIBUTION.md`.
- [ ] **Step 6: Verify** with `grep -nE "100\+|XTTS|Azure|your-username|TuniSign" README.md`. Expected: matches only inside the Roadmap section, none for `your-username`.
- [ ] **Step 7: Record `docs/demo.gif`** of one full consultation and commit `docs: README matches the code, with results and limits`.

---

## Not in this plan

- The "learn my signs" teach step and recorded Derja audio clips (optional in the spec).
- Clinical review of the questions and urgency rules: a doctor or medical student on the team's side does this before the pitch.
