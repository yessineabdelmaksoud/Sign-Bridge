"""Drive the consultation page the way a patient and doctor would, by tapping."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

# How a confirmed "Motif: Malade" appears in the doctor's running note.
NOTE_REASON = "<dt>Motif</dt><dd>Malade</dd>"
PAGE = str(Path(__file__).resolve().parents[1] / "Demo" / "pages" / "6_Consultation.py")


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGNBRIDGE_MODELS", str(tmp_path))  # no model files: tap-only
    at = AppTest.from_file(PAGE, default_timeout=30).run()
    assert not at.exception, at.exception
    return at


def tap(at, key):
    at.button(key=key).click().run()
    assert not at.exception, at.exception


def answer(at, key):
    tap(at, key)
    tap(at, "confirm_yes")


def text(at):
    return "\n".join(str(m.value) for m in at.markdown)


def test_full_consultation_by_tapping_builds_the_note(app):
    for key in ("opt_YES", "opt_ENFERMO", "opt_DOLOR", "opt_CHEST", "opt_SCALE_8", "opt_TODAY", "opt_DIABETES"):
        answer(app, key)
    tap(app, "multi_done")
    for key in ("opt_NO", "opt_YES", "opt_NO"):
        answer(app, key)
    consultation = app.session_state["consultation"]
    assert consultation.finished
    assert consultation.answers["symptom"] == "DOLOR" and consultation.answers["conditions"] == ["DIABETES"]
    page = text(app)
    assert "Intensité : 8/10" in page and "Douleur thoracique : à voir en priorité" in page
    assert "السبب : مريض" in page


def test_a_tapped_answer_is_not_in_the_note_until_confirmed(app):
    answer(app, "opt_YES")
    tap(app, "opt_ENFERMO")
    assert app.session_state["consultation"].has_pending
    assert NOTE_REASON not in text(app)
    tap(app, "confirm_yes")
    assert NOTE_REASON in text(app)


def test_not_correct_returns_to_the_same_question(app):
    answer(app, "opt_YES")
    tap(app, "opt_CITA")
    tap(app, "confirm_no")
    consultation = app.session_state["consultation"]
    assert consultation.current.id == "reason" and "reason" not in consultation.answers


def test_refusing_consent_stops_before_any_question(app):
    answer(app, "opt_NO")
    assert app.session_state["consultation"].finished
    assert "Consultation non commencée" in text(app)


def test_asking_for_an_interpreter_ends_the_consultation(app):
    answer(app, "opt_YES")
    tap(app, "ask_interpreter")
    assert "Interprète demandé" in text(app)
    assert not [b for b in app.button if b.key and b.key.startswith("opt_")]


def test_ending_the_consultation_forgets_every_answer(app):
    answer(app, "opt_YES")
    answer(app, "opt_ENFERMO")
    assert NOTE_REASON in text(app)
    tap(app, "end_consultation")
    consultation = app.session_state["consultation"]
    assert consultation.answers == {} and consultation.current.id == "consent"
    assert NOTE_REASON not in text(app)


def test_without_a_model_the_page_says_to_answer_by_tapping(app):
    answer(app, "opt_YES")
    assert "Répondez en touchant" in text(app)


def test_doctor_instruction_is_shown_large_to_the_patient(app):
    for key in ("opt_YES", "opt_CITA", "opt_TOSER", "opt_WEEK"):
        answer(app, key)
    tap(app, "multi_done")
    for key in ("opt_NO", "opt_NO", "opt_NO"):
        answer(app, key)
    tap(app, "ins_COME_BACK_3_DAYS")
    assert "Revenez dans 3 jours" in text(app) and "ارجع بعد 3 أيام" in text(app)
