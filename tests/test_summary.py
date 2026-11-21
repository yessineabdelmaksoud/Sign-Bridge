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
    s = build_summary(F, A)
    assert len(s.lines["ar"]) == len(s.lines["fr"]) == 9
    assert s.lines["ar"][0] == "السبب : مريض"


def test_flags_for_chest_pain_and_severity():
    assert {f["id"] for f in build_summary(F, A).flags} == {"chest_pain", "severe_pain"}


def test_partial_answers_and_empty_conditions():
    s = build_summary(F, {"consent": True, "reason": "CITA", "conditions": []})
    assert s.lines["fr"] == ["Motif : Rendez-vous", "Antécédents : Aucun"] and s.flags == []


def test_intensity_seven_is_not_flagged_as_severe():
    flags = build_summary(F, {**A, "intensity": 7, "location": "BACK"}).flags
    assert flags == []
