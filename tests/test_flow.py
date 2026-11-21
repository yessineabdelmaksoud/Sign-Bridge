import json
from pathlib import Path

import pytest

from consultation.flow import SIGN_SETS, Flow, FlowError

FLOW = Path(__file__).resolve().parents[1] / "Demo" / "consultation" / "flow.json"


def ids(flow, answers):
    out = []
    while (q := flow.next_question(answers)) is not None:
        out.append(q.id)
        answers[q.id] = answers.get("_fill", {}).get(q.id, True)
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
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(FlowError):
        Flow.load(p)


def test_condition_on_a_later_question_is_rejected(tmp_path):
    d = json.loads(FLOW.read_text(encoding="utf-8"))
    d["questions"][1]["show_if"] = {"symptom": ["DOLOR"]}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(FlowError):
        Flow.load(p)
