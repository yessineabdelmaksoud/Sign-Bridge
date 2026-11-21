from pathlib import Path

import pytest

from consultation.flow import Flow
from consultation.session import Consultation

FLOW = Path(__file__).resolve().parents[1] / "Demo" / "consultation" / "flow.json"


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def new(clock=None):
    return Consultation(Flow.load(FLOW), now=clock or Clock())


def test_pending_answer_is_not_an_answer_until_confirmed():
    c = new()
    c.propose(True)
    assert c.answers == {} and c.current.id == "consent"
    c.confirm()
    assert c.answers == {"consent": True} and c.current.id == "reason"


def test_a_pending_no_can_be_confirmed():
    c = new()
    c.propose(False)
    assert c.has_pending
    c.confirm()
    assert c.answers == {"consent": False} and c.finished


def test_confirm_without_pending_raises():
    with pytest.raises(RuntimeError):
        new().confirm()


def test_two_rejections_switch_to_tap_fallback():
    c = new()
    c.propose(True)
    c.confirm()
    for _ in range(2):
        c.propose("CITA")
        c.reject()
    assert c.needs_tap_fallback and "reason" not in c.answers
    c.propose("ENFERMO")
    c.confirm()
    assert not c.needs_tap_fallback


def test_conditions_collects_several_then_closes():
    c = new()
    for v in (True, "ENFERMO", "FIEBRE2", "TODAY"):
        c.propose(v)
        c.confirm()
    assert c.current.id == "conditions"
    c.propose("DIABETES")
    c.confirm()
    c.propose("ASMA")
    c.confirm()
    assert c.collected == ["DIABETES", "ASMA"] and "conditions" not in c.answers
    c.finish_multi()
    assert c.answers["conditions"] == ["DIABETES", "ASMA"] and c.current.id == "allergy_medicine"


def test_idle_session_expires_and_clear_removes_everything():
    clock = Clock()
    c = new(clock)
    c.propose(True)
    c.confirm()
    clock.t = 599
    assert not c.expired()
    clock.t = 600
    assert c.expired()
    c.clear()
    assert c.answers == {} and c.pending is None and c.current.id == "consent"
