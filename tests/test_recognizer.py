import json

import numpy as np

from consultation.recognizer import Prediction, SignRecognizer, StaticSignRecognizer, narrow
from tests.test_features import clip

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


class FakeModel:
    """Stands in for the Keras model: always answers with the same probabilities."""

    def __init__(self, probs):
        self.probs, self.seen = np.asarray(probs, np.float32), None

    def __call__(self, x, training=False):
        self.seen = np.asarray(x)
        return np.tile(self.probs, (len(self.seen), 1))


def recognizer(cls, tmp_path, probs, **meta):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"labels": L, "threshold": 0.5, **meta}), encoding="utf-8")
    return cls(tmp_path / "m.keras", path, model=FakeModel(probs))


def test_predict_feeds_one_clip_of_32_by_184_features(tmp_path):
    r = recognizer(SignRecognizer, tmp_path, [0.7, 0.1, 0.1, 0.05, 0.05])
    p = r.predict(clip(20), ["DOLOR", "FIEBRE2"])
    assert r.available and p.label == "DOLOR" and r._model.seen.shape == (1, 32, 184)


def test_weak_labels_are_never_proposed(tmp_path):
    r = recognizer(SignRecognizer, tmp_path, [0.7, 0.2, 0.05, 0.0, 0.05], weak_labels=["DOLOR"])
    assert r.predict(clip(20), ["DOLOR", "FIEBRE2"]).label == "FIEBRE2"


def test_clip_without_a_hand_gives_no_prediction(tmp_path):
    r = recognizer(SignRecognizer, tmp_path, [0.7, 0.1, 0.1, 0.05, 0.05])
    assert r.predict(clip(20, left=False), ["DOLOR"]) == Prediction(None, 0.0, [])


def test_static_recognizer_reads_its_gate_and_averages_frames(tmp_path):
    r = recognizer(StaticSignRecognizer, tmp_path, [0.1, 0.8, 0.05, 0.0, 0.05], signing_enabled=True)
    assert r.signing_enabled is True
    assert r.predict(clip(12), ["DOLOR", "FIEBRE2"]).label == "FIEBRE2"
    assert r._model.seen.shape == (12, 184)
