import json

import numpy as np

from consultation.numpy_model import DenseModel
from consultation.recognizer import SignRecognizer
from tests.test_features import clip
from training.mlp import train_mlp


def blobs(n_per_class=60, classes=3, dim=20, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 4, (classes, dim))
    x = np.concatenate([c + rng.normal(0, 1, (n_per_class, dim)) for c in centres]).astype(np.float32)
    y = np.repeat(np.arange(classes), n_per_class)
    return x, y


def test_training_learns_a_separable_problem():
    x, y = blobs()
    xv, yv = blobs(seed=1)  # same centres are not guaranteed, so validate on a held-out part instead
    order = np.random.default_rng(2).permutation(len(x))
    train, held = order[:140], order[140:]
    model, history = train_mlp(x[train], y[train], 3, hidden=(16,), epochs=40, seed=0,
                               x_val=x[held], y_val=y[held])
    probs = model(x[held])
    assert probs.shape == (len(held), 3) and np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert (probs.argmax(axis=1) == y[held]).mean() >= 0.95
    assert history["best_epoch"] >= 1


def test_model_file_round_trip_gives_identical_output(tmp_path):
    x, y = blobs()
    model, _ = train_mlp(x, y, 3, hidden=(8,), epochs=3, seed=0)
    model.save(tmp_path / "m.npz")
    again = DenseModel.load(tmp_path / "m.npz")
    assert np.allclose(model(x[:5]), again(x[:5]))


def test_model_accepts_clips_and_flattens_them():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 4, 6)).astype(np.float32)
    y = np.arange(30) % 2
    model, _ = train_mlp(x, y, 2, hidden=(8,), epochs=2, seed=0)
    assert model(x[:3]).shape == (3, 2)


def test_recognizer_loads_a_numpy_model(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 32, 184)).astype(np.float32)
    model, _ = train_mlp(x, np.arange(40) % 3, 3, hidden=(8,), epochs=1, seed=0)
    model.save(tmp_path / "m.npz")
    (tmp_path / "m.json").write_text(json.dumps({"labels": ["DOLOR", "FIEBRE2", "OTHER"], "threshold": 0.0}))
    r = SignRecognizer(tmp_path / "m.npz", tmp_path / "m.json")
    assert r.available, r.error
    assert r.predict(clip(20), ["DOLOR", "FIEBRE2"]).label in {"DOLOR", "FIEBRE2"}
