import json

import numpy as np

from tests.test_tunisian_signs import _take
from training.train_combined import load_tunisian


def test_tunisian_takes_are_loaded_with_a_steady_body(tmp_path):
    clip = _take(n=12)
    clip[5, 11, :2] = (0.95, 0.95)          # one wild shoulder guess
    np.save(tmp_path / "a.npy", clip)
    np.save(tmp_path / "b.npy", _take(n=3))  # too few hand frames: skipped
    (tmp_path / "index.json").write_text(json.dumps([
        {"code": "TN_OUI", "signer": 2, "file": "a.npy", "frames": 12, "hand_frames": 12},
        {"code": "TN_NON", "signer": 2, "file": "b.npy", "frames": 3, "hand_frames": 3}]))
    clips, labels, signers = load_tunisian(tmp_path, ["TN_OUI", "TN_NON"])
    assert clips.shape == (1, 32, 75, 3) and labels == ["TN_OUI"] and signers.tolist() == [2]
    assert np.allclose(clips[0, :, 11, :2], (0.7, 0.3))
