import json

import numpy as np

from consultation.landmarks import LEFT, RIGHT, empty_frame
from training.export_poses import POSE_POINTS, encode_sign, fill_gaps, normalise, smooth, trim


def body_frame(shift=0.0, scale=1.0, hand=True):
    """A frame with shoulders 0.2 apart at y=0.4, a nose above them, and optionally a left hand."""
    f = empty_frame()
    f[:33, :2] = 0.5
    f[0, :2] = (0.5, 0.25)
    f[11, :2], f[12, :2] = (0.6, 0.4), (0.4, 0.4)
    f[13, :2], f[14, :2] = (0.65, 0.55), (0.35, 0.55)
    f[15, :2], f[16, :2] = (0.66, 0.7), (0.34, 0.7)
    if hand:
        f[LEFT, 0] = np.linspace(0.6, 0.7, 21)
        f[LEFT, 1] = np.linspace(0.3, 0.35, 21)
    f[:, :2] = (f[:, :2] - 0.5) * scale + 0.5 + shift
    return f


def test_normalise_puts_the_shoulders_at_the_origin_one_unit_apart():
    pose, left, right = normalise(np.stack([body_frame()] * 4))
    shoulders = [POSE_POINTS.index(11), POSE_POINTS.index(12)]
    assert pose.shape == (4, len(POSE_POINTS), 2)
    assert np.allclose(pose[0, shoulders[0]], (0.5, 0.0), atol=1e-6) and np.allclose(pose[0, shoulders[1]], (-0.5, 0.0), atol=1e-6)
    assert np.allclose(pose[0, POSE_POINTS.index(0)], (0.0, -0.75), atol=1e-6)
    assert left.shape == (4, 21, 2) and np.isnan(right).all()


def test_normalise_ignores_where_the_signer_stands_and_how_big_they_are():
    a = normalise(np.stack([body_frame()] * 3))
    b = normalise(np.stack([body_frame(shift=0.07, scale=1.3)] * 3))
    for x, y in zip(a, b):
        assert np.allclose(x, y, atol=1e-5, equal_nan=True)   # frames are float32


def test_fill_gaps_bridges_short_dropouts_only():
    seq = np.zeros((12, 21, 2))
    seq[:, :, 0] = np.arange(12)[:, None]
    seq[3:5] = np.nan     # 2 frames: filled
    seq[7:12] = np.nan    # runs to the end: left empty
    out = fill_gaps(seq, max_gap=4)
    assert np.allclose(out[3:5, 0, 0], [3, 4])
    assert np.isnan(out[7:]).all()


def test_smooth_keeps_missing_frames_missing_and_steadies_jitter():
    seq = np.zeros((6, 2, 2))
    seq[:, 0, 0] = [0, 1, 0, 1, 0, 1]
    seq[2] = np.nan
    out = smooth(seq, window=3)
    assert np.isnan(out[2]).all()
    assert np.ptp(out[[0, 1, 3, 4, 5], 0, 0]) < 1.0


def test_trim_keeps_a_short_rest_before_and_after_the_sign():
    clip = np.stack([body_frame(hand=False)] * 10 + [body_frame()] * 8 + [body_frame(hand=False)] * 10)
    kept = trim(clip, pad=3)
    assert len(kept) == 3 + 8 + 3


def test_encoded_sign_is_compact_json_with_null_for_a_missing_hand():
    clip = np.stack([body_frame()] * 5 + [body_frame(hand=False)] * 2)
    sign = encode_sign(clip, pad=2)
    text = json.dumps(sign)
    frame = sign["frames"][0]
    assert len(frame["p"]) == 2 * len(POSE_POINTS) and len(frame["l"]) == 42 and frame["r"] is None
    assert all(isinstance(v, (int, float)) for v in frame["p"])
    assert "NaN" not in text and len(text) < 20_000
