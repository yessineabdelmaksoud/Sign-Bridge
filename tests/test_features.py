import numpy as np
import pytest

from consultation.features import NoHandError, clip_features, frame_features, trim_to_hands
from consultation.landmarks import LEFT, RIGHT, hand_present, mirror_clip


def frame(left=True, right=False, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.uniform(0.2, 0.8, (75, 3)).astype(np.float32)
    f[11, :2], f[12, :2] = (0.6, 0.4), (0.4, 0.4)
    if not left:
        f[LEFT] = np.nan
    if not right:
        f[RIGHT] = np.nan
    return f


def clip(n=20, **kw):
    return np.stack([frame(seed=i, **kw) for i in range(n)])


def test_shape_and_no_nan_when_hand_drops_out():
    c = clip(left=True)
    c[5:9, LEFT] = np.nan
    x = clip_features(c)
    assert x.shape == (32, 184) and np.isfinite(x).all()


def test_missing_hand_is_zero_with_flag():
    x = frame_features(frame(left=True, right=False))
    assert x[182] == 1.0 and x[183] == 0.0
    assert not x[56:98].any() and not x[140:182].any()


def test_translation_and_scale_invariant():
    f = frame(left=True, right=True)
    g = f.copy()
    g[:, :2] = g[:, :2] * 0.5 + 0.1
    assert np.allclose(frame_features(f), frame_features(g), atol=1e-4)


def test_trim_removes_rest_frames_and_rejects_empty():
    c = np.concatenate([clip(4, left=False), clip(10), clip(6, left=False)])
    assert len(trim_to_hands(c)) == 10
    with pytest.raises(NoHandError):
        trim_to_hands(clip(5, left=False))


def test_mirror_swaps_hands_and_is_its_own_inverse():
    c = clip(left=True, right=False)
    m = mirror_clip(c)
    assert np.isnan(m[0][LEFT]).all() and np.isfinite(m[0][RIGHT]).all()
    assert np.allclose(m[0][RIGHT][:, 0], 1.0 - c[0][LEFT][:, 0])
    assert np.allclose(mirror_clip(m), c, equal_nan=True)


def test_hand_present():
    assert hand_present(frame(left=True)) and not hand_present(frame(left=False))
