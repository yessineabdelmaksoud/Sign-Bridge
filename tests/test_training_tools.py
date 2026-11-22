import numpy as np

from consultation.landmarks import LEFT, RIGHT
from tests.test_features import clip
from training.tools import augment_clip, narrowed_accuracy, pick_threshold, stratified_folds, weak_labels

L = ["A", "B", "C", "D", "OTHER"]


def test_augment_keeps_shape_and_which_hands_are_visible():
    c = clip(32, left=True, right=False)
    out = augment_clip(c, np.random.default_rng(1))
    assert out.shape == c.shape and out.dtype == np.float32
    left, right = np.isfinite(out[:, LEFT]).all(), np.isfinite(out[:, RIGHT]).all()
    assert left != right  # one hand before, one hand after (possibly mirrored)
    assert not np.allclose(np.nan_to_num(out), np.nan_to_num(c))


def test_augment_is_repeatable_for_a_seed_and_varies_across_seeds():
    c = clip(32)
    a = augment_clip(c, np.random.default_rng(7))
    b = augment_clip(c, np.random.default_rng(7))
    d = augment_clip(c, np.random.default_rng(8))
    assert np.allclose(a, b, equal_nan=True) and not np.allclose(a, d, equal_nan=True)


def test_narrowed_accuracy_counts_only_samples_of_the_answer_set():
    probs = np.array([[0.6, 0.3, 0.05, 0.03, 0.02],    # true A -> A
                      [0.5, 0.4, 0.05, 0.03, 0.02],    # true B -> A (wrong)
                      [0.0, 0.0, 0.9, 0.05, 0.05]])    # true C, outside the set
    acc, n = narrowed_accuracy(probs, np.array([0, 1, 2]), L, ["A", "B"])
    assert n == 2 and acc == 0.5


def test_pick_threshold_is_the_lowest_that_keeps_wrong_accepts_under_five_percent():
    rng = np.random.default_rng(0)
    y = np.array([0] * 50 + [1] * 50)
    probs = np.zeros((100, 5))
    probs[np.arange(100), y] = 0.9
    probs[:, 4] = 0.1
    probs[:10] = [0.42, 0.48, 0.0, 0.0, 0.10]  # ten A clips that look slightly more like B
    # B scores 0.48 on those ten, so 0.40 and 0.45 accept the mistake (10% wrong); 0.50 rejects it.
    assert pick_threshold(probs, y, L, [["A", "B"]]) == 0.50
    probs[:10] = [0.05, 0.95, 0.0, 0.0, 0.0]   # ten confident mistakes: no threshold helps
    assert pick_threshold(probs, y, L, [["A", "B"]]) == 0.60


def test_weak_labels_are_those_under_seventy_percent():
    assert weak_labels({"A": (7, 10), "B": (6, 10), "C": (0, 0), "D": (1, 4)}) == ["B", "D"]


def test_stratified_folds_use_every_sample_once_and_balance_classes():
    y = np.array([0] * 10 + [1] * 5 + [2] * 7)
    folds = stratified_folds(y, 5, seed=3)
    assert sorted(np.concatenate(folds).tolist()) == list(range(22))
    assert all(1 <= (y[f] == 1).sum() <= 1 for f in folds) and all((y[f] == 0).sum() == 2 for f in folds)


def test_featurise_of_no_clips_is_an_empty_batch():
    from training.train_health import featurise, top_k

    empty = featurise(np.empty((0, 32, 75, 3), np.float32))
    assert empty.shape == (0, 32, 184)
    assert top_k(np.empty((0, 5)), np.empty(0, dtype=int), 1) == 0.0


def test_a_hand_resting_at_the_waist_is_dropped_and_a_raised_hand_kept():
    from consultation.landmarks import LEFT, RIGHT
    from tests.test_features import frame as make_frame
    from training.tools import drop_resting_hands

    f = make_frame(left=True, right=True)          # shoulders at y=0.4, 0.2 apart
    f[LEFT, 1] = 0.5                                # wrist 0.5 widths below the shoulders: signing
    f[RIGHT, 1] = 0.85                              # 2.25 widths below: resting on the desk
    out = drop_resting_hands(f)
    assert np.isfinite(out[LEFT]).all() and np.isnan(out[RIGHT]).all()
    assert np.allclose(out[:33], f[:33]) and np.isfinite(f[RIGHT]).all(), "the input is not changed"


def test_without_shoulders_hands_are_kept():
    from consultation.landmarks import RIGHT
    from tests.test_features import frame as make_frame
    from training.tools import drop_resting_hands

    f = make_frame(left=False, right=True)
    f[11:13] = np.nan
    f[RIGHT, 1] = 0.95
    assert np.isfinite(drop_resting_hands(f)[RIGHT]).all()
