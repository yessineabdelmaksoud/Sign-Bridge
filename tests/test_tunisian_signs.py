import json
from pathlib import Path

import numpy as np

from training.tunisian_signs import VOCAB_PATH, load_vocab, takes


def test_vocabulary_is_complete_and_unique():
    vocab = load_vocab()
    codes = [v["code"] for v in vocab]
    assert len(codes) == len(set(codes)) == 27
    assert all(c.startswith("TN_") for c in codes)
    assert all(v["fr"] and v["ar"] and "/" in v["folder"] for v in vocab)
    assert json.loads(Path(VOCAB_PATH).read_text(encoding="utf-8")) == vocab


def test_takes_are_runs_of_one_signer_in_file_order():
    # positions 0..11 in file order; signer of each frame:
    signers = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 2])
    runs = takes(signers, min_length=3)
    assert runs == [(0, [0, 1, 2]), (1, [3, 4, 5, 6]), (0, [7, 8, 9, 10])]


def test_a_single_frame_of_someone_else_does_not_split_a_take():
    signers = np.array([0, 0, 0, 1, 0, 0, 0])
    assert takes(signers, min_length=3, bridge=1) == [(0, [0, 1, 2, 4, 5, 6])]


def test_short_runs_are_dropped():
    signers = np.array([0, 0, 1, 1, 1, 2, 2])
    assert takes(signers, min_length=3) == [(1, [2, 3, 4])]


def _hand_at(x, y):
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[:, 0], pts[:, 1] = x, y
    pts[1:, 1] -= 0.05
    return pts


def test_detected_hands_go_to_the_nearer_wrist():
    from consultation.landmarks import LEFT, RIGHT, empty_frame
    from training.tunisian_signs import attach_hands

    frame = empty_frame()
    frame[:33, :2] = 0.5
    frame[15, :2] = (0.7, 0.6)   # signer's left wrist (image right)
    frame[16, :2] = (0.3, 0.6)   # signer's right wrist
    out = attach_hands(frame, [_hand_at(0.32, 0.62), _hand_at(0.69, 0.58)])
    assert np.allclose(out[RIGHT][0, :2], (0.32, 0.62)) and np.allclose(out[LEFT][0, :2], (0.69, 0.58))


def test_two_hands_near_one_wrist_take_both_sides():
    from consultation.landmarks import LEFT, RIGHT, empty_frame
    from training.tunisian_signs import attach_hands

    frame = empty_frame()
    frame[15, :2], frame[16, :2] = (0.7, 0.6), (0.3, 0.6)
    out = attach_hands(frame, [_hand_at(0.66, 0.5), _hand_at(0.6, 0.5)])
    assert np.isfinite(out[LEFT]).all() and np.isfinite(out[RIGHT]).all()


def test_detected_hand_replaces_a_missing_one_and_keeps_the_pose():
    from consultation.landmarks import LEFT, RIGHT, empty_frame
    from training.tunisian_signs import attach_hands

    frame = empty_frame()
    frame[:33, :2] = 0.5
    frame[15, :2], frame[16, :2] = (0.7, 0.6), (0.3, 0.6)
    out = attach_hands(frame, [_hand_at(0.31, 0.6)])
    assert np.isfinite(out[RIGHT]).all() and np.isnan(out[LEFT]).all()
    assert np.allclose(out[:33], frame[:33], equal_nan=True)
    assert attach_hands(frame, []) is not frame and np.allclose(attach_hands(frame, []), frame, equal_nan=True)


def test_a_take_without_a_body_is_skipped_for_the_stick_figure(tmp_path, monkeypatch):
    from consultation.landmarks import LEFT, empty_frame
    from training import tunisian_signs

    def frame(with_body):
        f = empty_frame()
        if with_body:
            f[:33, :2] = 0.5
            f[11, :2], f[12, :2] = (0.6, 0.4), (0.4, 0.4)
        f[LEFT, :2] = 0.6
        return f

    seq = tmp_path / "seq"
    seq.mkdir()
    np.save(seq / "a.npy", np.stack([frame(False)] * 30))   # most hand frames, but no body
    np.save(seq / "b.npy", np.stack([frame(True)] * 12))
    (seq / "index.json").write_text(json.dumps([
        {"code": "TN_OUI", "signer": 1, "file": "a.npy", "frames": 30, "hand_frames": 30},
        {"code": "TN_OUI", "signer": 2, "file": "b.npy", "frames": 12, "hand_frames": 12}]))
    poses = tmp_path / "poses.json"
    poses.write_text(json.dumps({"fps": 20, "signs": {}}))
    monkeypatch.setattr(tunisian_signs, "load_vocab", lambda: [{"code": "TN_OUI"}])
    assert tunisian_signs.add_poses(seq, poses) == ["TN_OUI"]
    assert len(json.loads(poses.read_text())["signs"]["TN_OUI"]["frames"]) >= 8


# --- a steady body for cropped images ------------------------------------------------

def _take(n=10, shoulders=((0.7, 0.3), (0.3, 0.3))):
    from consultation.landmarks import LEFT, RIGHT, empty_frame

    frames = []
    for t in range(n):
        f = empty_frame()
        f[:33, :2] = 0.5
        f[11, :2], f[12, :2] = shoulders
        f[LEFT, 0], f[LEFT, 1] = 0.65 + 0.01 * t, 0.6
        f[RIGHT, 0], f[RIGHT, 1] = 0.35, 0.6
        frames.append(f)
    return np.stack(frames)


def test_shoulders_are_held_steady_at_the_median_of_plausible_frames():
    from training.tunisian_signs import steady_clip

    clip = _take()
    clip[3, 11, :2] = (0.95, 0.9)      # a wild guess: far too low and wide
    clip[6, 12, :2] = np.nan
    out = steady_clip(clip)
    assert np.allclose(out[:, 11, :2], (0.7, 0.3)) and np.allclose(out[:, 12, :2], (0.3, 0.3))
    assert (out[:, 0, 1] < 0.3).all(), "the head sits above the shoulders"


def test_without_plausible_shoulders_a_standard_framing_is_used():
    from training.tunisian_signs import steady_clip

    clip = _take(shoulders=((0.52, 0.9), (0.5, 0.9)))   # 0.02 apart: not a pair of shoulders
    out = steady_clip(clip)
    width = np.linalg.norm(out[0, 11, :2] - out[0, 12, :2])
    assert 0.3 < width < 0.9 and np.isfinite(out[:, 0, :2]).all()


def test_hands_that_swap_sides_for_one_frame_are_put_back():
    from consultation.landmarks import LEFT, RIGHT
    from training.tunisian_signs import steady_clip

    clip = _take()
    clip[4, LEFT], clip[4, RIGHT] = clip[4, RIGHT].copy(), clip[4, LEFT].copy()
    out = steady_clip(clip)
    assert (out[:, LEFT.start, 0] > 0.5).all() and (out[:, RIGHT.start, 0] < 0.5).all()


def test_an_implausible_elbow_is_rebuilt_between_shoulder_and_hand():
    from consultation.landmarks import LEFT
    from training.tunisian_signs import steady_clip

    clip = _take()
    clip[:, 13, :2] = (0.1, 0.05)       # left elbow guessed above the head on the other side
    out = steady_clip(clip)
    shoulder, elbow, wrist = out[0, 11, :2], out[0, 13, :2], out[0, LEFT.start, :2]
    width = np.linalg.norm(out[0, 11, :2] - out[0, 12, :2])
    assert elbow[1] > shoulder[1], "the elbow hangs below the shoulder"
    assert np.linalg.norm(elbow - shoulder) < 1.3 * width and np.linalg.norm(elbow - wrist) < 1.3 * width


def test_a_take_with_both_hands_on_the_wrong_side_is_flipped_back():
    from consultation.landmarks import LEFT, RIGHT
    from training.tunisian_signs import fix_take_sides

    clip = _take()
    clip[:, LEFT, 0], clip[:, RIGHT, 0] = 0.3, 0.7     # signer's left hand on the image left: crossed all along
    out = fix_take_sides(clip)
    assert (out[:, LEFT.start, 0] > 0.5).all() and (out[:, RIGHT.start, 0] < 0.5).all()


def test_a_lone_hand_far_on_the_other_side_is_moved_to_its_side():
    from consultation.landmarks import LEFT, RIGHT
    from training.tunisian_signs import fix_take_sides

    clip = _take()
    clip[:, RIGHT] = np.nan
    clip[:, LEFT, 0] = 0.2                              # body midline at 0.5: this is the right hand
    out = fix_take_sides(clip)
    assert np.isnan(out[:, LEFT]).all() and (out[:, RIGHT.start, 0] < 0.3).all()


def test_hands_on_their_own_sides_are_left_alone():
    from training.tunisian_signs import fix_take_sides

    clip = _take()
    assert np.allclose(fix_take_sides(clip), clip, equal_nan=True)
