from consultation.capture import SignCapture
from tests.test_features import frame


def feed(cap, pattern, dt=1 / 30):
    out, t = [], 0.0
    for has_hand in pattern:
        r = cap.push(frame(left=bool(has_hand)), t)
        t += dt
        if r is not None:
            out.append(r)
    return out


def test_returns_one_clip_without_trailing_gap():
    clips = feed(SignCapture(), [0] * 5 + [1] * 20 + [0] * 10)
    assert len(clips) == 1 and len(clips[0]) == 20


def test_no_hand_never_returns_a_clip():
    cap = SignCapture()
    assert feed(cap, [0] * 300) == [] and not cap.recording


def test_short_blip_is_discarded():
    assert feed(SignCapture(), [1] * 5 + [0] * 10) == []


def test_cut_at_four_seconds():
    clips = feed(SignCapture(), [1] * 200)
    assert len(clips) >= 1 and 110 <= len(clips[0]) <= 122


def test_recording_flag_follows_the_hand():
    cap = SignCapture()
    feed(cap, [1] * 3)
    assert cap.recording
    feed(cap, [0] * 8)
    assert not cap.recording
