from consultation.camera import ClipSink
from tests.test_features import frame


def feed(sink, pattern, dt=1 / 30):
    t = 0.0
    for has_hand in pattern:
        sink.push(frame(left=bool(has_hand)), t)
        t += dt


def test_sink_hands_over_a_finished_sign_once():
    sink = ClipSink()
    feed(sink, [1] * 20 + [0] * 10)
    clip = sink.take()
    assert clip is not None and len(clip) == 20
    assert sink.take() is None


def test_sink_reports_what_the_camera_sees():
    sink = ClipSink()
    assert sink.status == "idle"
    feed(sink, [1] * 4)
    assert sink.status == "recording"
    feed(sink, [0] * 9)
    assert sink.status == "idle"


def test_sink_keeps_only_the_newest_sign_when_nobody_collects():
    sink = ClipSink()
    feed(sink, ([1] * 12 + [0] * 9) * 5)
    assert len(sink.take()) == 12 and sink.take() is None
