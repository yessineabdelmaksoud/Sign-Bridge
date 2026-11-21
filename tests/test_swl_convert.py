import io
import os
import pickle
import struct
import sys
import types
import zipfile
from collections import Counter

import numpy as np

from consultation.landmarks import LEFT, POSE, RIGHT
from training.swl_convert import frame_to_canonical, load_frames, parse_landmark_list, pickle_to_clip, select_ids
from training.swl_remote import RemoteZip

MP_MODULE = "mediapipe.framework.formats.landmark_pb2"


def encode_landmarks(points, extra=False):
    """Serialise points the way MediaPipe's NormalizedLandmarkList does."""
    out = b""
    for x, y, z in points:
        body = b"\x0d" + struct.pack("<f", x) + b"\x15" + struct.pack("<f", y) + b"\x1d" + struct.pack("<f", z)
        if extra:  # visibility and presence, as pose landmarks carry
            body += b"\x25" + struct.pack("<f", 0.9) + b"\x2d" + struct.pack("<f", 0.8)
        out += b"\x0a" + bytes([len(body)]) + body
    return out


def mediapipe_style_pickle(frames, monkeypatch):
    """Pickle frames that reference MediaPipe classes, without MediaPipe installed."""
    module = types.ModuleType(MP_MODULE)

    class NormalizedLandmarkList:
        def __init__(self, data=b""):
            self._data = data

        def __getstate__(self):
            return {"serialized": self._data}

        def __setstate__(self, state):
            self._data = state["serialized"]

    NormalizedLandmarkList.__module__ = MP_MODULE
    NormalizedLandmarkList.__qualname__ = "NormalizedLandmarkList"
    module.NormalizedLandmarkList = NormalizedLandmarkList
    parts = MP_MODULE.split(".")
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        monkeypatch.setitem(sys.modules, name, module if name == MP_MODULE else types.ModuleType(name))
    wrapped = [
        {"pose": None, "hands": None, "holistic_legacy": {
            key: None if value is None else NormalizedLandmarkList(value) for key, value in frame.items()}}
        for frame in frames
    ]
    data = pickle.dumps(wrapped, protocol=4)
    monkeypatch.undo()
    assert MP_MODULE not in sys.modules
    return data


def test_parse_landmark_list_reads_xyz_and_ignores_other_fields():
    pts = parse_landmark_list(encode_landmarks([(0.5, 0.6, -0.1), (0.25, 0.75, 0.0)], extra=True))
    assert pts.shape == (2, 3) and pts.dtype == np.float32
    assert np.allclose(pts, [(0.5, 0.6, -0.1), (0.25, 0.75, 0.0)])


def test_pickle_is_read_without_mediapipe(monkeypatch):
    frame = {"pose_landmarks": encode_landmarks([(0.5, 0.6, 0.0)] * 33, extra=True),
             "left_hand_landmarks": encode_landmarks([(0.3, 0.4, 0.0)] * 21),
             "right_hand_landmarks": None}
    data = mediapipe_style_pickle([frame, frame], monkeypatch)
    frames = load_frames(data)
    f = frame_to_canonical(frames[0])
    assert len(frames) == 2 and f.shape == (75, 3) and f.dtype == np.float32
    assert np.allclose(f[POSE][0, :2], [0.5, 0.6]) and np.allclose(f[LEFT][0, :2], [0.3, 0.4])
    assert np.isnan(f[RIGHT]).all()
    assert pickle_to_clip(data).shape == (2, 75, 3)


def test_loader_never_runs_code_from_the_pickle(tmp_path):
    marker = tmp_path / "pwned"

    class Evil:
        def __reduce__(self):
            return (os.mkdir, (str(marker),))

    data = pickle.dumps([{"holistic_legacy": {"pose_landmarks": Evil()}}])
    load_frames(data)
    assert not marker.exists()


class BytesReader:
    def __init__(self, data):
        self.data, self.size, self.calls = data, len(data), []

    def read_range(self, start, end):
        self.calls.append((start, end))
        return self.data[start:end]


def test_remote_zip_reads_one_member_with_one_request():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("MEDIAPIPE/a.pkl", os.urandom(40_000))
        z.writestr("MEDIAPIPE/b.pkl", b"hello" * 100)
    reader = BytesReader(buf.getvalue())
    remote = RemoteZip(reader)
    before = len(reader.calls)
    assert remote.read("MEDIAPIPE/b.pkl") == b"hello" * 100
    assert len(reader.calls) == before + 1
    assert set(remote.names()) == {"MEDIAPIPE/a.pkl", "MEDIAPIPE/b.pkl"}


def test_select_ids_takes_whole_classes_and_a_seeded_sample_of_the_rest():
    ann = {"train": [(f"t{i}", i % 30) for i in range(600)],
           "val": [(f"v{i}", i % 30) for i in range(90)],
           "test": [(f"x{i}", i % 30) for i in range(60)]}
    names = {i: f"S{i}" for i in range(30)}
    quota = {"train": 10, "val": 3, "test": 3}
    picked = select_ids(ann, names, ["S0", "S1"], other=quota, seed=42)
    count = Counter((label, split) for _, label, split in picked)
    assert count[("S0", "train")] == 20 and count[("S1", "val")] == 3 and count[("S1", "test")] == 2
    assert [count[("OTHER", s)] for s in ("train", "val", "test")] == [10, 3, 3]
    selected_ids = {i for i, label, _ in picked if label != "OTHER"}
    assert all(i not in selected_ids for i, label, _ in picked if label == "OTHER")
    assert picked == select_ids(ann, names, ["S0", "S1"], other=quota, seed=42)
