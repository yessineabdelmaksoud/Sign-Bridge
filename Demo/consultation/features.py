"""Turn canonical landmark clips into fixed-size model features.

Per frame (184 values, in this order):
  pose            14  points 0, 11-16 (x, y) relative to the shoulder midpoint,
                      in shoulder widths
  left hand       42  21 points (x, y), same origin and scale
  right hand      42
  left hand local 42  relative to its own wrist, in hand sizes
  right hand local 42
  presence         2  left present, right present
A missing hand is all zeros with presence 0, so features never contain NaN.
"""
import numpy as np

from consultation.landmarks import LEFT, RIGHT

FRAMES = 32
WIDTH = 184
_POSE_POINTS = [0, 11, 12, 13, 14, 15, 16]
_FLOOR = 1e-3


class NoHandError(ValueError):
    """Raised when a clip has no frame with a visible hand."""


def trim_to_hands(clip: np.ndarray) -> np.ndarray:
    """Keep the span from the first to the last frame that shows a hand."""
    has_hand = np.isfinite(clip[:, LEFT.start:RIGHT.stop]).any(axis=(1, 2))
    index = np.flatnonzero(has_hand)
    if index.size == 0:
        raise NoHandError("no hand visible in any frame")
    return clip[index[0]:index[-1] + 1]


def resample(clip: np.ndarray, frames: int = FRAMES) -> np.ndarray:
    """Pick `frames` frames spread evenly over the clip (nearest index)."""
    index = np.round(np.linspace(0, len(clip) - 1, frames)).astype(int)
    return clip[index]


def _hand_blocks(hand_xy: np.ndarray, origin: np.ndarray, scale: float):
    if not np.isfinite(hand_xy).all():
        zeros = np.zeros(42, dtype=np.float32)
        return zeros, zeros, 0.0
    body = (hand_xy - origin) / scale
    local = hand_xy - hand_xy[0]
    size = max(float(np.linalg.norm(local, axis=1).max()), _FLOOR)
    return body.ravel(), (local / size).ravel(), 1.0


def frame_features(frame: np.ndarray) -> np.ndarray:
    xy = np.asarray(frame, dtype=np.float32)[:, :2]
    origin = (xy[11] + xy[12]) / 2.0
    scale = max(float(np.nan_to_num(np.linalg.norm(xy[11] - xy[12]))), _FLOOR)
    pose = ((xy[_POSE_POINTS] - origin) / scale).ravel()
    left_body, left_local, left_on = _hand_blocks(xy[LEFT], origin, scale)
    right_body, right_local, right_on = _hand_blocks(xy[RIGHT], origin, scale)
    out = np.concatenate([pose, left_body, right_body, left_local, right_local, [left_on, right_on]])
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def clip_features(clip: np.ndarray) -> np.ndarray:
    """Trim to the signed span, resample to FRAMES, and featurise each frame."""
    frames = resample(trim_to_hands(np.asarray(clip, dtype=np.float32)))
    return np.stack([frame_features(f) for f in frames])
