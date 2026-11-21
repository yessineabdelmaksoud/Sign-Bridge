"""The canonical landmark frame shared by training, live capture and features.

A frame is a float32 array of shape (75, 3): rows 0-32 are MediaPipe pose
points, rows 33-53 the left hand, rows 54-74 the right hand. x and y are
image-normalised (0..1); a missing hand is all NaN.
"""
import numpy as np

POINTS = 75
POSE = slice(0, 33)
LEFT = slice(33, 54)
RIGHT = slice(54, 75)

# MediaPipe pose landmarks that come in left/right pairs.
_POSE_PAIRS = (
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
    (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
)


def empty_frame() -> np.ndarray:
    return np.full((POINTS, 3), np.nan, dtype=np.float32)


def hand_present(frame: np.ndarray) -> bool:
    """True if either hand has at least one finite coordinate."""
    return bool(np.isfinite(frame[LEFT.start:RIGHT.stop]).any())


def mirror_clip(clip: np.ndarray) -> np.ndarray:
    """Flip a clip left-to-right, as if the signer used the other hand."""
    out = np.array(clip, dtype=np.float32, copy=True)
    out[..., 0] = 1.0 - out[..., 0]
    left = out[:, LEFT].copy()
    out[:, LEFT] = out[:, RIGHT]
    out[:, RIGHT] = left
    for a, b in _POSE_PAIRS:
        out[:, [a, b]] = out[:, [b, a]]
    return out


def _fill(block: np.ndarray, landmark_list) -> None:
    if landmark_list is None:
        return
    points = getattr(landmark_list, "landmark", landmark_list)
    for i, point in enumerate(points):
        if i >= len(block):
            break
        block[i] = (point.x, point.y, point.z)


def canonical_from_holistic(results) -> np.ndarray:
    """Convert one MediaPipe Holistic result into a canonical frame."""
    frame = empty_frame()
    _fill(frame[POSE], getattr(results, "pose_landmarks", None))
    _fill(frame[LEFT], getattr(results, "left_hand_landmarks", None))
    _fill(frame[RIGHT], getattr(results, "right_hand_landmarks", None))
    return frame
