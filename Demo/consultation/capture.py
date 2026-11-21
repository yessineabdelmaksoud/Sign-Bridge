"""Cut one sign out of a stream of landmark frames.

A sign starts when a hand has been visible for a few frames in a row and ends
when the hands have been gone for a moment, or after a hard time limit.
"""
import numpy as np

from consultation.landmarks import hand_present


class SignCapture:
    START_FRAMES = 3     # consecutive frames with a hand before recording starts
    END_GAP_FRAMES = 8   # consecutive frames without a hand that end the sign
    MAX_SECONDS = 4.0    # hard limit, measured from the first recorded frame
    MIN_FRAMES = 8       # anything shorter is a blip, not a sign

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.recording = False
        self._lead = []      # (frame, time) seen before recording starts
        self._frames = []
        self._gap = 0
        self._t_start = 0.0

    def push(self, frame: np.ndarray, t: float):
        """Add one frame. Returns the finished clip (T, 75, 3) once, else None."""
        present = hand_present(frame)
        if not self.recording:
            if not present:
                self._lead = []
                return None
            self._lead.append((frame, t))
            if len(self._lead) >= self.START_FRAMES:
                self.recording = True
                self._t_start = self._lead[0][1]
                self._frames = [f for f, _ in self._lead]
                self._lead, self._gap = [], 0
            return None

        self._frames.append(frame)
        self._gap = 0 if present else self._gap + 1
        if self._gap >= self.END_GAP_FRAMES or t - self._t_start >= self.MAX_SECONDS:
            return self._finish()
        return None

    def _finish(self):
        frames = self._frames[:len(self._frames) - self._gap]
        self.reset()
        if len(frames) < self.MIN_FRAMES:
            return None
        return np.stack(frames)
