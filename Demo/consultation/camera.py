"""The camera path: frames in, at most one finished sign out.

`ClipSink` is the testable part and knows nothing about video. The video
processor at the bottom is a thin wrapper that turns each camera frame into
landmarks and feeds the sink. Frames are processed in memory and dropped;
only landmarks are kept, and only until the sign has been classified.
"""
from __future__ import annotations

import threading
import time

from consultation.capture import SignCapture
from consultation.landmarks import canonical_from_holistic, hand_present


class ClipSink:
    """Collects landmark frames and holds the most recent finished sign."""

    def __init__(self):
        self._capture = SignCapture()
        self._lock = threading.Lock()
        self._clip = None
        self.hand = False

    @property
    def status(self) -> str:
        """"recording" while a sign is being captured, otherwise "idle"."""
        return "recording" if self._capture.recording else "idle"

    def push(self, frame, t: float) -> None:
        self.hand = hand_present(frame)
        clip = self._capture.push(frame, t)
        if clip is not None:
            with self._lock:
                self._clip = clip

    def take(self):
        """The finished sign, once; None if there is nothing new."""
        with self._lock:
            clip, self._clip = self._clip, None
        return clip


def make_processor_class():
    """Build the video processor. Imports the camera stack only when the camera is switched on."""
    import av
    import cv2
    import mediapipe as mp
    from streamlit_webrtc import VideoProcessorBase

    class ConsultationProcessor(VideoProcessorBase):
        def __init__(self):
            self.sink = ClipSink()
            self._holistic = mp.solutions.holistic.Holistic(
                model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        def recv(self, frame):
            image = frame.to_ndarray(format="bgr24")
            results = self._holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            self.sink.push(canonical_from_holistic(results), time.monotonic())
            preview = cv2.flip(image, 1)  # mirror, so the patient sees themselves as in a mirror
            colour = (46, 16, 200) if self.sink.status == "recording" else (95, 138, 15) if self.sink.hand else (160, 160, 160)
            cv2.circle(preview, (30, 30), 13, colour, -1)
            return av.VideoFrame.from_ndarray(preview, format="bgr24")

    return ConsultationProcessor
