"""One consultation in progress.

The rule this class enforces: an answer the patient has not confirmed is not
an answer. `propose` holds a value as pending; only `confirm` records it.
"""
from __future__ import annotations

import time
from typing import Callable

from consultation.flow import Flow, Question


class Consultation:
    IDLE_SECONDS = 600
    MAX_SIGN_ATTEMPTS = 2

    def __init__(self, flow: Flow, now: Callable[[], float] = time.monotonic):
        self._flow = flow
        self._now = now
        self.clear()

    def clear(self) -> None:
        """Forget everything and go back to the first question."""
        self.answers: dict[str, object] = {}
        self.pending: object | None = None
        self.has_pending = False
        self.collected: list[str] = []   # confirmed picks of a multi-answer question in progress
        self._attempts = 0
        self._last_activity = self._now()

    @property
    def current(self) -> Question | None:
        return self._flow.next_question(self.answers)

    @property
    def finished(self) -> bool:
        return self.current is None

    @property
    def needs_tap_fallback(self) -> bool:
        return self._attempts >= self.MAX_SIGN_ATTEMPTS

    def propose(self, value) -> None:
        """Hold a recognised or tapped answer until the patient confirms it."""
        self._touch()
        self.pending, self.has_pending = value, True

    def confirm(self) -> None:
        if not self.has_pending:
            raise RuntimeError("there is no pending answer to confirm")
        question, value = self.current, self.pending
        self._drop_pending()
        self._attempts = 0
        if question is None:
            return
        if question.kind == "sign_multi":
            if value not in self.collected:
                self.collected.append(value)
        else:
            self.answers[question.id] = value

    def reject(self) -> None:
        """The patient said the proposed answer is wrong."""
        self._drop_pending()
        self._attempts += 1

    def finish_multi(self) -> None:
        """Close a multi-answer question; an empty list means "none"."""
        question = self.current
        if question is None or question.kind != "sign_multi":
            raise RuntimeError("the current question does not take several answers")
        self._drop_pending()
        self.answers[question.id] = list(self.collected)
        self.collected = []
        self._attempts = 0

    def expired(self) -> bool:
        return self._now() - self._last_activity >= self.IDLE_SECONDS

    def _drop_pending(self) -> None:
        self._touch()
        self.pending, self.has_pending = None, False

    def _touch(self) -> None:
        self._last_activity = self._now()
