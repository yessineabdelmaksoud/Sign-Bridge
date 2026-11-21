"""The consultation as data: questions, skip logic, labels and urgency rules.

Everything a clinician might want to review lives in flow.json. This module
only loads it, checks it, and answers "which question comes next?".
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PATH = Path(__file__).with_name("flow.json")
KINDS = {"yesno", "sign", "sign_multi", "bodymap", "scale", "choice"}
LANGS = ("fr", "ar")


class FlowError(ValueError):
    """The flow file is inconsistent."""


@dataclass(frozen=True)
class Question:
    id: str
    kind: str
    text: dict
    short: dict
    options: list = field(default_factory=list)
    show_if: dict | None = None
    hide_if: dict | None = None


def _matches(answer, codes) -> bool:
    if isinstance(answer, list):
        return any(a in codes for a in answer)
    return answer in codes


class Flow:
    def __init__(self, questions, labels, rules, instructions):
        self.questions: list[Question] = questions
        self._labels: dict = labels
        self.rules: list[dict] = rules
        self.instructions: list[str] = instructions

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PATH) -> "Flow":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        questions = [
            Question(
                id=q["id"], kind=q["kind"], text=q["text"], short=q["short"],
                options=list(q.get("options", [])),
                show_if=q.get("show_if"), hide_if=q.get("hide_if"),
            )
            for q in data["questions"]
        ]
        flow = cls(questions, data["labels"], data.get("rules", []), data.get("instructions", []))
        flow._check()
        return flow

    def _check(self) -> None:
        seen: list[str] = []
        for q in self.questions:
            if q.kind not in KINDS:
                raise FlowError(f"{q.id}: unknown kind {q.kind!r}")
            if q.id in seen:
                raise FlowError(f"{q.id}: duplicate question id")
            for code in q.options:
                self._require_label(code, q.id)
            for name, cond in (("show_if", q.show_if), ("hide_if", q.hide_if)):
                for other in cond or {}:
                    if other not in seen:
                        raise FlowError(f"{q.id}: {name} refers to {other!r}, which is not an earlier question")
            seen.append(q.id)
        for code in ("YES", "NO", "NONE", *self.instructions):
            self._require_label(code, "labels")
        for rule in self.rules:
            for other in (*rule.get("when", {}), *rule.get("min", {})):
                if other not in seen:
                    raise FlowError(f"rule {rule.get('id')}: unknown question {other!r}")

    def _require_label(self, code: str, where: str) -> None:
        entry = self._labels.get(code)
        if not entry or not all(entry.get(k) for k in (*LANGS, "icon")):
            raise FlowError(f"{where}: {code!r} has no complete label (fr, ar, icon)")

    def next_question(self, answers: dict) -> Question | None:
        """The first unanswered question that applies, or None when done."""
        if answers.get("consent") is False:
            return None
        for q in self.questions:
            if q.id in answers:
                continue
            if q.show_if and not all(_matches(answers.get(k), v) for k, v in q.show_if.items()):
                continue
            if q.hide_if and all(_matches(answers.get(k), v) for k, v in q.hide_if.items()):
                continue
            return q
        return None

    def question(self, question_id: str) -> Question:
        for q in self.questions:
            if q.id == question_id:
                return q
        raise KeyError(question_id)

    def label(self, code: str, lang: str) -> str:
        return self._labels[code][lang]

    def icon(self, code: str) -> str:
        return self._labels[code]["icon"]


SIGN_SETS: dict[str, list[str]] = {
    q.id: list(q.options) for q in Flow.load().questions if q.kind in ("sign", "sign_multi")
}
