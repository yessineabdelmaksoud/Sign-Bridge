"""The clinician's summary: fixed templates over confirmed answers, plus urgency flags.

No text is generated. Every line is "<short question label> : <answer label>",
and every flag comes from a rule in flow.json that a clinician can read and change.
"""
from __future__ import annotations

from dataclasses import dataclass

from consultation.flow import LANGS, Flow, Question


@dataclass(frozen=True)
class Summary:
    lines: dict   # {"fr": [...], "ar": [...]}
    flags: list   # [{"id", "fr", "ar"}, ...]


def _value(flow: Flow, question: Question, answer, lang: str) -> str:
    if question.kind == "yesno":
        return flow.label("YES" if answer else "NO", lang)
    if question.kind == "scale":
        return f"{answer}/10"
    if question.kind == "sign_multi":
        return ", ".join(flow.label(code, lang) for code in answer) if answer else flow.label("NONE", lang)
    return flow.label(answer, lang)


def _fires(rule: dict, answers: dict) -> bool:
    when, minimum = rule.get("when", {}), rule.get("min", {})
    if not when and not minimum:
        return False
    for question_id, codes in when.items():
        answer = answers.get(question_id)
        hit = any(a in codes for a in answer) if isinstance(answer, list) else answer in codes
        if not hit:
            return False
    for question_id, floor in minimum.items():
        answer = answers.get(question_id)
        if not isinstance(answer, int) or isinstance(answer, bool) or answer < floor:
            return False
    return True


def build_summary(flow: Flow, answers: dict) -> Summary:
    lines = {lang: [] for lang in LANGS}
    for question in flow.questions:
        if question.id == "consent" or question.id not in answers:
            continue
        for lang in LANGS:
            lines[lang].append(f"{question.short[lang]} : {_value(flow, question, answers[question.id], lang)}")
    flags = [{"id": r["id"], "fr": r["fr"], "ar": r["ar"]} for r in flow.rules if _fires(r, answers)]
    return Summary(lines=lines, flags=flags)
