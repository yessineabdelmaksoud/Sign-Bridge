"""How the consultation looks: one stylesheet and a few HTML builders.

The visual language is hospital wayfinding. The question is a blue bilingual
sign, Arabic above French as on a Tunisian hospital wall; the answers are sign
tiles with a pictogram. Fonts are whatever the device has, so the page needs
no connection.
"""
from __future__ import annotations

from html import escape

from consultation.flow import Flow, Question
from consultation.summary import Summary

LATIN = '"Inter", "Noto Sans", "Segoe UI", "Helvetica Neue", Arial, sans-serif'
ARABIC = '"Noto Kufi Arabic", "Noto Sans Arabic", "Noto Naskh Arabic", "Segoe UI", "Geeza Pro", Tahoma, sans-serif'

CSS = f"""
:root {{
  --sign: #12508F;      /* hospital signage blue */
  --sign-deep: #0C3A6B;
  --wall: #EEF2F6;      /* clinic wall */
  --ink: #10233A;
  --muted: #4E6074;
  --line: #C5D0DC;
  --ok: #0F8A5F;        /* pharmacy cross green */
  --alert: #C8102E;     /* emergency red */
  --amber: #E8A100;
}}
.stApp, [data-testid="stHeader"] {{ background: var(--wall); }}
.stApp {{ color: var(--ink); font-family: {LATIN}; }}
.block-container {{ padding-top: 2.4rem; padding-bottom: 2rem; max-width: 1380px; }}
.stApp p, .stApp li, .stApp span, .stApp label {{ color: var(--ink); }}

/* The question: one big bilingual sign. */
.sb-sign {{
  background: var(--sign); border-radius: 14px; padding: 1.6rem 2rem 1.7rem;
  margin-bottom: 1.1rem;
}}
.sb-sign .ar {{
  font-family: {ARABIC}; direction: rtl; text-align: right; color: #fff;
  font-size: clamp(1.7rem, 3vw, 2.7rem); font-weight: 800; line-height: 1.45; margin: 0;
}}
.sb-sign .fr {{
  color: #fff; font-size: clamp(1.25rem, 2.1vw, 1.85rem); font-weight: 700;
  line-height: 1.3; margin: .55rem 0 0;
}}
.sb-sign.quiet {{ background: #fff; border: 2px solid var(--line); }}
.sb-sign.quiet .ar, .sb-sign.quiet .fr {{ color: var(--ink); }}
.sb-sign.alert {{ background: var(--alert); }}

/* What the app understood, waiting for the patient's yes or no. */
.sb-heard {{
  background: #fff; border: 4px solid var(--ok); border-radius: 14px;
  padding: 1.4rem 2rem; margin-bottom: 1.1rem; display: flex; align-items: center; gap: 1.6rem;
}}
.sb-heard .pic {{ font-size: 5rem; line-height: 1; }}
.sb-heard .ar {{ font-family: {ARABIC}; direction: rtl; font-size: 2.3rem; font-weight: 800; margin: 0; }}
.sb-heard .fr {{ font-size: 1.7rem; font-weight: 700; margin: .2rem 0 0; }}
.sb-heard .ask {{ color: var(--muted); font-size: 1.05rem; margin: 0 0 .5rem; }}

/* Answer tiles are Streamlit buttons: pictogram, Arabic, French on three lines. */
.stApp [class*="st-key-opt_"] button,
.stApp [class*="st-key-guess_"] button,
.stApp [class*="st-key-ins_"] button {{
  background: #fff; color: var(--ink); border: 2px solid var(--ink); border-radius: 8px;
  width: 100%; min-height: 9.4rem; padding: .6rem .5rem;
}}
.stApp [class*="st-key-opt_"] button p,
.stApp [class*="st-key-guess_"] button p,
.stApp [class*="st-key-ins_"] button p,
.stApp [class*="st-key-confirm_"] button p,
.stApp [class*="st-key-multi_done"] button p {{
  display: block; margin: 0; white-space: pre-line; unicode-bidi: plaintext; text-align: center;
  font-weight: 700; font-size: 1.08rem; line-height: 1.4; color: inherit;
}}
.stApp [class*="st-key-opt_"] button p::first-line,
.stApp [class*="st-key-guess_"] button p::first-line,
.stApp [class*="st-key-ins_"] button p::first-line,
.stApp [class*="st-key-confirm_"] button p::first-line {{ font-size: 2.7rem; line-height: 1.45; }}
.stApp [class*="st-key-opt_"] button [data-testid="stMarkdownContainer"],
.stApp [class*="st-key-guess_"] button [data-testid="stMarkdownContainer"],
.stApp [class*="st-key-ins_"] button [data-testid="stMarkdownContainer"],
.stApp [class*="st-key-confirm_"] button [data-testid="stMarkdownContainer"],
.stApp .st-key-multi_done button [data-testid="stMarkdownContainer"] {{
  overflow: visible; text-overflow: clip; white-space: normal; width: 100%;
}}
.stApp [class*="st-key-opt_"] button:hover,
.stApp [class*="st-key-guess_"] button:hover,
.stApp [class*="st-key-ins_"] button:hover {{ background: #E3ECF7; border-color: var(--sign); color: var(--ink); }}
.stApp [class*="st-key-guess_"] button {{ border-color: var(--amber); border-width: 4px; }}
.stApp [class*="st-key-opt_SCALE_"] button {{ min-height: 5.4rem; padding: .2rem; }}
.stApp [class*="st-key-opt_SCALE_"] button p {{ white-space: nowrap; }}
.stApp .st-key-bodymap [class*="st-key-opt_"] button {{ min-height: 5.6rem; }}
.stApp .st-key-bodymap [class*="st-key-opt_"] button p::first-line {{ font-size: 1.9rem; }}
.stApp [class*="st-key-opt_SCALE_"] button p::first-line {{ font-size: 1.7rem; }}

.stApp [class*="st-key-confirm_"] button {{ width: 100%; min-height: 7.4rem; border-radius: 8px; border: 3px solid; }}
.stApp .st-key-confirm_yes button {{ background: var(--ok); border-color: var(--ok); color: #fff; }}
.stApp .st-key-confirm_yes button:hover {{ background: #0B6E4B; border-color: #0B6E4B; color: #fff; }}
.stApp .st-key-confirm_no button {{ background: #fff; border-color: var(--alert); color: var(--alert); }}
.stApp .st-key-confirm_no button:hover {{ background: #FCE9EC; color: var(--alert); }}
.stApp .st-key-multi_done button {{
  width: 100%; min-height: 4.4rem; background: var(--sign); border: 2px solid var(--sign); color: #fff; border-radius: 8px;
}}
.stApp .st-key-multi_done button:hover {{ background: var(--sign-deep); color: #fff; }}
/* Streamlit colours button text itself, so each filled button states its own text colour. */
.stApp .st-key-confirm_yes button p, .stApp .st-key-multi_done button p {{ color: #fff; }}
.stApp .st-key-confirm_no button p {{ color: var(--alert); }}
.stApp button:focus-visible {{ outline: 4px solid var(--amber); outline-offset: 2px; }}

/* The doctor's side: a plain sheet. */
.sb-note {{ background: #fff; border-left: 6px solid var(--sign); border-radius: 4px; padding: 1.1rem 1.3rem 1.2rem; }}
.sb-note h2 {{ font-size: 1.15rem; font-weight: 800; margin: 0 0 .7rem; color: var(--ink); padding: 0; }}
.sb-note dl {{ margin: 0; }}
.sb-note dt {{ color: var(--muted); font-size: .86rem; margin-top: .65rem; }}
.sb-note dd {{ margin: 0; font-size: 1.12rem; font-weight: 700; color: var(--ink); }}
.sb-note .empty {{ color: var(--muted); margin: 0; }}
.stApp .sb-flag, .stApp .sb-flag span {{
  background: var(--alert); color: #fff; font-weight: 800; border-radius: 4px;
  padding: .55rem .8rem; margin: 0 0 .5rem; font-size: 1.02rem;
}}
.sb-sheet {{ background: #fff; border: 2px solid var(--line); border-radius: 8px; padding: 1.2rem 1.5rem; }}
.sb-sheet .cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
.sb-sheet p {{ margin: .28rem 0; font-size: 1.06rem; }}
.sb-sheet .ar p {{ font-family: {ARABIC}; direction: rtl; text-align: right; }}
@media (max-width: 760px) {{ .sb-sheet .cols {{ grid-template-columns: 1fr; }} .sb-heard {{ flex-direction: column; }} }}

.sb-steps {{ display: flex; gap: .4rem; margin: 0 0 .9rem; }}
.sb-steps i {{ width: 2.1rem; height: .5rem; border-radius: 3px; background: var(--line); display: block; }}
.sb-steps i.done {{ background: var(--sign); }}
.sb-steps i.now {{ background: var(--amber); }}

.sb-hint {{ background: #FFF6DB; border-left: 6px solid var(--amber); border-radius: 4px; padding: .7rem 1rem; margin: 0 0 1rem; }}
.sb-hint p {{ margin: 0; font-weight: 600; }}
.sb-hint .ar {{ font-family: {ARABIC}; direction: rtl; text-align: right; }}
.sb-chips {{ display: flex; flex-wrap: wrap; gap: .5rem; margin: 0 0 1rem; }}
.sb-chips span {{ background: var(--ok); color: #fff; font-weight: 700; border-radius: 4px; padding: .3rem .7rem; }}
.sb-cam {{ display: flex; align-items: center; gap: .6rem; font-weight: 700; margin: .4rem 0; }}
.sb-cam i {{ width: .95rem; height: .95rem; border-radius: 50%; background: var(--line); display: block; }}
.sb-cam.ready i {{ background: var(--ok); }}
.sb-cam.rec i {{ background: var(--alert); animation: sb-pulse 1s ease-in-out infinite; }}
@keyframes sb-pulse {{ 50% {{ opacity: .25; }} }}
@media (prefers-reduced-motion: reduce) {{ .sb-cam.rec i {{ animation: none; }} }}
.stApp .sb-foot {{ color: var(--muted); font-size: .88rem; margin-top: 1.6rem; max-width: 70ch; }}
"""

_FACES = ("🙂", "🙂", "🙂", "🙂", "😐", "😐", "😐", "😣", "😣", "😫", "😫")


def sign_html(question: Question, tone: str = "") -> str:
    """The question as a bilingual sign: Arabic (right to left) above French."""
    return bilingual_sign(question.text["ar"], question.text["fr"], tone)


def bilingual_sign(arabic: str, french: str, tone: str = "") -> str:
    return (f'<div class="sb-sign {escape(tone)}">'
            f'<p class="ar" dir="rtl" lang="ar">{escape(arabic)}</p>'
            f'<p class="fr" lang="fr">{escape(french)}</p></div>')


def tile_label(flow: Flow, code: str) -> str:
    """Button text for an answer tile: pictogram, Arabic, French on three lines."""
    return f"{flow.icon(code)}\n{flow.label(code, 'ar')}\n{flow.label(code, 'fr')}"


def value_display(flow: Flow, question: Question, value) -> tuple[str, str, str]:
    """(pictogram, French, Arabic) for one answer."""
    if question.kind == "yesno":
        code = "YES" if value else "NO"
        return flow.icon(code), flow.label(code, "fr"), flow.label(code, "ar")
    if question.kind == "scale":
        return _FACES[max(0, min(10, int(value)))], f"{value}/10", f"{value}/10"
    return flow.icon(value), flow.label(value, "fr"), flow.label(value, "ar")


def heard_html(flow: Flow, question: Question, value) -> str:
    """What the app understood, shown back to the patient before it counts."""
    icon, french, arabic = value_display(flow, question, value)
    return (f'<div class="sb-heard"><div class="pic">{escape(icon)}</div><div>'
            f'<p class="ask">{escape(question.short["fr"])}</p>'
            f'<p class="ar" dir="rtl" lang="ar">{escape(arabic)}</p>'
            f'<p class="fr" lang="fr">{escape(french)}</p></div></div>')


def note_html(summary: Summary) -> str:
    """The doctor's running note: urgency first, then one line per confirmed answer."""
    parts = ['<div class="sb-note"><h2>Note du médecin</h2>']
    parts += [f'<p class="sb-flag">{escape(flag["fr"])}</p>' for flag in summary.flags]
    lines = summary.lines["fr"]
    if not lines:
        parts.append('<p class="empty">Les réponses confirmées par le patient apparaissent ici.</p>')
    else:
        parts.append("<dl>")
        for line in lines:
            label, _, value = line.partition(" : ")
            parts.append(f"<dt>{escape(label)}</dt><dd>{escape(value)}</dd>")
        parts.append("</dl>")
    parts.append("</div>")
    return "".join(parts)


def sheet_html(summary: Summary) -> str:
    """The finished summary in both languages, side by side."""
    flags = "".join(f'<p class="sb-flag">{escape(f["fr"])} / <span dir="rtl" lang="ar">{escape(f["ar"])}</span></p>'
                    for f in summary.flags)
    french = "".join(f"<p>{escape(line)}</p>" for line in summary.lines["fr"])
    arabic = "".join(f'<p dir="rtl" lang="ar">{escape(line)}</p>' for line in summary.lines["ar"])
    return (f'<div class="sb-sheet">{flags}<div class="cols"><div class="fr">{french}</div>'
            f'<div class="ar">{arabic}</div></div></div>')


def note_text(summary: Summary) -> str:
    """The summary as plain text, for the doctor to download or paste into the file."""
    lines = ["Note de consultation (SignBridge)", ""]
    lines += [f"! {flag['fr']}" for flag in summary.flags]
    if summary.flags:
        lines.append("")
    lines += summary.lines["fr"] + [""] + summary.lines["ar"]
    return "\n".join(lines) + "\n"


def steps_html(done: int, total: int) -> str:
    cells = "".join(f'<i class="{"done" if i < done else "now" if i == done else ""}"></i>' for i in range(total))
    return f'<div class="sb-steps" role="img" aria-label="Question {done + 1} sur {total}">{cells}</div>'


def hint_html(french: str, arabic: str) -> str:
    return (f'<div class="sb-hint"><p class="ar" dir="rtl" lang="ar">{escape(arabic)}</p>'
            f'<p lang="fr">{escape(french)}</p></div>')


def chips_html(flow: Flow, codes: list[str]) -> str:
    chips = "".join(f"<span>{escape(flow.icon(c))} {escape(flow.label(c, 'fr'))}</span>" for c in codes)
    return f'<div class="sb-chips">{chips}</div>'


def camera_status_html(state: str, french: str) -> str:
    return f'<div class="sb-cam {escape(state)}"><i></i><span>{escape(french)}</span></div>'
