"""Guided consultation between a deaf patient and a hearing clinician.

One shared screen. The patient answers each question by signing or by tapping
a picture and confirms what the app understood; only then does the answer
reach the doctor's note. Nothing on this page uses the network, and no camera
image is stored.
"""
import os
from pathlib import Path

import streamlit as st

from consultation import ui
from consultation.flow import Flow
from consultation.recognizer import SignRecognizer, StaticSignRecognizer
from consultation.session import Consultation
from consultation.summary import Summary, build_summary

st.set_page_config(page_title="SignBridge consultation", page_icon="🩺", layout="wide")
st.markdown(f"<style>{ui.CSS}</style>", unsafe_allow_html=True)

MODELS = Path(os.environ.get("SIGNBRIDGE_MODELS") or Path(__file__).resolve().parents[1] / "consultation" / "models")
BODY_ROWS = [[None, "HEAD", None], [None, "THROAT", None], ["ARM_R", "CHEST", "ARM_L"],
             [None, "ABDOMEN", "BACK"], ["LEG_R", None, "LEG_L"]]


@st.cache_resource
def load_flow() -> Flow:
    return Flow.load()


@st.cache_resource
def load_recognizers(folder: str):
    base = Path(folder)
    health_file = base / "health_signs.keras" if (base / "health_signs.keras").exists() else base / "health_signs.npz"
    return (SignRecognizer(health_file, base / "health_signs.json"),
            StaticSignRecognizer(base / "tunisl10.npz", base / "tunisl10.json"))


@st.cache_resource
def camera_processor():
    from consultation.camera import make_processor_class

    return make_processor_class()


flow = load_flow()
health, tunisian = load_recognizers(str(MODELS))
state = st.session_state
if "consultation" not in state:
    state.consultation = Consultation(flow)
for name, default in (("interpreter", False), ("instruction", None), ("camera_on", False), ("guesses", [])):
    state.setdefault(name, default)


# --- actions: each runs before the page is drawn again --------------------------

def start_over() -> None:
    st.session_state.consultation.clear()
    st.session_state.update(interpreter=False, instruction=None, camera_on=False, guesses=[])


def propose(value) -> None:
    st.session_state.consultation.propose(value)


def confirm() -> None:
    st.session_state.consultation.confirm()
    st.session_state.guesses = []


def reject() -> None:
    st.session_state.consultation.reject()


def finish_multi() -> None:
    st.session_state.consultation.finish_multi()
    st.session_state.guesses = []


def ask_interpreter() -> None:
    st.session_state.update(interpreter=True, camera_on=False)


def show_instruction(code: str) -> None:
    st.session_state.instruction = code


def toggle_camera() -> None:
    st.session_state.camera_on = not st.session_state.camera_on


if state.consultation.expired():
    start_over()
consultation: Consultation = state.consultation


# --- drawing --------------------------------------------------------------------

def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def tiles(codes, columns: int, prefix: str = "opt_", value_of=lambda code: code, action=propose) -> None:
    for start in range(0, len(codes), columns):
        for column, code in zip(st.columns(columns), codes[start:start + columns]):
            if code is not None:
                column.button(ui.tile_label(flow, code), key=f"{prefix}{code}", on_click=action,
                              args=(value_of(code),), use_container_width=True)


def signed_answers(question) -> tuple[object, list[str], dict] | None:
    """(recogniser, labels it may answer with, label -> answer value) for this question, if it can be signed."""
    if question.kind in ("sign", "sign_multi") and health.available:
        allowed = [code for code in question.options if code not in consultation.collected]
        return health, allowed, {code: code for code in allowed}
    if tunisian.signing_enabled:
        if question.kind == "yesno":
            wanted = (True, False)
        elif question.kind == "choice":
            wanted = tuple(question.options)
        else:
            return None
        mapping = {label: value for label, value in tunisian.to_answer.items() if value in wanted}
        if mapping:
            return tunisian, list(mapping), mapping
    return None


@st.fragment(run_every=0.5)
def watch_camera(context, question) -> None:
    """Twice a second: show what the camera sees and act on a finished sign."""
    processor = context.video_processor
    if processor is None:
        html(ui.camera_status_html("", "Caméra à l'arrêt. Appuyez sur START, puis signez votre réponse."))
        return
    sink = processor.sink
    if sink.status == "recording":
        html(ui.camera_status_html("rec", "Signe en cours"))
    elif sink.hand:
        html(ui.camera_status_html("ready", "Main visible"))
    else:
        html(ui.camera_status_html("ready", "Prête : levez la main pour signer"))
    clip = sink.take()
    target = signed_answers(question)
    if clip is None or target is None:
        return
    recogniser, allowed, mapping = target
    prediction = recogniser.predict(clip, allowed)
    st.session_state.guesses = [mapping[label] for label, _ in prediction.top3 if label in mapping]
    if prediction.label in mapping:
        st.session_state.consultation.propose(mapping[prediction.label])
    else:
        st.session_state.consultation.reject()
    st.rerun()


def camera_block(question) -> None:
    label = "Éteindre la caméra" if state.camera_on else "📷 Signer la réponse : activer la caméra"
    st.button(label, key="camera_toggle", on_click=toggle_camera)
    if not state.camera_on:
        return
    try:
        from streamlit_webrtc import WebRtcMode, webrtc_streamer

        processor = camera_processor()
    except ImportError:
        html(ui.hint_html("Caméra indisponible : installez streamlit-webrtc et mediapipe. Répondez en touchant une image.",
                          "الكاميرا موش متوفرة. جاوب باللمس على الصورة."))
        return
    context = webrtc_streamer(
        key="consultation_camera", mode=WebRtcMode.SENDRECV, video_processor_factory=processor,
        rtc_configuration={"iceServers": []}, media_stream_constraints={"video": True, "audio": False},
        async_processing=True)
    watch_camera(context, question)


def question_view(question) -> None:
    html(ui.sign_html(question))
    if question.kind in ("sign", "sign_multi") and not health.available:
        html(ui.hint_html("Répondez en touchant une image. La reconnaissance des signes n'est pas installée sur cet appareil.",
                          "جاوب باللمس على الصورة."))
    if consultation.needs_tap_fallback:
        html(ui.hint_html("Signe non reconnu. Touchez votre réponse.", "ما فهمتش الإشارة. المس الجواب."))
        guesses = [g for g in state.guesses if isinstance(g, str) and g in question.options]
        if guesses:
            tiles(guesses, 3, prefix="guess_")

    if question.kind == "yesno":
        tiles(["YES", "NO"], 2, value_of=lambda code: code == "YES")
    elif question.kind == "scale":
        for column, level in zip(st.columns(11), range(11)):
            column.button(str(level), key=f"opt_SCALE_{level}", on_click=propose, args=(level,),
                          use_container_width=True)
        html(ui.hint_html("0 = aucune douleur. 10 = la pire douleur.", "0 = ما فماش وجيعة. 10 = أقوى وجيعة."))
    elif question.kind == "bodymap":
        with st.container(key="bodymap"):
            for row in BODY_ROWS:
                tiles(row, 3)
    elif question.kind == "choice":
        tiles(question.options[:5], 3)
        if tunisian.signing_enabled:   # weekdays are offered only when they can be signed
            tiles(question.options[5:], 4)
    else:
        if consultation.collected:
            html(ui.chips_html(flow, consultation.collected))
        tiles([code for code in question.options if code not in consultation.collected], 4)
        if question.kind == "sign_multi":
            done = "✔ هذا الكل\nC'est tout" if consultation.collected else "🚫 لا شيء\nAucune"
            st.button(done, key="multi_done", on_click=finish_multi, use_container_width=True)

    if signed_answers(question) is not None:
        camera_block(question)


def confirmation_view(question) -> None:
    html(ui.heard_html(flow, question, consultation.pending))
    yes, no = st.columns(2)
    yes.button("✅\nنعم، صحيح\nOui, c'est ça", key="confirm_yes", on_click=confirm, use_container_width=True)
    no.button("❌\nلا، بدّل\nNon, changer", key="confirm_no", on_click=reject, use_container_width=True)


def finished_view() -> None:
    summary = build_summary(flow, consultation.answers)
    if state.instruction:
        code = state.instruction
        html(ui.bilingual_sign(f"{flow.icon(code)}  {flow.label(code, 'ar')}", f"{flow.icon(code)}  {flow.label(code, 'fr')}"))
    else:
        html(ui.bilingual_sign("كملنا الأسئلة. شكرا", "Questions terminées. Merci.", "quiet"))
    html(ui.sheet_html(summary))
    st.download_button("Télécharger la note", data=ui.note_text(summary), file_name="note-consultation.txt",
                       mime="text/plain", key="download_note")
    html('<p class="sb-foot">Réponse du médecin : touchez un message pour le montrer au patient.</p>')
    tiles(flow.instructions, 3, prefix="ins_", action=show_instruction)


def patient_side() -> None:
    if state.interpreter:
        html(ui.bilingual_sign("المريض يطلب مترجم", "Interprète demandé", "alert"))
        return
    question = consultation.current
    if question is None:
        if consultation.answers.get("consent") is False:
            html(ui.bilingual_sign("ما بديناش الاستشارة", "Consultation non commencée", "quiet"))
            html(ui.hint_html("Le patient n'a pas donné son accord pour la caméra et les questions.",
                              "المريض ما وافقش."))
        else:
            finished_view()
        return
    html(ui.steps_html(flow.questions.index(question), len(flow.questions)))
    if consultation.has_pending:
        confirmation_view(question)
    else:
        question_view(question)


def doctor_side() -> None:
    if state.interpreter:
        summary = Summary(lines={"fr": ["Interprète demandé : oui"], "ar": ["المريض يطلب مترجم"]}, flags=[])
    else:
        summary = build_summary(flow, consultation.answers)
    html(ui.note_html(summary))
    st.write("")
    st.button("🧏 Interprète / مترجم", key="ask_interpreter", on_click=ask_interpreter,
              disabled=state.interpreter, use_container_width=True)
    st.button("Terminer et effacer les réponses", key="end_consultation", on_click=start_over,
              use_container_width=True)


patient_column, doctor_column = st.columns([2.1, 1], gap="large")
with patient_column:
    patient_side()
with doctor_column:
    doctor_side()
html('<p class="sb-foot">Les signes médicaux reconnus sont en langue des signes espagnole (jeu de données SWL-LSE), '
     "pas en langue des signes tunisienne. Cet outil recueille des réponses ; il n'établit pas de diagnostic. "
     "La caméra lit la position des mains ; aucune image n'est enregistrée ni envoyée.</p>")
