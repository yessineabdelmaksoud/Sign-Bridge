from pathlib import Path

from consultation import ui
from consultation.flow import Flow
from consultation.summary import Summary, build_summary

F = Flow.load()


def test_sign_shows_arabic_right_to_left_above_french():
    html = ui.sign_html(F.question("reason"))
    assert 'dir="rtl"' in html and "علاش جيت اليوم؟" in html and "Pourquoi venez-vous" in html
    assert html.index("علاش") < html.index("Pourquoi")


def test_text_from_the_flow_file_is_escaped():
    summary = Summary(lines={"fr": ["Motif : <script>x</script>"], "ar": ["a"]}, flags=[])
    assert "<script>" not in ui.note_html(summary) and "&lt;script&gt;" in ui.note_html(summary)


def test_tile_label_is_pictogram_then_arabic_then_french():
    assert ui.tile_label(F, "DOLOR") == "😣\nوجيعة\nDouleur"


def test_value_display_for_each_kind_of_answer():
    assert ui.value_display(F, F.question("consent"), False) == ("❌", "Non", "لا")
    assert ui.value_display(F, F.question("symptom"), "TOSER") == ("😷", "Toux", "كحة")
    icon, fr, ar = ui.value_display(F, F.question("intensity"), 9)
    assert (fr, ar) == ("9/10", "9/10") and icon


def test_note_lists_confirmed_lines_and_puts_urgency_first():
    answers = {"consent": True, "reason": "URGENTE", "symptom": "RESPIRAR"}
    html = ui.note_html(build_summary(F, answers))
    assert "Motif" in html and "Urgent" in html
    assert html.index("Difficulté respiratoire") < html.index("Motif")


def test_empty_note_tells_the_doctor_what_will_appear():
    assert "apparaissent ici" in ui.note_html(build_summary(F, {}))


def test_note_as_plain_text_has_both_languages_and_flags():
    answers = {"consent": True, "reason": "URGENTE"}
    text = ui.note_text(build_summary(F, answers))
    assert "Motif : Urgent" in text and "السبب : حالة مستعجلة" in text and "Le patient signale une urgence" in text


def test_stylesheet_uses_no_remote_assets():
    assert "url(" not in ui.CSS and "@import" not in ui.CSS
