import io
import os

import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
from fpdf import FPDF


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Teaching Assistant",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.card {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(128,128,128,0.3);
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

def configure_gemini():

    try:
        api_key = st.secrets["GEMINI_API_KEY"]

    except Exception:
        st.error(
            "Gemini API key not found. "
            "Please add GEMINI_API_KEY to Streamlit secrets."
        )
        return None

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        "gemini-3.7-flash"
    )

    return model


model = configure_gemini()


# ============================================================
# GEMINI HELPER FUNCTION
# ============================================================

def ask_gemini(prompt):

    if model is None:
        return "Gemini model is not configured."

    try:

        response = model.generate_content(prompt)

        if response and response.text:
            return response.text

        return "Gemini did not return a response."

    except Exception as e:

        return f"Gemini API Error: {str(e)}"


# ============================================================
# PDF TEXT EXTRACTION (for the PDF Assistant tab — reading a PDF IN)
# ============================================================

def extract_pdf_text(uploaded_file):

    try:

        reader = PdfReader(uploaded_file)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return text

    except Exception as e:

        return f"PDF extraction error: {str(e)}"


# ============================================================
# PDF EXPORT (for turning generated answers INTO a downloadable PDF)
# ============================================================

# Optional Unicode fonts. If you want proper Hindi / Bengali rendering in the
# exported PDFs (not just on screen), download these free Google Noto fonts
# and place them in a "fonts/" folder next to this script:
#
#   fonts/NotoSans-Regular.ttf        (Latin / English)
#   fonts/NotoSans-Bold.ttf
#   fonts/NotoSansDevanagari-Regular.ttf   (Hindi)
#   fonts/NotoSansBengali-Regular.ttf      (Bengali)
#
# Without these files, PDF export still works, but non-Latin characters
# (Hindi/Bengali) will be replaced with "?" in the exported PDF, since the
# built-in PDF core fonts only support Latin-1 text. The on-screen
# st.markdown() output is NOT affected — this limitation is PDF-only.

FONTS_DIR = os.path.join(os.path.dirname(__file__), "fonts")

_UNICODE_FONT_CANDIDATES = [
    ("NotoSansDevanagari", "NotoSansDevanagari-Regular.ttf"),
    ("NotoSansBengali", "NotoSansBengali-Regular.ttf"),
    ("NotoSans", "NotoSans-Regular.ttf"),
]


def _register_unicode_fonts(pdf: FPDF):
    """
    Try to register any bundled Unicode TTF fonts. Returns the name of the
    'body' font to use, and whether any Unicode font was actually found.
    """

    registered_any = False
    body_font_name = "Helvetica"  # fpdf2 core font fallback (Latin-1 only)

    for font_name, filename in _UNICODE_FONT_CANDIDATES:

        font_path = os.path.join(FONTS_DIR, filename)

        if os.path.exists(font_path):

            try:
                pdf.add_font(font_name, "", font_path)
                registered_any = True
                body_font_name = font_name

            except Exception:
                pass

    return body_font_name, registered_any


def _safe_text(text: str, unicode_ok: bool) -> str:
    """
    If we don't have a Unicode font loaded, strip characters the core PDF
    fonts can't render so fpdf2 doesn't crash on Hindi/Bengali text.
    """

    if unicode_ok:
        return text

    return text.encode("latin-1", "replace").decode("latin-1")


import re

_RULE_LINE_RE = re.compile(r"^[\s\-_*=]{3,}$")          # e.g. "---", "___", "***"
_TABLE_SEP_LINE_RE = re.compile(r"^[\s|:\-]{3,}$")       # e.g. "|---|:---:|---|"


def _sanitize_markdown_for_pdf(content: str) -> str:
    """
    Strips Markdown constructs that add visual noise to a PDF export:
    table separator rows (|---|---|), horizontal rules (---, ***), and
    raw table pipes (turned into readable "a | b | c" text). Does NOT
    need to worry about line length/wrapping — that's handled precisely
    in _write_cell() below via actual font-width measurement.
    """

    cleaned_lines = []

    for line in content.split("\n"):

        stripped = line.strip()

        # Drop markdown table separator rows entirely (they carry no content).
        if _TABLE_SEP_LINE_RE.match(stripped) and "-" in stripped:
            continue

        # Collapse pure horizontal-rule lines to a blank line.
        if _RULE_LINE_RE.match(stripped):
            cleaned_lines.append("")
            continue

        # Turn "| a | b | c |" into "a | b | c" so pipes don't glue
        # to neighboring words.
        if stripped.startswith("|") and stripped.endswith("|"):
            stripped = stripped.strip("|")
            stripped = " | ".join(
                part.strip() for part in stripped.split("|")
            )

        cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines)


def _wrap_by_measured_width(pdf: FPDF, text: str, max_width: float) -> list:
    """
    Wraps `text` into lines that are each verified, via fpdf2's own
    get_string_width() for the CURRENTLY ACTIVE font, to fit within
    max_width. This deliberately bypasses fpdf2's internal multi_cell
    word-wrap algorithm, which has known edge cases — including on
    long unbroken tokens (URLs, hashes, table artifacts) and, on some
    fpdf2 versions, even on wrapmode='CHAR' itself — that raise
    "Not enough horizontal space to render a single character" instead
    of wrapping. By only ever handing multi_cell a string that already
    fits on one line, that whole class of error is avoided regardless
    of which fpdf2 version is installed.
    """

    if not text:
        return [""]

    words = text.split(" ")
    lines = []
    current = ""

    def fits(candidate: str) -> bool:
        return pdf.get_string_width(candidate) <= max_width

    for word in words:

        # A single "word" wider than the whole line (long URL, hash,
        # run-on punctuation) gets sliced into pieces that each fit,
        # via binary search on the measured width.
        while word and not fits(word):

            lo, hi, fit_len = 1, len(word), 1

            while lo <= hi:
                mid = (lo + hi) // 2
                if fits(word[:mid]):
                    fit_len = mid
                    lo = mid + 1
                else:
                    hi = mid - 1

            piece, word = word[:fit_len], word[fit_len:]

            if current:
                lines.append(current)
                current = ""

            lines.append(piece)

        candidate = f"{current} {word}".strip() if current else word

        if fits(candidate):
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines or [""]


def _write_cell(pdf: FPDF, text: str, line_height: int):
    """
    Renders `text`, pre-wrapped to fit the page width for the currently
    active font, one guaranteed-to-fit line per multi_cell call.
    """

    max_width = pdf.w - pdf.l_margin - pdf.r_margin

    for line in _wrap_by_measured_width(pdf, text, max_width):
        pdf.multi_cell(0, line_height, line)


def generate_pdf_bytes(content: str, title: str) -> bytes:
    """
    Converts a markdown-ish text response (from Gemini) into a simple,
    readably-formatted PDF. Handles #/##/### headings and -/* bullet points;
    everything else is treated as a normal paragraph.
    """

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    body_font, unicode_ok = _register_unicode_fonts(pdf)

    heading_font = body_font if unicode_ok else "Helvetica"

    # Title
    pdf.set_font(heading_font, size=16)
    _write_cell(pdf, _safe_text(title, unicode_ok), 10)
    pdf.ln(2)

    pdf.set_font(body_font if unicode_ok else "Helvetica", size=12)

    content = _sanitize_markdown_for_pdf(content)

    for raw_line in content.split("\n"):

        line = raw_line.strip()

        if not line:
            pdf.ln(3)
            continue

        if line.startswith("### "):
            pdf.set_font(heading_font, size=13)
            _write_cell(pdf, _safe_text(line[4:], unicode_ok), 8)
            pdf.set_font(body_font if unicode_ok else "Helvetica", size=12)

        elif line.startswith("## "):
            pdf.set_font(heading_font, size=14)
            _write_cell(pdf, _safe_text(line[3:], unicode_ok), 9)
            pdf.set_font(body_font if unicode_ok else "Helvetica", size=12)

        elif line.startswith("# "):
            pdf.set_font(heading_font, size=16)
            _write_cell(pdf, _safe_text(line[2:], unicode_ok), 10)
            pdf.set_font(body_font if unicode_ok else "Helvetica", size=12)

        elif line.startswith(("- ", "* ")):
            bullet_text = "  •  " + line[2:].replace("**", "")
            _write_cell(pdf, _safe_text(bullet_text, unicode_ok), 7)

        else:
            clean = line.replace("**", "").replace("__", "")
            _write_cell(pdf, _safe_text(clean, unicode_ok), 7)

    output = pdf.output(dest="S")

    # fpdf2 versions differ on str vs bytearray return type
    if isinstance(output, (bytes, bytearray)):
        return bytes(output)

    return output.encode("latin-1")


def render_pdf_download_button(content: str, title: str, filename: str, key: str):
    """
    Renders a 'Download as PDF' button for a piece of generated content.
    Call this right after st.markdown(content) in any tab.
    """

    if not content or not content.strip():
        return

    try:
        pdf_bytes = generate_pdf_bytes(content, title)

    except Exception as e:
        st.warning(f"Could not prepare PDF for download: {e}")
        return

    st.download_button(
        label="📄 Download as PDF",
        data=pdf_bytes,
        file_name=filename,
        mime="application/pdf",
        key=key,
        use_container_width=True,
    )

    if not os.path.isdir(FONTS_DIR) or not os.listdir(FONTS_DIR):
        st.caption(
            "ℹ️ Add Unicode fonts to a `fonts/` folder for proper "
            "Hindi/Bengali PDF export — see comments in app.py."
        )


# ============================================================
# SESSION STATE
# ============================================================

_DEFAULT_STATE = {
    "pdf_text": "",
    "teacher_response": "",
    "notes_response": "",
    "questions_response": "",
    "mcqs_response": "",
    "pdf_qa_response": "",
}

for _key, _default in _DEFAULT_STATE.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎓 AI Teaching Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'An intelligent teaching and learning assistant powered by Gemini'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Assistant Settings")

user_mode = st.sidebar.selectbox(
    "Select Mode",
    [
        "👨‍🏫 Teacher",
        "👨‍🎓 Student"
    ]
)

difficulty = st.sidebar.selectbox(
    "Difficulty Level",
    [
        "Easy",
        "Medium",
        "Hard",
        "Mixed"
    ]
)

education_level = st.sidebar.selectbox(
    "Education Level",
    [
        "School",
        "Diploma",
        "Undergraduate",
        "Postgraduate"
    ]
)

response_language = st.sidebar.selectbox(
    "Response Language",
    [
        "English",
        "Hindi",
        "Bengali"
    ]
)


# ============================================================
# MAIN TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "💬 AI Teacher",
        "📚 Notes",
        "❓ Questions",
        "📝 MCQs",
        "📄 PDF Assistant"
    ]
)


# ============================================================
# TAB 1 — AI TEACHER
# ============================================================

with tab1:

    st.header("💬 Ask the AI Teacher")

    col1, col2 = st.columns(2)

    with col1:

        subject = st.text_input(
            "Subject",
            placeholder="Example: Machine Learning"
        )

    with col2:

        topic = st.text_input(
            "Topic",
            placeholder="Example: Random Forest"
        )

    question = st.text_area(
        "What do you want to learn?",
        placeholder=(
            "Example: Explain Random Forest with a simple example."
        ),
        height=150
    )

    if st.button(
        "🤖 Ask Gemini",
        key="ask_teacher",
        use_container_width=True
    ):

        if not question.strip():

            st.warning("Please enter your question.")

        else:

            prompt = f"""
You are an expert AI Teaching Assistant.

Education level:
{education_level}

Subject:
{subject}

Topic:
{topic}

Difficulty:
{difficulty}

Response language:
{response_language}

Student/teacher mode:
{user_mode}

Question:
{question}

Instructions:

1. Explain the concept clearly.
2. Use simple language where appropriate.
3. Give examples.
4. Use headings and bullet points.
5. Include important points.
6. If useful, include a real-world example.
7. Do not provide irrelevant information.
"""

            with st.spinner("Gemini is preparing the answer..."):

                answer = ask_gemini(prompt)

            st.session_state.teacher_response = answer

    if st.session_state.teacher_response:

        st.subheader("🤖 AI Teacher Response")

        st.markdown(st.session_state.teacher_response)

        render_pdf_download_button(
            st.session_state.teacher_response,
            title=f"AI Teacher — {subject or 'Response'}",
            filename="ai_teacher_response.pdf",
            key="pdf_download_teacher",
        )


# ============================================================
# TAB 2 — NOTES GENERATOR
# ============================================================

with tab2:

    st.header("📚 AI Notes Generator")

    notes_subject = st.text_input(
        "Subject",
        placeholder="Example: Artificial Intelligence",
        key="notes_subject"
    )

    notes_topic = st.text_input(
        "Topic",
        placeholder="Example: Neural Networks",
        key="notes_topic"
    )

    notes_length = st.selectbox(
        "Notes Length",
        [
            "Short",
            "Medium",
            "Detailed"
        ]
    )

    if st.button(
        "📚 Generate Notes",
        key="generate_notes",
        use_container_width=True
    ):

        if not notes_topic.strip():

            st.warning("Please enter a topic.")

        else:

            prompt = f"""
You are an expert college teaching assistant.

Create {notes_length.lower()} study notes.

Subject:
{notes_subject}

Topic:
{notes_topic}

Education Level:
{education_level}

Difficulty:
{difficulty}

Language:
{response_language}

Include:

1. Introduction
2. Definition
3. Important concepts
4. Detailed explanation
5. Examples
6. Advantages
7. Disadvantages
8. Applications
9. Important points for examinations
10. Short conclusion

Format the answer using clear headings and bullet points.
"""

            with st.spinner("Generating notes..."):

                notes = ask_gemini(prompt)

            st.session_state.notes_response = notes

    if st.session_state.notes_response:

        st.subheader("📖 Generated Notes")

        st.markdown(st.session_state.notes_response)

        render_pdf_download_button(
            st.session_state.notes_response,
            title=f"Notes — {notes_topic or 'Study Notes'}",
            filename="study_notes.pdf",
            key="pdf_download_notes",
        )


# ============================================================
# TAB 3 — QUESTION GENERATOR
# ============================================================

with tab3:

    st.header("❓ Question Paper Generator")

    q_subject = st.text_input(
        "Subject",
        key="q_subject"
    )

    q_topics = st.text_area(
        "Topics / Units",
        placeholder=(
            "Unit 1: Introduction\n"
            "Unit 2: Supervised Learning\n"
            "Unit 3: Classification"
        )
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        number_questions = st.number_input(
            "Number of Questions",
            min_value=1,
            max_value=50,
            value=10
        )

    with col2:

        marks = st.selectbox(
            "Marks per Question",
            [
                1,
                2,
                5,
                10,
                15
            ]
        )

    with col3:

        question_type = st.selectbox(
            "Question Type",
            [
                "Short Answer",
                "Long Answer",
                "Very Long Answer",
                "Viva Questions"
            ]
        )

    if st.button(
        "❓ Generate Questions",
        key="generate_questions",
        use_container_width=True
    ):

        if not q_topics.strip():

            st.warning("Please enter topics.")

        else:

            prompt = f"""
You are an expert university question-paper designer.

Generate {number_questions} questions.

Subject:
{q_subject}

Topics:
{q_topics}

Question type:
{question_type}

Marks per question:
{marks}

Difficulty:
{difficulty}

Education level:
{education_level}

Language:
{response_language}

Requirements:

1. Questions must be related to the provided topics.
2. Avoid duplicate questions.
3. Match the selected difficulty.
4. Questions should be academically meaningful.
5. Include a mixture of conceptual and application-based
   questions where appropriate.
6. Number every question.
"""

            with st.spinner("Generating questions..."):

                questions = ask_gemini(prompt)

            st.session_state.questions_response = questions

    if st.session_state.questions_response:

        st.subheader("📝 Generated Questions")

        st.markdown(st.session_state.questions_response)

        render_pdf_download_button(
            st.session_state.questions_response,
            title=f"Question Paper — {q_subject or 'Exam'}",
            filename="question_paper.pdf",
            key="pdf_download_questions",
        )


# ============================================================
# TAB 4 — MCQ GENERATOR
# ============================================================

with tab4:

    st.header("📝 AI MCQ Generator")

    mcq_subject = st.text_input(
        "Subject",
        key="mcq_subject"
    )

    mcq_topic = st.text_input(
        "Topic",
        key="mcq_topic"
    )

    number_mcqs = st.number_input(
        "Number of MCQs",
        min_value=1,
        max_value=50,
        value=10
    )

    show_answers = st.checkbox(
        "Show Answers",
        value=True
    )

    if st.button(
        "🧠 Generate MCQs",
        key="generate_mcqs",
        use_container_width=True
    ):

        if not mcq_topic.strip():

            st.warning("Please enter a topic.")

        else:

            answer_instruction = (
                "Include the correct answer and explanation."
                if show_answers
                else
                "Do not reveal the answers."
            )

            prompt = f"""
You are an expert educational assessment assistant.

Generate {number_mcqs} multiple-choice questions.

Subject:
{mcq_subject}

Topic:
{mcq_topic}

Difficulty:
{difficulty}

Education Level:
{education_level}

Language:
{response_language}

Each question must contain:

Question
A. Option
B. Option
C. Option
D. Option

{answer_instruction}

Make sure:

- There is exactly one best answer.
- Questions are factually correct.
- Do not repeat questions.
- Questions should test understanding rather than only memorization.
"""

            with st.spinner("Generating MCQs..."):

                mcqs = ask_gemini(prompt)

            st.session_state.mcqs_response = mcqs

    if st.session_state.mcqs_response:

        st.subheader("🧠 Generated MCQs")

        st.markdown(st.session_state.mcqs_response)

        render_pdf_download_button(
            st.session_state.mcqs_response,
            title=f"MCQs — {mcq_topic or 'Quiz'}",
            filename="mcqs.pdf",
            key="pdf_download_mcqs",
        )


# ============================================================
# TAB 5 — PDF ASSISTANT
# ============================================================

with tab5:

    st.header("📄 Ask Questions From Your PDF")

    uploaded_pdf = st.file_uploader(
        "Upload a study material PDF",
        type=["pdf"]
    )

    if uploaded_pdf:

        if st.button(
            "📖 Read PDF",
            use_container_width=True
        ):

            with st.spinner("Reading PDF..."):

                pdf_text = extract_pdf_text(
                    uploaded_pdf
                )

            st.session_state.pdf_text = pdf_text

            if pdf_text.strip():

                st.success(
                    "PDF successfully processed."
                )

                st.info(
                    f"Extracted approximately "
                    f"{len(pdf_text)} characters."
                )

            else:

                st.error(
                    "Could not extract text from this PDF."
                )

    if st.session_state.pdf_text:

        pdf_question = st.text_area(
            "Ask something about the uploaded material",
            placeholder=(
                "Example: Explain the main concepts "
                "covered in Unit 1."
            ),
            height=130
        )

        if st.button(
            "🤖 Ask About PDF",
            use_container_width=True
        ):

            if not pdf_question.strip():

                st.warning(
                    "Please enter a question."
                )

            else:

                # Limit the amount of text sent in one request.
                material = st.session_state.pdf_text[:50000]

                prompt = f"""
You are an AI Teaching Assistant.

Answer the student's question using the study
material provided below.

Study Material:
----------------
{material}
----------------

Student Question:
{pdf_question}

Difficulty:
{difficulty}

Education Level:
{education_level}

Language:
{response_language}

Instructions:

1. Base the answer primarily on the provided material.
2. Explain clearly.
3. Do not invent information.
4. If the answer cannot be found in the material,
   clearly say so.
5. Use examples when useful.
6. Use headings and bullet points where appropriate.
"""

                with st.spinner(
                    "Analyzing your study material..."
                ):

                    pdf_answer = ask_gemini(prompt)

                st.session_state.pdf_qa_response = pdf_answer

    if st.session_state.pdf_qa_response:

        st.subheader(
            "🤖 AI Answer"
        )

        st.markdown(st.session_state.pdf_qa_response)

        render_pdf_download_button(
            st.session_state.pdf_qa_response,
            title="PDF Assistant — Answer",
            filename="pdf_assistant_answer.pdf",
            key="pdf_download_pdf_qa",
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🎓 AI Teaching Assistant | "
    "Powered by Python, Streamlit and Gemini"
)
