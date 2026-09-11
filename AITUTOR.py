import io
import os
import re
import tempfile
import time

import pandas as pd
import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
from fpdf import FPDF

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    YOUTUBE_TRANSCRIPT_AVAILABLE = True
except ImportError:
    YOUTUBE_TRANSCRIPT_AVAILABLE = False


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

    # NOTE: this must be a model that supports video input for the
    # "Upload Video File" transcription path to work (gemini-1.5/2.0/2.5
    # flash and pro models all do). Check https://ai.google.dev/models
    # for the current list if you change this.
    model = genai.GenerativeModel(
        "gemini-2.0-flash"
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


def is_gemini_error(response_text: str) -> bool:
    """
    True if ask_gemini() returned an error/placeholder string rather than
    real generated content. Used to skip offering a PDF download for
    something that isn't actually study material.
    """

    if not response_text:
        return True

    return response_text.startswith((
        "Gemini API Error:",
        "Gemini model is not configured.",
        "Gemini did not return a response.",
    ))


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
# VIDEO LECTURE HELPERS
# ============================================================
#
# Two ways a video lecture gets turned into text the model can reason
# over:
#   1. YouTube link  -> youtube-transcript-api pulls the real captions.
#   2. Uploaded file  -> the file is sent to Gemini's File API and the
#      model itself is asked to produce a timestamped transcript.
#
# Either path produces the same shape of data: a list of "segments"
# (each with a start time, text, and a short label for the section
# picker UI) plus a flattened `full_text` version.

_YOUTUBE_ID_RE = re.compile(
    r"(?:youtu\.be/|youtube\.com/(?:watch\?v=|embed/|shorts/))([A-Za-z0-9_-]{11})"
)


def extract_youtube_video_id(url: str):
    """Pulls the 11-character video ID out of a YouTube URL, or None."""

    if not url:
        return None

    match = _YOUTUBE_ID_RE.search(url.strip())
    return match.group(1) if match else None


def _format_timestamp(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 60:02d}:{total % 60:02d}"


def _build_full_text(segments):
    return "\n".join(
        f"[{_format_timestamp(seg['start'])}] {seg['text']}" for seg in segments
    )


def fetch_youtube_transcript(video_id: str):
    """
    Returns (segments, full_text, error) for a YouTube video, grouping
    raw caption lines into ~45 second chunks so the section picker
    shows meaningful pieces instead of every individual caption line.
    """

    if not YOUTUBE_TRANSCRIPT_AVAILABLE:
        return [], "", (
            "The `youtube-transcript-api` package is not installed. "
            "Add it to requirements.txt to enable YouTube transcripts."
        )

    try:
        raw = YouTubeTranscriptApi.get_transcript(video_id)

    except Exception as e:
        return [], "", f"Could not fetch transcript: {str(e)}"

    chunk_seconds = 45
    segments = []
    current_start = None
    current_text = []

    for entry in raw:

        start = entry["start"]

        if current_start is None:
            current_start = start

        if start - current_start > chunk_seconds and current_text:
            segments.append({"start": current_start, "text": " ".join(current_text).strip()})
            current_start = start
            current_text = []

        current_text.append(entry["text"])

    if current_text:
        segments.append({"start": current_start, "text": " ".join(current_text).strip()})

    for seg in segments:
        preview = seg["text"][:60] + ("…" if len(seg["text"]) > 60 else "")
        seg["label"] = f"[{_format_timestamp(seg['start'])}] {preview}"

    return segments, _build_full_text(segments), None


def upload_and_transcribe_video(uploaded_file):
    """
    Uploads a locally-uploaded video file to Gemini's File API and asks
    the model to produce a timestamped transcript. Returns
    (segments, full_text, error).
    """

    if model is None:
        return [], "", "Gemini model is not configured."

    tmp_path = None

    try:
        suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        video_ref = genai.upload_file(path=tmp_path)

        # Gemini needs a little time to finish processing video files.
        while getattr(video_ref, "state", None) and video_ref.state.name == "PROCESSING":
            time.sleep(2)
            video_ref = genai.get_file(video_ref.name)

        if getattr(video_ref, "state", None) and video_ref.state.name == "FAILED":
            return [], "", "Gemini could not process this video file."

        prompt = (
            "Transcribe this lecture video. Break the transcript into "
            "sections of roughly 30-60 seconds. Return ONLY plain text, "
            "one section per line, each formatted exactly as:\n"
            "[MM:SS] section text here\n"
            "Do not add commentary, headings, or markdown."
        )

        response = model.generate_content([video_ref, prompt])
        raw_text = response.text if response and response.text else ""

        segments = []

        for line in raw_text.strip().split("\n"):

            m = re.match(r"^\[(\d{1,2}):(\d{2})\]\s*(.*)$", line.strip())

            if not m:
                continue

            minutes, secs, text = m.groups()
            start = int(minutes) * 60 + int(secs)
            preview = text.strip()[:60] + ("…" if len(text.strip()) > 60 else "")

            segments.append({
                "start": start,
                "text": text.strip(),
                "label": f"[{_format_timestamp(start)}] {preview}",
            })

        full_text = _build_full_text(segments) if segments else raw_text

        return segments, full_text, None

    except Exception as e:
        return [], "", f"Video transcription error: {str(e)}"

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


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


def _safe_multi_cell(pdf: FPDF, w: float, h: int, text: str, max_attempts: int = 8):
    """
    Last-resort safety net: if multi_cell still raises an exception for
    some edge case not covered above, progressively halve the line
    instead of letting the whole PDF export crash. In the worst case a
    stubborn line gets truncated — the download always succeeds.
    """

    remaining = text

    for _ in range(max_attempts):

        try:
            pdf.multi_cell(w, h, remaining)
            return

        except Exception:
            if len(remaining) <= 1:
                return
            remaining = remaining[: max(1, len(remaining) // 2)]

    # Final fallback: try a single blank line rather than propagate.
    try:
        pdf.multi_cell(w, h, "")
    except Exception:
        pass


def _write_cell(pdf: FPDF, text: str, line_height: int):
    """
    Renders `text`, pre-wrapped to fit the page width for the currently
    active font, one guaranteed-to-fit line per multi_cell call.

    Three defensive measures beyond the width-measurement wrap itself:
      1. A safety buffer subtracted from the wrap width, to account for
         multi_cell's internal cell padding (c_margin, ~1mm per side)
         which get_string_width() does not know about. Without this,
         a line built to exactly fill the page width can be a hair too
         wide once multi_cell reserves its own internal padding.
      2. pdf.set_x() forced back to the left margin before every single
         line, so we never depend on multi_cell's (version-dependent)
         behavior of resetting the cursor after a previous call.
      3. An explicit non-zero cell width passed to multi_cell (instead
         of the "0 = auto to right margin" shorthand), which internally
         is computed from the CURRENT x position on some fpdf2 versions
         — meaning it can silently shrink if x drifted. Passing the
         width explicitly removes that dependency entirely.
      4. _safe_multi_cell as a hard backstop, so no remaining edge case
         can crash the export outright.
    """

    content_width = pdf.w - pdf.l_margin - pdf.r_margin

    c_margin = getattr(pdf, "c_margin", 1.0) or 1.0
    safety_buffer = max(2 * c_margin + 1.0, 3.0)  # generous, cheap insurance
    usable_width = max(content_width - safety_buffer, content_width * 0.5)

    for line in _wrap_by_measured_width(pdf, text, usable_width):
        pdf.set_x(pdf.l_margin)
        _safe_multi_cell(pdf, content_width, line_height, line)


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
    "lesson_response": "",
    "assignment_response": "",
    "questions_response": "",
    "teaching_material_response": "",
    "student_learning_response": "",
    "student_doubt_response": "",
    "notes_response": "",
    "mcqs_response": "",
    "quiz_questions": [],
    "quiz_answers": {},
    "quiz_score": None,
    "student_analysis_response": "",
    "uploaded_student_data": None,
    "pdf_qa_response": "",
    "video_transcript": "",
    "video_segments": [],
    "video_title": "",
    "video_error": "",
    "video_material_response": "",
    "video_qa_response": "",
    "video_notes_response": "",
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
    ["Easy", "Medium", "Hard", "Mixed"]
)

education_level = st.sidebar.selectbox(
    "Education Level",
    ["School", "Diploma", "Undergraduate", "Postgraduate"]
)

response_language = st.sidebar.selectbox(
    "Response Language",
    ["English", "Hindi", "Bengali"]
)

st.sidebar.info(
    "Teacher Mode provides teaching, assessment and student-analysis tools. "
    "Student Mode provides learning, doubt-solving and practice tools."
)


# ============================================================
# COMMON HELPER
# ============================================================

def show_generated_content(
    state_key: str,
    heading: str,
    title: str,
    filename: str,
    pdf_key: str
):
    content = st.session_state.get(state_key, "")

    if content:
        st.subheader(heading)
        st.markdown(content)

        if is_gemini_error(content):
            st.caption(
                "⚠️ This looks like an API error, not generated content — "
                "no PDF export offered."
            )
        else:
            render_pdf_download_button(
                content,
                title=title,
                filename=filename,
                key=pdf_key,
            )


# ============================================================
# SHARED VIDEO LECTURE UI HELPER
# ============================================================

def render_video_source_picker(key_prefix: str):
    """
    Renders the "how do I get a video in here" UI (YouTube link or file
    upload) and, once processed, stores the transcript/segments in
    session_state so any tab (teacher or student) can use them.
    """

    source_choice = st.radio(
        "Video Source",
        ["🔗 YouTube Link", "📁 Upload Video File"],
        horizontal=True,
        key=f"{key_prefix}_video_source_choice"
    )

    if source_choice == "🔗 YouTube Link":

        youtube_url = st.text_input(
            "YouTube video URL",
            placeholder="https://www.youtube.com/watch?v=...",
            key=f"{key_prefix}_youtube_url"
        )

        if st.button(
            "📥 Fetch Transcript",
            key=f"{key_prefix}_fetch_youtube",
            use_container_width=True
        ):

            video_id = extract_youtube_video_id(youtube_url)

            if not video_id:
                st.warning("Please enter a valid YouTube video URL.")
            else:
                with st.spinner("Fetching transcript..."):
                    segments, full_text, error = fetch_youtube_transcript(video_id)

                if error:
                    st.session_state.video_error = error
                    st.session_state.video_transcript = ""
                    st.session_state.video_segments = []
                else:
                    st.session_state.video_transcript = full_text
                    st.session_state.video_segments = segments
                    st.session_state.video_title = youtube_url
                    st.session_state.video_error = ""
                    st.rerun()

        if youtube_url and extract_youtube_video_id(youtube_url):
            st.video(youtube_url)

    else:

        uploaded_video = st.file_uploader(
            "Upload a lecture video",
            type=["mp4", "mov", "avi", "mkv", "webm"],
            key=f"{key_prefix}_video_upload"
        )

        if uploaded_video:

            st.video(uploaded_video)

            if st.button(
                "🤖 Transcribe with Gemini",
                key=f"{key_prefix}_transcribe_video",
                use_container_width=True
            ):

                with st.spinner(
                    "Uploading and transcribing video — this can take a "
                    "minute for longer lectures..."
                ):
                    segments, full_text, error = upload_and_transcribe_video(uploaded_video)

                if error:
                    st.session_state.video_error = error
                    st.session_state.video_transcript = ""
                    st.session_state.video_segments = []
                else:
                    st.session_state.video_transcript = full_text
                    st.session_state.video_segments = segments
                    st.session_state.video_title = uploaded_video.name
                    st.session_state.video_error = ""
                    st.rerun()

    if st.session_state.video_error:
        st.error(st.session_state.video_error)

    if st.session_state.video_transcript:

        st.success(
            f"✅ Transcript ready for **{st.session_state.video_title}** "
            f"({len(st.session_state.video_segments)} sections)"
        )

        with st.expander("📜 View full transcript"):
            st.text(st.session_state.video_transcript)


def get_video_content_for_prompt(key_prefix: str, mode: str) -> str:
    """
    Returns the text to feed into a generation prompt, according to the
    chosen content-usage mode:
      - "Full transcript"   -> everything
      - "Selected sections" -> only the sections the user checked
      - "Reference only"    -> just the title/topic, no transcript body
    """

    if mode == "Reference only (just the video title/topic)":
        return f"(Video lecture titled/linked: {st.session_state.video_title} — " \
               f"use general knowledge of this topic, full transcript not provided.)"

    if mode == "Selected sections only" and st.session_state.video_segments:

        chosen = st.session_state.get(f"{key_prefix}_selected_segments", [])
        segments_by_label = {
            seg["label"]: seg for seg in st.session_state.video_segments
        }
        chosen_segments = [segments_by_label[label] for label in chosen if label in segments_by_label]

        if chosen_segments:
            return _build_full_text(chosen_segments)

    return st.session_state.video_transcript


# ============================================================
# TEACHER MODE
# ============================================================

if user_mode == "👨‍🏫 Teacher":

    teacher_tabs = st.tabs([
        "📘 Create Lesson",
        "📝 Question Paper",
        "📚 Assignments",
        "👥 Analyze Students",
        "🎨 Teaching Material",
        "📄 PDF Assistant",
        "🎥 Video Lecture",
    ])

    # --------------------------------------------------------
    # TEACHER — CREATE LESSON
    # --------------------------------------------------------

    with teacher_tabs[0]:

        st.header("📘 Create a Lesson Plan")

        col1, col2 = st.columns(2)

        with col1:
            lesson_subject = st.text_input(
                "Subject",
                placeholder="Example: Machine Learning",
                key="lesson_subject"
            )

        with col2:
            lesson_topic = st.text_input(
                "Topic",
                placeholder="Example: Decision Trees",
                key="lesson_topic"
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            lesson_duration = st.selectbox(
                "Class Duration",
                ["30 minutes", "45 minutes", "60 minutes", "90 minutes"],
                key="lesson_duration"
            )

        with col2:
            lesson_level = st.selectbox(
                "Student Level",
                ["Beginner", "Intermediate", "Advanced"],
                key="lesson_level"
            )

        with col3:
            lesson_style = st.selectbox(
                "Teaching Style",
                [
                    "Lecture",
                    "Interactive",
                    "Activity Based",
                    "Discussion Based",
                    "Mixed"
                ],
                key="lesson_style"
            )

        lesson_objectives = st.text_area(
            "Learning Objectives",
            placeholder=(
                "Example: Students should understand the concept, "
                "build a simple model and explain its applications."
            ),
            key="lesson_objectives"
        )

        if st.button(
            "📘 Create Lesson Plan",
            key="create_lesson",
            use_container_width=True
        ):

            if not lesson_topic.strip():
                st.warning("Please enter a lesson topic.")
            else:

                prompt = f"""
You are an expert university teacher and lesson-plan designer.

Create a complete lesson plan.

Subject:
{lesson_subject}

Topic:
{lesson_topic}

Class Duration:
{lesson_duration}

Student Level:
{lesson_level}

Teaching Style:
{lesson_style}

Learning Objectives:
{lesson_objectives}

Education Level:
{education_level}

Difficulty:
{difficulty}

Language:
{response_language}

Include:

1. Lesson title
2. Learning objectives
3. Prerequisites
4. Introduction / warm-up
5. Key concepts to teach
6. Step-by-step teaching sequence
7. Examples
8. Real-world application
9. Interactive classroom activity
10. Questions the teacher can ask during class
11. Common student mistakes
12. Quick assessment
13. Homework
14. Summary
15. Estimated time for each section

Make the plan practical and classroom-ready.
"""

                with st.spinner("Creating lesson plan..."):
                    st.session_state.lesson_response = ask_gemini(prompt)

        show_generated_content(
            "lesson_response",
            "📘 Generated Lesson Plan",
            f"Lesson Plan — {st.session_state.get('lesson_topic', 'Lesson')}",
            "lesson_plan.pdf",
            "pdf_download_lesson",
        )

    # --------------------------------------------------------
    # TEACHER — QUESTION PAPER
    # --------------------------------------------------------

    with teacher_tabs[1]:

        st.header("📝 Question Paper Generator")

        q_subject = st.text_input(
            "Subject",
            key="teacher_q_subject"
        )

        q_topics = st.text_area(
            "Topics / Units",
            placeholder=(
                "Unit 1: Introduction\n"
                "Unit 2: Supervised Learning\n"
                "Unit 3: Classification"
            ),
            key="teacher_q_topics"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            number_questions = st.number_input(
                "Number of Questions",
                min_value=1,
                max_value=50,
                value=10,
                key="teacher_number_questions"
            )

        with col2:
            marks = st.selectbox(
                "Marks per Question",
                [1, 2, 5, 10, 15],
                key="teacher_marks"
            )

        with col3:
            question_type = st.selectbox(
                "Question Type",
                [
                    "Short Answer",
                    "Long Answer",
                    "Very Long Answer",
                    "Viva Questions",
                    "Mixed"
                ],
                key="teacher_question_type"
            )

        include_answers = st.checkbox(
            "Include answer key / marking hints",
            value=False,
            key="teacher_include_answers"
        )

        if st.button(
            "📝 Generate Question Paper",
            key="generate_teacher_questions",
            use_container_width=True
        ):

            if not q_topics.strip():
                st.warning("Please enter topics or units.")
            else:

                answer_part = (
                    "After the question paper, provide an answer key "
                    "or concise marking hints."
                    if include_answers
                    else
                    "Do not provide answers unless necessary."
                )

                prompt = f"""
You are an expert university examination-paper designer.

Create a professional question paper.

Subject:
{q_subject}

Topics / Units:
{q_topics}

Number of Questions:
{number_questions}

Marks per Question:
{marks}

Question Type:
{question_type}

Difficulty:
{difficulty}

Education Level:
{education_level}

Language:
{response_language}

Requirements:

1. Cover the supplied topics.
2. Avoid duplicate questions.
3. Balance conceptual and application-based questions.
4. Match the selected difficulty.
5. Number every question.
6. Make the paper suitable for the selected education level.
7. Keep marks consistent.
8. Format it like a real examination paper.

{answer_part}
"""

                with st.spinner("Generating question paper..."):
                    st.session_state.questions_response = ask_gemini(prompt)

        show_generated_content(
            "questions_response",
            "📝 Generated Question Paper",
            f"Question Paper — {st.session_state.get('teacher_q_subject', 'Exam')}",
            "question_paper.pdf",
            "pdf_download_teacher_questions",
        )

    # --------------------------------------------------------
    # TEACHER — ASSIGNMENTS
    # --------------------------------------------------------

    with teacher_tabs[2]:

        st.header("📚 Assignment Generator")

        col1, col2 = st.columns(2)

        with col1:
            assignment_subject = st.text_input(
                "Subject",
                placeholder="Example: Artificial Intelligence",
                key="assignment_subject"
            )

        with col2:
            assignment_topic = st.text_input(
                "Topic",
                placeholder="Example: Neural Networks",
                key="assignment_topic"
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            assignment_type = st.selectbox(
                "Assignment Type",
                [
                    "Theory",
                    "Programming",
                    "Case Study",
                    "Research",
                    "Practical",
                    "Mixed"
                ],
                key="assignment_type"
            )

        with col2:
            assignment_count = st.number_input(
                "Number of Tasks",
                min_value=1,
                max_value=30,
                value=5,
                key="assignment_count"
            )

        with col3:
            assignment_time = st.selectbox(
                "Expected Completion",
                ["1 hour", "2 hours", "1 day", "3 days", "1 week"],
                key="assignment_time"
            )

        assignment_requirements = st.text_area(
            "Special Requirements",
            placeholder="Example: Include one real-world case study and one coding task.",
            key="assignment_requirements"
        )

        if st.button(
            "📚 Generate Assignment",
            key="generate_assignment",
            use_container_width=True
        ):

            if not assignment_topic.strip():
                st.warning("Please enter an assignment topic.")
            else:

                prompt = f"""
You are an expert college teacher.

Create a complete student assignment.

Subject:
{assignment_subject}

Topic:
{assignment_topic}

Assignment Type:
{assignment_type}

Number of Tasks:
{assignment_count}

Expected Completion:
{assignment_time}

Difficulty:
{difficulty}

Education Level:
{education_level}

Language:
{response_language}

Special Requirements:
{assignment_requirements}

Include:

1. Assignment title
2. Background / introduction
3. Learning objectives
4. Instructions for students
5. Clearly numbered tasks
6. Mix of conceptual and practical work where appropriate
7. Expected submission format
8. Evaluation / marking rubric
9. Academic integrity reminder
10. Expected learning outcomes

Make it realistic and classroom-ready.
"""

                with st.spinner("Generating assignment..."):
                    st.session_state.assignment_response = ask_gemini(prompt)

        show_generated_content(
            "assignment_response",
            "📚 Generated Assignment",
            f"Assignment — {st.session_state.get('assignment_topic', 'Assignment')}",
            "assignment.pdf",
            "pdf_download_assignment",
        )

    # --------------------------------------------------------
    # TEACHER — ANALYZE STUDENTS
    # --------------------------------------------------------

    with teacher_tabs[3]:

        st.header("👥 Student Performance Analysis")

        st.write(
            "Upload a CSV containing student records. The assistant can "
            "identify risk patterns and generate student-specific suggestions."
        )

        student_file = st.file_uploader(
            "Upload Student CSV",
            type=["csv"],
            key="student_csv"
        )

        if student_file:

            try:
                import pandas as pd

                df = pd.read_csv(student_file)
                st.session_state.uploaded_student_data = df

                st.success(
                    f"Student data loaded successfully: {len(df)} records."
                )

                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )

                student_id_column = None

                for column in df.columns:
                    if str(column).strip().lower() in [
                        "student id",
                        "student_id",
                        "studentid"
                    ]:
                        student_id_column = column
                        break

                if student_id_column:

                    student_ids = df[student_id_column].astype(str).tolist()

                    selected_id = st.selectbox(
                        "Select Student ID",
                        student_ids,
                        key="selected_student_id"
                    )

                    selected_row = df[
                        df[student_id_column].astype(str) == str(selected_id)
                    ]

                    if not selected_row.empty:

                        student = selected_row.iloc[0]

                        st.subheader("📊 Selected Student")

                        st.dataframe(
                            selected_row,
                            use_container_width=True,
                            hide_index=True
                        )

                        attendance = None
                        risk_value = None

                        for column in df.columns:
                            normalized = str(column).strip().lower()

                            if normalized == "attendance_percent":
                                attendance = student[column]

                            if normalized == "at_risk":
                                risk_value = student[column]

                        low_attendance = False

                        if attendance is not None:
                            try:
                                low_attendance = float(attendance) < 75
                            except Exception:
                                low_attendance = False

                        at_risk = (
                            str(risk_value).strip().lower()
                            in ["yes", "true", "1"]
                            if risk_value is not None
                            else False
                        )

                        if low_attendance or at_risk:

                            st.warning(
                                "⚠️ This student requires attention: "
                                "Attendance < 75% or At_Risk is Yes."
                            )

                            if st.button(
                                "🤖 Generate Student-Specific Suggestions",
                                key="generate_student_suggestions",
                                use_container_width=True
                            ):

                                student_details = "\n".join(
                                    f"{column}: {student[column]}"
                                    for column in df.columns
                                )

                                prompt = f"""
You are an academic student-support assistant.

Analyze ONLY the selected student's information.

Student record:
----------------
{student_details}
----------------

Attendance below 75%:
{low_attendance}

At-risk flag:
{at_risk}

Education level:
{education_level}

Language:
{response_language}

Generate practical, personalized academic suggestions.

Include:

1. Main areas of concern
2. Possible academic impact
3. Attendance improvement suggestions if relevant
4. Study-time suggestions if relevant
5. Exam preparation suggestions
6. Subject-learning suggestions based on the available fields
7. Short-term action plan
8. Weekly improvement plan
9. Encouraging conclusion

Do not give generic advice unrelated to the student's actual record.
Do not invent missing student information.
"""

                                with st.spinner(
                                    "Analyzing selected student..."
                                ):
                                    st.session_state.student_analysis_response = ask_gemini(
                                        prompt
                                    )

                        else:

                            st.success(
                                "✅ This student does not currently meet "
                                "the configured Gemini-support condition."
                            )

                        show_generated_content(
                            "student_analysis_response",
                            "🤖 Personalized Student Suggestions",
                            f"Student Analysis — {selected_id}",
                            "student_analysis.pdf",
                            "pdf_download_student_analysis",
                        )

                    else:
                        st.warning("Selected student could not be found.")

                else:
                    st.error(
                        "The CSV needs a Student ID column "
                        "(for example: Student ID)."
                    )

            except Exception as e:
                st.error(f"Could not read the student CSV: {e}")

        st.markdown("---")
        st.subheader("📈 Class-Level Risk Overview")

        df = st.session_state.get("uploaded_student_data")

        if df is not None:

            try:

                attendance_column = next(
                    (
                        c for c in df.columns
                        if str(c).strip().lower() == "attendance_percent"
                    ),
                    None
                )

                risk_column = next(
                    (
                        c for c in df.columns
                        if str(c).strip().lower() == "at_risk"
                    ),
                    None
                )

                if attendance_column or risk_column:

                    risk_mask = False

                    if attendance_column:
                        numeric_attendance = pd.to_numeric(
                            df[attendance_column],
                            errors="coerce"
                        )
                        risk_mask = numeric_attendance < 75

                    if risk_column:
                        flag_mask = (
                            df[risk_column]
                            .astype(str)
                            .str.strip()
                            .str.lower()
                            .isin(["yes", "true", "1"])
                        )

                        risk_mask = risk_mask | flag_mask

                    risk_count = int(risk_mask.sum())
                    total_count = len(df)

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Total Students", total_count)

                    with col2:
                        st.metric("Students Requiring Attention", risk_count)

                    with col3:
                        percentage = (
                            round((risk_count / total_count) * 100, 1)
                            if total_count else 0
                        )
                        st.metric("Attention Rate", f"{percentage}%")

                else:
                    st.info(
                        "Class-level risk overview needs "
                        "Attendance_Percent and/or At_Risk."
                    )

            except Exception as e:
                st.warning(f"Could not calculate class risk overview: {e}")

    # --------------------------------------------------------
    # TEACHER — TEACHING MATERIAL
    # --------------------------------------------------------

    with teacher_tabs[4]:

        st.header("🎨 Teaching Material Generator")

        col1, col2 = st.columns(2)

        with col1:
            material_subject = st.text_input(
                "Subject",
                key="material_subject",
                placeholder="Example: Data Science"
            )

        with col2:
            material_topic = st.text_input(
                "Topic",
                key="material_topic",
                placeholder="Example: Data Preprocessing"
            )

        material_type = st.selectbox(
            "Material Type",
            [
                "Lecture Notes",
                "Class Handout",
                "Presentation Outline",
                "Blackboard / Whiteboard Plan",
                "Classroom Activity",
                "Case Study",
                "Revision Sheet",
                "Lab Exercise",
                "Viva Preparation Material"
            ],
            key="material_type"
        )

        material_length = st.selectbox(
            "Material Length",
            ["Short", "Medium", "Detailed"],
            key="material_length"
        )

        material_extra = st.text_area(
            "Additional Instructions",
            placeholder="Example: Include examples suitable for BTech students.",
            key="material_extra"
        )

        use_video_for_material = False

        if st.session_state.video_transcript:

            use_video_for_material = st.checkbox(
                f"🎥 Base this on the extracted video lecture "
                f"(\"{st.session_state.video_title}\")",
                key="use_video_for_material"
            )

        else:
            st.caption(
                "ℹ️ Extract a transcript in the **🎥 Video Lecture** tab first "
                "to generate material that follows an actual lecture."
            )

        if st.button(
            "🎨 Generate Teaching Material",
            key="generate_teaching_material",
            use_container_width=True
        ):

            if not material_topic.strip() and not use_video_for_material:
                st.warning("Please enter a topic (or use a video lecture as the source).")
            else:

                video_block = ""

                if use_video_for_material:
                    video_block = f"""
This material MUST follow the actual lecture below — cover the same
concepts, in the same order, and do not introduce content the lecture
does not cover.

Video Lecture Transcript:
----------------
{st.session_state.video_transcript[:50000]}
----------------
"""

                prompt = f"""
You are an expert teacher and academic-content designer.

Create {material_length.lower()} teaching material.

Subject:
{material_subject}

Topic:
{material_topic}

Material Type:
{material_type}

Education Level:
{education_level}

Difficulty:
{difficulty}

Language:
{response_language}

Additional Instructions:
{material_extra}
{video_block}
Make the material classroom-ready.

Where appropriate include:
- Learning objectives
- Key definitions
- Important concepts
- Examples
- Diagrams described in words
- Real-world applications
- Classroom activities
- Common misconceptions
- Important examination points
- Summary

For a presentation outline, organize the output slide-by-slide.
For a lab exercise, include objective, requirements, procedure and expected output.
For a case study, include scenario, questions and expected learning outcomes.
"""

                with st.spinner("Generating teaching material..."):
                    st.session_state.teaching_material_response = ask_gemini(
                        prompt
                    )

        show_generated_content(
            "teaching_material_response",
            "🎨 Generated Teaching Material",
            f"Teaching Material — {st.session_state.get('material_topic', 'Topic')}",
            "teaching_material.pdf",
            "pdf_download_teaching_material",
        )

    # --------------------------------------------------------
    # TEACHER — PDF ASSISTANT
    # --------------------------------------------------------

    with teacher_tabs[5]:

        st.header("📄 Teacher PDF Assistant")

        uploaded_pdf = st.file_uploader(
            "Upload a teaching material PDF",
            type=["pdf"],
            key="teacher_pdf_upload"
        )

        if uploaded_pdf:

            if st.button(
                "📖 Read PDF",
                key="teacher_read_pdf",
                use_container_width=True
            ):

                with st.spinner("Reading PDF..."):
                    st.session_state.pdf_text = extract_pdf_text(
                        uploaded_pdf
                    )

                if st.session_state.pdf_text.strip():

                    st.success("PDF successfully processed.")

                    st.info(
                        f"Extracted approximately "
                        f"{len(st.session_state.pdf_text)} characters."
                    )

                else:
                    st.error("Could not extract text from this PDF.")

        if st.session_state.pdf_text:

            pdf_question = st.text_area(
                "Ask something about the teaching material",
                placeholder="Example: Create five discussion questions from Unit 1.",
                height=130,
                key="teacher_pdf_question"
            )

            if st.button(
                "🤖 Ask About PDF",
                key="teacher_pdf_ask",
                use_container_width=True
            ):

                if not pdf_question.strip():
                    st.warning("Please enter a question.")
                else:

                    material = st.session_state.pdf_text[:50000]

                    prompt = f"""
You are an AI Teaching Assistant helping a teacher.

Use the supplied teaching material to answer the request.

Teaching Material:
----------------
{material}
----------------

Teacher Request:
{pdf_question}

Education Level:
{education_level}

Language:
{response_language}

Instructions:

1. Base the response primarily on the supplied material.
2. Do not invent information.
3. Clearly say when the material does not contain the answer.
4. Make the result practical for teaching.
5. Use headings and bullet points where appropriate.
"""

                    with st.spinner("Analyzing teaching material..."):
                        st.session_state.pdf_qa_response = ask_gemini(prompt)

        show_generated_content(
            "pdf_qa_response",
            "🤖 AI Answer",
            "Teacher PDF Assistant — Answer",
            "teacher_pdf_answer.pdf",
            "pdf_download_teacher_pdf",
        )

    # --------------------------------------------------------
    # TEACHER — VIDEO LECTURE
    # --------------------------------------------------------

    with teacher_tabs[6]:

        st.header("🎥 Turn a Video Lecture Into Teaching Material")

        st.caption(
            "Paste a YouTube link or upload a lecture recording, extract "
            "its transcript, then generate material that actually follows "
            "what was taught."
        )

        render_video_source_picker(key_prefix="teacher")

        if st.session_state.video_segments:

            with st.expander("✂️ Pick specific sections (optional)"):
                st.multiselect(
                    "Only these sections will be used when "
                    "'Selected sections only' is chosen below",
                    options=[seg["label"] for seg in st.session_state.video_segments],
                    key="teacher_selected_segments"
                )

        if st.session_state.video_transcript:

            st.divider()

            col1, col2 = st.columns(2)

            with col1:
                video_material_subject = st.text_input(
                    "Subject",
                    placeholder="Example: Machine Learning",
                    key="video_material_subject"
                )

            with col2:
                video_material_type = st.selectbox(
                    "Material Type",
                    [
                        "Lecture Notes",
                        "Presentation Outline",
                        "Handout",
                        "Revision Sheet",
                        "Quiz Prep Summary",
                        "Case Study",
                    ],
                    key="video_material_type"
                )

            video_content_mode = st.selectbox(
                "How should the video be used?",
                [
                    "Full transcript",
                    "Selected sections only",
                    "Reference only (just the video title/topic)",
                ],
                key="video_content_mode",
                help=(
                    "Full transcript: material mirrors the whole video. "
                    "Selected sections only: uses just what you picked "
                    "above. Reference only: uses the video as a topic "
                    "pointer without feeding the transcript text."
                )
            )

            video_material_extra = st.text_area(
                "Additional Instructions",
                placeholder="Example: Add 3 practice questions per section.",
                key="video_material_extra"
            )

            if st.button(
                "🎨 Generate Material From Video",
                key="generate_video_material",
                use_container_width=True
            ):

                video_content = get_video_content_for_prompt(
                    "teacher", video_content_mode
                )

                prompt = f"""
You are an expert teacher and academic-content designer.

Create classroom-ready {video_material_type.lower()} based on the video
lecture content below. Follow the same concepts and order as the
source content — do not invent material the source does not cover
(unless the mode is "reference only", in which case use the topic as
a general guide).

Subject:
{video_material_subject}

Video Lecture Content ({video_content_mode}):
----------------
{video_content[:50000]}
----------------

Education Level:
{education_level}

Difficulty:
{difficulty}

Language:
{response_language}

Additional Instructions:
{video_material_extra}

Where appropriate include:
- Learning objectives
- Key definitions and concepts (matching what the lecture covered)
- Examples used in the lecture (or equivalent ones)
- A short summary
- Suggested practice questions
"""

                with st.spinner("Generating material from the video..."):
                    st.session_state.video_material_response = ask_gemini(prompt)

        show_generated_content(
            "video_material_response",
            "🎨 Generated Teaching Material",
            f"Teaching Material — {st.session_state.get('video_title', 'Video Lecture')}",
            "video_teaching_material.pdf",
            "pdf_download_video_material",
        )


# ============================================================
# STUDENT MODE
# ============================================================

else:

    student_tabs = st.tabs([
        "🧠 Learn Topic",
        "❓ Ask Doubts",
        "📚 Generate Notes",
        "📝 Practice MCQs",
        "🎯 Take Quiz",
        "📄 PDF Assistant",
        "🎥 Video Lecture",
    ])

    # --------------------------------------------------------
    # STUDENT — LEARN TOPIC
    # --------------------------------------------------------

    with student_tabs[0]:

        st.header("🧠 Learn a Topic")

        col1, col2 = st.columns(2)

        with col1:
            learn_subject = st.text_input(
                "Subject",
                placeholder="Example: Machine Learning",
                key="learn_subject"
            )

        with col2:
            learn_topic = st.text_input(
                "Topic",
                placeholder="Example: Random Forest",
                key="learn_topic"
            )

        learning_goal = st.selectbox(
            "What do you want to do?",
            [
                "Understand the basics",
                "Prepare for an exam",
                "Understand with examples",
                "Learn step by step",
                "Revise quickly"
            ],
            key="learning_goal"
        )

        if st.button(
            "🧠 Start Learning",
            key="start_learning",
            use_container_width=True
        ):

            if not learn_topic.strip():
                st.warning("Please enter a topic.")
            else:

                prompt = f"""
You are a friendly AI tutor.

Teach the student the following topic.

Subject:
{learn_subject}

Topic:
{learn_topic}

Student Goal:
{learning_goal}

Education Level:
{education_level}

Difficulty:
{difficulty}

Language:
{response_language}

Teach from basic to advanced only as appropriate.

Include:

1. Simple introduction
2. Definition
3. Why the topic matters
4. Step-by-step explanation
5. Simple example
6. Real-world example
7. Important terms
8. Common mistakes
9. Exam-important points
10. Quick recap
11. Three self-check questions

Use clear student-friendly language.
"""

                with st.spinner("Preparing your lesson..."):
                    st.session_state.student_learning_response = ask_gemini(
                        prompt
                    )

        show_generated_content(
            "student_learning_response",
            "🧠 Your Lesson",
            f"Learning — {st.session_state.get('learn_topic', 'Topic')}",
            "student_learning.pdf",
            "pdf_download_student_learning",
        )

    # --------------------------------------------------------
    # STUDENT — ASK DOUBTS
    # --------------------------------------------------------

    with student_tabs[1]:

        st.header("❓ Ask Your Doubt")

        doubt_subject = st.text_input(
            "Subject",
            key="doubt_subject",
            placeholder="Example: Computer Networks"
        )

        doubt_topic = st.text_input(
            "Topic",
            key="doubt_topic",
            placeholder="Example: TCP vs UDP"
        )

        doubt = st.text_area(
            "Describe your doubt",
            placeholder=(
                "Example: I understand TCP is reliable, but why is UDP "
                "used for streaming?"
            ),
            height=170,
            key="student_doubt"
        )

        if st.button(
            "🤖 Explain My Doubt",
            key="explain_doubt",
            use_container_width=True
        ):

            if not doubt.strip():
                st.warning("Please describe your doubt.")
            else:

                prompt = f"""
You are a patient personal AI tutor.

A student has asked the following doubt.

Subject:
{doubt_subject}

Topic:
{doubt_topic}

Student's Doubt:
{doubt}

Education Level:
{education_level}

Difficulty:
{difficulty}

Language:
{response_language}

Answer the exact doubt.

Instructions:

1. First identify what the student is confused about.
2. Explain it step by step.
3. Use a simple analogy if useful.
4. Give an example.
5. Correct any misconception gently.
6. End with a short takeaway.
7. Do not overwhelm the student with unrelated theory.
"""

                with st.spinner("Solving your doubt..."):
                    st.session_state.student_doubt_response = ask_gemini(
                        prompt
                    )

        show_generated_content(
            "student_doubt_response",
            "🤖 Explanation",
            "Student Doubt — Explanation",
            "doubt_explanation.pdf",
            "pdf_download_student_doubt",
        )

    # --------------------------------------------------------
    # STUDENT — NOTES
    # --------------------------------------------------------

    with student_tabs[2]:

        st.header("📚 Generate Study Notes")

        notes_subject = st.text_input(
            "Subject",
            placeholder="Example: Artificial Intelligence",
            key="student_notes_subject"
        )

        notes_topic = st.text_input(
            "Topic",
            placeholder="Example: Neural Networks",
            key="student_notes_topic"
        )

        notes_length = st.selectbox(
            "Notes Length",
            ["Short", "Medium", "Detailed"],
            key="student_notes_length"
        )

        notes_purpose = st.selectbox(
            "Purpose",
            [
                "General Study",
                "Exam Preparation",
                "Quick Revision",
                "Viva Preparation"
            ],
            key="student_notes_purpose"
        )

        if st.button(
            "📚 Generate Notes",
            key="generate_student_notes",
            use_container_width=True
        ):

            if not notes_topic.strip():
                st.warning("Please enter a topic.")
            else:

                prompt = f"""
You are an expert student study assistant.

Create {notes_length.lower()} study notes.

Subject:
{notes_subject}

Topic:
{notes_topic}

Purpose:
{notes_purpose}

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
4. Clear explanation
5. Examples
6. Advantages
7. Disadvantages
8. Applications
9. Important examination points
10. Quick revision summary
11. Five key terms

Use student-friendly formatting.
"""

                with st.spinner("Generating study notes..."):
                    st.session_state.notes_response = ask_gemini(prompt)

        show_generated_content(
            "notes_response",
            "📖 Your Study Notes",
            f"Notes — {st.session_state.get('student_notes_topic', 'Study Notes')}",
            "study_notes.pdf",
            "pdf_download_student_notes",
        )

    # --------------------------------------------------------
    # STUDENT — PRACTICE MCQs
    # --------------------------------------------------------

    with student_tabs[3]:

        st.header("📝 Practice MCQs")

        col1, col2 = st.columns(2)

        with col1:
            practice_subject = st.text_input(
                "Subject",
                key="practice_subject",
                placeholder="Example: Python"
            )

        with col2:
            practice_topic = st.text_input(
                "Topic",
                key="practice_topic",
                placeholder="Example: OOP"
            )

        practice_count = st.number_input(
            "Number of MCQs",
            min_value=1,
            max_value=20,
            value=5,
            key="practice_count"
        )

        if st.button(
            "📝 Generate Practice Questions",
            key="generate_practice_mcqs",
            use_container_width=True
        ):

            if not practice_topic.strip():
                st.warning("Please enter a topic.")
            else:

                prompt = f"""
Generate {practice_count} multiple-choice practice questions.

Subject:
{practice_subject}

Topic:
{practice_topic}

Education Level:
{education_level}

Difficulty:
{difficulty}

Language:
{response_language}

Return ONLY valid JSON in this exact structure:

[
  {{
    "question": "Question text",
    "options": [
      "Option A",
      "Option B",
      "Option C",
      "Option D"
    ],
    "answer": 0,
    "explanation": "Short explanation"
  }}
]

Rules:
- answer is the zero-based index of the correct option.
- Exactly four options per question.
- Exactly one correct answer.
- No duplicate questions.
- Questions should test understanding.
"""

                with st.spinner("Creating practice MCQs..."):

                    raw = ask_gemini(prompt)

                    try:
                        import json

                        cleaned = raw.strip()

                        if cleaned.startswith("```"):
                            cleaned = cleaned.replace("```json", "", 1)
                            cleaned = cleaned.replace("```", "", 1).strip()

                        parsed = json.loads(cleaned)

                        if isinstance(parsed, list):
                            st.session_state.quiz_questions = parsed
                            st.session_state.quiz_answers = {}
                            st.session_state.quiz_score = None
                        else:
                            st.error("Gemini returned an unexpected MCQ format.")

                    except Exception:
                        st.error(
                            "Could not parse the MCQs automatically. "
                            "Please generate them again."
                        )
                        st.code(raw)

        if st.session_state.quiz_questions:

            st.subheader("📝 Practice")

            for index, item in enumerate(
                st.session_state.quiz_questions
            ):

                question_text = item.get(
                    "question",
                    f"Question {index + 1}"
                )

                options = item.get("options", [])

                if len(options) != 4:
                    continue

                answer = st.radio(
                    f"{index + 1}. {question_text}",
                    options,
                    key=f"practice_answer_{index}"
                )

                st.session_state.quiz_answers[index] = answer

                if st.checkbox(
                    "Show explanation",
                    key=f"show_practice_explanation_{index}"
                ):
                    st.info(
                        item.get("explanation", "No explanation provided.")
                    )

    # --------------------------------------------------------
    # STUDENT — TAKE QUIZ
    # --------------------------------------------------------

    with student_tabs[4]:

        st.header("🎯 Take a Quiz")

        quiz_subject = st.text_input(
            "Subject",
            key="quiz_subject",
            placeholder="Example: Data Structures"
        )

        quiz_topic = st.text_input(
            "Topic",
            key="quiz_topic",
            placeholder="Example: Binary Trees"
        )

        quiz_count = st.number_input(
            "Number of Questions",
            min_value=1,
            max_value=15,
            value=5,
            key="quiz_count"
        )

        quiz_time = st.selectbox(
            "Quiz Difficulty",
            ["Easy", "Medium", "Hard", "Mixed"],
            key="quiz_difficulty"
        )

        if st.button(
            "🎯 Start New Quiz",
            key="start_new_quiz",
            use_container_width=True
        ):

            if not quiz_topic.strip():
                st.warning("Please enter a quiz topic.")
            else:

                prompt = f"""
Create {quiz_count} quiz questions for a student.

Subject:
{quiz_subject}

Topic:
{quiz_topic}

Difficulty:
{quiz_time}

Education Level:
{education_level}

Language:
{response_language}

Return ONLY valid JSON:

[
  {{
    "question": "Question",
    "options": ["A", "B", "C", "D"],
    "answer": 0,
    "explanation": "Why this is correct"
  }}
]

Rules:
- answer must be the zero-based correct option index.
- Exactly four options.
- Exactly one correct answer.
- No duplicate questions.
"""

                with st.spinner("Building your quiz..."):

                    raw = ask_gemini(prompt)

                    try:
                        import json

                        cleaned = raw.strip()

                        if cleaned.startswith("```"):
                            cleaned = cleaned.replace("```json", "", 1)
                            cleaned = cleaned.replace("```", "", 1).strip()

                        parsed = json.loads(cleaned)

                        if isinstance(parsed, list) and parsed:

                            st.session_state.quiz_questions = parsed
                            st.session_state.quiz_answers = {}
                            st.session_state.quiz_score = None

                            st.rerun()

                        else:
                            st.error("Invalid quiz format returned by Gemini.")

                    except Exception:
                        st.error(
                            "Could not create the interactive quiz. "
                            "Please try again."
                        )
                        st.code(raw)

        if st.session_state.quiz_questions:

            st.markdown("### 🎯 Answer all questions")

            for index, item in enumerate(
                st.session_state.quiz_questions
            ):

                options = item.get("options", [])

                if len(options) != 4:
                    continue

                selected = st.radio(
                    f"{index + 1}. {item.get('question', '')}",
                    options,
                    key=f"quiz_answer_{index}"
                )

                st.session_state.quiz_answers[index] = selected

            if st.button(
                "✅ Submit Quiz",
                key="submit_quiz",
                use_container_width=True
            ):

                score = 0

                for index, item in enumerate(
                    st.session_state.quiz_questions
                ):

                    options = item.get("options", [])
                    correct_index = item.get("answer")

                    if (
                        index in st.session_state.quiz_answers
                        and isinstance(correct_index, int)
                        and 0 <= correct_index < len(options)
                    ):

                        if (
                            st.session_state.quiz_answers[index]
                            == options[correct_index]
                        ):
                            score += 1

                st.session_state.quiz_score = score

            if st.session_state.quiz_score is not None:

                total = len(st.session_state.quiz_questions)
                score = st.session_state.quiz_score

                percentage = round(
                    (score / total) * 100,
                    1
                ) if total else 0

                st.success(
                    f"🎉 Your Score: {score}/{total} "
                    f"({percentage}%)"
                )

                for index, item in enumerate(
                    st.session_state.quiz_questions
                ):

                    options = item.get("options", [])
                    correct_index = item.get("answer")

                    if (
                        isinstance(correct_index, int)
                        and 0 <= correct_index < len(options)
                    ):

                        correct_option = options[correct_index]

                        if (
                            st.session_state.quiz_answers.get(index)
                            == correct_option
                        ):
                            st.success(
                                f"Q{index + 1}: Correct"
                            )
                        else:
                            st.error(
                                f"Q{index + 1}: Incorrect. "
                                f"Correct answer: {correct_option}"
                            )

                        st.caption(
                            item.get(
                                "explanation",
                                "No explanation provided."
                            )
                        )

    # --------------------------------------------------------
    # STUDENT — PDF ASSISTANT
    # --------------------------------------------------------

    with student_tabs[5]:

        st.header("📄 Ask Questions From Your PDF")

        uploaded_pdf = st.file_uploader(
            "Upload a study material PDF",
            type=["pdf"],
            key="student_pdf_upload"
        )

        if uploaded_pdf:

            if st.button(
                "📖 Read PDF",
                key="student_read_pdf",
                use_container_width=True
            ):

                with st.spinner("Reading PDF..."):

                    st.session_state.pdf_text = extract_pdf_text(
                        uploaded_pdf
                    )

                if st.session_state.pdf_text.strip():

                    st.success("PDF successfully processed.")

                    st.info(
                        f"Extracted approximately "
                        f"{len(st.session_state.pdf_text)} characters."
                    )

                else:
                    st.error("Could not extract text from this PDF.")

        if st.session_state.pdf_text:

            pdf_question = st.text_area(
                "Ask something about your study material",
                placeholder=(
                    "Example: Explain the main concepts "
                    "covered in Unit 1."
                ),
                height=130,
                key="student_pdf_question"
            )

            if st.button(
                "🤖 Ask About PDF",
                key="student_pdf_ask",
                use_container_width=True
            ):

                if not pdf_question.strip():
                    st.warning("Please enter a question.")
                else:

                    material = st.session_state.pdf_text[:50000]

                    prompt = f"""
You are an AI student tutor.

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
                        st.session_state.pdf_qa_response = ask_gemini(
                            prompt
                        )

        show_generated_content(
            "pdf_qa_response",
            "🤖 AI Answer",
            "Student PDF Assistant — Answer",
            "student_pdf_answer.pdf",
            "pdf_download_student_pdf",
        )

    # --------------------------------------------------------
    # STUDENT — VIDEO LECTURE
    # --------------------------------------------------------

    with student_tabs[6]:

        st.header("🎥 Learn From a Video Lecture")

        st.caption(
            "Paste a YouTube link or upload a lecture recording, then ask "
            "questions or generate notes straight from what was taught."
        )

        render_video_source_picker(key_prefix="student")

        if st.session_state.video_segments:

            with st.expander("✂️ Focus on specific sections (optional)"):
                st.multiselect(
                    "Only these sections will be used when "
                    "'Selected sections only' is chosen below",
                    options=[seg["label"] for seg in st.session_state.video_segments],
                    key="student_selected_segments"
                )

        if st.session_state.video_transcript:

            st.divider()

            video_use_mode = st.selectbox(
                "Use which part of the video?",
                ["Full transcript", "Selected sections only"],
                key="student_video_use_mode"
            )

            video_action = st.radio(
                "What do you want to do?",
                ["Ask a question", "Generate notes"],
                horizontal=True,
                key="student_video_action"
            )

            if video_action == "Ask a question":

                video_question = st.text_area(
                    "Ask something about this video lecture",
                    placeholder="Example: Explain the second concept covered in the video.",
                    height=130,
                    key="student_video_question"
                )

                if st.button(
                    "🤖 Ask About Video",
                    key="student_video_ask",
                    use_container_width=True
                ):

                    if not video_question.strip():
                        st.warning("Please enter a question.")
                    else:

                        video_content = get_video_content_for_prompt(
                            "student", video_use_mode
                        )

                        prompt = f"""
You are an AI student tutor.

Answer the student's question using the video lecture transcript
provided below.

Video Lecture Transcript:
----------------
{video_content[:50000]}
----------------

Student Question:
{video_question}

Difficulty:
{difficulty}

Education Level:
{education_level}

Language:
{response_language}

Instructions:

1. Base the answer primarily on the video transcript.
2. Reference the approximate timestamp when helpful.
3. Do not invent information.
4. If the answer cannot be found in the transcript, clearly say so.
5. Use examples when useful.
"""

                        with st.spinner("Analyzing the video lecture..."):
                            st.session_state.video_qa_response = ask_gemini(prompt)

                show_generated_content(
                    "video_qa_response",
                    "🤖 AI Answer",
                    "Video Lecture — Answer",
                    "video_lecture_answer.pdf",
                    "pdf_download_video_qa",
                )

            else:

                if st.button(
                    "📚 Generate Notes From Video",
                    key="student_video_notes",
                    use_container_width=True
                ):

                    video_content = get_video_content_for_prompt(
                        "student", video_use_mode
                    )

                    prompt = f"""
You are an AI study-notes generator.

Create clear, well-organized study notes from the video lecture
transcript below.

Video Lecture Transcript:
----------------
{video_content[:50000]}
----------------

Difficulty:
{difficulty}

Education Level:
{education_level}

Language:
{response_language}

Instructions:

1. Organize notes with headings and bullet points.
2. Follow the order the lecture covered topics in.
3. Include a short summary at the end.
4. Do not invent information not present in the transcript.
"""

                    with st.spinner("Generating notes from the video..."):
                        st.session_state.video_notes_response = ask_gemini(prompt)

                show_generated_content(
                    "video_notes_response",
                    "📚 Notes From Video",
                    f"Video Notes — {st.session_state.get('video_title', 'Lecture')}",
                    "video_notes.pdf",
                    "pdf_download_video_notes",
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🎓 AI Teaching Assistant | Teacher Mode + Student Mode | "
    "Powered by Gemini + Streamlit"
)
