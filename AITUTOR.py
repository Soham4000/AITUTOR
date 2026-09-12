import io
import os

import pandas as pd
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
        "gemini-3.5-flash"
    )

    return model


model = configure_gemini()


# ============================================================
# AGENT MODE — TOOL SCHEMAS
# ============================================================
#
# These are the actions the agent is allowed to choose between. Unlike
# every other tab in this app, the agent decides WHICH of these to call,
# WHAT arguments to pass, WHETHER TO CALL SEVERAL AT ONCE, and WHEN it's
# done — based on a natural-language goal instead of a button click.

AGENT_SYSTEM_INSTRUCTION = """You are an autonomous teaching-prep agent \
embedded in a teacher's classroom-prep tool.

Guidelines:
- Break the teacher's goal into the smallest set of concrete actions
  needed to satisfy it.
- Call multiple independent tools in the SAME turn when their results
  don't depend on each other (for example, practice MCQs and a graded
  quiz can usually be generated together once the topic is known).
  Sequence tools that genuinely depend on each other's output (for
  example, check student data or read lesson material BEFORE deciding
  what a lesson plan should emphasize).
- Only call ask_clarifying_question when the goal is genuinely
  ambiguous or missing information you cannot reasonably infer (for
  example, no subject or topic given at all). Call it ALONE, never
  combined with other tool calls, and only once per genuine gap.
  Prefer acting on a sensible default over asking.
- When every needed artifact has been produced, respond with a concise
  plain-text summary of what you made and why, with no further tool
  calls.
"""

AGENT_TOOLS = [
    {
        "name": "check_student_risk_data",
        "description": (
            "Check the uploaded student CSV for which students are "
            "struggling (low attendance or flagged at_risk). Call this "
            "FIRST when preparing for a class so material can be tailored "
            "to students who need help."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        },
    },
    {
        "name": "analyze_class_performance",
        "description": (
            "Run a deeper statistical pass over the uploaded student CSV: "
            "averages/min/max for every numeric column, and how at-risk "
            "students compare to the class average on each one. Use this "
            "instead of (or in addition to) check_student_risk_data when "
            "you need to understand PATTERNS across the class, not just "
            "who is flagged."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        },
    },
    {
        "name": "read_lesson_material",
        "description": (
            "Read the previously uploaded lesson/teaching PDF (from the "
            "PDF Assistant tab) to see what was already taught, so a "
            "review targets gaps instead of repeating everything."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        },
    },
    {
        "name": "generate_lesson_plan",
        "description": (
            "Create a classroom-ready lesson plan for a topic. Use this "
            "when the class needs fresh instruction or a targeted review."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "topic": {"type": "STRING"},
                "duration": {"type": "STRING"},
                "focus": {
                    "type": "STRING",
                    "description": (
                        "What to emphasize, e.g. 'conceptual review' or "
                        "'practical/hands-on review of weak points'."
                    ),
                },
            },
            "required": ["topic"],
        },
    },
    {
        "name": "generate_mcqs",
        "description": (
            "Generate multiple-choice practice questions on a topic. Use "
            "this when students would benefit from low-stakes self-check "
            "practice before a graded quiz or exam."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "topic": {"type": "STRING"},
                "count": {"type": "INTEGER"},
                "difficulty": {"type": "STRING"},
            },
            "required": ["topic", "count"],
        },
    },
    {
        "name": "generate_quiz",
        "description": (
            "Generate a graded quiz on a topic. Use this as a final "
            "assessment step, typically after practice material has "
            "already been prepared."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "topic": {"type": "STRING"},
                "count": {"type": "INTEGER"},
            },
            "required": ["topic"],
        },
    },
    {
        "name": "generate_assignment",
        "description": (
            "Create a full student assignment (tasks, instructions, "
            "rubric) for a topic. Use this when the goal calls for "
            "homework or take-home practice, not just in-class material."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "subject": {"type": "STRING"},
                "topic": {"type": "STRING"},
                "assignment_type": {
                    "type": "STRING",
                    "description": (
                        "Theory, Programming, Case Study, Research, "
                        "Practical, or Mixed."
                    ),
                },
                "count": {
                    "type": "INTEGER",
                    "description": "Number of tasks in the assignment.",
                },
            },
            "required": ["topic"],
        },
    },
    {
        "name": "generate_question_paper",
        "description": (
            "Create a formal examination-style question paper covering "
            "one or more topics/units. Use this when the goal explicitly "
            "calls for an exam or test paper, distinct from a lightweight "
            "practice quiz."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "subject": {"type": "STRING"},
                "topics": {
                    "type": "STRING",
                    "description": "Topics or units to cover, comma-separated.",
                },
                "count": {"type": "INTEGER"},
                "marks": {"type": "INTEGER"},
            },
            "required": ["topics"],
        },
    },
    {
        "name": "generate_teaching_material",
        "description": (
            "Create supporting teaching material other than a lesson plan "
            "or assessment — lecture notes, a handout, a revision sheet, "
            "a case study, or a lab exercise. Use this when the goal asks "
            "for reference/handout material rather than a structured "
            "lesson plan or a graded item."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "subject": {"type": "STRING"},
                "topic": {"type": "STRING"},
                "material_type": {
                    "type": "STRING",
                    "description": (
                        "Lecture Notes, Class Handout, Revision Sheet, "
                        "Case Study, or Lab Exercise."
                    ),
                },
            },
            "required": ["topic"],
        },
    },
    {
        "name": "ask_clarifying_question",
        "description": (
            "Pause and ask the teacher a direct question when the goal is "
            "genuinely ambiguous or missing information you cannot "
            "reasonably infer (e.g. no subject/topic stated anywhere). "
            "MUST be called alone, never combined with other tool calls "
            "in the same turn. Do not use this for preferences that have "
            "a sensible default — only for real gaps."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "question": {"type": "STRING"},
            },
            "required": ["question"],
        },
    },
]

AGENT_MAX_STEPS = 12  # safety cap so a confused loop can't run forever


def configure_gemini_agent():
    """Separate model instance WITH tools attached. The plain `model`
    above is left untouched so every existing tab keeps working exactly
    as before."""

    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None

    genai.configure(api_key=api_key)

    return genai.GenerativeModel(
        "gemini-3.5-flash",
        tools=AGENT_TOOLS,
        system_instruction=AGENT_SYSTEM_INSTRUCTION,
    )


agent_model = configure_gemini_agent()


# ------------------------------------------------------------
# Self-correcting JSON generation
# ------------------------------------------------------------
#
# Gemini occasionally wraps JSON in prose or code fences, or drops a
# field. Instead of failing the whole agent step, this retries with the
# broken output fed back in as an error correction — the agent's own
# artifacts get a chance to fix themselves before giving up.

def _clean_json_block(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "", 1)
        cleaned = cleaned.replace("```", "", 1).strip()
    return cleaned


def _ask_for_json(base_prompt: str, max_tries: int = 3):
    """Returns (parsed_list, raw_text, error). error is None on success."""

    import json as _json

    prompt = base_prompt
    last_raw = ""

    for attempt in range(1, max_tries + 1):
        raw = ask_gemini(prompt)
        last_raw = raw

        if is_gemini_error(raw):
            return None, raw, raw

        try:
            parsed = _json.loads(_clean_json_block(raw))
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed, raw, None
        except Exception:
            pass

        prompt = (
            base_prompt
            + "\n\nYour previous response was not a valid JSON array matching "
            "the required schema. Return ONLY the JSON array and nothing "
            "else — no prose, no code fences.\n\nYour previous output was:\n"
            + raw[:800]
        )

    return None, last_raw, f"Could not produce valid JSON after {max_tries} tries."


# ------------------------------------------------------------
# Deeper class-performance analysis
# ------------------------------------------------------------

def _find_risk_column(df):
    for col in df.columns:
        if str(col).strip().lower() == "at_risk":
            return col
    return None


def _analyze_class_performance(df) -> str:
    lines = [f"Total students: {len(df)}"]

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if not numeric_cols:
        lines.append("No numeric columns found to analyze.")

    for col in numeric_cols:
        try:
            lines.append(
                f"{col}: class average={df[col].mean():.1f}, "
                f"min={df[col].min()}, max={df[col].max()}"
            )
        except Exception:
            continue

    risk_col = _find_risk_column(df)

    if risk_col and numeric_cols:
        risky = df[df[risk_col].astype(str).str.lower().isin(["yes", "true", "1"])]

        if len(risky):
            for col in numeric_cols:
                try:
                    risky_avg = risky[col].mean()
                    overall_avg = df[col].mean()
                    gap = risky_avg - overall_avg
                    lines.append(
                        f"At-risk students average {col}={risky_avg:.1f} "
                        f"vs class average {overall_avg:.1f} "
                        f"(gap: {gap:+.1f})"
                    )
                except Exception:
                    continue
        else:
            lines.append("No students are currently flagged at-risk.")
    elif not risk_col:
        lines.append("No At_Risk column found, so risk-gap comparison was skipped.")

    return "\n".join(lines)


# ------------------------------------------------------------
# Tool dispatch
# ------------------------------------------------------------

def execute_agent_tool(name, args, ctx):
    """
    Dispatch table: actually runs the tool the agent chose to call.

    ctx carries whatever state the tools need:
        ctx["pdf_text"]        -> extracted lesson PDF text (or "")
        ctx["student_df"]      -> uploaded student DataFrame (or None)
        ctx["education_level"] -> from sidebar
        ctx["language"]        -> from sidebar
        ctx["outputs"]         -> dict this function fills in with results
        ctx["quiz_questions"]  -> parsed quiz JSON, when generate_quiz
                                   succeeds, for interactive rendering

    Returns a short string result that gets fed back to the model so it
    can decide the next step.
    """

    if name == "check_student_risk_data":
        df = ctx.get("student_df")
        if df is None:
            return "No student CSV has been uploaded. Skip risk-based tailoring."

        risk_col = _find_risk_column(df)

        if risk_col is None:
            return "No At_Risk column found in the uploaded CSV. Skip risk-based tailoring."

        risk_rows = df[df[risk_col].astype(str).str.lower().isin(["yes", "true", "1"])]
        id_col = df.columns[0]

        if len(risk_rows):
            return (
                f"{len(risk_rows)} of {len(df)} students are flagged at-risk. "
                f"Struggling students: {risk_rows[id_col].tolist()}"
            )
        return f"No at-risk students found among {len(df)} records."

    if name == "analyze_class_performance":
        df = ctx.get("student_df")
        if df is None:
            return "No student CSV has been uploaded. Skip class-performance analysis."
        return _analyze_class_performance(df)

    if name == "read_lesson_material":
        text = ctx.get("pdf_text", "")
        if not text.strip():
            return "No lesson PDF has been uploaded/read yet in the PDF Assistant tab."
        return f"Lesson material excerpt (first 800 chars): {text[:800]}"

    if name == "generate_lesson_plan":
        topic = args.get("topic", "the topic")
        duration = args.get("duration", "45 minutes")
        focus = args.get("focus", "balanced coverage")
        prompt = f"""You are an expert teacher. Create a {duration} lesson plan
for the topic "{topic}", focused on: {focus}.
Education level: {ctx.get('education_level')}
Language: {ctx.get('language')}
Include objectives, sequence, examples, and a quick assessment."""
        result = ask_gemini(prompt)
        ctx.setdefault("outputs", {})["Lesson Plan"] = result
        return f"Lesson plan generated ({len(result)} chars). Stored as 'Lesson Plan'."

    if name == "generate_mcqs":
        topic = args.get("topic", "the topic")
        count = args.get("count", 5)
        difficulty = args.get("difficulty", "Medium")
        prompt = f"""Generate {count} multiple-choice practice questions on "{topic}"
at {difficulty} difficulty, in {ctx.get('language')}.
Return ONLY valid JSON:
[{{"question": "...", "options": ["A","B","C","D"], "answer": 0, "explanation": "..."}}]"""
        parsed, raw, error = _ask_for_json(prompt)
        if error:
            return f"Failed to generate valid MCQs after retries: {error}"
        ctx.setdefault("outputs", {})["Practice MCQs"] = raw
        ctx["mcq_questions"] = parsed
        return f"{len(parsed)} MCQs generated and validated as JSON. Stored as 'Practice MCQs'."

    if name == "generate_quiz":
        topic = args.get("topic", "the topic")
        count = args.get("count", 5)
        prompt = f"""Generate {count} graded quiz questions on "{topic}", in {ctx.get('language')}.
Return ONLY valid JSON:
[{{"question": "...", "options": ["A","B","C","D"], "answer": 0, "explanation": "..."}}]"""
        parsed, raw, error = _ask_for_json(prompt)
        if error:
            return f"Failed to generate valid quiz after retries: {error}"
        ctx.setdefault("outputs", {})["Graded Quiz"] = raw
        ctx["quiz_questions"] = parsed
        return f"{len(parsed)}-question quiz generated and validated as JSON. Stored as 'Graded Quiz', ready to take interactively."

    if name == "generate_assignment":
        subject = args.get("subject", "")
        topic = args.get("topic", "the topic")
        assignment_type = args.get("assignment_type", "Mixed")
        count = args.get("count", 5)
        prompt = f"""You are an expert college teacher. Create a complete student
assignment.
Subject: {subject}
Topic: {topic}
Assignment Type: {assignment_type}
Number of Tasks: {count}
Education Level: {ctx.get('education_level')}
Language: {ctx.get('language')}
Include: title, background, learning objectives, numbered tasks,
submission format, and a marking rubric."""
        result = ask_gemini(prompt)
        ctx.setdefault("outputs", {})["Assignment"] = result
        return f"Assignment generated ({len(result)} chars). Stored as 'Assignment'."

    if name == "generate_question_paper":
        subject = args.get("subject", "")
        topics = args.get("topics", "")
        count = args.get("count", 10)
        marks = args.get("marks", 5)
        prompt = f"""You are an expert examination-paper designer. Create a
professional question paper.
Subject: {subject}
Topics/Units: {topics}
Number of Questions: {count}
Marks per Question: {marks}
Education Level: {ctx.get('education_level')}
Language: {ctx.get('language')}
Number every question and format it like a real examination paper."""
        result = ask_gemini(prompt)
        ctx.setdefault("outputs", {})["Question Paper"] = result
        return f"Question paper generated ({len(result)} chars). Stored as 'Question Paper'."

    if name == "generate_teaching_material":
        subject = args.get("subject", "")
        topic = args.get("topic", "the topic")
        material_type = args.get("material_type", "Lecture Notes")
        prompt = f"""You are an expert teacher and academic-content designer.
Create {material_type} for the topic "{topic}".
Subject: {subject}
Education Level: {ctx.get('education_level')}
Language: {ctx.get('language')}
Make it classroom-ready with definitions, examples, and a summary."""
        result = ask_gemini(prompt)
        ctx.setdefault("outputs", {})["Teaching Material"] = result
        return f"Teaching material generated ({len(result)} chars). Stored as 'Teaching Material'."

    if name == "ask_clarifying_question":
        # Normally intercepted by the loop before reaching dispatch; this
        # is a defensive fallback only.
        return "Waiting for the teacher's answer."

    return f"Unknown tool: {name}"


# ------------------------------------------------------------
# The agent loop
# ------------------------------------------------------------
#
# Supports two things a single-call-at-a-time loop can't:
#   1. PARALLEL tool calls — if Gemini returns more than one function
#      call in a single turn, all of them are executed and their results
#      are sent back together in one message, instead of one round trip
#      per tool.
#   2. PAUSING for a clarifying question — the loop can hand control back
#      to the teacher mid-run instead of guessing, then resume exactly
#      where it left off once an answer is provided. This is why the
#      loop is split into start/continue instead of one blocking call:
#      Streamlit reruns the whole script on every interaction, so the
#      live `chat` session has to be handed back to the caller to stash
#      in st.session_state rather than looped over internally.

def _extract_function_calls(response):
    candidate = response.candidates[0]
    return [part.function_call for part in candidate.content.parts if part.function_call]


def _final_text_from_response(response):
    candidate = response.candidates[0]
    return "".join(part.text for part in candidate.content.parts if part.text)


def _drive_agent_loop(chat, response, ctx, log, step):

    while step < AGENT_MAX_STEPS:
        step += 1

        calls = _extract_function_calls(response)

        if not calls:
            final_text = _final_text_from_response(response)
            log(f"**Step {step}:** Agent finished — no further actions needed.")
            return {
                "status": "done",
                "chat": chat,
                "final_text": final_text,
                "step": step,
            }

        clarifying = next(
            (c for c in calls if c.name == "ask_clarifying_question"), None
        )

        if clarifying is not None:
            question = dict(clarifying.args).get("question", "Could you clarify the goal?")
            log(f"**Step {step}:** Agent needs clarification: {question}")
            return {
                "status": "needs_input",
                "chat": chat,
                "pending_question": question,
                "pending_call_name": clarifying.name,
                "step": step,
            }

        response_parts = []

        for call in calls:
            args = dict(call.args)
            log(f"**Step {step}:** Agent calls `{call.name}({args})`")

            result = execute_agent_tool(call.name, args, ctx)
            log(f"**Step {step} result ({call.name}):** {result}")

            response_parts.append(
                genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=call.name,
                        response={"result": result},
                    )
                )
            )

        try:
            response = chat.send_message(genai.protos.Content(parts=response_parts))
        except Exception as e:
            log(f"Agent API error: {e}")
            return {"status": "error", "chat": chat, "step": step}

    log("**Stopped:** hit the safety cap on steps.")
    final_text = _final_text_from_response(response)
    return {"status": "done", "chat": chat, "final_text": final_text, "step": step}


def start_agent(goal: str, ctx: dict, log=print):
    """Starts a new agent chat and runs it until it finishes, needs a
    clarifying answer, or errors. Returns a state dict to stash in
    st.session_state."""

    if agent_model is None:
        log("Agent Mode is not available: Gemini API key not configured.")
        return {"status": "error", "chat": None}

    chat = agent_model.start_chat()

    try:
        response = chat.send_message(goal)
    except Exception as e:
        log(f"Agent API error: {e}")
        return {"status": "error", "chat": None}

    return _drive_agent_loop(chat, response, ctx, log, step=0)


def continue_agent_with_answer(chat, pending_call_name: str, answer: str, ctx: dict, log=print, step: int = 0):
    """Resumes a paused agent after the teacher answers a clarifying
    question, continuing from exactly where it left off."""

    try:
        response = chat.send_message(
            genai.protos.Content(
                parts=[genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=pending_call_name,
                        response={"result": answer},
                    )
                )]
            )
        )
    except Exception as e:
        log(f"Agent API error: {e}")
        return {"status": "error", "chat": chat, "step": step}

    return _drive_agent_loop(chat, response, ctx, log, step=step)


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
    "agent_final_summary": "",
    "agent_outputs": {},
    "agent_log": [],
    "agent_chat": None,
    "agent_step": 0,
    "agent_pending_question": "",
    "agent_pending_call_name": "",
    "agent_quiz_questions": [],
    "agent_quiz_answers": {},
    "agent_quiz_score": None,
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
        "🤖 Agent Mode",
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

        if st.button(
            "🎨 Generate Teaching Material",
            key="generate_teaching_material",
            use_container_width=True
        ):

            if not material_topic.strip():
                st.warning("Please enter a topic.")
            else:

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
    # TEACHER — AGENT MODE
    # --------------------------------------------------------

    with teacher_tabs[6]:

        st.header("🤖 Autonomous Course Prep Agent")

        st.write(
            "Describe a goal instead of picking a tool. The agent decides "
            "which actions to use, in what order, whether to run several "
            "at once, and — if the goal is genuinely unclear — it will "
            "ask you instead of guessing."
        )

        with st.expander("What can the agent do on its own?"):
            st.markdown(
                "- Check the uploaded student CSV for at-risk students\n"
                "- Run a deeper class-performance analysis (averages, "
                "gaps between at-risk and class-wide numbers)\n"
                "- Read the previously uploaded lesson PDF\n"
                "- Generate a lesson plan\n"
                "- Generate practice MCQs\n"
                "- Generate a graded quiz (takeable right here, not just "
                "a text dump)\n"
                "- Generate a full assignment\n"
                "- Generate a formal question paper\n"
                "- Generate other teaching material (notes, handouts, "
                "case studies, revision sheets)\n"
                "- Ask you a clarifying question if it genuinely can't "
                "tell what you want\n\n"
                "It can call several of these in the same step when they "
                "don't depend on each other, instead of one at a time."
            )

        agent_goal = st.text_area(
            "Describe what you want prepared",
            placeholder=(
                "Example: Prepare my class for next week's midterm on "
                "Chapter 5: Neural Networks"
            ),
            height=100,
            key="agent_goal"
        )

        def _make_agent_logger():
            log_lines = list(st.session_state.get("agent_log", []))
            log_area = st.empty()
            log_area.markdown("\n\n".join(log_lines) if log_lines else "")

            def _log(msg):
                log_lines.append(msg)
                st.session_state.agent_log = log_lines
                log_area.markdown("\n\n".join(log_lines))

            return _log

        run_disabled = agent_model is None

        if run_disabled:
            st.error(
                "Agent Mode needs GEMINI_API_KEY configured in "
                "Streamlit secrets."
            )

        if st.button(
            "🚀 Run Agent",
            key="run_agent_button",
            use_container_width=True,
            disabled=run_disabled,
        ):

            if not agent_goal.strip():
                st.warning("Please describe a goal for the agent.")
            else:

                st.session_state.agent_log = []
                st.session_state.agent_final_summary = ""
                st.session_state.agent_outputs = {}
                st.session_state.agent_quiz_questions = []
                st.session_state.agent_quiz_answers = {}
                st.session_state.agent_quiz_score = None
                st.session_state.agent_pending_question = ""
                st.session_state.agent_pending_call_name = ""

                ctx = {
                    "pdf_text": st.session_state.get("pdf_text", ""),
                    "student_df": st.session_state.get("uploaded_student_data"),
                    "education_level": education_level,
                    "language": response_language,
                    "outputs": {},
                }

                log = _make_agent_logger()

                with st.spinner("Agent is planning and acting..."):
                    state = start_agent(agent_goal, ctx, log=log)

                st.session_state.agent_chat = state.get("chat")
                st.session_state.agent_step = state.get("step", 0)
                st.session_state.agent_outputs = ctx.get("outputs", {})

                if state["status"] == "needs_input":
                    st.session_state.agent_pending_question = state["pending_question"]
                    st.session_state.agent_pending_call_name = state["pending_call_name"]
                elif state["status"] == "done":
                    st.session_state.agent_final_summary = state["final_text"]
                    if ctx.get("quiz_questions"):
                        st.session_state.agent_quiz_questions = ctx["quiz_questions"]
                else:
                    st.session_state.agent_final_summary = (
                        "The agent stopped early due to an error — see the log above."
                    )

                st.rerun()

        # --------------------------------------------------------
        # Clarifying-question pause/resume
        # --------------------------------------------------------

        if st.session_state.get("agent_pending_question"):

            st.markdown("---")
            st.info(
                f"🤔 **Agent needs clarification:** "
                f"{st.session_state.agent_pending_question}"
            )

            clarifying_answer = st.text_input(
                "Your answer",
                key="agent_clarifying_answer"
            )

            if st.button(
                "↩️ Send Answer & Continue",
                key="agent_send_answer",
                use_container_width=True
            ):

                if not clarifying_answer.strip():
                    st.warning("Please enter an answer.")
                else:

                    ctx = {
                        "pdf_text": st.session_state.get("pdf_text", ""),
                        "student_df": st.session_state.get("uploaded_student_data"),
                        "education_level": education_level,
                        "language": response_language,
                        "outputs": st.session_state.get("agent_outputs", {}),
                    }

                    log = _make_agent_logger()

                    with st.spinner("Agent is continuing..."):
                        state = continue_agent_with_answer(
                            st.session_state.agent_chat,
                            st.session_state.agent_pending_call_name,
                            clarifying_answer,
                            ctx,
                            log=log,
                            step=st.session_state.get("agent_step", 0),
                        )

                    st.session_state.agent_chat = state.get("chat")
                    st.session_state.agent_step = state.get("step", 0)
                    st.session_state.agent_outputs = ctx.get("outputs", {})
                    st.session_state.agent_pending_question = ""
                    st.session_state.agent_pending_call_name = ""

                    if state["status"] == "needs_input":
                        st.session_state.agent_pending_question = state["pending_question"]
                        st.session_state.agent_pending_call_name = state["pending_call_name"]
                    elif state["status"] == "done":
                        st.session_state.agent_final_summary = state["final_text"]
                        if ctx.get("quiz_questions"):
                            st.session_state.agent_quiz_questions = ctx["quiz_questions"]
                    else:
                        st.session_state.agent_final_summary = (
                            "The agent stopped early due to an error — see the log above."
                        )

                    st.rerun()

        # --------------------------------------------------------
        # Final summary + generated artifacts
        # --------------------------------------------------------

        if st.session_state.get("agent_final_summary"):

            st.markdown("---")
            st.subheader("✅ Agent Summary")
            st.markdown(st.session_state.agent_final_summary)

            outputs = st.session_state.get("agent_outputs", {})

            for artifact_name, artifact_content in outputs.items():

                st.subheader(artifact_name)

                if artifact_name in ("Practice MCQs", "Graded Quiz"):
                    st.caption(
                        "Raw generated JSON — see the interactive quiz "
                        "below if this is the graded quiz."
                    )
                    st.code(artifact_content, language="json")
                else:
                    st.markdown(artifact_content)

                render_pdf_download_button(
                    artifact_content,
                    title=artifact_name,
                    filename=f"{artifact_name.lower().replace(' ', '_')}.pdf",
                    key=f"pdf_download_agent_{artifact_name}",
                )

            # ----------------------------------------------------
            # Interactive quiz — the agent's JSON gets rendered as
            # a real, takeable quiz instead of a text dump.
            # ----------------------------------------------------

            if st.session_state.get("agent_quiz_questions"):

                st.markdown("---")
                st.subheader("🎯 Take the Agent-Generated Quiz")

                questions = st.session_state.agent_quiz_questions

                for index, item in enumerate(questions):

                    options = item.get("options", [])
                    if len(options) != 4:
                        continue

                    selected = st.radio(
                        f"{index + 1}. {item.get('question', '')}",
                        options,
                        key=f"agent_quiz_answer_{index}"
                    )
                    st.session_state.agent_quiz_answers[index] = selected

                if st.button(
                    "✅ Submit Quiz",
                    key="agent_submit_quiz",
                    use_container_width=True
                ):

                    score = 0
                    for index, item in enumerate(questions):
                        options = item.get("options", [])
                        correct_index = item.get("answer")

                        if (
                            index in st.session_state.agent_quiz_answers
                            and isinstance(correct_index, int)
                            and 0 <= correct_index < len(options)
                            and st.session_state.agent_quiz_answers[index] == options[correct_index]
                        ):
                            score += 1

                    st.session_state.agent_quiz_score = score

                if st.session_state.get("agent_quiz_score") is not None:

                    total = len(questions)
                    score = st.session_state.agent_quiz_score
                    percentage = round((score / total) * 100, 1) if total else 0

                    st.success(f"🎉 Score: {score}/{total} ({percentage}%)")

                    for index, item in enumerate(questions):
                        options = item.get("options", [])
                        correct_index = item.get("answer")

                        if isinstance(correct_index, int) and 0 <= correct_index < len(options):
                            correct_option = options[correct_index]

                            if st.session_state.agent_quiz_answers.get(index) == correct_option:
                                st.success(f"Q{index + 1}: Correct")
                            else:
                                st.error(f"Q{index + 1}: Incorrect. Correct answer: {correct_option}")

                            st.caption(item.get("explanation", "No explanation provided."))


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


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🎓 AI Teaching Assistant | Teacher Mode + Student Mode | "
    "Powered by Gemini + Streamlit"
)
