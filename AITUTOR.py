import io
import os
import json
import sqlite3
from datetime import datetime, timezone

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
#
# Design direction: "Academic ledger" — near-black backgrounds and
# honor-roll gold, serif headings for gravitas, clean sans for
# legibility. The actual dark theme (backgrounds, text contrast, form
# inputs, alerts, dropdowns) is set via .streamlit/config.toml, NOT
# here — Streamlit's own components assume a light theme unless the
# theme config says otherwise, so painting backgrounds black with CSS
# alone leaves things like alert boxes and dropdowns with light text on
# light surfaces in spots this file can't fully predict. This CSS only
# layers the accent styling (fonts, banner, cards, tabs, chat bubbles)
# on top of that base theme.
#
# REQUIRED: see config_toml_for_dark_theme.toml (provided alongside
# this file) — copy it to .streamlit/config.toml in your repo root.

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --surface: #15171C;
    --surface-2: #1E2128;
    --border: #2A2D35;
    --gold-600: #C6912F;
    --gold-500: #D9A441;
    --gold-200: #F3DCA0;
    --text-100: #F5F1E8;
    --text-400: #A9ADB8;
    --text-on-gold: #1A1200;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, .main-title {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    letter-spacing: -0.01em;
}

/* ---------- Header banner ---------- */

.app-banner {
    background: linear-gradient(120deg, var(--surface) 0%, var(--surface-2) 100%);
    border-radius: 10px;
    padding: 28px 36px;
    margin-bottom: 22px;
    border-bottom: 3px solid var(--gold-500);
}

.main-title {
    font-size: 34px;
    font-weight: 700;
    color: var(--text-100);
    text-align: left;
    margin: 0 0 4px 0;
}

.subtitle {
    text-align: left;
    font-size: 15px;
    font-family: 'Inter', sans-serif;
    color: var(--gold-200);
    margin: 0;
}

/* ---------- Generic content card ---------- */

.card {
    padding: 18px 22px;
    border-radius: 8px;
    border: 1px solid var(--border);
    border-left: 4px solid var(--gold-500);
    background: var(--surface);
    margin-bottom: 18px;
}

/* ---------- Sidebar ---------- */

section[data-testid="stSidebar"] {
    border-right: 1px solid var(--border);
}

/* ---------- Tabs ---------- */

.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    color: var(--text-400);
    padding: 10px 16px;
}

.stTabs [aria-selected="true"] {
    color: var(--text-100);
    border-bottom: 3px solid var(--gold-500) !important;
}

/* ---------- Buttons ---------- */

.stButton > button {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--text-100);
    transition: background 0.15s ease, border-color 0.15s ease, transform 0.05s ease;
}

.stButton > button:hover {
    background: var(--surface-2);
    border-color: var(--gold-500);
}

.stButton > button:active {
    transform: scale(0.99);
}

.stButton > button[kind="primary"],
div[data-testid="stDownloadButton"] > button {
    background: var(--gold-500);
    border-color: var(--gold-600);
    color: var(--text-on-gold);
}

.stButton > button[kind="primary"]:hover,
div[data-testid="stDownloadButton"] > button:hover {
    background: var(--gold-600);
}

/* ---------- Metrics ---------- */

div[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-top: 3px solid var(--gold-500);
    border-radius: 8px;
    padding: 12px 14px;
}

div[data-testid="stMetricValue"] {
    color: var(--text-100);
    font-family: 'Fraunces', serif;
}

/* ---------- Expanders ---------- */

details[data-testid="stExpander"] {
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--surface);
}

/* ---------- Chat (Agent Mode) ---------- */

div[data-testid="stChatMessage"] {
    border-radius: 10px;
    padding: 4px 6px;
    margin-bottom: 6px;
}

div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: var(--surface-2);
}

div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--gold-500);
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
embedded in a teacher's classroom-prep tool. This is an ongoing chat — the \
teacher may send you a new request at any point, including after you've \
already completed earlier ones. Treat each new request from the teacher as \
a fresh goal to plan for, while remembering everything already produced \
earlier in the conversation.

Guidelines:
- For EVERY new request the teacher sends (the very first one, and any
  follow-up), your first action in response MUST be present_plan: list
  the concrete steps (and which tools you intend to call for each) that
  you plan to take. Call it ALONE. Wait for the teacher's approval
  before taking any other action for that request.
- After a plan is approved, break that request into the smallest set of
  concrete actions needed to satisfy it.
- Call multiple independent tools in the SAME turn when their results
  don't depend on each other (for example, practice MCQs and a graded
  quiz can usually be generated together once the topic is known).
  Sequence tools that genuinely depend on each other's output (for
  example, check student data or read lesson material BEFORE deciding
  what a lesson plan should emphasize).
- Only call ask_clarifying_question when a request is genuinely
  ambiguous or missing information you cannot reasonably infer (for
  example, no subject or topic given at all). Call it ALONE, never
  combined with other tool calls, and only once per genuine gap.
  Prefer acting on a sensible default over asking. If you need to ask a
  clarifying question, it may come either before or after the plan, but
  present_plan must still be your first call for that request.
- A follow-up request may reference or build on artifacts you already
  produced earlier in this conversation (e.g. "make that quiz harder",
  "also add a case study") — use that context instead of starting over.
- When every needed artifact for the CURRENT request has been produced,
  respond with a concise plain-text summary of what you made and why,
  with no further tool calls.
"""

AGENT_TOOLS = [
    {
        "name": "present_plan",
        "description": (
            "Present the concrete steps you intend to take to satisfy "
            "the goal, before doing anything else. MUST be your first "
            "tool call, called ALONE, every run. The teacher approves "
            "or the run stops here."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "steps": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                    "description": (
                        "Short, plain-language steps, e.g. "
                        "'Check which students are struggling', "
                        "'Generate a review-focused lesson plan on X'."
                    ),
                },
            },
            "required": ["steps"],
        },
    },
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
# Course-pack compiler — merges every artifact the agent produced
# into a single combined document, so the teacher can download one
# PDF instead of clicking through each one individually.
# ------------------------------------------------------------

def build_course_pack_text(outputs: dict) -> str:
    sections = []
    for name, content in outputs.items():
        sections.append(f"# {name}\n\n{content}")
    return "\n\n---\n\n".join(sections)


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

    if name == "present_plan":
        # Normally intercepted by the loop before reaching dispatch; this
        # is a defensive fallback only.
        return "Plan approved by the teacher."

    return f"Unknown tool: {name}"


# ============================================================
# STUDENT AGENT MODE — a second agent persona
# ============================================================
#
# Same underlying engine (present_plan / ask_clarifying_question
# interception, parallel tool calls, self-review) as the teacher's
# agent — just a different tool set, system instruction, and Gemini
# model instance, so students get their own study-focused actions
# instead of the teacher's classroom-prep ones. Shared tools
# (generate_mcqs, generate_quiz, read_lesson_material, present_plan,
# ask_clarifying_question) are delegated straight to execute_agent_tool
# rather than reimplemented.

STUDENT_AGENT_SYSTEM_INSTRUCTION = """You are an autonomous personal \
study agent embedded in a student's learning tool. This is an ongoing \
chat — the student may send you a new request at any point, including \
after you've already completed earlier ones. Treat each new request as \
a fresh goal to plan for, while remembering everything already produced \
earlier in the conversation.

Guidelines:
- For EVERY new request the student sends (the very first one, and any
  follow-up), your first action in response MUST be present_plan: list
  the concrete steps (and which tools you intend to call for each) that
  you plan to take. Call it ALONE. Wait for the student's approval
  before taking any other action for that request.
- After a plan is approved, break that request into the smallest set of
  concrete actions needed to satisfy it.
- Call multiple independent tools in the SAME turn when their results
  don't depend on each other (for example, study notes and practice
  MCQs can usually be generated together once the topic is known).
  Sequence tools that genuinely depend on each other's output (for
  example, read previously uploaded study material BEFORE deciding
  what a lesson should emphasize).
- Only call ask_clarifying_question when a request is genuinely
  ambiguous or missing information you cannot reasonably infer (for
  example, no subject or topic given at all). Call it ALONE, never
  combined with other tool calls, and only once per genuine gap.
  Prefer acting on a sensible default over asking. If you need to ask a
  clarifying question, it may come either before or after the plan, but
  present_plan must still be your first call for that request.
- A follow-up request may reference or build on artifacts you already
  produced earlier in this conversation (e.g. "make the notes shorter",
  "now quiz me on that") — use that context instead of starting over.
- When every needed artifact for the CURRENT request has been produced,
  respond with a concise plain-text summary of what you made and why,
  with no further tool calls.
"""

STUDENT_AGENT_TOOLS = [
    next(t for t in AGENT_TOOLS if t["name"] == "present_plan"),
    next(t for t in AGENT_TOOLS if t["name"] == "ask_clarifying_question"),
    {
        "name": "read_lesson_material",
        "description": (
            "Read the previously uploaded study material PDF (from the "
            "PDF Assistant tab) to see what it covers, so a lesson or "
            "notes can target it directly instead of teaching blind."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        },
    },
    {
        "name": "learn_topic",
        "description": (
            "Teach the student a topic from the basics, tailored to "
            "their stated learning goal. Use this for 'teach me', "
            "'explain', or 'help me understand' style requests."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "subject": {"type": "STRING"},
                "topic": {"type": "STRING"},
                "learning_goal": {
                    "type": "STRING",
                    "description": (
                        "Understand the basics, Prepare for an exam, "
                        "Understand with examples, Learn step by step, "
                        "or Revise quickly."
                    ),
                },
            },
            "required": ["topic"],
        },
    },
    {
        "name": "explain_doubt",
        "description": (
            "Explain a specific doubt or point of confusion the student "
            "has. Use this when the request is a targeted question about "
            "something confusing, rather than a request to learn a whole "
            "topic from scratch."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "subject": {"type": "STRING"},
                "topic": {"type": "STRING"},
                "doubt": {"type": "STRING"},
            },
            "required": ["doubt"],
        },
    },
    {
        "name": "generate_notes",
        "description": (
            "Create study notes for a topic. Use this when the student "
            "wants something to read/revise from, not an interactive "
            "explanation or a quiz."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "subject": {"type": "STRING"},
                "topic": {"type": "STRING"},
                "length": {
                    "type": "STRING",
                    "description": "Short, Medium, or Detailed.",
                },
                "purpose": {
                    "type": "STRING",
                    "description": (
                        "General Study, Exam Preparation, Quick "
                        "Revision, or Viva Preparation."
                    ),
                },
            },
            "required": ["topic"],
        },
    },
    next(t for t in AGENT_TOOLS if t["name"] == "generate_mcqs"),
    next(t for t in AGENT_TOOLS if t["name"] == "generate_quiz"),
]

_STUDENT_DELEGATED_TOOLS = {
    "present_plan", "ask_clarifying_question", "read_lesson_material",
    "generate_mcqs", "generate_quiz",
}


def configure_gemini_student_agent():
    """Separate model instance for the student persona — its own tool
    set and system instruction, independent of the teacher's agent_model."""

    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None

    genai.configure(api_key=api_key)

    return genai.GenerativeModel(
        "gemini-3.5-flash",
        tools=STUDENT_AGENT_TOOLS,
        system_instruction=STUDENT_AGENT_SYSTEM_INSTRUCTION,
    )


student_agent_model = configure_gemini_student_agent()


def execute_student_agent_tool(name, args, ctx):
    """Dispatch table for the student agent. Tools shared with the
    teacher agent (same name, same behavior) are delegated straight to
    execute_agent_tool instead of being reimplemented here."""

    if name in _STUDENT_DELEGATED_TOOLS:
        return execute_agent_tool(name, args, ctx)

    if name == "learn_topic":
        subject = args.get("subject", "")
        topic = args.get("topic", "the topic")
        learning_goal = args.get("learning_goal", "Understand the basics")
        prompt = f"""You are a friendly AI tutor. Teach the student the
following topic.
Subject: {subject}
Topic: {topic}
Student Goal: {learning_goal}
Education Level: {ctx.get('education_level')}
Difficulty: {ctx.get('difficulty')}
Language: {ctx.get('language')}
Include: simple introduction, definition, why it matters, step-by-step
explanation, a simple example, a real-world example, important terms,
common mistakes, exam-important points, a quick recap, and three
self-check questions."""
        result = ask_gemini(prompt)
        ctx.setdefault("outputs", {})[f"Lesson: {topic}"] = result
        return f"Lesson on '{topic}' generated ({len(result)} chars)."

    if name == "explain_doubt":
        subject = args.get("subject", "")
        topic = args.get("topic", "")
        doubt = args.get("doubt", "")
        prompt = f"""You are a patient personal AI tutor. A student has
asked the following doubt.
Subject: {subject}
Topic: {topic}
Student's Doubt: {doubt}
Education Level: {ctx.get('education_level')}
Difficulty: {ctx.get('difficulty')}
Language: {ctx.get('language')}
Instructions: first identify what the student seems confused about,
explain it step by step, use a simple analogy if useful, give an
example, correct any misconception gently, and end with a short
takeaway."""
        result = ask_gemini(prompt)
        ctx.setdefault("outputs", {})["Doubt Explanation"] = result
        return f"Doubt explained ({len(result)} chars). Stored as 'Doubt Explanation'."

    if name == "generate_notes":
        subject = args.get("subject", "")
        topic = args.get("topic", "the topic")
        length = args.get("length", "Medium")
        purpose = args.get("purpose", "General Study")
        prompt = f"""You are an expert student study assistant. Create
{length.lower()} study notes.
Subject: {subject}
Topic: {topic}
Purpose: {purpose}
Education Level: {ctx.get('education_level')}
Difficulty: {ctx.get('difficulty')}
Language: {ctx.get('language')}
Include: introduction, definition, important concepts, a clear
explanation, examples, advantages, disadvantages, applications,
important examination points, a quick revision summary, and five key
terms."""
        result = ask_gemini(prompt)
        ctx.setdefault("outputs", {})["Study Notes"] = result
        return f"Study notes generated ({len(result)} chars). Stored as 'Study Notes'."

    return f"Unknown tool: {name}"


# ------------------------------------------------------------
# Friendly, human-readable labels for the activity log — turns
# "generate_lesson_plan({'topic': 'X'})" into something a teacher
# actually enjoys watching scroll by.
# ------------------------------------------------------------

TOOL_LABELS = {
    "present_plan": "📋 Proposing a plan",
    "ask_clarifying_question": "❓ Asking a clarifying question",
    "check_student_risk_data": "📊 Checking which students are struggling",
    "analyze_class_performance": "📈 Analyzing class-wide performance patterns",
    "read_lesson_material": "📄 Reading the uploaded material",
    "generate_lesson_plan": "📘 Drafting a lesson plan",
    "generate_mcqs": "📝 Writing practice MCQs",
    "generate_quiz": "🎯 Building a graded quiz",
    "generate_assignment": "📚 Preparing an assignment",
    "generate_question_paper": "🗒️ Drafting a question paper",
    "generate_teaching_material": "🎨 Creating teaching material",
    "learn_topic": "🧠 Teaching the topic from scratch",
    "explain_doubt": "💡 Explaining the doubt",
    "generate_notes": "📓 Writing study notes",
}


def _tool_label(name: str) -> str:
    return TOOL_LABELS.get(name, f"🔧 Calling {name}")


# ------------------------------------------------------------
# The agent loop
# ------------------------------------------------------------
#
# Supports three things a single-call-at-a-time loop can't:
#   1. PARALLEL tool calls — if Gemini returns more than one function
#      call in a single turn, all of them are executed and their results
#      are sent back together in one message, instead of one round trip
#      per tool.
#   2. PAUSING for a plan approval or a clarifying question — the loop
#      can hand control back to the teacher mid-run instead of guessing
#      or barreling ahead, then resume exactly where it left off once an
#      answer is provided. This is why the loop is split into
#      start/continue instead of one blocking call: Streamlit reruns the
#      whole script on every interaction, so the live `chat` session has
#      to be handed back to the caller to stash in st.session_state
#      rather than looped over internally.
#   3. SELF-REVIEW — once the agent thinks it's done, it's asked once,
#      using its own tools if needed, to check its work against the
#      original goal before actually finishing (see run_self_review).

def _extract_function_calls(response):
    candidate = response.candidates[0]
    return [part.function_call for part in candidate.content.parts if part.function_call]


def _final_text_from_response(response):
    candidate = response.candidates[0]
    return "".join(part.text for part in candidate.content.parts if part.text)


def _drive_agent_loop(chat, response, ctx, log, step, tool_dispatch=execute_agent_tool):

    while step < AGENT_MAX_STEPS:
        step += 1

        calls = _extract_function_calls(response)

        if not calls:
            final_text = _final_text_from_response(response)
            log(f"**Step {step}:** ✅ Agent finished — no further actions needed.")
            return {
                "status": "done",
                "chat": chat,
                "final_text": final_text,
                "step": step,
            }

        plan_call = next(
            (c for c in calls if c.name == "present_plan"), None
        )

        if plan_call is not None:
            steps = list(dict(plan_call.args).get("steps", []))
            log(f"**Step {step}:** {_tool_label('present_plan')}")
            return {
                "status": "plan_ready",
                "chat": chat,
                "plan_steps": steps,
                "pending_call_name": plan_call.name,
                "step": step,
            }

        clarifying = next(
            (c for c in calls if c.name == "ask_clarifying_question"), None
        )

        if clarifying is not None:
            question = dict(clarifying.args).get("question", "Could you clarify the goal?")
            log(f"**Step {step}:** {_tool_label('ask_clarifying_question')}: {question}")
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
            log(f"**Step {step}:** {_tool_label(call.name)}")

            result = tool_dispatch(call.name, args, ctx)
            log(f"↳ {result}")

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


def run_self_review(chat, ctx: dict, log, step: int, goal: str, tool_dispatch=execute_agent_tool):
    """
    One bounded self-correction pass: once the agent thinks it's done,
    it's asked to check its own work against the ORIGINAL goal and, if
    it finds a real gap, fix it using its tools — before actually
    finishing. This runs at most once per run (the caller only invokes
    it right after a "done" status, never recursively from inside
    itself), so it can't turn into an endless self-review loop.
    """

    review_prompt = f"""Before finishing, double-check your work against the
ORIGINAL GOAL you were given:

"{goal}"

If anything is missing, incomplete, or noticeably weak relative to that
goal, use the available tools now to fix it. If everything you already
produced fully satisfies the goal, reply with a short confirmation and
do NOT call any tools."""

    log("**Self-review:** 🔍 Checking the work against the original goal...")

    try:
        response = chat.send_message(review_prompt)
    except Exception as e:
        log(f"Agent API error during self-review: {e}")
        return {"status": "error", "chat": chat, "step": step}

    return _drive_agent_loop(chat, response, ctx, log, step=step, tool_dispatch=tool_dispatch)


def _agent_turn_to_replay_text(turn: dict) -> str:
    """Flattens a saved agent_conversation turn into plain text, for
    replaying prior conversation into a fresh Gemini chat session via
    start_chat(history=...). This restores the model's actual memory of
    what was discussed, not just what the UI displays — an approximate
    reconstruction (tool-call mechanics aren't replayed, only their
    outcomes), but far better than starting genuinely cold."""

    if turn.get("role") == "user":
        return turn.get("content", "")

    kind = turn.get("kind")
    if kind == "plan":
        steps = turn.get("plan_steps", [])
        return "Proposed plan:\n" + "\n".join(f"- {s}" for s in steps)
    if kind == "clarify":
        return f"Asked: {turn.get('question', '')}"
    return turn.get("content", "")


def send_agent_message(chat, message: str, ctx: dict, log=print, step: int = 0, model=None, tool_dispatch=execute_agent_tool, history_turns=None):
    """
    Sends a message on the agent's chat — starting a brand-new chat if
    `chat` is None, or continuing an already-open one otherwise. This is
    what makes the agent genuinely interactive: a follow-up like "also
    make a quiz" or "make that harder" reuses the SAME conversation, so
    the model still remembers everything produced earlier, instead of
    starting cold every time.

    `model` lets a different agent persona (e.g. the student study agent)
    reuse this exact loop instead of duplicating it; defaults to the
    teacher's agent_model for backward compatibility.

    `history_turns`, if given, is prior agent_conversation turns (loaded
    from the database, from BEFORE this browser session existed) to
    replay into a brand-new chat session — so resuming a saved
    conversation restores real conversational memory, not just what's
    shown on screen. Only used when `chat` is None.
    """

    model = model or agent_model

    if model is None:
        log("Agent Mode is not available: Gemini API key not configured.")
        return {"status": "error", "chat": None}

    ctx["goal"] = message  # used by the self-review pass for this request

    if chat is None:
        if history_turns:
            replay = [
                {
                    "role": "user" if t.get("role") == "user" else "model",
                    "parts": [_agent_turn_to_replay_text(t)],
                }
                for t in history_turns
            ]
            chat = model.start_chat(history=replay)
        else:
            chat = model.start_chat()

    try:
        response = chat.send_message(message)
    except Exception as e:
        log(f"Agent API error: {e}")
        return {"status": "error", "chat": chat, "step": step}

    state = _drive_agent_loop(chat, response, ctx, log, step=step, tool_dispatch=tool_dispatch)

    if state["status"] == "done":
        state = run_self_review(chat, ctx, log, state["step"], message, tool_dispatch=tool_dispatch)

    return state


def start_agent(goal: str, ctx: dict, log=print):
    """Back-compat wrapper: starts a brand-new agent chat."""
    return send_agent_message(None, goal, ctx, log=log, step=0)


def continue_agent_with_answer(chat, pending_call_name: str, answer: str, ctx: dict, log=print, step: int = 0, tool_dispatch=execute_agent_tool):
    """Resumes a paused agent after the teacher/student approves a plan
    or answers a clarifying question, continuing from exactly where it
    left off."""

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

    state = _drive_agent_loop(chat, response, ctx, log, step=step, tool_dispatch=tool_dispatch)

    if state["status"] == "done":
        goal = ctx.get("goal", "the stated goal")
        state = run_self_review(chat, ctx, log, state["step"], goal, tool_dispatch=tool_dispatch)

    return state


# ============================================================
# GENERIC AGENT CHAT UI
# ============================================================
#
# Renders a full interactive-agent chat tab: conversation history,
# plan-approval / clarifying-question pause-resume, a chat input for
# new/follow-up goals, completion metrics, a combined course-pack PDF,
# JSON export/import of the chat history, and (if a graded quiz was
# produced) an interactive quiz to take. Shared by BOTH the teacher's
# and the student's Agent Mode tabs — state_prefix keeps their session
# state, widget keys, and conversations completely independent.

def _make_status_logger(label: str):
    """Live st.status()-backed logger scoped to a single turn."""
    turn_log_lines = []
    status_box = st.status(label, expanded=True)
    log_area = status_box.empty()
    log_area.markdown("_Starting up..._")

    def _log(msg):
        turn_log_lines.append(msg)
        log_area.markdown("\n\n".join(turn_log_lines))

    return _log, status_box, turn_log_lines


def _finish_status_box(status_box, state):
    if state["status"] == "done":
        status_box.update(label="✅ Done", state="complete")
    elif state["status"] == "plan_ready":
        status_box.update(label="📋 Waiting for plan approval", state="complete")
    elif state["status"] == "needs_input":
        status_box.update(label="❓ Waiting for your answer", state="complete")
    else:
        status_box.update(label="⚠️ Stopped early", state="error")


def _render_agent_artifact(artifact_name: str, artifact_content: str, key_suffix: str):
    st.markdown(f"**{artifact_name}**")
    if artifact_name in ("Practice MCQs", "Graded Quiz"):
        st.caption(
            "Raw generated JSON — see the interactive quiz below if "
            "this is the graded quiz."
        )
        st.code(artifact_content, language="json")
    else:
        st.markdown(artifact_content)

    render_pdf_download_button(
        artifact_content,
        title=artifact_name,
        filename=f"{artifact_name.lower().replace(' ', '_')}.pdf",
        key=f"pdf_download_{key_suffix}_{artifact_name}",
    )


def render_agent_chat_tab(state_prefix: str, agent_model_instance, tool_dispatch, ctx_builder):
    """
    state_prefix:        unique session-state/widget-key prefix for this
                          agent chat (e.g. "agent" for Teacher, "student_agent"
                          for Student). Also the `panel` name it's saved
                          under in SQLite, so each persona's history is
                          kept separate.
    agent_model_instance: the genai.GenerativeModel configured with this
                          persona's tools + system instruction.
    tool_dispatch:        the execute_*_agent_tool function for this persona.
    ctx_builder:          callable() -> dict returning fresh context for a
                          new turn (pdf_text, student_df, sidebar values,
                          etc.) — 'outputs' is filled in automatically from
                          this tab's accumulated session state.

    History is saved to / loaded from SQLite (see PERSISTENT CHAT HISTORY
    above), keyed by the sidebar's "Your Name / ID" field, so it survives
    app restarts and follows the same person across devices — same
    treatment as the plain chat panels get.
    """

    conv_key = f"{state_prefix}_conversation"
    chat_key = f"{state_prefix}_chat"
    step_key = f"{state_prefix}_step"
    outputs_key = f"{state_prefix}_outputs"
    pending_q_key = f"{state_prefix}_pending_question"
    pending_call_key = f"{state_prefix}_pending_call_name"
    pending_plan_key = f"{state_prefix}_pending_plan_steps"
    quiz_q_key = f"{state_prefix}_quiz_questions"
    quiz_a_key = f"{state_prefix}_quiz_answers"
    quiz_score_key = f"{state_prefix}_quiz_score"
    loaded_flag_key = f"{state_prefix}_history_loaded"

    defaults = [
        (conv_key, []), (chat_key, None), (step_key, 0), (outputs_key, {}),
        (pending_q_key, ""), (pending_call_key, ""), (pending_plan_key, []),
        (quiz_q_key, []), (quiz_a_key, {}), (quiz_score_key, None),
    ]
    for key, default in defaults:
        if key not in st.session_state:
            st.session_state[key] = default

    def _reconstruct_from_turns(turns):
        """Rebuilds outputs/quiz state from a list of saved turns —
        shared by the initial DB load and the manual reload button."""
        st.session_state[conv_key] = turns
        st.session_state[chat_key] = None
        st.session_state[outputs_key] = {}
        for t in turns:
            if t.get("kind") == "done":
                st.session_state[outputs_key].update(t.get("outputs", {}))
        if "Graded Quiz" in st.session_state[outputs_key]:
            try:
                st.session_state[quiz_q_key] = json.loads(
                    _clean_json_block(st.session_state[outputs_key]["Graded Quiz"])
                )
            except Exception:
                pass

    # Load this user's saved conversation once per browser session.
    if not st.session_state.get(loaded_flag_key):
        st.session_state[loaded_flag_key] = True
        loaded_turns = load_chat_history(get_current_user_id(), state_prefix)
        if loaded_turns:
            _reconstruct_from_turns(loaded_turns)

    def _append_turn(turn):
        st.session_state[conv_key].append(turn)
        save_chat_turn(get_current_user_id(), state_prefix, turn["role"], turn)

    def _build_ctx():
        ctx = ctx_builder()
        ctx["outputs"] = dict(st.session_state.get(outputs_key, {}))
        return ctx

    def _apply_state(state, ctx, steps_this_turn):
        st.session_state[chat_key] = state.get("chat")
        st.session_state[step_key] = state.get("step", 0)
        st.session_state[outputs_key] = ctx.get("outputs", {})
        st.session_state[pending_q_key] = ""
        st.session_state[pending_call_key] = ""
        st.session_state[pending_plan_key] = []

        if state["status"] == "plan_ready":
            st.session_state[pending_plan_key] = state["plan_steps"]
            st.session_state[pending_call_key] = state["pending_call_name"]
            return {"role": "assistant", "kind": "plan", "plan_steps": state["plan_steps"], "steps": steps_this_turn}
        elif state["status"] == "needs_input":
            st.session_state[pending_q_key] = state["pending_question"]
            st.session_state[pending_call_key] = state["pending_call_name"]
            return {"role": "assistant", "kind": "clarify", "question": state["pending_question"], "steps": steps_this_turn}
        elif state["status"] == "done":
            if ctx.get("quiz_questions"):
                st.session_state[quiz_q_key] = ctx["quiz_questions"]
            return {
                "role": "assistant", "kind": "done", "content": state["final_text"],
                "steps": steps_this_turn, "outputs": dict(ctx.get("outputs", {})),
            }
        else:
            return {
                "role": "assistant", "kind": "error",
                "content": "The agent stopped early due to an error — see the steps above.",
                "steps": steps_this_turn,
            }

    # ---- conversation history ----
    for turn_index, turn in enumerate(st.session_state[conv_key]):
        if turn["role"] == "user":
            with st.chat_message("user"):
                st.markdown(turn["content"])
            continue

        with st.chat_message("assistant"):
            if turn.get("steps"):
                with st.expander("🛠️ Steps taken", expanded=False):
                    st.markdown("\n\n".join(turn["steps"]))

            if turn["kind"] == "plan":
                st.info("📋 **Proposed a plan:**")
                for s in turn["plan_steps"]:
                    st.markdown(f"- {s}")
                st.caption(
                    "(This was the plan at the time — see below if "
                    "it's still awaiting your approval.)"
                )
            elif turn["kind"] == "clarify":
                st.info(f"🤔 **Asked:** {turn['question']}")
            elif turn["kind"] == "error":
                st.error(turn["content"])
            else:  # done
                st.success(turn["content"])
                for artifact_name, artifact_content in turn.get("outputs", {}).items():
                    _render_agent_artifact(artifact_name, artifact_content, key_suffix=f"{state_prefix}hist{turn_index}")

    # ---- pending plan approval ----
    if st.session_state[pending_plan_key]:

        with st.chat_message("assistant"):
            st.info("📋 **Proposed plan — approve to run it:**")
            for s in st.session_state[pending_plan_key]:
                st.markdown(f"- {s}")

            col_approve, col_cancel = st.columns(2)
            with col_approve:
                approve_clicked = st.button("▶️ Approve & Run", key=f"{state_prefix}_approve_plan", use_container_width=True)
            with col_cancel:
                cancel_clicked = st.button("✖️ Cancel", key=f"{state_prefix}_cancel_plan", use_container_width=True)

            if cancel_clicked:
                _append_turn({
                    "role": "assistant", "kind": "error",
                    "content": "Run cancelled before execution.", "steps": [],
                })
                st.session_state[pending_plan_key] = []
                st.session_state[pending_call_key] = ""
                st.rerun()

            if approve_clicked:
                ctx = _build_ctx()
                log, status_box, steps_this_turn = _make_status_logger("🤖 Executing the approved plan...")
                state = continue_agent_with_answer(
                    st.session_state[chat_key],
                    st.session_state[pending_call_key],
                    "Approved. Proceed with the plan.",
                    ctx, log=log, step=st.session_state[step_key],
                    tool_dispatch=tool_dispatch,
                )
                turn = _apply_state(state, ctx, steps_this_turn)
                _finish_status_box(status_box, state)
                _append_turn(turn)
                st.rerun()

    # ---- pending clarifying question ----
    elif st.session_state[pending_q_key]:

        with st.chat_message("assistant"):
            st.info(f"🤔 **Needs clarification:** {st.session_state[pending_q_key]}")

            answer_text = st.text_input("Your answer", key=f"{state_prefix}_clarifying_answer")

            if st.button("↩️ Send Answer", key=f"{state_prefix}_send_answer", use_container_width=True):
                if not answer_text.strip():
                    st.warning("Please enter an answer.")
                else:
                    _append_turn({"role": "user", "content": answer_text})
                    ctx = _build_ctx()
                    log, status_box, steps_this_turn = _make_status_logger("🤖 Continuing...")
                    state = continue_agent_with_answer(
                        st.session_state[chat_key],
                        st.session_state[pending_call_key],
                        answer_text, ctx, log=log, step=st.session_state[step_key],
                        tool_dispatch=tool_dispatch,
                    )
                    turn = _apply_state(state, ctx, steps_this_turn)
                    _finish_status_box(status_box, state)
                    _append_turn(turn)
                    st.rerun()

    # ---- new message input ----
    awaiting_response = bool(st.session_state[pending_plan_key] or st.session_state[pending_q_key])

    new_message = st.chat_input(
        "Ask the agent to prepare something, or send a follow-up...",
        key=f"{state_prefix}_chat_input",
        disabled=(agent_model_instance is None or awaiting_response),
    )

    if agent_model_instance is None:
        st.error("Agent Mode needs GEMINI_API_KEY configured in Streamlit secrets.")

    if new_message:
        # Prior turns BEFORE this new message — if there's no live chat
        # yet but these exist, this is a saved conversation resuming in
        # a fresh browser session, so replay them into the new chat.
        prior_turns = list(st.session_state[conv_key])

        _append_turn({"role": "user", "content": new_message})
        with st.chat_message("user"):
            st.markdown(new_message)

        ctx = _build_ctx()
        log, status_box, steps_this_turn = _make_status_logger("🤖 Working on it...")

        state = send_agent_message(
            st.session_state[chat_key], new_message, ctx, log=log, step=0,
            model=agent_model_instance, tool_dispatch=tool_dispatch,
            history_turns=prior_turns,
        )

        turn = _apply_state(state, ctx, steps_this_turn)
        _finish_status_box(status_box, state)
        _append_turn(turn)
        st.rerun()

    # ---- metrics, course pack, export/import, clear ----
    if st.session_state[outputs_key]:

        st.markdown("---")

        metric_cols = st.columns(3)
        with metric_cols[0]:
            st.metric("Artifacts produced", len(st.session_state[outputs_key]))
        with metric_cols[1]:
            st.metric("Steps taken (total)", st.session_state[step_key])
        with metric_cols[2]:
            st.metric("Interactive quiz", "Ready" if st.session_state[quiz_q_key] else "—")

        try:
            pack_bytes = generate_pdf_bytes(
                build_course_pack_text(st.session_state[outputs_key]),
                title="Course Prep Pack",
            )
            st.download_button(
                label="📦 Download Full Course Pack (all artifacts, one PDF)",
                data=pack_bytes,
                file_name=f"{state_prefix}_course_pack.pdf",
                mime="application/pdf",
                key=f"pdf_download_{state_prefix}_course_pack",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"Could not prepare the combined course pack: {e}")

        st.caption(
            "This conversation is also saved automatically on the server "
            "under your sidebar ID — the JSON download/upload below is "
            "just for taking a copy with you or viewing it in a browser "
            "session that hasn't loaded your ID's saved history yet."
        )

        export_col, import_col = st.columns(2)

        with export_col:
            try:
                history_json = json.dumps(st.session_state[conv_key], indent=2)
                st.download_button(
                    "💾 Download Chat History (JSON)",
                    data=history_json,
                    file_name=f"{state_prefix}_chat_history.json",
                    mime="application/json",
                    key=f"{state_prefix}_history_download",
                    use_container_width=True,
                )
            except Exception as e:
                st.warning(f"Could not prepare chat history export: {e}")

        with import_col:
            uploaded_history = st.file_uploader(
                "📤 Load previous chat history",
                type=["json"],
                key=f"{state_prefix}_history_upload",
            )
            if uploaded_history is not None:
                try:
                    loaded = json.loads(uploaded_history.read().decode("utf-8"))
                    if isinstance(loaded, list):
                        st.session_state[conv_key] = loaded
                        st.success("Chat history loaded. Scroll up to view it.")
                        st.rerun()
                    else:
                        st.error("That file doesn't look like a saved agent chat history.")
                except Exception as e:
                    st.error(f"Could not load that history file: {e}")

        reload_col, clear_col = st.columns(2)

        with reload_col:
            if st.button("🔄 Reload History for This ID", key=f"{state_prefix}_reload_history"):
                _reconstruct_from_turns(load_chat_history(get_current_user_id(), state_prefix))
                st.rerun()

        with clear_col:
            if st.button("🗑️ Clear Conversation", key=f"{state_prefix}_clear_conversation"):
                clear_chat_history(get_current_user_id(), state_prefix)
                for key, default in defaults:
                    st.session_state[key] = default
                st.rerun()

    # ---- interactive quiz ----
    if st.session_state[quiz_q_key]:

        st.markdown("---")
        st.subheader("🎯 Take the Agent-Generated Quiz")

        questions = st.session_state[quiz_q_key]

        for index, item in enumerate(questions):
            options = item.get("options", [])
            if len(options) != 4:
                continue
            selected = st.radio(
                f"{index + 1}. {item.get('question', '')}",
                options, key=f"{state_prefix}_quiz_answer_{index}"
            )
            st.session_state[quiz_a_key][index] = selected

        if st.button("✅ Submit Quiz", key=f"{state_prefix}_submit_quiz", use_container_width=True):
            score = 0
            for index, item in enumerate(questions):
                options = item.get("options", [])
                correct_index = item.get("answer")
                if (
                    index in st.session_state[quiz_a_key]
                    and isinstance(correct_index, int)
                    and 0 <= correct_index < len(options)
                    and st.session_state[quiz_a_key][index] == options[correct_index]
                ):
                    score += 1
            st.session_state[quiz_score_key] = score

        if st.session_state[quiz_score_key] is not None:
            total = len(questions)
            score = st.session_state[quiz_score_key]
            percentage = round((score / total) * 100, 1) if total else 0
            st.success(f"🎉 Score: {score}/{total} ({percentage}%)")

            for index, item in enumerate(questions):
                options = item.get("options", [])
                correct_index = item.get("answer")
                if isinstance(correct_index, int) and 0 <= correct_index < len(options):
                    correct_option = options[correct_index]
                    if st.session_state[quiz_a_key].get(index) == correct_option:
                        st.success(f"Q{index + 1}: Correct")
                    else:
                        st.error(f"Q{index + 1}: Incorrect. Correct answer: {correct_option}")
                    st.caption(item.get("explanation", "No explanation provided."))


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
# PERSISTENT CHAT HISTORY (SQLite)
# ============================================================
#
# st.session_state only lives for one browser session — it's gone the
# moment the tab closes, and every device/browser hitting the same
# deployed app gets its own empty history. This stores every chat turn
# server-side in a SQLite file instead, keyed by a user ID the person
# enters in the sidebar, so the same ID sees the same history from any
# device, and it survives the app process restarting.
#
# HONEST LIMITATION: on Streamlit Community Cloud specifically, the
# local filesystem persists while the app is just sleeping/waking, but
# a full redeploy (new commit, or "reboot app") rebuilds the container
# and wipes local files — including this .db file. For history that
# survives redeploys too, swap CHAT_DB_PATH's sqlite3 connection for a
# hosted database (e.g. Postgres via SQLAlchemy) — every function below
# is isolated to this one section specifically so that swap only
# touches this file, nothing that calls save_chat_turn/load_chat_history
# needs to change.

CHAT_DB_PATH = os.path.join(os.path.dirname(__file__), "chat_history.db")


def get_db_connection():
    conn = sqlite3.connect(CHAT_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_chat_db():
    conn = get_db_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                panel TEXT NOT NULL,
                role TEXT NOT NULL,
                turn_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chat_turns_user_panel
            ON chat_turns (user_id, panel, id)
        """)
        conn.commit()
    finally:
        conn.close()


init_chat_db()


def get_current_user_id() -> str:
    """The sidebar 'Your ID' field controls whose history is loaded/saved.
    Defaults to 'guest' so the app works with zero setup — but everyone
    who leaves it as 'guest' shares one history bucket, so multi-user
    deployments should ask people to set a unique ID."""
    return (st.session_state.get("chat_user_id") or "guest").strip() or "guest"


def save_chat_turn(user_id: str, panel: str, role: str, turn: dict):
    """Persists one turn. `turn` is any JSON-serializable dict — the
    simple chat panels store {"role", "content"}; Agent Mode stores its
    richer turn structure (kind, plan_steps, outputs, etc.) as-is."""
    conn = get_db_connection()
    try:
        conn.execute(
            "INSERT INTO chat_turns (user_id, panel, role, turn_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                user_id,
                panel,
                role,
                json.dumps(turn),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    except Exception as e:
        st.warning(f"Could not save chat history: {e}")
    finally:
        conn.close()


def load_chat_history(user_id: str, panel: str) -> list:
    conn = get_db_connection()
    try:
        rows = conn.execute(
            "SELECT turn_json FROM chat_turns WHERE user_id = ? AND panel = ? ORDER BY id ASC",
            (user_id, panel),
        ).fetchall()
        return [json.loads(row["turn_json"]) for row in rows]
    except Exception as e:
        st.warning(f"Could not load saved chat history: {e}")
        return []
    finally:
        conn.close()


def clear_chat_history(user_id: str, panel: str):
    conn = get_db_connection()
    try:
        conn.execute(
            "DELETE FROM chat_turns WHERE user_id = ? AND panel = ?",
            (user_id, panel),
        )
        conn.commit()
    except Exception as e:
        st.warning(f"Could not clear saved chat history: {e}")
    finally:
        conn.close()


# ============================================================
# SHARED CHAT PANEL — interactive, multi-turn, with persisted history
# ============================================================
#
# For places where a real back-and-forth (with follow-ups that remember
# earlier context) is more useful than a single one-shot answer: doubt
# solving, PDF Q&A. Backed by the plain `model` (no tools — this isn't
# Agent Mode's tool-calling loop, just an ordinary multi-turn chat).
# Each panel keeps its own independent conversation + chat session,
# scoped by `state_prefix`, so multiple panels on the same page don't
# collide. History is loaded from / saved to SQLite (see above) so it
# outlives the browser session and follows the same user ID across
# devices — only the live Gemini chat session itself (needed for
# multi-turn context) is per-browser-session, since that object can't
# be serialized to a database.

def render_chat_panel(state_prefix: str, placeholder: str, seed_prompt_builder=None, empty_hint: str = None):
    """
    state_prefix:        unique key prefix for this panel's session state
                          AND the `panel` name it's saved under in SQLite.
    placeholder:         placeholder text for the chat input box.
    seed_prompt_builder: optional callable(first_message) -> str that
                          wraps a message with extra context (e.g. PDF
                          material, subject/topic, tone instructions).
    empty_hint:          optional caption shown before the first message.
    """

    conv_key = f"{state_prefix}_conversation"
    chat_key = f"{state_prefix}_chat_session"
    user_id = get_current_user_id()

    if conv_key not in st.session_state:
        st.session_state[conv_key] = load_chat_history(user_id, state_prefix)
    if chat_key not in st.session_state:
        st.session_state[chat_key] = None

    if not st.session_state[conv_key] and empty_hint:
        st.caption(empty_hint)

    for turn in st.session_state[conv_key]:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    user_message = st.chat_input(placeholder, key=f"{state_prefix}_chat_input")

    if user_message:
        user_turn = {"role": "user", "content": user_message}
        st.session_state[conv_key].append(user_turn)
        save_chat_turn(user_id, state_prefix, "user", user_turn)

        with st.chat_message("user"):
            st.markdown(user_message)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if model is None:
                    answer = "Gemini model is not configured."
                else:
                    needs_new_session = st.session_state[chat_key] is None

                    if needs_new_session:
                        # History before this new message (i.e. everything
                        # loaded from the database, minus the message just
                        # appended above) — if there's any, this is a
                        # resumed conversation in a fresh browser session.
                        prior_turns = st.session_state[conv_key][:-1]

                        if prior_turns:
                            # Restore the model's actual conversational
                            # memory, not just the visible transcript:
                            # re-prime it with context, then replay the
                            # real exchange so far.
                            replay = []
                            if seed_prompt_builder:
                                context_only = seed_prompt_builder(
                                    "(Continuing a previous conversation — no new question yet, just re-establishing context.)"
                                )
                                replay.append({"role": "user", "parts": [context_only]})
                                replay.append({"role": "model", "parts": ["Understood — I have the context and I'm ready to continue."]})
                            for t in prior_turns:
                                api_role = "user" if t["role"] == "user" else "model"
                                replay.append({"role": api_role, "parts": [t["content"]]})
                            st.session_state[chat_key] = model.start_chat(history=replay)
                            outgoing = user_message
                        else:
                            st.session_state[chat_key] = model.start_chat()
                            outgoing = (
                                seed_prompt_builder(user_message)
                                if seed_prompt_builder else user_message
                            )
                    else:
                        outgoing = user_message

                    try:
                        response = st.session_state[chat_key].send_message(outgoing)
                        answer = (
                            response.text if response and response.text
                            else "Gemini did not return a response."
                        )
                    except Exception as e:
                        answer = f"Gemini API Error: {e}"

            st.markdown(answer)

        assistant_turn = {"role": "assistant", "content": answer}
        st.session_state[conv_key].append(assistant_turn)
        save_chat_turn(user_id, state_prefix, "assistant", assistant_turn)

        if not is_gemini_error(answer):
            render_pdf_download_button(
                answer,
                title="Chat Answer",
                filename=f"{state_prefix}_answer.pdf",
                key=f"pdf_download_{state_prefix}_{len(st.session_state[conv_key])}",
            )

    reload_col, clear_col = st.columns(2)

    with reload_col:
        if st.button("🔄 Reload History", key=f"{state_prefix}_reload"):
            st.session_state[conv_key] = load_chat_history(user_id, state_prefix)
            st.session_state[chat_key] = None
            st.rerun()

    with clear_col:
        if st.session_state[conv_key]:
            if st.button("🗑️ Clear Chat", key=f"{state_prefix}_clear_chat"):
                clear_chat_history(user_id, state_prefix)
                st.session_state[conv_key] = []
                st.session_state[chat_key] = None
                st.rerun()


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
    "notes_response": "",
    "mcqs_response": "",
    "quiz_questions": [],
    "quiz_answers": {},
    "quiz_score": None,
    "student_analysis_response": "",
    "uploaded_student_data": None,
}

for _key, _default in _DEFAULT_STATE.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="app-banner">'
    '<div class="main-title">🎓 AI Teaching Assistant</div>'
    '<div class="subtitle">An intelligent teaching and learning assistant, powered by Gemini</div>'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Assistant Settings")

st.session_state.chat_user_id = st.sidebar.text_input(
    "Your Name / ID (for saved chat history)",
    value=st.session_state.get("chat_user_id", "guest"),
    help=(
        "Chats are saved on the server under this ID, so they survive "
        "app restarts and follow you across devices. Anyone who leaves "
        "this as 'guest' shares one saved history — set a unique ID to "
        "keep yours separate. If you change this mid-session, use a "
        "panel's 'Reload History' button (or refresh the page) to pick "
        "up that ID's saved chats."
    ),
)

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

            st.markdown("---")
            st.write(
                "Chat about the material below — ask a question, then "
                "send follow-ups and it'll remember the conversation."
            )

            def _teacher_pdf_seed(first_question):
                material = st.session_state.pdf_text[:50000]
                return f"""You are an AI Teaching Assistant helping a teacher.
This is an ongoing conversation — use the supplied teaching material to
answer this request and any follow-ups that come later.

Teaching Material:
----------------
{material}
----------------

Education Level: {education_level}
Language: {response_language}

Instructions for every answer in this conversation:
1. Base responses primarily on the supplied material.
2. Do not invent information.
3. Clearly say when the material does not contain the answer.
4. Make results practical for teaching.
5. Use headings and bullet points where appropriate.

Teacher's first request:
{first_question}"""

            render_chat_panel(
                state_prefix="teacher_pdf",
                placeholder="Ask something about the material, or send a follow-up...",
                seed_prompt_builder=_teacher_pdf_seed,
                empty_hint="Ask a question about the uploaded material to start the conversation.",
            )

    # --------------------------------------------------------
    # TEACHER — AGENT MODE
    # --------------------------------------------------------

    with teacher_tabs[6]:

        st.header("🤖 Autonomous Course Prep Agent")

        st.write(
            "Chat with the agent instead of filling a form. It remembers "
            "everything produced earlier in this conversation, so you can "
            "send follow-ups like *\"also make a quiz\"* or *\"make that "
            "harder\"* and it'll build on what's already there — planning "
            "each new request and waiting for your approval before acting, "
            "asking you when something's genuinely unclear, and "
            "double-checking its own work before calling itself done."
        )

        with st.expander("What can the agent do on its own?"):
            st.markdown(
                "- 📋 Propose a plan for each new request and wait for "
                "your approval before doing anything\n"
                "- 📊 Check the uploaded student CSV for at-risk students\n"
                "- 📈 Run a deeper class-performance analysis (averages, "
                "gaps between at-risk and class-wide numbers)\n"
                "- 📄 Read the previously uploaded lesson PDF\n"
                "- 📘 Generate a lesson plan\n"
                "- 📝 Generate practice MCQs\n"
                "- 🎯 Generate a graded quiz (takeable right here, not "
                "just a text dump)\n"
                "- 📚 Generate a full assignment\n"
                "- 🗒️ Generate a formal question paper\n"
                "- 🎨 Generate other teaching material (notes, handouts, "
                "case studies, revision sheets)\n"
                "- ❓ Ask you a clarifying question if it genuinely can't "
                "tell what you want\n"
                "- 🔍 Review its own output against your request before "
                "finishing, and fix gaps it finds\n\n"
                "It can call several of these in the same step when they "
                "don't depend on each other, instead of one at a time — "
                "and it keeps the whole conversation in memory, so later "
                "requests can build on earlier artifacts."
            )

        def _teacher_agent_ctx_builder():
            return {
                "pdf_text": st.session_state.get("pdf_text", ""),
                "student_df": st.session_state.get("uploaded_student_data"),
                "education_level": education_level,
                "language": response_language,
            }

        render_agent_chat_tab(
            state_prefix="agent",
            agent_model_instance=agent_model,
            tool_dispatch=execute_agent_tool,
            ctx_builder=_teacher_agent_ctx_builder,
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
        "🤖 Agent Mode",
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

        st.markdown("---")

        def _doubt_seed(first_doubt):
            return f"""You are a patient personal AI tutor. This is an
ongoing conversation — continue naturally for any follow-up questions
the student sends after this one.

Subject: {doubt_subject}
Topic: {doubt_topic}
Education Level: {education_level}
Difficulty: {difficulty}
Language: {response_language}

Instructions for every answer in this conversation:
1. First identify what the student seems confused about.
2. Explain it step by step.
3. Use a simple analogy if useful.
4. Give an example.
5. Correct any misconception gently.
6. End with a short takeaway.
7. Do not overwhelm the student with unrelated theory.

Student's first doubt:
{first_doubt}"""

        render_chat_panel(
            state_prefix="student_doubt",
            placeholder="Describe your doubt, or ask a follow-up...",
            seed_prompt_builder=_doubt_seed,
            empty_hint="Describe your doubt below to start chatting with your tutor.",
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

            st.markdown("---")
            st.write(
                "Chat about the material below — ask a question, then "
                "send follow-ups and it'll remember the conversation."
            )

            def _student_pdf_seed(first_question):
                material = st.session_state.pdf_text[:50000]
                return f"""You are an AI student tutor. This is an ongoing
conversation — answer this question and any follow-ups using the study
material provided below.

Study Material:
----------------
{material}
----------------

Difficulty: {difficulty}
Education Level: {education_level}
Language: {response_language}

Instructions for every answer in this conversation:
1. Base answers primarily on the provided material.
2. Explain clearly.
3. Do not invent information.
4. If the answer cannot be found in the material, clearly say so.
5. Use examples when useful.
6. Use headings and bullet points where appropriate.

Student's first question:
{first_question}"""

            render_chat_panel(
                state_prefix="student_pdf",
                placeholder="Ask something about your study material, or send a follow-up...",
                seed_prompt_builder=_student_pdf_seed,
                empty_hint="Ask a question about the uploaded material to start the conversation.",
            )

    # --------------------------------------------------------
    # STUDENT — AGENT MODE
    # --------------------------------------------------------

    with student_tabs[6]:

        st.header("🤖 Personal Study Agent")

        st.write(
            "Chat with the agent instead of filling a form. It remembers "
            "everything produced earlier in this conversation, so follow-ups "
            "like *\"now quiz me on that\"* or *\"explain it more simply\"* "
            "build on what's already there — planning each new request and "
            "waiting for your approval before acting, asking you when "
            "something's genuinely unclear, and double-checking its own "
            "work before calling itself done."
        )

        with st.expander("What can the agent do on its own?"):
            st.markdown(
                "- 📋 Propose a plan for each new request and wait for "
                "your approval before doing anything\n"
                "- 📄 Read your previously uploaded study material PDF\n"
                "- 🧠 Teach you a topic from the basics\n"
                "- 💡 Explain a specific doubt or point of confusion\n"
                "- 📓 Write study notes on a topic\n"
                "- 📝 Generate practice MCQs\n"
                "- 🎯 Generate a graded quiz (takeable right here, not "
                "just a text dump)\n"
                "- ❓ Ask you a clarifying question if it genuinely can't "
                "tell what you want\n"
                "- 🔍 Review its own output against your request before "
                "finishing, and fix gaps it finds\n\n"
                "It can call several of these in the same step when they "
                "don't depend on each other, instead of one at a time — "
                "and it keeps the whole conversation in memory, so later "
                "requests can build on earlier lessons and notes."
            )

        def _student_agent_ctx_builder():
            return {
                "pdf_text": st.session_state.get("pdf_text", ""),
                "education_level": education_level,
                "difficulty": difficulty,
                "language": response_language,
            }

        render_agent_chat_tab(
            state_prefix="student_agent",
            agent_model_instance=student_agent_model,
            tool_dispatch=execute_student_agent_tool,
            ctx_builder=_student_agent_ctx_builder,
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🎓 AI Teaching Assistant | Teacher Mode + Student Mode | "
    "Powered by Gemini + Streamlit"
)
