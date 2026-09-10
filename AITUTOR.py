import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader


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
# PDF TEXT EXTRACTION
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
# SESSION STATE
# ============================================================

if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

if "last_response" not in st.session_state:
    st.session_state.last_response = ""


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

            st.session_state.last_response = answer

            st.subheader("🤖 AI Teacher Response")

            st.markdown(answer)


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

            st.subheader("📖 Generated Notes")

            st.markdown(notes)


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

            st.subheader("📝 Generated Questions")

            st.markdown(questions)


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

            st.subheader("🧠 Generated MCQs")

            st.markdown(mcqs)


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

                st.subheader(
                    "🤖 AI Answer"
                )

                st.markdown(pdf_answer)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🎓 AI Teaching Assistant | "
    "Powered by Python, Streamlit and Gemini"
)
