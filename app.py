import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
from datetime import datetime


# ============================================================
# CAREERTWIN AI
# AI-Powered Personalized Career Intelligence System
# ============================================================

# -------------------- PAGE CONFIG --------------------

st.set_page_config(
    page_title="CareerTwin AI",
    page_icon="🧠",
    layout="wide"
)


# -------------------- AI MODEL --------------------

@st.cache_resource
def load_ai_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_ai_model()


# -------------------- SKILLS --------------------

SKILLS = [
    "python",
    "java",
    "c++",
    "sql",
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "generative ai",
    "nlp",
    "natural language processing",
    "computer vision",
    "tensorflow",
    "pytorch",
    "keras",
    "pandas",
    "numpy",
    "scikit-learn",
    "fastapi",
    "flask",
    "django",
    "docker",
    "kubernetes",
    "aws",
    "azure",
    "gcp",
    "google cloud",
    "git",
    "github",
    "html",
    "css",
    "javascript",
    "react",
    "rest api",
    "streamlit",
    "linux",
    "terraform",
    "mongodb",
    "mysql",
    "postgresql",
    "data science",
    "data analysis",
    "power bi",
    "tableau",
    "excel",
    "opencv",
    "llm",
    "rag",
    "langchain",
    "transformers",
    "hugging face"
]


# -------------------- DATABASE --------------------

def create_database():

    connection = sqlite3.connect("careertwin.db")

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            match_score REAL,
            skills TEXT,
            missing_skills TEXT
        )
    """)

    connection.commit()
    connection.close()


def save_analysis(score, skills, missing_skills):

    connection = sqlite3.connect("careertwin.db")

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO analysis
        (date, match_score, skills, missing_skills)
        VALUES (?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        score,
        ", ".join(skills),
        ", ".join(missing_skills)
    ))

    connection.commit()
    connection.close()


create_database()


# -------------------- PDF TEXT EXTRACTION --------------------

def extract_pdf_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# -------------------- SKILL EXTRACTION --------------------

def extract_skills(text):

    text = text.lower()

    found_skills = []

    for skill in SKILLS:

        if skill in text:
            found_skills.append(skill)

    return sorted(set(found_skills))


# -------------------- AI MATCHING --------------------

def calculate_match_score(resume_text, job_description):

    resume_embedding = model.encode([resume_text])

    job_embedding = model.encode([job_description])

    similarity = cosine_similarity(
        resume_embedding,
        job_embedding
    )[0][0]

    score = similarity * 100

    return round(score, 2)


# -------------------- CAREER LEVEL --------------------

def get_career_level(score):

    if score >= 85:

        return (
            "Excellent Match",
            "You are strongly aligned with this role."
        )

    elif score >= 70:

        return (
            "Good Match",
            "You have a good foundation. Improve the missing skills."
        )

    elif score >= 50:

        return (
            "Moderate Match",
            "You need additional preparation."
        )

    else:

        return (
            "Low Match",
            "There are significant skill gaps."
        )


# -------------------- ROADMAP --------------------

def generate_roadmap(missing_skills):

    roadmap = []

    for skill in missing_skills[:6]:

        roadmap.append(
            f"Learn {skill.title()} and build a small practical project."
        )

    return roadmap


# -------------------- INTERVIEW QUESTIONS --------------------

def generate_questions(skills):

    questions = []

    if "python" in skills:

        questions.append(
            "Explain the difference between a list, tuple and dictionary in Python."
        )

    if "machine learning" in skills:

        questions.append(
            "Explain supervised and unsupervised learning."
        )

    if "deep learning" in skills:

        questions.append(
            "What is a neural network?"
        )

    if "sql" in skills:

        questions.append(
            "Explain JOIN operations in SQL."
        )

    if "docker" in skills:

        questions.append(
            "What is Docker and why is it useful?"
        )

    if "aws" in skills:

        questions.append(
            "Explain commonly used AWS services."
        )

    if "nlp" in skills:

        questions.append(
            "What is Natural Language Processing?"
        )

    questions.append(
        "Explain one of your projects and the challenges you faced."
    )

    questions.append(
        "Why are you interested in this role?"
    )

    return questions[:8]


# ============================================================
# USER INTERFACE
# ============================================================

st.title("🧠 CareerTwin AI")

st.subheader(
    "AI-Powered Personalized Career Intelligence System"
)

st.write(
    "Upload your resume and enter a target job description "
    "to analyze your career readiness."
)


# -------------------- SIDEBAR --------------------

st.sidebar.title("🎯 CareerTwin AI")

st.sidebar.info(
    """
    CareerTwin AI analyzes your resume,
    compares it with a target job,
    identifies skill gaps and creates
    a personalized roadmap.
    """
)

st.sidebar.markdown("---")

st.sidebar.write("Technology Stack")

st.sidebar.write(
    "🐍 Python\n\n"
    "🤖 Machine Learning\n\n"
    "🧠 NLP\n\n"
    "🔤 Sentence Transformers\n\n"
    "🗃️ SQLite\n\n"
    "🌐 Streamlit"
)


# -------------------- INPUT --------------------

st.header("📥 Enter Your Information")

resume_file = st.file_uploader(
    "📄 Upload your Resume",
    type=["pdf"]
)

job_description = st.text_area(
    "💼 Paste Target Job Description",
    height=250,
    placeholder="Paste the job description here..."
)


# -------------------- ANALYSIS --------------------

if st.button(
    "🚀 Analyze My Career",
    use_container_width=True
):

    if resume_file is None:

        st.error("Please upload your resume PDF.")
        st.stop()

    if not job_description.strip():

        st.error("Please enter a job description.")
        st.stop()

    with st.spinner("Reading your resume..."):

        resume_text = extract_pdf_text(resume_file)

    if not resume_text.strip():

        st.error(
            "Could not extract text from this PDF."
        )

        st.stop()

    with st.spinner("AI is analyzing your profile..."):

        resume_skills = extract_skills(resume_text)

        job_skills = extract_skills(job_description)

        missing_skills = [
            skill
            for skill in job_skills
            if skill not in resume_skills
        ]

        match_score = calculate_match_score(
            resume_text,
            job_description
        )

    career_level, career_message = get_career_level(
        match_score
    )

    save_analysis(
        match_score,
        resume_skills,
        missing_skills
    )

    # ========================================================
    # RESULTS
    # ========================================================

    st.success(
        "Career analysis completed successfully! 🎉"
    )

    st.header("🎯 Career Readiness")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Job Match",
            f"{match_score}%"
        )

    with col2:

        st.metric(
            "Skills Found",
            len(resume_skills)
        )

    with col3:

        st.metric(
            "Skill Gaps",
            len(missing_skills)
        )

    st.progress(
        min(match_score / 100, 1.0)
    )

    st.info(
        f"**{career_level}**\n\n"
        f"{career_message}"
    )

    # ========================================================
    # YOUR SKILLS
    # ========================================================

    st.header("🛠️ Your Skills")

    if resume_skills:

        columns = st.columns(3)

        for index, skill in enumerate(resume_skills):

            with columns[index % 3]:

                st.success(
                    f"✅ {skill.title()}"
                )

    else:

        st.warning(
            "No recognized technical skills detected."
        )

    # ========================================================
    # JOB SKILLS
    # ========================================================

    st.header("💼 Job Requirements")

    if job_skills:

        columns = st.columns(3)

        for index, skill in enumerate(job_skills):

            with columns[index % 3]:

                st.write(
                    f"📌 {skill.title()}"
                )

    # ========================================================
    # SKILL GAPS
    # ========================================================

    st.header("⚠️ Skill Gaps")

    if missing_skills:

        for skill in missing_skills:

            st.error(
                f"❌ {skill.title()}"
            )

    else:

        st.success(
            "🎉 No major skill gaps detected!"
        )

    # ========================================================
    # ROADMAP
    # ========================================================

    st.header("📚 Personalized Learning Roadmap")

    roadmap = generate_roadmap(missing_skills)

    if roadmap:

        for index, item in enumerate(
            roadmap,
            start=1
        ):

            st.write(
                f"### Step {index}"
            )

            st.write(
                f"➡️ {item}"
            )

    else:

        st.success(
            "Focus on interview preparation and practical projects."
        )

    # ========================================================
    # INTERVIEW QUESTIONS
    # ========================================================

    st.header("🎤 Interview Preparation")

    questions = generate_questions(resume_skills)

    for index, question in enumerate(
        questions,
        start=1
    ):

        st.write(
            f"**Q{index}. {question}**"
        )

    # ========================================================
    # CAREER RECOMMENDATION
    # ========================================================

    st.header("🤖 CareerTwin Recommendation")

    if match_score >= 85:

        st.success(
            """
            ⭐ Strong candidate profile.

            Focus on advanced projects,
            interview preparation and practical experience.
            """
        )

    elif match_score >= 70:

        st.info(
            """
            👍 Good candidate profile.

            Close the identified skill gaps
            and strengthen your projects.
            """
        )

    elif match_score >= 50:

        st.warning(
            """
            📚 Moderate candidate profile.

            Follow the learning roadmap and
            build practical projects.
            """
        )

    else:

        st.error(
            """
            🚨 Several important skill gaps were detected.

            Follow the roadmap before targeting this role.
            """
        )

    # ========================================================
    # CAREER TWIN
    # ========================================================

    st.header("🧠 Your CareerTwin")

    st.write(
        f"""
        **Career Readiness:** {match_score}%

        **Skills Detected:** {len(resume_skills)}

        **Job Skills:** {len(job_skills)}

        **Skill Gaps:** {len(missing_skills)}

        **Profile Status:** {career_level}
        """
    )


# -------------------- FOOTER --------------------

st.markdown("---")

st.caption(
    "🧠 CareerTwin AI | AI-Powered Career Intelligence"
)