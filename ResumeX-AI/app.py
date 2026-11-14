import streamlit as st
import fitz
import io
import re
import pandas as pd
import plotly.express as px

from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==================================================
# 1. PAGE SETTINGS
# ==================================================

st.set_page_config(
    page_title="ResumeX AI",
    page_icon="🎯",
    layout="wide"
)

# ==================================================
# 2. PREMIUM DESIGN
# ==================================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#F5F3FF,#EFF6FF,#FDF2F8);
}
h1 {
    color: #312E81;
    font-weight: 850;
}
h2,h3 {
    color: #4338CA;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#1E1B4B,#4338CA,#7E22CE);
}
[data-testid="stSidebar"] * {
    color: white !important;
}
[data-testid="stMetric"] {
    background: linear-gradient(135deg,#FFFFFF,#EDE9FE);
    padding: 18px;
    border-radius: 16px;
    border: 1px solid #DDD6FE;
    box-shadow: 0 5px 15px #4338CA15;
}
[data-testid="stMetricValue"] {
    color: #4F46E5;
    font-weight: 800;
}
.stButton>button {
    background: linear-gradient(90deg,#4F46E5,#9333EA);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: bold;
    min-height: 42px;
}
.stButton>button:hover {
    background: #312E81;
    color: white;
}
[data-testid="stFileUploader"] {
    background: white;
    padding: 12px;
    border-radius: 12px;
    border: 1px dashed #8B5CF6;
}
[data-testid="stChatMessage"] {
    border-radius: 14px;
}
div[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg,#6366F1,#A855F7,#EC4899);
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# 3. SKILL DATABASE
# ==================================================

SKILLS = [
    "python", "java", "javascript", "react",
    "node.js", "sql", "mysql", "mongodb",
    "html", "css", "machine learning",
    "deep learning", "artificial intelligence",
    "data analysis", "data science",
    "power bi", "tableau", "excel",
    "statistics", "communication", "leadership",
    "project management", "teamwork",
    "problem solving", "cloud computing",
    "aws", "azure", "docker", "kubernetes",
    "git", "linux", "rest api", "agile",
    "marketing", "sales", "financial analysis",
    "accounting", "customer service",
    "cybersecurity", "testing", "automation",
    "natural language processing",
    "time management", "critical thinking",
    "business analysis", "data visualization",
    "c++", "c#", "php", "figma", "seo"
]

# ==================================================
# 4. SESSION STATE
# ==================================================

for key, value in {
    "result": None,
    "resume": "",
    "job": "",
    "history": []
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ==================================================
# 5. RESUME EXTRACTION
# ==================================================

def extract_resume(file):

    content = file.getvalue()

    if file.name.lower().endswith(".pdf"):

        pdf = fitz.open(
            stream=content,
            filetype="pdf"
        )

        text = "\n".join(
            page.get_text() for page in pdf
        )

        pdf.close()
        return text

    if file.name.lower().endswith(".docx"):

        doc = Document(io.BytesIO(content))

        parts = [p.text for p in doc.paragraphs]

        for table in doc.tables:
            for row in table.rows:
                parts.append(
                    " ".join(cell.text for cell in row.cells)
                )

        return "\n".join(parts)

    return ""

# ==================================================
# 6. SKILL DETECTION
# ==================================================

def has_skill(text, skill):

    pattern = (
        r"(?<![a-z0-9])"
        + re.escape(skill)
        + r"(?![a-z0-9])"
    )

    return bool(re.search(pattern, text.lower()))

# ==================================================
# 7. ANALYSIS ENGINE
# ==================================================

def analyse_resume(resume, job):

    resume_lower = resume.lower()
    job_lower = job.lower()

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000
        )

        matrix = vectorizer.fit_transform([
            resume_lower,
            job_lower
        ])

        similarity = cosine_similarity(
            matrix[0:1],
            matrix[1:2]
        )[0][0]

    except ValueError:
        similarity = 0

    required = [
        s for s in SKILLS
        if has_skill(job_lower, s)
    ]

    matched = [
        s for s in required
        if has_skill(resume_lower, s)
    ]

    missing = [
        s for s in required
        if s not in matched
    ]

    skill_score = (
        len(matched) / len(required)
        if required else similarity
    )

    score = round(
        (similarity * 0.70 + skill_score * 0.30) * 100
    )

    flags = []

    sections = {
        "Education": ["education", "academic"],
        "Experience": ["experience", "employment"],
        "Skills": ["skills"],
        "Projects": ["projects"]
    }

    for section, words in sections.items():
        if not any(word in resume_lower for word in words):
            flags.append(
                f"Check your {section} section."
            )

    if not re.search(r"\d+%|\d+\+|\$[\d,]+", resume):
        flags.append(
            "Consider adding genuine measurable achievements."
        )

    if len(resume.split()) < 100:
        flags.append("Your resume appears short.")

    if len(resume.split()) > 1200:
        flags.append("Consider shortening irrelevant content.")

    tips = []

    if missing:
        tips.append(
            "Review missing skills and add them only if you genuinely possess them."
        )

    tips.extend([
        "Create a job-specific professional summary.",
        "Use strong action verbs in experience descriptions.",
        "Add measurable achievements where accurate.",
        "Highlight relevant projects and technical skills.",
        "Keep formatting clear and consistent."
    ])

    return {
        "score": min(max(score, 0), 100),
        "similarity": round(similarity * 100),
        "skill_score": round(skill_score * 100),
        "required": required,
        "matched": matched,
        "missing": missing,
        "flags": flags,
        "tips": tips,
        "words": len(resume.split())
    }

# ==================================================
# 8. CHATBOT
# ==================================================

def chatbot_reply(question):

    r = st.session_state.result
    q = question.lower()

    if any(x in q for x in ["score", "rating", "match", "percentage"]):
        return (
            f"Your job match score is {r['score']}/100.\n\n"
            f"Text similarity: {r['similarity']}%.\n\n"
            f"Skills match: {r['skill_score']}%.\n\n"
            "This is a heuristic score, not a verified ATS result."
        )

    elif any(x in q for x in ["missing", "gap", "skill"]):
        return (
            "Missing or unverified skills:\n\n"
            + (
                "\n".join("- " + s.title() for s in r["missing"])
                if r["missing"]
                else "No missing skills detected in our skill dictionary."
            )
        )

    elif any(x in q for x in ["improve", "better", "change"]):
        return (
            "Your improvement roadmap:\n\n"
            + "\n".join(
                f"{i+1}. {tip}"
                for i, tip in enumerate(r["tips"])
            )
        )

    elif any(x in q for x in ["interview", "question", "prepare"]):

        skills = r["missing"][:5] or r["matched"][:5]

        if not skills:
            skills = ["professional experience", "projects"]

        return (
            "Interview questions to practise:\n\n"
            + "\n".join(
                f"- Explain your practical experience with {s}."
                for s in skills
            )
        )

    elif any(x in q for x in ["recruiter", "reject", "risk", "red flag"]):
        return (
            "Potential resume concerns:\n\n"
            + (
                "\n".join("- " + f for f in r["flags"])
                if r["flags"]
                else "No basic structural concerns detected."
            )
        )

    elif any(x in q for x in ["hello", "hi", "hey"]):
        return (
            "Hello! 👋 Welcome to ResumeX Career Assistant. "
            "Ask me about your resume score, skills, improvements "
            "or interview preparation."
        )

    else:
        return (
            "I can help with your resume score, missing skills, "
            "recruiter feedback and interview preparation. "
            "Try asking: How can I improve my resume?"
        )

# ==================================================
# 9. SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown("""
    <div style="text-align:center;padding:15px;">
        <div style="font-size:45px;">🎯</div>
        <h1 style="color:white;">ResumeX AI</h1>
        <p style="color:#DDD6FE;">
            Career Intelligence Platform
        </p>
        <p style="color:#C4B5FD;font-size:11px;">
            SMARTER CAREERS. BETTER OPPORTUNITIES.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Home",
            "📊 Dashboard",
            "🧠 Skills Analysis",
            "⚠️ Recruiter Feedback",
            "🤖 AI Chatbot",
            "🚀 Project Showcase"
        ]
    )

    st.divider()

    st.markdown("""
    <div style="text-align:center;padding:12px;">
        <h3 style="color:white;">✨ ResumeX Pro</h3>
        <p style="color:#DDD6FE;font-size:12px;">
            Your career starts here.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==================================================
# 10. HOME PAGE
# ==================================================

if page == "🏠 Home":

    st.markdown("""
    <div style="
        background:linear-gradient(120deg,#312E81,#6D28D9,#9333EA);
        padding:35px;
        border-radius:22px;
        margin-bottom:25px;
    ">
        <p style="color:#DDD6FE;letter-spacing:2px;">
            YOUR PERSONAL CAREER INTELLIGENCE
        </p>
        <h1 style="color:white;font-size:38px;">
            Welcome to ResumeX AI ✨
        </h1>
        <p style="color:#EDE9FE;font-size:17px;">
            Unlock your potential. Discover your skills.
            Build a resume that gets noticed.
        </p>
        <p style="color:#DDD6FE;">
            🎯 Resume Insights | 📊 Job Matching | 🚀 Career Growth
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Analyse Your Resume")

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader(
            "📄 Upload Resume",
            type=["pdf", "docx"]
        )

    with col2:
        st.markdown("### 💼 Target Job")
        st.write("Paste your target job description below.")

    job = st.text_area(
        "Job Description",
        height=180,
        placeholder="Paste job requirements and responsibilities..."
    )

    if st.button("🚀 Analyse My Resume", use_container_width=True):

        if uploaded_file is None:
            st.error("Please upload your resume.")

        elif not job.strip():
            st.error("Please enter the job description.")

        else:
            try:
                if uploaded_file.size > 10 * 1024 * 1024:
                    st.error("Maximum file size is 10 MB.")
                    st.stop()

                with st.spinner("Analysing your resume..."):

                    resume = extract_resume(uploaded_file)

                    if len(resume.strip()) < 30:
                        st.error("Could not extract readable resume text.")
                        st.stop()

                    result = analyse_resume(resume, job)

                    st.session_state.resume = resume
                    st.session_state.job = job
                    st.session_state.result = result
                    st.session_state.history = []

                st.success("Resume analysis completed!")
                st.balloons()

                a, b, c = st.columns(3)

                a.metric("🎯 Match Score", f"{result['score']}/100")
                b.metric("✅ Matched Skills", len(result["matched"]))
                c.metric("⚠️ Missing Skills", len(result["missing"]))

                st.progress(result["score"] / 100)

                st.info("Open Dashboard for your complete analysis.")

            except Exception as e:
                st.error(f"Error: {e}")

# ==================================================
# 11. DASHBOARD
# ==================================================

elif page == "📊 Dashboard":

    if st.session_state.result is None:
        st.warning("Please analyse your resume first.")

    else:

        r = st.session_state.result

        st.title("📊 Career Analytics Dashboard")
        st.write("Your resume performance at a glance.")

        a, b, c, d = st.columns(4)

        a.metric("🎯 Match Score", f"{r['score']}/100")
        b.metric("🧩 Matched Skills", len(r["matched"]))
        c.metric("⚠️ Missing Skills", len(r["missing"]))
        d.metric("📄 Similarity", f"{r['similarity']}%")

        st.markdown("### Overall Job Match")
        st.progress(r["score"] / 100)

        st.caption("Heuristic matching indicator, not an actual ATS score.")

        st.divider()

        left, right = st.columns(2)

        with left:

            st.markdown("### 🎯 Skills Coverage")

            if r["required"]:

                df = pd.DataFrame({
                    "Status": ["Matched", "Missing"],
                    "Count": [len(r["matched"]), len(r["missing"])]
                })

                fig = px.pie(
                    df,
                    names="Status",
                    values="Count",
                    hole=0.65,
                    color="Status",
                    color_discrete_map={
                        "Matched": "#10B981",
                        "Missing": "#F59E0B"
                    }
                )

                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#312E81"
                )

                st.plotly_chart(fig, use_container_width=True)

            else:
                st.info("No recognised skills in the job description.")

        with right:

            st.markdown("### 📈 Performance Breakdown")

            df2 = pd.DataFrame({
                "Metric": ["Text Similarity", "Skills Match", "Overall Match"],
                "Score": [r["similarity"], r["skill_score"], r["score"]]
            })

            fig2 = px.bar(
                df2,
                x="Metric",
                y="Score",
                color="Metric",
                text="Score",
                range_y=[0, 110],
                color_discrete_sequence=["#6366F1", "#06B6D4", "#A855F7"]
            )

            fig2.update_traces(
                texttemplate="%{text}%",
                textposition="outside"
            )

            fig2.update_layout(
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#312E81"
            )

            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### ✨ Improvement Suggestions")

        for tip in r["tips"]:
            st.info(tip)

        report = f"""
RESUMEX AI CAREER REPORT

Match Score: {r['score']}/100
Text Similarity: {r['similarity']}%
Skills Match: {r['skill_score']}%
Resume Word Count: {r['words']}

MATCHED SKILLS:
{', '.join(r['matched'])}

MISSING OR UNVERIFIED SKILLS:
{', '.join(r['missing'])}

POTENTIAL CONCERNS:
{chr(10).join(r['flags'])}

RECOMMENDATIONS:
{chr(10).join(r['tips'])}

Note: Heuristic analysis, not an actual ATS or hiring prediction.
"""

        st.download_button(
            "📥 Download Resume Report",
            data=report,
            file_name="ResumeX_Report.txt",
            mime="text/plain",
            use_container_width=True
        )

# ==================================================
# 12. SKILLS ANALYSIS
# ==================================================

elif page == "🧠 Skills Analysis":

    if st.session_state.result is None:
        st.warning("Analyse your resume first.")

    else:

        r = st.session_state.result

        st.title("🧠 Skills Intelligence Center")

        left, right = st.columns(2)

        with left:

            st.markdown("### ✅ Matching Skills")

            if r["matched"]:
                for skill in r["matched"]:
                    st.success("✓ " + skill.title())
            else:
                st.info("No matching skills found.")

        with right:

            st.markdown("### ⚠️ Missing Skills")

            if r["missing"]:
                for skill in r["missing"]:
                    st.warning(skill.title())
            else:
                st.success("No missing skills detected.")

        st.divider()

        st.markdown("### 📚 Skill Development Plan")

        for i, skill in enumerate(r["missing"], 1):
            st.write(f"**{i}. {skill.title()}**")
            st.write(
                "Learn this skill if it is relevant to your career goals. "
                "Add it to your resume only when you have genuine evidence."
            )

        st.caption(
            "A missing keyword does not prove that you lack the skill."
        )

# ==================================================
# 13. RECRUITER FEEDBACK
# ==================================================

elif page == "⚠️ Recruiter Feedback":

    if st.session_state.result is None:
        st.warning("Analyse your resume first.")

    else:

        r = st.session_state.result

        st.title("⚠️ Recruiter Feedback Center")

        st.markdown("### Potential Resume Concerns")

        if r["flags"]:
            for flag in r["flags"]:
                st.warning(flag)
        else:
            st.success("No basic structural issues detected.")

        st.divider()

        st.markdown("### 🚀 Improvement Roadmap")

        for i, tip in enumerate(r["tips"], 1):

            with st.expander(f"Improvement {i}: {tip}"):
                st.write(
                    "Review this suggestion and apply it "
                    "using accurate information about your experience."
                )

        st.info(
            "These are potential presentation issues, "
            "not predictions of actual recruiter rejection."
        )

# ==================================================
# 14. CHATBOT
# ==================================================

elif page == "🤖 AI Chatbot":

    if st.session_state.result is None:

        st.warning("Analyse your resume before using the chatbot.")

    else:

        st.title("🤖 ResumeX Career Assistant")

        st.markdown("""
        <div style="
            background:linear-gradient(120deg,#312E81,#9333EA);
            padding:22px;
            border-radius:18px;
            color:white;
        ">
            <h3 style="color:white;">
                Hello! Welcome to ResumeX Assistant 👋
            </h3>
            <p style="color:#EDE9FE;">
                Ask about your resume, skills, job matching
                and interview preparation.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 💬 Quick Questions")

        questions = [
            "What is my resume score?","Which skills are missing?",
            "How can I improve my resume?",
            "What interview questions should I prepare?",
            "What are my recruiter red flags?"
        ]

        for i, question in enumerate(questions):

            if st.button(
                question,
                key=f"quick_{i}",
                use_container_width=True
            ):

                answer = chatbot_reply(question)

                st.session_state.history.extend([
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ])

                st.rerun()

        st.divider()

        st.markdown("### 💬 Your Conversation")

        for msg in st.session_state.history:

            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        question = st.chat_input(
            "Ask your career assistant..."
        )

        if question:

            answer = chatbot_reply(question)

            st.session_state.history.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ])

            st.rerun()

        if st.button("🗑️ Clear Chat History"):

            st.session_state.history = []
            st.rerun()

        st.caption(
            "ResumeX Career Assistant | Free rule-based chatbot"
        )


# ==================================================
# 15. PROJECT SHOWCASE
# ==================================================

elif page == "🚀 Project Showcase":

    st.title("🚀 ResumeX AI")

    st.markdown("""
<div style="background:linear-gradient(120deg,#1E1B4B,#6D28D9,#9333EA);
padding:35px;border-radius:22px;color:white;">

<p style="color:#DDD6FE;letter-spacing:2px;">
CAREER INTELLIGENCE PLATFORM
</p>

<h1 style="color:white;font-size:40px;">
ResumeX AI
</h1>

<h3 style="color:#E9D5FF;">
Smarter Resume Analysis. Smarter Career Decisions.
</h3>

<p style="color:#EDE9FE;">
Discover skill gaps, understand job compatibility
and improve your resume quality.
</p>

</div>
""", unsafe_allow_html=True)

    st.divider()

    st.markdown("## 💡 Problem We Solve")

    st.write(
        "Job seekers often struggle to understand "
        "how their resume aligns with specific job descriptions."
    )

    st.markdown("## ✨ Key Features")

    c1, c2, c3 = st.columns(3)

    c1.info("📄 Resume Analysis")
    c2.success("🎯 Job Matching")
    c3.warning("🧩 Skill Detection")

    c1, c2, c3 = st.columns(3)

    c1.info("📊 Interactive Dashboard")
    c2.success("🤖 Career Chatbot")
    c3.warning("🚀 Improvement Plan")

    st.divider()

    st.markdown("## 🛠️ Technology Stack")

    st.write(
        "Python | Streamlit | Scikit-learn | Plotly | NLP"
    )

    st.markdown("## 🎯 Project Vision")

    st.write(
        "Making career preparation more accessible, "
        "transparent and data-driven."
    )

    st.success("ResumeX AI | Career Technology Project")


# ==================================================
# 16. FOOTER
# ==================================================

st.divider()

st.markdown("""
<div style="
    text-align:center;
    padding:20px;
    color:#6D28D9;
">
    <h3>💜 ResumeX AI</h3>
    <p>Smarter Careers. Better Opportunities.</p>
    <small>Your Career Intelligence Platform</small>
</div>
""", unsafe_allow_html=True)