import streamlit as st
import pdfminer.high_level
import os
from resume_utils import (
    extract_skills,
    resume_score,
    ats_similarity,
    extract_basic_info,
    generate_self_intro
)

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Smart Resume Analyser",
    page_icon="🤖",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Smart Resume Analyser + Jarvis")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTRO_PATH = os.path.join(BASE_DIR, "intro.txt")

# ---------------- UI ----------------
resume_file = st.file_uploader("📄 Upload Resume (PDF)", type="pdf")
jd_text = st.text_area("🧾 Paste Job Description (Optional)")

# ---------------- LOGIC ----------------
if resume_file:
    resume_text = pdfminer.high_level.extract_text(resume_file)

    st.subheader("🛠️ Extracted Skills")
    skills = extract_skills(resume_text)
    st.write(skills)

    st.subheader("📊 Resume Score")
    score = resume_score(resume_text)
    st.success(f"{score}/100")

    if jd_text:
        ats = ats_similarity(resume_text, jd_text)
        st.subheader("🤖 ATS Match")
        st.info(f"{ats}%")

    info = extract_basic_info(resume_text)
    intro_text = generate_self_intro(info)

    st.subheader("🗣️ AI Generated Self Introduction")
    st.success(intro_text)

    # SAVE FOR JARVIS
    with open(INTRO_PATH, "w", encoding="utf-8") as f:
        f.write(intro_text)

    st.toast("Introduction saved for Jarvis 🎤", icon="✅")
