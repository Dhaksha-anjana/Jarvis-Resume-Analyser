import streamlit as st
import pdfminer.high_level
import os
import json
import plotly.graph_objects as go
import plotly.express as px
from streamlit_lottie import st_lottie

from resume_utils import (
    extract_skills,
    resume_score,
    ats_similarity,
    extract_basic_info,
    generate_self_intro
)

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Jarvis Resume AI",
    page_icon="🤖",
    layout="wide"
)

# ---------------- 🔥 IRON MAN HUD BACKGROUND ----------------
st.markdown("""
<style>
.stApp {background: transparent;}
[data-testid="stAppViewContainer"] {background: transparent;}
[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 2rem;}

#bg-lottie {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: -1;
    pointer-events: none;
}

#bg-lottie lottie-player {
    width: 100%;
    height: 100%;
    opacity: 0.25;
}
</style>
""", unsafe_allow_html=True)

st.components.v1.html("""
<div id="bg-lottie">
 <lottie-player
    src="https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json"
    background="transparent"
    speed="1"
    loop
    autoplay>
</lottie-player>
</div>

<script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
""", height=0)



# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOTTIE_DIR = os.path.join(BASE_DIR, "lottie")
INTRO_PATH = os.path.join(BASE_DIR, "intro.txt")

# ---------------- LOAD LOTTIE ----------------
def load_lottie(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

jarvis_lottie = load_lottie(os.path.join(LOTTIE_DIR, "jarvis.json"))
voice_lottie = load_lottie(os.path.join(LOTTIE_DIR, "voice.json"))

# ---------------- VOICE OUTPUT ----------------
def speak_js(text):
    st.components.v1.html(f"""
    <script>
    var msg = new SpeechSynthesisUtterance("{text}");
    msg.lang = "en-IN";
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(msg);
    </script>
    """, height=0)

# ---------------- HEADER ----------------
colA, colB = st.columns([1, 3])

with colA:
    st_lottie(jarvis_lottie, height=160, key="jarvis")

with colB:
    st.title(" Jarvis Resume Analytics Dashboard")
    st.caption("AI Powered ATS + Resume Intelligence")

# ---------------- INPUTS ----------------
resume_file = st.file_uploader("📄 Upload Resume (PDF)", type="pdf")
jd_text = st.text_area("🧾 Paste Job Description (Optional)")

# ---------------- PROCESS ----------------
if resume_file:
    resume_text = pdfminer.high_level.extract_text(resume_file)

    skills = extract_skills(resume_text)
    score = resume_score(resume_text)

    ats = None
    if jd_text:
        ats = ats_similarity(resume_text, jd_text)

    info = extract_basic_info(resume_text)
    intro_text = generate_self_intro(info)

    col1, col2, col3 = st.columns([1.2, 1.5, 1.2])

    # -------- COLUMN 1 : SKILLS --------
    with col1:
        st.subheader("🛠️ Skills")

        if skills:
            for s in skills[:8]:
                st.progress(1.0, text=s.upper())
        else:
            st.warning("No skills detected")

        st.subheader("📈 Skill Distribution")

        fig_skills = px.pie(
            names=skills if skills else ["No Skills"],
            values=[1]*len(skills) if skills else [1],
            hole=0.6
        )
        fig_skills.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_skills, use_container_width=True)

    # -------- COLUMN 2 : SCORE + ATS + VOICE --------
    with col2:
        st.subheader("🎯 Resume Score")

        fig_score = go.Figure(go.Pie(
            values=[score, 100-score],
            hole=0.75,
            marker_colors=["#22c55e", "#1f2937"],
            textinfo="none"
        ))

        fig_score.update_layout(
            annotations=[dict(text=f"{score}%", x=0.5, y=0.5, font_size=28, showarrow=False)],
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            margin=dict(t=10, b=10, l=10, r=10)
        )

        st.plotly_chart(fig_score, use_container_width=True)

        if ats is not None:
            st.subheader("🤖 ATS Match")

            fig_ats = go.Figure(go.Pie(
                values=[ats, 100-ats],
                hole=0.75,
                marker_colors=["#38bdf8", "#1f2937"],
                textinfo="none"
            ))

            fig_ats.update_layout(
                annotations=[dict(text=f"{ats}%", x=0.5, y=0.5, font_size=24, showarrow=False)],
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                margin=dict(t=10, b=10, l=10, r=10)
            )

            st.plotly_chart(fig_ats, use_container_width=True)

        st.subheader("🧠 AI Introduction")
        st.success(intro_text)

        st_lottie(voice_lottie, height=90, key="voice")
        speak_js(intro_text)

        with open(INTRO_PATH, "w", encoding="utf-8") as f:
            f.write(intro_text)

        st.toast("Introduction saved for Jarvis 🎤", icon="✅")

    # -------- COLUMN 3 : SECTION ANALYSIS --------
    with col3:
        st.subheader("📊 Resume Sections")

        sections = ["Objective", "Skills", "Projects", "Experience", "Education"]
        section_scores = [20 if sec.lower() in resume_text.lower() else 5 for sec in sections]

        fig_bar = px.bar(
            x=section_scores,
            y=sections,
            orientation="h",
            color=section_scores,
            color_continuous_scale="Turbo"
        )

        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            margin=dict(t=10, b=10, l=10, r=10),
            coloraxis_showscale=False
        )

        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("📌 Quick Stats")
        st.write(f"**Total Skills:** {len(skills)}")
        st.write(f"**Resume Score:** {score}/100")
        if ats is not None:
            st.write(f"**ATS Match:** {ats}%")
