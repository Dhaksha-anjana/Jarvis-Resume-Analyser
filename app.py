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
    advanced_ats_similarity,
    keyword_density_analysis,
    calculate_competitiveness,
    extract_basic_info,
    generate_self_intro
)
from llm_utils import validate_api_key, generate_interview_questions, get_chat_response
# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Jarvis Resume AI",
    page_icon="🤖",
    layout="wide"
)
# ---------------- 🔥 MODERN ANIMATED BACKGROUND ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(-45deg, #09090b, #1e1b4b, #111827, #020617);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    color: #f8fafc;
}
@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
[data-testid="stAppViewContainer"] {background: transparent;}
[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)
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

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("⚙️ LLM Integration")
    st.markdown("Unlock advanced AI capabilities:")
    api_key = st.text_input("Gemini API Key", type="password", help="Get a free key from Google AI Studio")
    if api_key:
        if validate_api_key(api_key):
            st.success("API Key Validated!")
        else:
            st.error("Invalid API Key!")
    st.caption("To use Chatbot & Interview Predictor.")
# ---------------- INPUTS ----------------
resume_file = st.file_uploader("📄 Upload Resume (PDF)", type="pdf")
ROLE_JDS = {
    "Custom Job Description": "",
    "Data Scientist": "Looking for a Data Scientist. Skills required: python, machine learning, deep learning, nlp, sql, tensorflow, keras.",
    "Web Developer": "Seeking a Web Developer. Skills required: react, node, django, flask, python, mysql, html, css.",
    "Software Engineer": "Software Engineer needed. Skills required: python, java, c++, sql, algorithms, data structures.",
    "Machine Learning Engineer": "Machine Learning Engineer. Required: python, machine learning, deep learning, nlp, tensorflow, keras.",
    "Android Developer": "Looking for Android Developer with java, kotlin, android sdk, mobile development, sqlite."
}
selected_role = st.selectbox("🎯 Select Target Role", list(ROLE_JDS.keys()))
if selected_role == "Custom Job Description":
    jd_text = st.text_area("🧾 Paste Job Description (Optional)")
else:
    jd_text = ROLE_JDS[selected_role]
    st.info(f"Using predefined keywords for {selected_role}: {jd_text}")
# ---------------- PROCESS ----------------
if resume_file:
    resume_text = pdfminer.high_level.extract_text(resume_file)
    skills = extract_skills(resume_text)
    score = resume_score(resume_text)
    ats_data = None
    keyword_density = []
    if jd_text:
        ats_data = advanced_ats_similarity(resume_text, jd_text)
        keyword_density = keyword_density_analysis(resume_text, jd_text)
    info = extract_basic_info(resume_text, selected_role)
    info['score'] = score
    if ats_data is not None:
        info['ats_match'] = ats_data['total']
    
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
        
        if ats_data is not None:
            st.subheader("📊 Keyword Density Analysis")
            if keyword_density:
                for item in keyword_density:
                    status = item['status']
                    label = f"**{item['skill']}** → {status} (Count: {item['count']})"
                    
                    if "Good" in status:
                        st.success(label)
                    elif "Missing" in status:
                        skill_url = item['skill'].lower().replace(' ', '+').replace('#', '%23').replace('++', '%2B%2B')
                        gfg_link = f"[GeeksforGeeks Course 📚](https://www.geeksforgeeks.org/courses?search={skill_url})"
                        st.error(f"**{item['skill']}** → {status} | {gfg_link}")
                    elif "Overused" in status:
                        st.warning(label)
            else:
                st.info("No specific keywords detected in JD.")
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
        if ats_data is not None:
            st.subheader("🤖 Advanced ATS Match")
            
            ats_total = ats_data["total"]
            fig_ats = go.Figure(go.Pie(
                values=[ats_total, max(0, 100-ats_total)],
                hole=0.75,
                marker_colors=["#38bdf8", "#1f2937"],
                textinfo="none"
            ))
            fig_ats.update_layout(
                annotations=[dict(text=f"{ats_total}%", x=0.5, y=0.5, font_size=24, showarrow=False)],
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                margin=dict(t=10, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_ats, use_container_width=True)
            # Show breakdown
            st.markdown("### 📊 Scoring Breakdown")
            st.progress(ats_data["skills_match"]/40, text=f"Skills ({ats_data['skills_match']}/40%)")
            st.progress(ats_data["keyword_match"]/20, text=f"Keywords ({ats_data['keyword_match']}/20%)")
            st.progress(ats_data["action_metrics"]/20, text=f"Action Verbs/Metrics ({ats_data['action_metrics']}/20%)")
            st.progress(ats_data["section_match"]/20, text=f"Formatting ({ats_data['section_match']}/20%)")
        st.subheader("🧠 AI Introduction")
        st.success(intro_text)
        
        st.download_button(
            label="Download Introduction (.txt)",
            data=intro_text,
            file_name="jarvis_introduction.txt",
            mime="text/plain",
            type="primary"
        )
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
        
        comp_score = calculate_competitiveness(score, selected_role)
        st.info(f"🏆 {comp_score}")
        
        if ats_data is not None:
            st.write(f"**ATS Match:** {ats_data['total']}%")
            
            missing_skills = [item['skill'] for item in keyword_density if "Missing" in item['status']]
            
            if missing_skills:
                st.markdown("---")
                with st.expander("🎯 Predict Interview Questions"):
                    if not api_key:
                        st.warning("Please enter a valid Gemini API Key in the sidebar.")
                    else:
                        st.info("I will generate rigorous questions to test your knowledge gaps.")
                        if st.button("Generate Technical Questions"):
                            with st.spinner("Analyzing skill gaps..."):
                                questions = generate_interview_questions(missing_skills, jd_text, api_key)
                                st.markdown(questions)
    st.markdown("---")
    st.markdown("### 📺 Recommended Resources for Resume Creation")
    colY1, colY2, colY3 = st.columns(3)
    with colY1:
        st.video("https://www.youtube.com/watch?v=Tt08KmFfIYQ")
    with colY2:
        st.video("https://www.youtube.com/watch?v=NELLXWFRUU0")
    with colY3:
        st.video("https://www.youtube.com/watch?v=XlXP_C2nsXI")

    # ---------------- CHATBOT ----------------
    st.markdown("---")
    st.header("💬 Chat with Jarvis (Resume Q&A)")
    
    if not api_key:
        st.warning("Please input your Gemini API Key in the sidebar to activate the Chatbot.")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        st.caption("💡 Try asking these quick questions:")
        q1, q2, q3 = st.columns(3)
        sample_prompt = None
        
        if q1.button("🎯 Summarize Strongest Points"):
            sample_prompt = "Give me a three-sentence summary of this candidate's strongest points."
        if q2.button("📊 Assess Seniority Level"):
            sample_prompt = "Based on this resume, is this candidate junior, mid-level, or senior?"
        if q3.button("🔌 Check API Experience"):
            sample_prompt = "Does this candidate have experience connecting APIs?"

        # Display history
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        # Chat interaction
        prompt = st.chat_input("Ask Jarvis anything specific about this resume...")
        final_prompt = prompt or sample_prompt
        
        if final_prompt:
            st.session_state.messages.append({"role": "user", "content": final_prompt})
            
            # If it's a typed prompt, display immediately (sample prompts will display natively on rerun)
            if not sample_prompt:
                st.chat_message("user").write(final_prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    chat_hist = st.session_state.messages[:-1]
                    response_text = get_chat_response(final_prompt, resume_text, chat_hist, api_key)
                    st.write(response_text)
            
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            
            if sample_prompt:
                st.rerun()