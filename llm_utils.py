import google.generativeai as genai
import re

_CACHED_MODEL_NAME = None

def _get_best_model(api_key):
    global _CACHED_MODEL_NAME
    if _CACHED_MODEL_NAME:
        return _CACHED_MODEL_NAME
        
    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        available = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        
        for name in available:
            if "flash" in name:
                _CACHED_MODEL_NAME = name
                return name
                
        if available:
            _CACHED_MODEL_NAME = available[0]
            return _CACHED_MODEL_NAME
            
    except Exception as e:
        print(f"DEBUG: Model listing failed: {e}")
        pass
        
    return "gemini-1.5-flash"

def validate_api_key(api_key):
    if not api_key or len(api_key) < 10:
        return False
    try:
        genai.configure(api_key=api_key)
        # Attempt to list models - this triggers a real API credential check
        for _ in genai.list_models():
            break
        return True
    except Exception as e:
        print(f"DEBUG: Key validation failed: {e}")
        return False

def generate_interview_questions(missing_skills, jd_text, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(_get_best_model(api_key))
        
        prompt = (
            f"You are an expert technical recruiter interviewing a candidate for a role defined by this Job Description:\n{jd_text}\n\n"
            f"The candidate's resume shows a gap in these specific skills required by the job: {', '.join(missing_skills)}\n\n"
            "Please generate 5 specific, rigorous technical interview questions designed to test their potential capability or learnability regarding these EXACT missing skills. "
            "Address the questions directly to the candidate, and do not include the answers. Provide only the 5 questions formatted as a clear markdown list."
        )
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            return get_local_interview_questions(missing_skills)
        return f"⚠️ Error generating questions: {error_msg}"

def get_local_interview_questions(missing_skills):
    questions = "### 🎯 Technical Skill-Gap Analysis\n\n"
    questions += "I have analyzed your profile against the job requirements. Here are some key technical questions you should prepare for based on your identified skill gaps:\n\n"
    
    for skill in missing_skills[:5]:
        questions += f"- **{skill}**: Can you explain the core fundamentals of {skill} and how you would integrate it into a production-ready pipeline?\n"
        questions += f"- **{skill}**: What are the most common challenges you've faced when working with {skill}?\n"
        
    return questions

def get_chat_response(prompt, resume_text, chat_history, api_key, stream=False):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(_get_best_model(api_key))
        
        # Build chat context
        system_context = (
            "You are J.A.R.V.I.S., a highly advanced, exceptionally formal, and polite British AI assistant. "
            "Always address the user respectfully as 'Sir' or 'Madam'. "
            "Give extremely concise, direct, and immediate answers to minimize thinking time. Avoid unnecessary elaboration. "
            "You must answer user questions based heavily on the specific contents of this resume text:\n\n"
            f"=== RESUME CONTENT ===\n{resume_text}\n=====================\n\n"
            "If the resume doesn't specify something, clearly and swiftly state that."
        )
        
        messages = [{"role": "user", "parts": [system_context]}, {"role": "model", "parts": ["Understood, Sir. I am ready to assist you immediately."]}]
        
        # Feed previous chat history
        for msg in chat_history:
            role = "user" if msg["role"] == "user" else "model"
            messages.append({"role": role, "parts": [msg["content"]]})
            
        # Add the newest prompt
        messages.append({"role": "user", "parts": [prompt]})
        
        if stream:
            response = model.generate_content(messages, stream=True)
            return response
        else:
            response = model.generate_content(messages)
            return response.text
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            return get_local_assistant_response(prompt, resume_text)
        return f"⚠️ Chat processing error: {error_msg}"

def get_local_assistant_response(prompt, resume_text):
    prompt = prompt.lower()
    text_lower = resume_text.lower()
    
    if "junior" in prompt or "senior" in prompt or "level" in prompt:
        if "experience" in text_lower and len(re.findall(r"\d+\+? years?", text_lower)) > 0:
            return "Based on my local analysis, the candidate shows signs of being an Experienced Professional, though the API is currently unavailable for a deeper linguistic assessment."
        return "The candidate appears to be at a Junior or Student level. They are currently focused on academic projects and internships."
        
    if "summarize" in prompt or "summary" in prompt or "strongest" in prompt:
        from resume_utils import extract_skills
        skills = extract_skills(resume_text)
        top = ", ".join(skills[:4]) if skills else "various technical domains"
        return f"In brief: The candidate is highly proficient in {top}. They have hands-on project experience and a strong academic foundation in AI & Data Science. Their profile suggests an execution-oriented mindset."
        
    if "skill" in prompt or "experience" in prompt:
        from resume_utils import extract_skills
        skills = extract_skills(resume_text)
        return f"The profile mentions several technical skills including: {', '.join(skills[:8])}. I am ready to evaluate any of these in detail."

    return "I am currently analyzing your profile using local heuristics. I can verify that the resume is loaded and I am ready to answer questions about skills or seniority levels."

def generate_styled_intro(style, resume_text, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(_get_best_model(api_key))
        
        style_prompts = {
            "Jarvis Mode": (
                "Provide a very concise, 3-sentence summary of this exact candidate. "
                "Explicitly list their technical skills, specific internships, and project names. "
                "Speak in the third-person about the candidate, written perfectly in the voice of J.A.R.V.I.S. from Iron Man, addressing your creator. "
                "Do not use markdown formatting."
            ),
            "Strong Professional": (
                "Generate a strong professional self-introduction for an interview. "
                "Structure: Greeting, Education, Technical Experience (internships/projects), and Future Interest. "
                "Tone: Formal, humble but confident. Use first-person 'I'. "
                "Focus on accuracy based on the resume. No markdown."
            ),
            "Short & Crisp": (
                "Generate a very short and crisp elevator pitch (under 50 words). "
                "Focus ONLY on the candidate's name, core domain (AI/ML/Web), most significant internship/project, and primary passion. "
                "Tone: Energetic and direct. No markdown."
            ),
            "Ruthless Upgrade": (
                "Generate a high-impact, 'ruthless' self-introduction. "
                "Tone: Extremely confident, execution-oriented, and ambitious. "
                "Key Phrases to include (rephrased if needed): 'I don't just learn concepts — I implement them', 'taking ideas from concept to production', 'execution mindset'. "
                "Emphasize the MERN job portal, disease prediction system, and IITM internship as evidence of real-world impact. No markdown."
            )
        }
        
        prompt = style_prompts.get(style, style_prompts["Strong Professional"])
        
        instructions = (
            f"{prompt}\n\n"
            "Use ONLY the details from this resume to ensure 100% accuracy:\n"
            f"=== RESUME CONTENT ===\n{resume_text}\n=====================\n"
            "Keep the response plain text without any special characters or markdown."
        )
        
        response = model.generate_content(instructions)
        return response.text.replace("*", "").replace("#", "").strip()
    except Exception as e:
        return f"⚠️ Error generating {style} intro: {str(e)}"
