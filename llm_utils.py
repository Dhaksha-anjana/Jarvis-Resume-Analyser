import google.generativeai as genai

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
            
    except Exception:
        pass
        
    return "gemini-1.5-flash-latest"

def validate_api_key(api_key):
    if not api_key:
        return False
    try:
        genai.configure(api_key=api_key)
        # Perform a lightweight call to verify key
        model = genai.GenerativeModel(_get_best_model(api_key))
        return True
    except Exception:
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
        return f"⚠️ Error generating questions: {str(e)}"

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
        return f"⚠️ Chat processing error: {str(e)}"
