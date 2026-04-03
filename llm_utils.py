import google.generativeai as genai

def validate_api_key(api_key):
    if not api_key:
        return False
    try:
        genai.configure(api_key=api_key)
        # Perform a lightweight call to verify key
        model = genai.GenerativeModel('gemini-1.5-flash')
        return True
    except Exception:
        return False

def generate_interview_questions(missing_skills, jd_text, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
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

def get_chat_response(prompt, resume_text, chat_history, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Build chat context
        system_context = (
            "You are Jarvis, an expert AI recruiter assistant analyzing a candidate's resume. "
            "You must answer user questions based heavily on the specific contents of this resume text:\n\n"
            f"=== RESUME CONTENT ===\n{resume_text}\n=====================\n\n"
            "Be professional, concise, and helpful. If the resume doesn't specify something, clearly state that."
        )
        
        messages = [{"role": "user", "parts": [system_context]}, {"role": "model", "parts": ["Understood. I am ready to analyze the resume."]}]
        
        # Feed previous chat history
        for msg in chat_history:
            role = "user" if msg["role"] == "user" else "model"
            messages.append({"role": role, "parts": [msg["content"]]})
            
        # Add the newest prompt
        messages.append({"role": "user", "parts": [prompt]})
        
        response = model.generate_content(messages)
        return response.text
    except Exception as e:
        return f"⚠️ Chat processing error: {str(e)}"
