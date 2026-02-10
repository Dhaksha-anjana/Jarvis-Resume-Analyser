import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- DOWNLOAD ----------------
nltk.download("stopwords")
STOPWORDS = stopwords.words("english")

# ---------------- SKILLS DB ----------------
SKILLS_DB = [
    "python", "java", "c++",
    "machine learning", "deep learning", "nlp",
    "streamlit", "flask", "django",
    "sql", "mysql",
    "tensorflow", "keras",
    "react", "node"
]

# ---------------- FUNCTIONS ----------------
def extract_skills(text):
    text = text.lower()
    found = []
    for skill in SKILLS_DB:
        if skill in text:
            found.append(skill)
    return list(set(found))


def resume_score(text):
    score = 0
    sections = ["objective", "skills", "projects", "experience", "education"]
    for sec in sections:
        if sec in text.lower():
            score += 20
    return score


def ats_similarity(resume_text, jd_text):
    tfidf = TfidfVectorizer(stop_words=STOPWORDS)
    vectors = tfidf.fit_transform([resume_text, jd_text])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(similarity * 100, 2)


def extract_basic_info(text):
    text = text.lower()

    name_match = re.search(r"name[:\-]?\s*(.*)", text)
    name = name_match.group(1).title() if name_match else "the candidate"

    role = "Data Science"
    if "web" in text:
        role = "Web Development"
    elif "android" in text:
        role = "Android Development"

    skills = extract_skills(text)
    top_skills = ", ".join(skills[:3]) if skills else "various technologies"

    level = "fresher"
    if "intern" in text:
        level = "intern"
    if "experience" in text:
        level = "experienced professional"

    return {
        "name": name,
        "skills": top_skills,
        "role": role,
        "level": level
    }


def generate_self_intro(info):
    return (
        f"Hi, my name is {info['name']}. "
        f"I am a {info['level']} with skills in {info['skills']}. "
        f"I am interested in opportunities related to {info['role']}. "
        f"I am passionate about learning and applying real world technologies."
    )
