import nltk
import re
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, RegexpParser
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import os

NLTK_PATH = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.append(NLTK_PATH)
STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ---------------- SKILLS DB ----------------
SKILLS_DB = [
    "python", "java", "c++", "c#", "c", "ruby", "php", "javascript", "typescript", "swift", "kotlin", "go", "rust", "r", "dart",
    "machine learning", "deep learning", "nlp", "computer vision", "artificial intelligence", "data analysis", "data science",
    "streamlit", "flask", "django", "fastapi", "spring boot", "express", "react", "node", "angular", "vue", "next.js", "svelte",
    "sql", "mysql", "postgresql", "mongodb", "sqlite", "redis", "cassandra", "dynamodb", "firebase", "supabase", "oracle",
    "tensorflow", "keras", "pytorch", "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn", "nltk", "spacy", "huggingface",
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "terraform", "ansible", "jenkins", "github actions", "gitlab ci",
    "git", "github", "gitlab", "bitbucket", "linux", "unix", "bash", "powershell",
    "html", "css", "sass", "less", "tailwind css", "bootstrap", "material ui",
    "agile", "scrum", "kanban", "jira", "confluence", "trello",
    "data structures", "algorithms", "object oriented programming", "oop", "system design", "microservices"
]

# ---------------- ACTION VERBS ----------------
ACTION_VERBS = [
    "achieved", "improved", "trained", "mentored", "managed", "created", 
    "resolved", "developed", "increased", "decreased", "negotiated", "launched",
    "implemented", "designed", "led", "spearheaded", "executed", "optimized"
]

# ---------------- PREPROCESS (MODULE 1) ----------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in STOPWORDS and len(w) > 2]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]  # Morphology
    return tokens

# ---------------- WORDNET SYNONYMS (MODULE 3) ----------------
def get_synonyms(word):
    syns = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            syns.add(l.name().lower())
    return syns

# ---------------- SKILL EXTRACTION (POS + CHUNKING) ----------------
def extract_skills(text):
    tokens = preprocess(text)
    pos = pos_tag(tokens)

    # Chunk grammar for noun phrases
    grammar = "NP: {<JJ>*<NN+>}"
    cp = RegexpParser(grammar)
    tree = cp.parse(pos)

    noun_phrases = []
    for subtree in tree.subtrees(filter=lambda t: t.label() == "NP"):
        phrase = " ".join([word for word, tag in subtree.leaves()])
        noun_phrases.append(phrase)

    found = set()
    text_lower = text.lower()

    # Match with skills DB accurately using regex bounding without stripping symbols
    for skill in SKILLS_DB:
        skill_lower = skill.lower()
        pattern = r'(?<![a-zA-Z0-9_])' + re.escape(skill_lower) + r'(?![a-zA-Z0-9_])'
        if re.search(pattern, text_lower):
            found.add(skill)

    # Match noun phrases
    for np in noun_phrases:
        if np in SKILLS_DB:
            found.add(np)

    return list(found)

# ---------------- RESUME SCORE (STANDALONE) ----------------
def resume_score(text):
    score = 0
    text_lower = text.lower()
    
    # 1. Completeness (30%)
    sections = ["objective", "skills", "projects", "experience", "education", "summary", "certifications"]
    sec_count = sum(1 for sec in sections if sec in text_lower)
    score += min(30, (sec_count / 5) * 30) # Max 30 points for having at least 5 standard sections
    
    # 2. Length/Density formatting (20%)
    word_count = len(text.split())
    if 300 < word_count < 1500:
        score += 20 # Ideal length
    elif word_count <= 300:
        score += 10 # Too short
    else:
        score += 10 # Extremely long
        
    # 3. Action Verbs & Quantifiable Metrics (30%)
    verbs_found = sum(1 for verb in ACTION_VERBS if verb in text_lower)
    metrics_found = len(re.findall(r'\d+%?|\$\d+', text_lower))
    
    if verbs_found >= 5: score += 15
    elif verbs_found > 0: score += 7
        
    if metrics_found >= 3: score += 15
    elif metrics_found > 0: score += 7
    
    # 4. Readability & Bullet Points (20%)
    bullet_count = text_lower.count('•') + text_lower.count('-')
    if bullet_count >= 8:
        score += 20
    elif bullet_count > 0:
        score += 10
        
    return round(min(100, score), 2)

# ---------------- N-GRAM ATS SIMILARITY (MODULE 2) ----------------
def ats_similarity(resume_text, jd_text):
    resume_tokens = preprocess(resume_text)
    jd_tokens = preprocess(jd_text)

    # TF-IDF similarity
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform([" ".join(resume_tokens), " ".join(jd_tokens)])
    tfidf_score = cosine_similarity(vectors[0], vectors[1])[0][0]

    # Bigram overlap
    resume_bigrams = set(ngrams(resume_tokens, 2))
    jd_bigrams = set(ngrams(jd_tokens, 2))

    overlap = len(resume_bigrams & jd_bigrams)
    total = len(jd_bigrams) if jd_bigrams else 1
    bigram_score = overlap / total

    # WordNet synonym boost
    synonym_match = 0
    for word in jd_tokens:
        syns = get_synonyms(word)
        if any(s in resume_tokens for s in syns):
            synonym_match += 1

    synonym_score = synonym_match / len(jd_tokens) if jd_tokens else 0

    final_score = (0.5 * tfidf_score) + (0.3 * bigram_score) + (0.2 * synonym_score)

    return round(final_score * 100, 2)

# ---------------- ADVANCED ATS SCORING ----------------
def advanced_ats_similarity(resume_text, jd_text):
    resume_tokens = preprocess(resume_text)
    jd_tokens = preprocess(jd_text)
    
    # 1. Skills Match (40%)
    jd_skills = extract_skills(jd_text)
    resume_skills = extract_skills(resume_text)
    
    if jd_skills:
        skill_overlap = len(set(jd_skills) & set(resume_skills))
        skills_score = (skill_overlap / len(jd_skills)) * 40
    else:
        skills_score = 40 if resume_skills else 0
        
    skills_score = min(40, skills_score)
    
    # 2. Keyword Context / TF-IDF Match (20%)
    if not jd_tokens:
        keyword_score = 0
    else:
        tfidf = TfidfVectorizer()
        try:
            vectors = tfidf.fit_transform([" ".join(resume_tokens), " ".join(jd_tokens)])
            cosine_sim = cosine_similarity(vectors[0], vectors[1])[0][0]
            keyword_score = cosine_sim * 20
        except ValueError:
            keyword_score = 0
            
    # 3. Action Verbs & Metrics (20%)
    resume_lower = resume_text.lower()
    verbs_found = sum(1 for verb in ACTION_VERBS if verb in resume_lower)
    metrics_found = len(re.findall(r'\d+%?', resume_text))
    
    action_metrics_score = 0
    if verbs_found >= 3:
        action_metrics_score += 10
    elif verbs_found > 0:
        action_metrics_score += 5
        
    if metrics_found >= 3:
        action_metrics_score += 10
    elif metrics_found > 0:
        action_metrics_score += 5
        
    # 4. Section Match / Completeness (20%)
    sections = ["objective", "skills", "projects", "experience", "education"]
    sections_found = sum(1 for sec in sections if sec in resume_lower)
    section_score = (sections_found / len(sections)) * 20
    
    total_score = skills_score + keyword_score + action_metrics_score + section_score
    
    return {
        "total": round(total_score, 2),
        "skills_match": round(skills_score, 2),
        "keyword_match": round(keyword_score, 2),
        "action_metrics": round(action_metrics_score, 2),
        "section_match": round(section_score, 2)
    }

def keyword_density_analysis(resume_text, jd_text):
    jd_skills = extract_skills(jd_text)
    
    analysis = []
    text_lower = resume_text.lower()
    
    for skill in jd_skills:
        skill_lower = skill.lower()
        
        # Safe exact match boundary for technical skills containing symbols (+, #, etc.)
        pattern = r'(?<![a-zA-Z0-9_])' + re.escape(skill_lower) + r'(?![a-zA-Z0-9_])'
        count = len(re.findall(pattern, text_lower))
            
        status = "Good ✅"
        if count == 0:
            status = "Missing ❌"
        elif count > 4:
            status = "Overused ⚠️"
            
        analysis.append({"skill": skill.title(), "count": count, "status": status})
        
    sort_order = {"Missing ❌": 0, "Overused ⚠️": 1, "Good ✅": 2}
    analysis.sort(key=lambda x: sort_order.get(x["status"], 3))
    return analysis

def calculate_competitiveness(score, role):
    if score >= 90: percent = "Top 5%"
    elif score >= 80: percent = "Top 15%"
    elif score >= 70: percent = "Top 30%"
    elif score >= 50: percent = "Top 50%"
    else: percent = "Bottom 50%"
    
    return f"You are in the {percent} for {role} roles."

# ---------------- BASIC INFO ----------------
def extract_basic_info(text, selected_role=None):
    text_lower = text.lower()

    name_match = re.search(r"name[:\-]?\s*(.*)", text_lower)
    name = name_match.group(1).title() if name_match else "the candidate"

    if selected_role and selected_role != "Custom Job Description":
        role = selected_role
    else:
        role = "Data Science"
        if "web" in text_lower:
            role = "Web Development"
        elif "android" in text_lower:
            role = "Android Development"

    skills = extract_skills(text_lower)
    top_skills = ", ".join(skills[:3]) if skills else "various technologies"

    level = "fresher"
    if "intern" in text_lower:
        level = "intern"
    if "experience" in text_lower:
        level = "experienced professional"

    return {
        "name": name,
        "skills_full": skills,
        "skills": top_skills,
        "role": role,
        "level": level
    }

# ---------------- SELF INTRO ----------------
def generate_self_intro(info):
    skills_list = info.get("skills_full", [])
    
    # Formulate a smooth sentence for skills
    if len(skills_list) > 1:
        skills_text = ", ".join(skills_list[:5]) + f", and {skills_list[5] if len(skills_list) > 5 else skills_list[-1]}"
    elif skills_list:
        skills_text = skills_list[0]
    else:
        skills_text = "various emerging tech stacks"
        
    level_str = info['level']
    article = "an" if level_str.startswith(("a", "e", "i", "o", "u")) else "a"
    
    intro = (
        f"Hello! My name is {info['name']}. "
        f"I am {article} {level_str} with extensive technical expertise in {skills_text}. "
    )
    
    if len(skills_list) >= 6:
        intro += "I am also deeply familiar with several other tools across the modern software engineering stack. "
        
    intro += f"My primary focus revolves around {info['role']}. "
    
    if 'score' in info:
        intro += f"Jarvis has analyzed my resume and awarded it an overall quality score of {info['score']} out of 100. "
        
    if 'ats_match' in info:
        intro += f"Furthermore, my profile is assessed as a solid {info['ats_match']} percent match for the current targeted job requirements. "
        
    intro += "I am highly passionate about continuing to learn, adapting to new challenges, and applying my background to build real-world, efficient solutions."
    return intro
