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
    "python", "java", "c++",
    "machine learning", "deep learning", "nlp",
    "streamlit", "flask", "django",
    "sql", "mysql",
    "tensorflow", "keras",
    "react", "node"
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

    # Match with skills DB
    for skill in SKILLS_DB:
        skill_tokens = preprocess(skill)
        if all(word in tokens for word in skill_tokens):
            found.add(skill)

    # Match noun phrases
    for np in noun_phrases:
        if np in SKILLS_DB:
            found.add(np)

    return list(found)

# ---------------- RESUME SCORE ----------------
def resume_score(text):
    score = 0
    sections = ["objective", "skills", "projects", "experience", "education"]
    for sec in sections:
        if sec in text.lower():
            score += 20
    return score

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

# ---------------- BASIC INFO ----------------
def extract_basic_info(text):
    text_lower = text.lower()

    name_match = re.search(r"name[:\-]?\s*(.*)", text_lower)
    name = name_match.group(1).title() if name_match else "the candidate"

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
        "skills": top_skills,
        "role": role,
        "level": level
    }

# ---------------- SELF INTRO ----------------
def generate_self_intro(info):
    return (
        f"Hi, my name is {info['name']}. "
        f"I am a {info['level']} with skills in {info['skills']}. "
        f"I am interested in opportunities related to {info['role']}. "
        f"I am passionate about learning and applying real world technologies."
    )
