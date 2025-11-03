import streamlit as st
import pickle
import re
from PyPDF2 import PdfReader
import numpy as np
from collections import Counter

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Resume Analyzer", page_icon="🤖", layout="centered")

# ---------------- CUSTOM STYLING ----------------
st.markdown("""
    <style>
        .stApp { background-color: #0E1117; color: white; }
        .main-title { text-align: center; font-size: 40px; font-weight: 700; color: #1DB954; }
        .sub-header { text-align: center; color: #BBBBBB; margin-bottom: 30px; }
        .footer { text-align: center; font-size: 14px; margin-top: 2rem; color: gray; }
        .highlight { color: #1DB954; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>🤖 AI Resume Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Upload your resume and get predicted job role, confidence level, and skill insights!</p>", unsafe_allow_html=True)

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("📄 Upload your resume (.pdf or .txt)", type=["pdf", "txt"])

# ---------------- TEXT EXTRACTION ----------------
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        vectorizer, model = pickle.load(f)
    return vectorizer, model

# ---------------- SKILL EXTRACTION ----------------
TECH_SKILLS = [
    "python", "java", "c++", "c", "javascript", "html", "css", "react", "angular",
    "node", "express", "flask", "django", "tensorflow", "pytorch", "machine learning",
    "deep learning", "nlp", "sql", "mysql", "mongodb", "aws", "azure", "linux", 
    "docker", "kubernetes", "git", "github", "data analysis", "tableau", "powerbi"
]

def extract_skills(text):
    text_lower = text.lower()
    found_skills = [skill for skill in TECH_SKILLS if skill in text_lower]
    skill_counts = Counter(found_skills)
    return dict(skill_counts)

# ---------------- PREDICTION ----------------
if uploaded_file:
    try:
        if uploaded_file.type == "text/plain":
            resume_text = uploaded_file.read().decode("utf-8")
        else:
            resume_text = extract_text_from_pdf(uploaded_file)

        st.success("✅ Resume uploaded successfully!")

        with st.spinner("🔍 Analyzing your resume..."):
            vectorizer, model = load_model()

            # Clean text
            clean_text = re.sub(r'[^a-zA-Z ]', ' ', resume_text).lower()
            features = vectorizer.transform([clean_text])
            prediction = model.predict(features)

            # Confidence
            if hasattr(model, "predict_proba"):
                confidence = np.max(model.predict_proba(features)) * 100
            else:
                confidence = 80  # fallback estimate

            # Skill extraction
            skills_found = extract_skills(clean_text)

        # ---------------- DISPLAY RESULTS ----------------
        st.markdown(f"### 🧠 Predicted Job Role: <span class='highlight'>{prediction[0]}</span>", unsafe_allow_html=True)
        st.write(f"#### 🔢 Confidence: **{confidence:.2f}%**")
        st.progress(int(confidence))

        # ---------------- SKILL ANALYSIS ----------------
        st.markdown("### 🧩 Technical Skill Analysis")
        if skills_found:
            total_skills = len(TECH_SKILLS)
            matched_skills = len(skills_found)
            percentage = (matched_skills / total_skills) * 100

            st.write(f"Detected **{matched_skills} / {total_skills}** relevant skills")
            st.progress(int(percentage))
            st.write(f"Skill Match Percentage: **{percentage:.2f}%**")

            st.write("#### 🧠 Skills Found:")
            st.success(", ".join(skills_found.keys()))
        else:
            st.warning("No common technical skills detected. Try uploading a more detailed resume!")

        st.markdown("---")
        st.markdown("<p class='footer'>Developed with ❤️ by Sumant Bobade</p>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
