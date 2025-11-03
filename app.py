import streamlit as st
import pickle
import re

st.title("Resume Classifer")

uploaded_file = st.file_uploader("Upload your resume (.txt)", type=['txt'])

if uploaded_file:
    resume_text = uploaded_file.read().decode('utf-8')
    
    with open("model.pkl","rb") as f:
        vectorizer, model = pickle.load(f)
        
    text = re.sub(r'[^a-zA-Z ]', '', resume_text).lower()
    features = vectorizer.transform([text])
    prediction = model.predict(features)
    
    st.write(f"### Predicted Job Role: `{prediction[0]}`")