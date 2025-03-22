import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import io

# Download NLTK resources for NLP tasks
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    pdf = PdfReader(io.BytesIO(uploaded_file.read()))  # Handle Streamlit's file object
    text = ""
    for page in pdf.pages:
        if page.extract_text():
            text += page.extract_text() + " "
    return text

# Function to preprocess text for removal of special characters
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    words = text.split()

    # Handle missing stopwords
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))

    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Function to extract skills and keywords
def extract_skills(text):
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    skills = [word for word, tag in tagged if tag.startswith('NN')]
    return set(skills)

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    processed_job_desc = preprocess_text(job_description)
    processed_resumes = [preprocess_text(resume) for resume in resumes]

    # Handle empty resumes
    if not any(processed_resumes):
        return [0] * len(resumes)  # Assign zero scores to all resumes

    documents = [processed_job_desc] + processed_resumes
    vectorizer = TfidfVectorizer()
    
    try:
        vectors = vectorizer.fit_transform(documents).toarray()
    except ValueError:
        return [0] * len(resumes)  # Return zero scores if vectorization fails
    
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    
    return cosine_similarities

# Streamlit app
st.title("AI Resume Screening & Candidate Ranking System")

# Job description input
st.header("Job Description")
job_description = st.text_area("Enter the job description")

# File uploader
st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    st.header("Ranking Resumes")
    
    resumes = []
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        resumes.append(text)

    # Rank resumes
    scores = rank_resumes(job_description, resumes)

    # Create a DataFrame for ranking
    results = pd.DataFrame({
        "Resume": [file.name for file in uploaded_files], 
        "Score": scores
    })

    # Sort by Score (Descending) and add Rank column
    results = results.sort_values(by="Score", ascending=False).reset_index(drop=True)
    results.index += 1  # Start ranking from 1

    # Rename index as Rank
    results.index.name = "Rank"

    # Display results in a clean table format
    st.dataframe(results.style.format({"Score": "{:.2f}"}))  