# AI Resume Screening System

An AI-powered Resume Screening and Candidate Ranking System using NLP and machine learning techniques. This tool extracts the text from PDF resumes, preprocesses the content, and ranks candidates based on the similarity between their resumes and the job description using TF-IDF and cosine similarity.

## Key Features
- Upload multiple resumes in PDF format  
- Extracts and preprocess text for analysis  
- Removes special characters and stopwords  
- Extract relevant skills and keywords from resumes  
- Rank candidates based on job description similarity  
- Interactive UI built with Streamlit  

## Tech Stack
- **Python** (NLTK, Scikit-learn, Pandas, PyPDF2, Regex)  
- **Streamlit** (for the web-based interface)  

## Installation & Usage

### 1. Clone the repository:
```bash
git clone https://github.com/yourusername/AI-Resume-Screening-System.git
cd AI-Resume-Screening-System
```

### 2. Set up a Virtual Environment (Recommended)

#### For Windows:
```bash
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

#### For macOS/Linux:
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### 3. Run the application:
```bash
streamlit run <filename>.py
```

## Future Enhancements
🔹 Improve NLP techniques for better skill extraction  
🔹 Add support for different file formats (e.g., DOCX)  
🔹 Implement a more advanced ranking algorithm  

## License
This project is licensed under the MIT License. Feel free to use and modify it according to your needs.

---

Contributions and feedback are welcome! 

