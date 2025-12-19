import streamlit as st
import subprocess
import sys
import fitz  # PyMuPDF
from openai import OpenAI
import os

# -----------------------------
# Load OpenAI API key safely
# -----------------------------

def get_openai_client():
    api_key = None

    # 1. Try Streamlit secrets (Cloud)
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]

    # 2. Fallback to environment variable (local)
    elif os.getenv("OPENAI_API_KEY"):
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return None

    return OpenAI(api_key=api_key)


client = get_openai_client()

@st.cache_resource
def load_spacy_model():
    model_name = "en_core_web_sm"
    try:
        return spacy.load(model_name)
    except OSError:
        # Model not found — download it
        subprocess.check_call(
            [sys.executable, "-m", "spacy", "download", model_name]
        )
        return spacy.load(model_name)

nlp = load_spacy_model()

# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title="Automated Resume Analyzer", layout="wide")

st.title("Automated Resume Analyzer")
st.write(
    "Upload your resume (PDF) and paste a job description to receive "
    "a fit score, keyword analysis, and a tailored cover letter snippet."
)

resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
job_desc = st.text_area("Paste Job Description", height=200)

# -----------------------------
# Helper functions
# -----------------------------

def extract_text_from_pdf(pdf_file):
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text.strip()
    except Exception as e:
        st.error(f"PDF parsing error: {e}")
        return ""


def analyze_fit(resume_text, job_desc):
    resume_doc = nlp(resume_text)
    job_doc = nlp(job_desc)

    job_keywords = [chunk.text.lower() for chunk in job_doc.noun_chunks]

    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(k) for k in job_keywords]
    matcher.add("JOB_KEYWORDS", patterns)

    matches = matcher(resume_doc)
    found = {resume_doc[s:e].text.lower() for _, s, e in matches}

    keyword_score = (len(found) / len(job_keywords) * 100) if job_keywords else 0
    similarity_score = resume_doc.similarity(job_doc) * 100
    final_score = (keyword_score + similarity_score) / 2

    missing = set(job_keywords) - found

    return final_score, sorted(found), sorted(missing)


def generate_cover_letter(resume_text, job_desc):
    if not client:
        return (
            "Cover letter generation is unavailable because the OpenAI API "
            "key is not configured or the region is unsupported."
        )

    prompt = (
        "Write ONE concise, professional cover letter paragraph.\n\n"
        f"RESUME:\n{resume_text[:1200]}\n\n"
        f"JOB DESCRIPTION:\n{job_desc}\n"
    )

    try:
        response = client.responses.create(
            model="gpt-5-nano",
            input=prompt
        )
        return response.output_text.strip()
    except Exception as e:
        return f"Cover letter generation failed: {e}"

# -----------------------------
# Main flow
# -----------------------------

if st.button("Analyze"):
    if not resume_file or not job_desc.strip():
        st.error("Please upload a resume and enter a job description.")
        st.stop()

    with st.spinner("Analyzing resume..."):
        resume_text = extract_text_from_pdf(resume_file)

    if not resume_text:
        st.error("Could not extract text from the resume.")
        st.stop()

    score, found, missing = analyze_fit(resume_text, job_desc)

    st.subheader("Fit Score")
    st.progress(score / 100)
    st.write(f"**{score:.2f}% overall match**")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Matched Keywords")
        st.write(", ".join(found) if found else "None")

    with col2:
        st.subheader("Missing Keywords")
        st.write(", ".join(missing) if missing else "Excellent coverage")

    st.subheader("Tailored Cover Letter Snippet")
    st.write(generate_cover_letter(resume_text, job_desc))
