import streamlit as st
import spacy
import fitz  # PyMuPDF
from spacy.matcher import PhraseMatcher
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

st.title("Automated Resume Analyzer")
st.write("Upload your resume (PDF) and paste a job description to get a fit score, suggestions, and a cover letter.")

# Upload resume
resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

# Job description input
job_desc = st.text_area("Paste Job Description")

def extract_text_from_pdf(pdf_file):
    if pdf_file is None:
        return ""
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def analyze_fit(resume_text, job_desc):
    # Process texts
    resume_doc = nlp(resume_text)
    job_doc = nlp(job_desc)
    
    # Extract job keywords (nouns/phrases)
    job_keywords = [chunk.text.lower() for chunk in job_doc.noun_chunks]
    
    # Matcher for exact phrases
    matcher = PhraseMatcher(nlp.vocab)
    patterns = [nlp.make_doc(kw) for kw in job_keywords]
    matcher.add("JobKeywords", patterns)
    
    matches = matcher(resume_doc)
    found_keywords = set(resume_doc[match[1]:match[2]].text.lower() for match in matches)
    
    # Fit score: ratio of matched keywords
    score = len(found_keywords) / len(job_keywords) * 100 if job_keywords else 0
    
    # Missing keywords
    missing = set(job_keywords) - found_keywords
    
    # Semantic similarity (bonus)
    similarity = resume_doc.similarity(job_doc) * 100
    
    # Combined score
    final_score = (score + similarity) / 2
    
    # Underwriter-specific keywords (optional enhancement)
    underwriter_keywords = ["risk assessment", "insurance policy", "actuarial analysis"]
    if "underwriter" in job_desc.lower():
        for kw in underwriter_keywords:
            if kw not in found_keywords:
                missing.add(kw)
    
    return final_score, list(found_keywords), list(missing)

def generate_cover_letter(resume_text, job_desc):
    if not os.getenv("OPENAI_API_KEY"):
        return "OpenAI API key not set. Please set the OPENAI_API_KEY environment variable."

    prompt = (
        f"Based on this resume: {resume_text[:1000]}...\n\n"
        f"And this job description: {job_desc}\n\n"
        "Generate a short, tailored cover letter paragraph highlighting fit."
    )

    try:
        response = client.responses.create(
            model="gpt-5-nano",
            input=prompt
        )
        return response.output_text
    except Exception as e:
        return f"Error generating cover letter: {e}"


# Button to analyze
if st.button("Analyze"):
    if resume_file is not None and job_desc:
        try:
            st.write("Analyzing...")
            resume_text = extract_text_from_pdf(resume_file)
            if resume_text:
                st.subheader("Extracted Resume Text")
                st.text(resume_text[:500] + "...")  # Preview
                
                score, found, missing = analyze_fit(resume_text, job_desc)
                st.subheader("Fit Score")
                st.progress(score / 100)
                st.write(f"Your resume fits the job at **{score:.2f}%** (based on keywords and similarity).")
                
                st.subheader("Matched Keywords")
                st.write(", ".join(found) or "None")
                
                st.subheader("Suggestions: Missing Keywords to Add")
                st.write(", ".join(missing) or "Great match!")
                
                cover_letter = generate_cover_letter(resume_text, job_desc)
                st.subheader("Tailored Cover Letter Snippet")
                st.write(cover_letter)
                
                # Optional feedback
                feedback = st.text_input("Feedback?")
                if feedback:
                    with open('feedback.txt', 'a') as f:
                        f.write(feedback + '\n')
                    st.success("Feedback saved!")
            else:
                st.error("Could not extract text from the resume. Please try another file.")
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
    else:
        st.error("Please upload a resume and enter a job description.")