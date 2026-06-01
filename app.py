import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

st.set_page_config(page_title="Job Recommender", layout="wide")

def extract_pdf_text(uploaded_file) -> str:
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        return "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
    except Exception as e:
        st.error(f"PDF Parsing Error: {e}")
        return ""

def truncate_text(text: str, max_words: int = 1200) -> str:
    words = str(text).split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:int(max_words * 0.6)]) + "\n\n" + " ".join(words[-int(max_words * 0.4):])

@st.cache_resource(show_spinner=False)
def get_model(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

@st.cache_data(show_spinner=False)
def load_data(jobs_csv="data/job_title_des.csv", model_name="all-MiniLM-L6-v2"):
    df = pd.read_csv(jobs_csv)
    if 'Job Title' not in df.columns or 'Job Description' not in df.columns:
        raise ValueError("Missing required columns in CSV")
    df['Job Description'] = df['Job Description'].astype(str)
    model = get_model(model_name)
    embeddings = model.encode(df['Job Description'].tolist(), show_progress_bar=True, convert_to_numpy=True)
    return df, embeddings

def get_recommendations(user_text, df, embeddings, model_name, min_score, top_k):
    user_text = truncate_text(user_text)
    user_embedding = get_model(model_name).encode([user_text], convert_to_numpy=True)
    scores = cosine_similarity(user_embedding, embeddings)[0]
    
    eligible = np.where(scores >= min_score)[0]
    if not eligible.size:
        return []
        
    top_indices = eligible[np.argsort(scores[eligible])][-top_k:][::-1]
    return [
        {
            "job_title": df.iloc[idx]['Job Title'],
            "job_description": df.iloc[idx]['Job Description'],
            "score": float(scores[idx])
        }
        for idx in top_indices
    ]

def main():
    st.title("Job Recommendation System")

    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Recommendations", 1, 10, 5)
        min_score = st.slider("Min Similarity", 0.0, 1.0, 0.3, 0.01)

    try:
        with st.spinner("Loading database... (this may take a moment on first run)"):
            jobs_df, job_embeddings = load_data()
    except Exception as e:
        st.error(f"Startup Error: {e}")
        return

    st.sidebar.text(f"Database: {len(jobs_df)} jobs")

    col1, col2 = st.columns([3, 1])
    with col1:
        pasted_text = st.text_area("Paste Resume", height=300)
    with col2:
        uploaded_file = st.file_uploader("Upload Resume", type=["txt", "pdf"])

    user_text = ""
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            user_text = extract_pdf_text(uploaded_file)
        else:
            try:
                user_text = uploaded_file.getvalue().decode("utf-8")
            except:
                user_text = str(uploaded_file.getvalue())
    else:
        user_text = pasted_text

    if st.button("Find Jobs"):
        if not user_text.strip():
            st.warning("Please provide resume content")
            return
            
        with st.spinner("Analyzing resume..."):
            results = get_recommendations(
                user_text, jobs_df, job_embeddings, 
                "all-MiniLM-L6-v2", min_score, top_k
            )

        if not results:
            st.info("No matching jobs found. Try lowering the threshold.")
            return

        st.subheader("Top Matches")
        for i, res in enumerate(results, 1):
            st.markdown(f"**{i}. {res['job_title']}** (Score: `{res['score']:.3f}`)")
            with st.expander("Show details"):
                st.write(res['job_description'])
            st.divider()

if __name__ == "__main__":
    main()
