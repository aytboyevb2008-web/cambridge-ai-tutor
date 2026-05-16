import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import requests

# ---- SETTINGS (will come from Streamlit Secrets) ----
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = "cambridge-notes"
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
# ----------------------------------------------------

# ---- PAGE CONFIGURATION (MUST be first Streamlit command) ----
st.set_page_config(page_title="CAIE A-Level Tutor", page_icon="🎓")

# ⭐ ===== INSERT THE CUSTOM CSS HERE ===== ⭐
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }

    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e0e0e0;
        padding: 15px 20px;
        font-size: 1rem;
    }

    .answer-box {
        background-color: #f8f9fa;
        padding: 25px;
        border-radius: 15px;
        border-left: 5px solid #1f77b4;
        margin: 20px 0;
        font-size: 1.05rem;
        line-height: 1.7;
    }

    .streamlit-expanderHeader {
        font-weight: 600;
        color: #1f77b4;
    }

    .mark-scheme-box {
        background-color: #eef6ff;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #cce5ff;
        margin: 20px 0;
    }
    
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #999;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)
# ⭐ ===== END OF CUSTOM CSS ===== ⭐

# ---- LOAD MODEL AND CONNECT TO PINECONE ----
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# ... rest of your functions (embed_query, retrieve_context, generate_answer) ...

# ---- STREAMLIT UI ----
# Replace the old title with styled HTML versions
st.markdown('<h1 class="main-title">🎓 Cambridge A-Level AI Tutor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Ask anything from the syllabus. The AI answers <b>only</b> from your official notes and past papers.</p>', unsafe_allow_html=True)

question = st.text_input("Your question:")

if question:
    with st.spinner("Searching your notes..."):
        contexts, sources, pages = retrieve(question)
        answer = ask_groq(question, contexts)

    # Display the answer using the styled box
    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

    with st.expander("📚 Sources"):
        for s, p in zip(sources, pages):
            st.write(f"- {s} (page {p})")
    
    # Add the footer
    st.markdown('<p class="footer">Powered by your Cambridge notes & past papers</p>', unsafe_allow_html=True)
