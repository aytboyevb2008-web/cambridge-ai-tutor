import streamlit as st
import requests
import time
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# ---- SETTINGS ----
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = "cambridge-notes"
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# ---- PAGE CONFIG ----
st.set_page_config(page_title="CAIE A-Level Tutor", page_icon="🎓")

# ---- CUSTOM CSS ----
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
        color: #1a1a1a;
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
</style>
""", unsafe_allow_html=True)

# ---- LOAD MODEL & CONNECT TO PINECONE ----
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# ---- FUNCTIONS ----
def retrieve(query_text, top_k=10):
    query_emb = model.encode(query_text).tolist()
    results = index.query(vector=query_emb, top_k=top_k, include_metadata=True)
    contexts, sources, pages = [], [], []
    for match in results["matches"]:
        text = match["metadata"]["text"]
        words = text.split()
        if len(words) > 300:
            text = " ".join(words[:300]) + "..."
        contexts.append(text)
        sources.append(match["metadata"]["source"])
        pages.append(match["metadata"].get("page", "?"))
    return contexts, sources, pages

def ask_groq(question, contexts):
    prompt = f"""You are a Cambridge A-Level tutor. Answer the student's question using ONLY the provided context.
If the answer is not in the context, say: "I don't have this in my notes. Please check your textbook."
Provide a comprehensive, detailed explanation suitable for an A‑Level student. Cover key definitions, examples, and related concepts found in the context. Use full sentences and structured paragraphs.

Context:
{chr(10).join(contexts)}

Question: {question}
Answer:"""

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "llama-3.1-8b-instant",   # or "mixtral-8x7b-32768" if you prefer
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 600
    }
    try:
        resp = requests.post(GROQ_URL, headers=headers, json=data, timeout=15)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        else:
            # silently return error message
            return "⚠️ Sorry, the AI service is temporarily unavailable."
    except Exception:
        return "⚠️ Network error. Please try again."

# ---- UI ----
if "last_question_time" not in st.session_state:
    st.session_state.last_question_time = 0

COOLDOWN_SECONDS = 15

# Styled headers
st.markdown('<h1 class="main-title">🎓 Cambridge A-Level AI Tutor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Ask anything from the syllabus. The AI answers <b>only</b> from your official notes and past papers.</p>', unsafe_allow_html=True)

question = st.text_input("Your question:")

if question:
    elapsed = time.time() - st.session_state.last_question_time
    remaining = COOLDOWN_SECONDS - elapsed
    if remaining > 0:
        st.warning(f"⏳ Please wait {remaining:.0f} seconds before asking another question.")
        st.stop()

    with st.spinner("Searching your notes..."):
        contexts, sources, pages = retrieve(question)
        answer = ask_groq(question, contexts)

    # Display answer in a nice box
    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

    with st.expander("📚 Sources"):
        for s, p in zip(sources, pages):
            st.write(f"- {s} (page {p})")

    st.session_state.last_question_time = time.time()
