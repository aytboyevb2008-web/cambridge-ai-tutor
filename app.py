import streamlit as st
import requests
import urllib.parse
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

    .mark-scheme-box {
        background-color: #eef6ff;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #cce5ff;
        margin: 20px 0;
        color: #1a1a1a;
    }
    
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #999;
        font-size: 0.9rem;
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

# ---- GROQ API ----
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# ---- FUNCTIONS ----
def retrieve(query_text, top_k=15):
    """Embed query, search Pinecone, return contexts."""
    query_emb = model.encode(query_text).tolist()
    results = index.query(
        vector=query_emb,
        top_k=top_k,
        include_metadata=True
    )
    contexts = [match["metadata"]["text"] for match in results["matches"]]
    sources = [match["metadata"]["source"] for match in results["matches"]]
    pages = [match["metadata"].get("page", "?") for match in results["matches"]]
    return contexts, sources, pages

import time  # add this at the top of your file if not already imported

def ask_groq(question, contexts):
    """Send context + question to Groq LLM with retry and error handling."""
    prompt = f"""You are a Cambridge A-Level tutor. Answer the student's question using ONLY the provided context.
If the answer is not in the context, say: "I don't have this in my notes. Please check your textbook."
Be concise and exam-focused.

Context:
{chr(10).join(contexts)}

Question: {question}
Answer:"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 600
    }
    
    # Try up to 3 times with a wait
    max_retries = 3
    for attempt in range(max_retries):
        resp = requests.post(GROQ_URL, headers=headers, json=data)
        
        # If successful, return answer
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        
        # If rate limited, parse the wait time from the error
        error_info = resp.json().get("error", {})
        if error_info.get("code") == "rate_limit_exceeded":
            wait_time = 10  # wait 10 seconds (the API suggested 4.3s, but we play safe)
            if attempt < max_retries - 1:
                st.warning(f"Rate limit hit. Waiting {wait_time} seconds... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
                continue
        
        # Other errors – break and show message
        break
    
    # If all retries failed, return a friendly error message
    return "⚠️ Sorry, the AI service is temporarily unavailable. Please try again in a minute."

# ---- UI ----
st.markdown('<h1 class="main-title">🎓 Cambridge A-Level AI Tutor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Ask anything from the syllabus. The AI answers <b>only</b> from your official notes and past papers.</p>', unsafe_allow_html=True)

question = st.text_input("Your question:")

if question:
    with st.spinner("Searching your notes..."):
        contexts, sources, pages = retrieve(question)
        answer = ask_groq(question, contexts)

    # Display answer
    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

    # Display sources
    with st.expander("📚 Sources"):
        for s, p in zip(sources, pages):
            st.write(f"- {s} (page {p})")

    # ---- CAIE Finder Search Link ----
    encoded_question = urllib.parse.quote(question)
    caie_search_url = f"https://www.caiefinder.com/search?q={encoded_question}"
    
    st.write(f"DEBUG: question = '{question}'")
    st.write(f"DEBUG: encoded = '{encoded_question}'")

    st.markdown(f"""
        <div class="mark-scheme-box">
            <h4>📋 Search CAIE Past Papers for this question</h4>
            <p>Click below to find relevant past papers and mark schemes:</p>
            <a href="{caie_search_url}" target="_blank" style="display: inline-block; padding: 12px 24px; background: #1f77b4; color: white; text-decoration: none; border-radius: 8px; font-weight: 600;">
                🔍 Search CAIE Finder
            </a>
            <p style="margin-top: 10px; font-size: 0.85rem; color: #666;">Search query: <code>{question}</code></p>
        </div>
    """, unsafe_allow_html=True)
    # Only create the link if question is not empty
    if question.strip():   # ensures no whitespace-only queries
        encoded_question = urllib.parse.quote(question.strip())
        caie_search_url = f"https://www.caiefinder.com/search?q={encoded_question}"

        st.markdown(f"""
            <div class="mark-scheme-box">
                <h4>📋 Search CAIE Past Papers for this question</h4>
                <p>Click below to find relevant past papers and mark schemes:</p>
                <a href="{caie_search_url}" target="_blank" style="...">🔍 Search CAIE Finder</a>
                <p style="...">Search query: <code>{question}</code></p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("The question field is empty. Cannot create search link.")

# Sidebar manual search
manual_query = st.sidebar.text_input("Enter topic or paper code", key="manual_search")

if st.sidebar.button("Search CAIE Finder"):
    if manual_query and manual_query.strip():
        encoded = urllib.parse.quote(manual_query.strip())
        search_url = f"https://www.caiefinder.com/search?q={encoded}"
        st.sidebar.success(f"Searching for: {manual_query}")
        st.sidebar.markdown(f"[🔍 Open CAIE Finder Results]({search_url})")
    else:
        st.sidebar.warning("Please enter a search term first.")
