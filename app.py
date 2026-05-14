import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import requests

# ---- SETTINGS (will come from Streamlit Secrets) ----
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = "cambridge-notes"   # your index name
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
# ----------------------------------------------------

# Load embedding model (cached, so it loads only once)
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Connect to Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Groq free endpoint
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def retrieve(query_text, top_k=5):
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

def ask_groq(question, contexts):
    """Send context + question to Groq LLM."""
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
    resp = requests.post(GROQ_URL, headers=headers, json=data)
    return resp.json()["choices"][0]["message"]["content"]

# ---- Streamlit UI ----
st.set_page_config(page_title="CAIE A-Level Tutor", page_icon="🎓")
st.title("🎓 Cambridge A-Level AI Tutor")
st.markdown("Ask anything from the syllabus. The AI answers **only** from your official notes and past papers.")

question = st.text_input("Your question:")

if question:
    with st.spinner("Searching your notes..."):
        contexts, sources, pages = retrieve(question)
        answer = ask_groq(question, contexts)

    st.success("Answer")
    st.write(answer)

    with st.expander("📚 Sources"):
        for s, p in zip(sources, pages):
            st.write(f"- {s} (page {p})")