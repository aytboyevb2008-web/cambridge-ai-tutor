import streamlit as st
import requests
import time
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import base64

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

    .streak-badge {
        background-color: #1f77b4;
        color: white;
        padding: 8px 18px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 1rem;
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

def ask_groq(question, contexts, detail="detailed", simple=False):
    """Generate answer with optional style controls."""
    # Base prompt
    prompt = f"""You are a Cambridge A-Level tutor. Answer the student's question using ONLY the provided context.
If the answer is not in the context, say: "I don't have this in my notes. Please check your textbook."
"""

    # Add style instructions
    if simple:
        prompt += "Explain this as if the student is a 10‑year‑old. Use very simple words and relatable analogies.\n"
    elif detail == "concise":
        prompt += "Be brief and to‑the‑point. Give a short answer in 2‑3 sentences.\n"
    else:  # detailed
        prompt += "Provide a comprehensive, detailed explanation suitable for an A‑Level student. Cover key definitions, examples, and related concepts found in the context. Use full sentences and structured paragraphs.\n"

    prompt += f"""
Context:
{chr(10).join(contexts)}

Question: {question}
Answer:"""

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "llama-3.1-8b-instant",   # or "mixtral-8x7b-32768"
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 400 if (simple or detail == "concise") else 800
    }
    try:
        resp = requests.post(GROQ_URL, headers=headers, json=data, timeout=15)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        else:
            return "⚠️ Sorry, the AI service is temporarily unavailable."
    except Exception:
        return "⚠️ Network error. Please try again."

# ---- UI ----
# Initialize session states
if "last_question_time" not in st.session_state:
    st.session_state.last_question_time = 0
if "question_count" not in st.session_state:
    st.session_state.question_count = 0
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "last_question" not in st.session_state:
    st.session_state.last_question = ""

COOLDOWN_SECONDS = 15

# Styled headers
st.markdown('<h1 class="main-title">🎓 Cambridge A-Level AI Tutor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Ask anything from the syllabus. The AI answers <b>only</b> from your official notes and past papers.</p>', unsafe_allow_html=True)

# ---- CREATIVITY CONTROLS ----
col1, col2 = st.columns(2)
with col1:
    detail_level = st.radio("📝 Answer Detail", ["detailed", "concise"], horizontal=True, index=0)
with col2:
    simple_mode = st.toggle("🧒 Explain Like I'm 5", value=False)

# Streak display
if st.session_state.question_count > 0:
    st.markdown(f'<span class="streak-badge">🔥 {st.session_state.question_count} questions answered this session</span>', unsafe_allow_html=True)

question = st.text_input("Your question:")

if question:
    # Cooldown check
    elapsed = time.time() - st.session_state.last_question_time
    remaining = COOLDOWN_SECONDS - elapsed
    if remaining > 0:
        st.warning(f"⏳ Please wait {remaining:.0f} seconds before asking another question.")
        st.stop()

    # Avoid duplicate API call if question is repeated (cache)
    if question == st.session_state.last_question:
        answer = st.session_state.last_answer
        # don't increment counter or reset timer for same question
        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
    else:
        with st.spinner("Searching your notes..."):
            contexts, sources, pages = retrieve(question)
            answer = ask_groq(question, contexts, detail=detail_level, simple=simple_mode)

        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

        with st.expander("📚 Sources"):
            for s, p in zip(sources, pages):
                st.write(f"- {s} (page {p})")

        # Save to session
        st.session_state.last_question = question
        st.session_state.last_answer = answer
        st.session_state.question_count += 1
        st.session_state.last_question_time = time.time()

               # ---- TEXT-TO-SPEECH (base64 – no escaping issues) ----
        encoded = base64.b64encode(answer.encode()).decode()
        tts_html = f"""
        <button id="readAloudBtn" data-text="{encoded}" onclick="readAloudBase64()" style="padding:8px 16px; background:#1f77b4; color:white; border:none; border-radius:6px; cursor:pointer; margin-top:10px;">
            🔊 Read Aloud
        </button>
        <script>
        function readAloudBase64() {{
            const btn = document.getElementById('readAloudBtn');
            const encoded = btn.getAttribute('data-text');
            const decoded = atob(encoded);
            const msg = new SpeechSynthesisUtterance(decoded);
            msg.lang = 'en-US';
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(msg);
        }}
        </script>
        """
        st.components.v1.html(tts_html, height=80)

# ---- MOTIVATIONAL MESSAGES (after streak milestones) ----
if st.session_state.question_count in [5, 10, 20, 30]:
    st.balloons()
    st.success(f"🎉 Amazing! You've asked {st.session_state.question_count} questions. Keep up the great work!")
elif st.session_state.question_count > 0 and st.session_state.question_count % 10 == 0:
    st.info(f"💪 {st.session_state.question_count} questions – you're on fire!")
