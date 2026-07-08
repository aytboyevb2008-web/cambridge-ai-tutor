import streamlit as st
import requests
import time
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import base64
import time
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

def ask_groq(question, contexts, detail="detailed", simple=False, language="English"):
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



def summarize_topic(question, contexts, language="English"):
    """Generate a bullet-point revision summary using the same context."""
    if language == "Oʻzbekcha":
        prompt = f"""Siz Cambridge A-Level o'qituvchisisiz. "{question}" mavzusi bo'yicha qisqa takrorlash eslatmalarini yarating.
Faqat berilgan kontekstdan foydalaning. Quyidagi tuzilishda yozing:
- Bir jumlali ta'rif
- 4–6 asosiy tushuncha, misollar va muhim tafsilotlar uchun band ro'yxati
- Agar kontekstda umumiy xatolar haqida ma'lumot bo'lsa, "💡 Imtihon maslahati" bilan yakunlang.

Kontekst:
{chr(10).join(contexts)}

Takrorlash Xulosasi:"""
    else:
        prompt = f"""You are a Cambridge A-Level tutor. Create a concise revision summary for the topic: "{question}".
Use ONLY the provided context. Structure the summary with:
- A one-sentence definition
- 4–6 bullet points covering the key concepts, examples, and important details
- End with a "💡 Exam Tip" if the context contains common pitfalls.

Context:
{chr(10).join(contexts)}

Revision Summary:"""

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "llama-3.1-8b-instant",   # or your preferred model
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 500
    }
    try:
        resp = requests.post(GROQ_URL, headers=headers, json=data, timeout=15)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        else:
            return "⚠️ Could not generate summary. Please try again."
    except Exception:
        return "⚠️ Network error."

def check_syllabus_coverage(topics, contexts_cache=None):
    """Use the LLM to check whether each topic is covered by the notes.
    Returns covered list, not_covered list, and percentage.
    """
    covered = []
    not_covered = []
    total = len(topics)
    if total == 0:
        return covered, not_covered, 0

    # If we don't have a pre-retrieved context for all notes, we'll get a sample of chunks
    if contexts_cache is None:
        # Retrieve a broad set of chunks (up to 50) to represent the whole knowledge base
        # We'll use a dummy query to get diverse chunks
        sample_emb = model.encode("Cambridge A-Level syllabus").tolist()
        results = index.query(vector=sample_emb, top_k=50, include_metadata=True)
        contexts_cache = [m["metadata"]["text"] for m in results["matches"]]

    progress_bar = st.progress(0)
    for i, topic in enumerate(topics):
        topic = topic.strip()
        if not topic:
            continue

        # Ask the LLM: is this topic covered?
        prompt = f"""You are an assistant that checks if a specific syllabus topic is present in a collection of study notes.
Read the notes and answer only "Yes" if the topic is clearly explained or discussed, otherwise answer "No".
Topic: {topic}

Notes (excerpts):
{chr(10).join(contexts_cache[:10])}  # limit to 10 chunks to save tokens

Is the topic "{topic}" covered in these notes? (Yes/No):"""

        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        data = {
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 5
        }
        try:
            resp = requests.post(GROQ_URL, headers=headers, json=data, timeout=15)
            if resp.status_code == 200:
                answer = resp.json()["choices"][0]["message"]["content"].strip().lower()
                if answer.startswith("yes"):
                    covered.append(topic)
                else:
                    not_covered.append(topic)
            else:
                not_covered.append(topic)
        except Exception:
            not_covered.append(topic)

        progress_bar.progress((i + 1) / total)
        time.sleep(0.5)  # small delay to avoid rate limits

    progress_bar.empty()
    percent = (len(covered) / total * 100) if total else 0
    return covered, not_covered, percent

# ---- SIDEBAR: Syllabus Coverage Tracker ----
st.sidebar.header("📈 Syllabus Coverage")

# Subject syllabus lists (add more if needed)
SYLLABUS_TOPICS = {
    "Computer Science (9618)": [
        "Information representation",
        "Communication and Internet technologies",
        "Hardware",
        "Processor fundamentals",
        "Assembly language",
        "System software",
        "Security, privacy and data integrity",
        "Ethics and ownership",
        "Databases",
        "Algorithm design and problem-solving",
        "Programming",
        "Data types and structures",
        "Logic gates and circuits",
        "Computer architecture",
        "Networking",
        "Internet principles",
        "Digital logic",
        "Artificial intelligence"
    ],
    "Physics (9702)": [
        "Physical quantities and units",
        "Measurement techniques",
        "Kinematics",
        "Dynamics",
        "Forces, density and pressure",
        "Work, energy and power",
        "Deformation of solids",
        "Waves",
        "Superposition",
        "Electric fields",
        "Current of electricity",
        "D.C. circuits",
        "Particle and nuclear physics",
        "Motion in a circle",
        "Gravitational fields",
        "Temperature",
        "Ideal gases",
        "Thermodynamics",
        "Oscillations",
        "Communication"
    ],
    "Chemistry (9701)": [
        "Atomic structure",
        "Bonding",
        "States of matter",
        "Chemical energetics",
        "Electrochemistry",
        "Equilibria",
        "Reaction kinetics",
        "Periodicity",
        "Group 2",
        "Group 17",
        "Nitrogen and sulfur",
        "Introduction to organic chemistry",
        "Hydrocarbons",
        "Halogenoalkanes",
        "Alcohols",
        "Carbonyl compounds",
        "Carboxylic acids and derivatives",
        "Nitrogen compounds",
        "Polymerisation",
        "Analytical techniques"
    ],
    "Biology (9700)": [
        "Cell structure",
        "Biological molecules",
        "Enzymes",
        "Cell membranes and transport",
        "The mitotic cell cycle",
        "Nucleic acids and protein synthesis",
        "Transport in plants",
        "Transport in mammals",
        "Gas exchange and smoking",
        "Infectious diseases",
        "Immunity",
        "Energy and respiration",
        "Photosynthesis",
        "Homeostasis",
        "Coordination",
        "Inherited change",
        "Selection and evolution",
        "Biodiversity",
        "Genetic technology"
    ]
}

subject_choice = st.sidebar.selectbox(
    "Choose a subject syllabus:",
    list(SYLLABUS_TOPICS.keys()),
    index=0,
    help="Select the Cambridge subject to check coverage against."
)

topics = SYLLABUS_TOPICS.get(subject_choice, [])
topics_text = "\n".join(topics)

with st.sidebar.expander("📋 View / Edit Topics"):
    edited_topics = st.text_area(
        "Syllabus topics (one per line):",
        value=topics_text,
        height=200,
        help="You can edit these topics or add new ones. One topic per line.",
        key="syllabus_topics_editor"
    )
    topics_list = [t.strip() for t in edited_topics.split("\n") if t.strip()]

if st.sidebar.button("Check Coverage", use_container_width=True):
    if topics_list:
        with st.spinner("AI is analyzing your notes for syllabus topics..."):
            covered, not_covered, percent = check_syllabus_coverage(topics_list)
        
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"### 📊 Coverage: {percent:.1f}%")
        st.sidebar.progress(percent / 100)
        
        with st.sidebar.expander("✅ Covered Topics"):
            for t in covered:
                st.markdown(f"- {t}")
        
        with st.sidebar.expander("❌ Not Covered"):
            for t in not_covered:
                st.markdown(f"- {t}")
        
        if percent < 50:
            st.sidebar.warning("⚠️ Less than half the syllabus is covered. Consider adding more notes.")
        elif percent >= 90:
            st.sidebar.success("🎉 Excellent! Most topics are covered.")
    else:
        st.sidebar.warning("No topics to check. Please add some topics.")
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
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []
if "last_pages" not in st.session_state:
    st.session_state.last_pages = []
if "last_contexts" not in st.session_state:
    st.session_state.last_contexts = []
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
# 🌐 Language selector
language = st.selectbox(
    "🌐 Language / Til",
    options=["English", "Oʻzbekcha"],
    index=0,
    help="Choose the language for answers."
)
# Streak display
if st.session_state.question_count > 0:
    st.markdown(f'<span class="streak-badge">🔥 {st.session_state.question_count} questions answered this session</span>', unsafe_allow_html=True)

question = st.text_input("Your question:")
# ---- SIDEBAR: Past Paper Search ----
# ---- SIDEBAR: Past Paper Search ----
st.sidebar.header("🔎 Search CAIE Past Papers")

# ℹ️ Instructions expander
with st.sidebar.expander("ℹ️ Search Tips"):
    st.markdown("""
    **For the best results:**
    - Write the **exact topic** (e.g., *MAC address*, *Doppler effect*, *stacks and queues*).
    - Do **not** write the syllabus code (e.g., 9701) in the search box – use the **Subject** and **Level** dropdowns instead.
    - Use the correct subject and level to narrow down papers.
    - You can search for a specific paper code if you know it (e.g., *9702/21/O/N/23*).
    """)

subject = st.sidebar.selectbox(
    "Subject",
    options=[
        "computerscience", "physics", "chemistry", "biology",
        "mathematics", "economics", "business", "accounting",
        "english", "history", "geography", "psychology"
    ],
    index=0,
    help="Choose the Cambridge subject exactly as it appears on CAIE Finder."
)

zone = st.sidebar.selectbox(
    "Level",
    options=["a", "o", "as"],
    index=0,
    format_func=lambda x: {"a": "A-Level", "o": "O-Level", "as": "AS-Level"}[x],
    help="A-Level, O-Level, or AS-Level."
)

search_term = st.sidebar.text_input(
    "Topic or paper code",
    placeholder="e.g., MAC address or 9702/21/O/N/23",
    key="sidebar_search_term"
)

# Placeholder for the result link – it will be refreshed on every search
result_placeholder = st.sidebar.empty()

if st.sidebar.button("🔍 Search CAIE Finder", use_container_width=True):
    if search_term and search_term.strip():
        import urllib.parse
        encoded = urllib.parse.quote(search_term.strip())
        url = f"https://caiefinder.com/search/?subs={subject}&zone={zone}&search={encoded}"
        # Replace the placeholder content with the new link
        result_placeholder.markdown(f"""
            <a href="{url}" target="_blank" style="
                display: inline-block;
                padding: 10px 20px;
                background: #1f77b4;
                color: white;
                text-decoration: none;
                border-radius: 6px;
                font-weight: 600;
                margin-top: 10px;
            ">Open CAIE Finder Results</a>
        """, unsafe_allow_html=True)
    else:
        result_placeholder.warning("Please enter a search term first.")
if question:
    # ---- CACHE CHECK (before cooldown) ----
    if (question == st.session_state.get("last_question", None)
    and detail_level == st.session_state.get("last_detail", None)
    and simple_mode == st.session_state.get("last_simple", None)
    and language == st.session_state.get("last_language", None)):
        # Same question – serve cached answer instantly, ignore cooldown
        answer = st.session_state.last_answer
    else:
        # New question – apply cooldown
        elapsed = time.time() - st.session_state.get("last_question_time", 0)
        remaining = COOLDOWN_SECONDS - elapsed
        if remaining > 0:
            # Toast notification (non‑blocking)
            st.toast(f"⏳ Please wait {remaining:.0f} seconds before asking a new question.", icon="⏳")
            # Stop processing the new question, but keep old answer visible
            st.stop()
        
        # Process new question
        with st.spinner("Searching your notes..."):
            contexts, sources, pages = retrieve(question)
            answer = ask_groq(question, contexts, detail=detail_level, simple=simple_mode, language=language)
        
        # Save to session
        st.session_state.last_question = question
        st.session_state.last_answer = answer
        st.session_state.last_sources = sources       
        st.session_state.last_pages = pages
        st.session_state.last_contexts = contexts
        st.session_state.question_count += 1
        st.session_state.last_question_time = time.time()

    # ---- DISPLAY ANSWER (always run, even if cached) ----
    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

    with st.expander("📚 Sources"):
        for s, p in zip(st.session_state.last_sources, st.session_state.last_pages):
            st.write(f"- {s} (page {p})")
                   
                # ---- TEXT-TO-SPEECH (manual voice selector, pause) ----
        encoded = base64.b64encode(answer.encode()).decode()
        tts_html = f"""
        <div style="margin-top: 10px;">
            <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
                <select id="voiceSelect" style="padding:6px; border-radius:4px; border:1px solid #ccc;">
                    <option value="">Loading voices...</option>
                </select>
                <button id="ttsPlayBtn" onclick="startSpeaking()" style="padding:8px 16px; background:#1f77b4; color:white; border:none; border-radius:6px; cursor:pointer;">
                    🔊 Read Aloud
                </button>
                <button id="ttsPauseBtn" onclick="togglePause()" style="display:none; padding:8px 16px; background:#ffc107; color:#1a1a1a; border:none; border-radius:6px; cursor:pointer;">
                    ⏸️ Pause
                </button>
            </div>
        </div>
        <script>
        const encodedText = "{encoded}";
        const fullText = atob(encodedText);
        let utterance = null;
        let isPaused = false;
        const voiceSelect = document.getElementById('voiceSelect');

        function populateVoices() {{
            const voices = window.speechSynthesis.getVoices();
            if (voices.length === 0) return;
            voiceSelect.innerHTML = '';
            // Add only English voices (you can remove this filter if you want all)
            const englishVoices = voices.filter(v => v.lang.startsWith('en'));
            if (englishVoices.length === 0) {{
                // fallback: show all voices
                voices.forEach(v => {{
                    const option = document.createElement('option');
                    option.value = v.name;
                    option.textContent = `${{v.name}} (${{v.lang}})`;
                    voiceSelect.appendChild(option);
                }});
            }} else {{
                englishVoices.forEach(v => {{
                    const option = document.createElement('option');
                    option.value = v.name;
                    option.textContent = `${{v.name}} (${{v.lang}})`;
                    voiceSelect.appendChild(option);
                }});
            }}
        }}

        // Populate on load and when voices change
        populateVoices();
        window.speechSynthesis.onvoiceschanged = () => {{
            populateVoices();
        }};

        function getSelectedVoice() {{
            const voices = window.speechSynthesis.getVoices();
            const selectedName = voiceSelect.value;
            return voices.find(v => v.name === selectedName) || null;
        }}

        function startSpeaking() {{
            window.speechSynthesis.cancel();
            setTimeout(() => {{
                utterance = new SpeechSynthesisUtterance(fullText);
                utterance.lang = 'en-US';
                const voice = getSelectedVoice();
                if (voice) utterance.voice = voice;
                utterance.rate = 0.95;
                utterance.pitch = 1.0;
                utterance.onend = () => {{
                    document.getElementById('ttsPlayBtn').style.display = 'inline-block';
                    document.getElementById('ttsPauseBtn').style.display = 'none';
                    isPaused = false;
                }};
                utterance.onerror = (e) => {{
                    console.error('Speech error:', e);
                    document.getElementById('ttsPlayBtn').style.display = 'inline-block';
                    document.getElementById('ttsPauseBtn').style.display = 'none';
                    isPaused = false;
                }};
                window.speechSynthesis.speak(utterance);
                document.getElementById('ttsPlayBtn').style.display = 'none';
                document.getElementById('ttsPauseBtn').style.display = 'inline-block';
                document.getElementById('ttsPauseBtn').textContent = '⏸️ Pause';
                isPaused = false;
            }}, 50);
        }}

        function togglePause() {{
            if (!utterance) return;
            if (isPaused) {{
                window.speechSynthesis.resume();
                document.getElementById('ttsPauseBtn').textContent = '⏸️ Pause';
                isPaused = false;
            }} else {{
                window.speechSynthesis.pause();
                document.getElementById('ttsPauseBtn').textContent = '▶️ Resume';
                isPaused = true;
            }}
        }}
        </script>
        """
        st.components.v1.html(tts_html, height=100)
       # ---- TOPIC SUMMARIZER ----
    summary_key = f"summary_{question}"
    if summary_key not in st.session_state:
        st.session_state[summary_key] = None

    col_summary_btn, _ = st.columns([1, 3])
    with col_summary_btn:
        if st.button("📝 Summarize This Topic"):
            # Use separate cooldown for summarizer (optional, but you can keep a short one)
            elapsed = time.time() - st.session_state.get("last_summary_time", 0)
            if elapsed < COOLDOWN_SECONDS:
                st.toast(f"⏳ Please wait {COOLDOWN_SECONDS - elapsed:.0f} seconds before summarizing again.", icon="⏳")
            else:
                with st.spinner("Generating revision summary..."):
                    summary = summarize_topic(question, st.session_state.last_contexts, language=language)
                    st.session_state[summary_key] = summary
                    st.session_state.last_summary_time = time.time()

    if st.session_state[summary_key]:
        st.markdown("---")
        st.markdown("### 📝 Revision Summary")
        st.markdown(f'<div class="answer-box" style="border-left: 5px solid #28a745;">{st.session_state[summary_key]}</div>', unsafe_allow_html=True)
# ---- MOTIVATIONAL MESSAGES (after streak milestones) ----
if st.session_state.question_count in [5, 10, 20, 30]:
    st.balloons()
    st.success(f"🎉 Amazing! You've asked {st.session_state.question_count} questions. Keep up the great work!")
elif st.session_state.question_count > 0 and st.session_state.question_count % 10 == 0:
    st.info(f"💪 {st.session_state.question_count} questions – you're on fire!")
