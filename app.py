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
# In your app.py, create a section for past papers
st.sidebar.header("🔎 Past Paper Search")
subject_code = st.sidebar.text_input("Enter Subject Code (e.g., 9701)", "9701")
search_query = st.sidebar.text_input("Enter Topic or Paper Info", "digital certificate")

if st.sidebar.button("Search on CAIE Finder"):
    # Construct the search URL
    base_url = "https://www.caiefinder.com/search"
    # This is the pattern the site uses. If it changes, you might need to adjust.
    full_url = f"{base_url}?q={subject_code}+{search_query.replace(' ', '+')}"
    
    # Show the link to the user
    st.sidebar.markdown(f"[Open Search Results for '{subject_code} {search_query}']({full_url})", unsafe_allow_html=True)
    # Optionally, you can also display a message in the main area
    st.info(f"Opening CAIE Finder search for '{subject_code} {search_query}' in a new tab.")
def handle_question_with_papers(question, contexts):
    """
    This function handles a question in two steps:
    1. Get the AI-generated answer from your notes.
    2. Provide a direct link to find related past papers on caiefinder.com.
    """
    # 1. Generate the answer from your existing RAG pipeline
    ai_answer = ask_groq(question, contexts) # This is your existing function
    
    # 2. Create the caiefinder search link
    # Extract likely subject/topic from the question (you can make this smarter later)
    # For now, a simple link using the question as the query will work
    search_url = f"https://www.caiefinder.com/search?q={question.replace(' ', '+')}"
    
    # 3. Display the answer
    st.markdown(f'<div class="answer-box">{ai_answer}</div>', unsafe_allow_html=True)
    
    # 4. Display the link to search for related past papers
    st.markdown(f"""
        <div class="mark-scheme-box">
            <h4>📋 Want to practice past paper questions on this topic?</h4>
            <p>Click the link below to search for relevant past papers and mark schemes:</p>
            <a href="{search_url}" target="_blank">🔍 Search CAIE Past Papers for: {question}</a>
        </div>
    """, unsafe_allow_html=True)
    
    return ai_answer
