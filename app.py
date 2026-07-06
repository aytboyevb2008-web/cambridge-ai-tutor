def ask_groq(question, contexts):
    prompt = f"""You are a Cambridge A-Level tutor. Answer the student's question using ONLY the provided context.
If the answer is not in the context, say: "I don't have this in my notes. Please check your textbook."
Provide a comprehensive, detailed explanation suitable for an A‑Level student. Cover key definitions, examples, and related concepts found in the context. Use full sentences and structured paragraphs.

Context:
{chr(10).join(contexts)}

Question: {question}
Answer:"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mixtral-8x7b-32768",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 600
    }
    
    try:
        resp = requests.post(GROQ_URL, headers=headers, json=data, timeout=15)
        # Show full response in the app for debugging
        st.write("### DEBUG: Groq API response status:", resp.status_code)
        st.json(resp.json())  # display the full JSON
        
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        else:
            return f"❌ Groq error: {resp.status_code} – see above"
    except Exception as e:
        return f"❌ Network/request failed: {str(e)}"
