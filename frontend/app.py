import os
import streamlit as st
import httpx

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="DocBot", page_icon="🤖", layout="centered")

st.title("🤖 DocBot")
st.caption("Ask me to find files in your Google Drive!")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("e.g., Find the financial report from last week"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call backend
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            try:
                payload = {
                    "message": prompt,
                    "history": [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                    ],
                }
                response = httpx.post(f"{API_URL}/chat", json=payload, timeout=60.0)
                response.raise_for_status()
                data = response.json()
                assistant_reply = data.get("content", "Sorry, I didn't get that.")
            except Exception as e:
                assistant_reply = f"⚠️ Error: {e}"

        st.markdown(assistant_reply)
        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
