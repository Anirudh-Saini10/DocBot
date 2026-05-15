import os
import streamlit as st
import httpx

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="DocBot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 720px !important;
    }

    /* Header */
    .docbot-header {
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .docbot-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #E2E8F0;
        margin-bottom: 0.25rem;
        letter-spacing: -0.5px;
    }
    .docbot-subtitle {
        font-size: 1rem;
        color: #94A3B8;
        font-weight: 400;
    }

    /* Chat bubbles */
    .stChatMessage {
        background: transparent !important;
        border: none !important;
        padding: 0.5rem 0 !important;
    }

    /* User message */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) .stMarkdown {
        background: #1E293B !important;
        border-radius: 1rem 1rem 0.25rem 1rem !important;
        padding: 0.75rem 1rem !important;
        color: #F1F5F9 !important;
        border: 1px solid #334155 !important;
    }

    /* Assistant message */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) .stMarkdown {
        background: #0F172A !important;
        border-radius: 1rem 1rem 1rem 0.25rem !important;
        padding: 0.75rem 1rem !important;
        color: #E2E8F0 !important;
        border: 1px solid #1E293B !important;
    }

    /* Links in assistant messages */
    .stChatMessage a {
        color: #60A5FA !important;
        text-decoration: none !important;
        font-weight: 500 !important;
    }
    .stChatMessage a:hover {
        text-decoration: underline !important;
    }

    /* Chat input */
    .stChatInputContainer {
        background: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 1rem !important;
        padding: 0.25rem !important;
    }
    .stChatInputContainer textarea {
        background: transparent !important;
        color: #F1F5F9 !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-color: #60A5FA transparent transparent transparent !important;
    }

    /* Sidebar */
    .css-1d391kg, .css-1lcbmhc {
        background: #0F172A !important;
    }

    /* Buttons */
    .stButton > button {
        background: #1E293B !important;
        color: #94A3B8 !important;
        border: 1px solid #334155 !important;
        border-radius: 0.5rem !important;
        font-size: 0.8rem !important;
        padding: 0.4rem 1rem !important;
    }
    .stButton > button:hover {
        background: #334155 !important;
        color: #F1F5F9 !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #0F172A;
    }
    ::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 3px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 DocBot")
    st.markdown(
        """
        **Your AI-powered Google Drive assistant.**

        Ask me anything like:
        - *"Find the Q3 financial report"*
        - *"Show me PDFs from last week"*
        - *"Search for documents about bounceup"*
        """
    )
    st.divider()
    st.markdown("**Tips**")
    st.markdown("- Be specific with file names")
    st.markdown("- Mention file types (PDF, Sheet, Doc)")
    st.markdown("- Include dates for recent files")
    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.markdown(
        "<p style='font-size:0.75rem;color:#64748B;text-align:center;margin-top:1rem;'>Made by <b>Aniruddha Saini</b></p>",
        unsafe_allow_html=True,
    )

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="docbot-header">
        <div class="docbot-title">🤖 DocBot</div>
        <div class="docbot-subtitle">Ask me to find files in your Google Drive</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Chat State ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Display Chat History ────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat Input ──────────────────────────────────────────────────────────────
if prompt := st.chat_input("e.g., Find the financial report from last week"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call backend
    with st.chat_message("assistant"):
        with st.spinner("Searching your Drive..."):
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

# ── Empty State ─────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(3)
    example_prompts = [
        "Find my project proposal PDF",
        "Show spreadsheets from last week",
        "Search for bounceup documents",
    ]
    for i, col in enumerate(cols):
        with col:
            if st.button(example_prompts[i], use_container_width=True, key=f"example_{i}"):
                st.session_state.messages.append({"role": "user", "content": example_prompts[i]})
                st.rerun()
