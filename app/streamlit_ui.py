import streamlit as st
import requests
import time
from datetime import datetime


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="John Bentley AI Assistant", 
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        'About': "Leadership AI Assistant powered by John Bentley's expertise"
    }
)

API_BASE = st.secrets.get("API_BASE", "http://127.0.0.1:8000")
API_KEY = st.secrets.get("X_API_KEY", "")

HEADERS = {"Content-Type": "application/json"}
if API_KEY:
    HEADERS["x-api-key"] = API_KEY

# ---- ENHANCED CSS ----
st.markdown(
    """
    <style>
      /* Modern dark gradient background */
      .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        background-attachment: fixed;
      }

      /* Content container */
      .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 3rem;
      }

      /* Header section */
      .jb-header {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(147, 51, 234, 0.1) 100%);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
      }

      .jb-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
      }
      
      .jb-subtitle {
        color: rgba(255, 255, 255, 0.7);
        font-size: 1rem;
        line-height: 1.5;
      }

      /* Chat container - glassmorphism effect */
      .jb-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 1.5rem;
        backdrop-filter: blur(20px);
        box-shadow: 
          0 8px 32px rgba(0, 0, 0, 0.4),
          inset 0 1px 0 rgba(255, 255, 255, 0.1);
        min-height: 400px;
      }

      /* Chat messages styling */
      [data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        animation: fadeIn 0.3s ease-in;
      }

      /* User message - subtle blue tint */
      [data-testid="stChatMessage"][data-testid*="user"] {
        background: rgba(59, 130, 246, 0.08);
        border-left: 3px solid rgba(59, 130, 246, 0.5);
      }

      /* Assistant message - subtle purple tint */
      [data-testid="stChatMessage"]:not([data-testid*="user"]) {
        background: rgba(147, 51, 234, 0.06);
        border-left: 3px solid rgba(147, 51, 234, 0.4);
      }

      /* Timestamp styling */
      .message-timestamp {
        font-size: 0.75rem;
        color: rgba(255, 255, 255, 0.4);
        margin-top: 0.5rem;
        font-style: italic;
      }

      /* Fade-in animation */
      @keyframes fadeIn {
        from {
          opacity: 0;
          transform: translateY(10px);
        }
        to {
          opacity: 1;
          transform: translateY(0);
        }
      }

      /* Chat input styling */
      [data-testid="stChatInput"] {
        margin-top: 1rem;
      }

      [data-testid="stChatInput"] > div {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 16px;
        transition: all 0.3s ease;
      }

      [data-testid="stChatInput"] > div:focus-within {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(59, 130, 246, 0.5);
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
      }

      /* New Chat button styling */
      .stButton > button {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(147, 51, 234, 0.15) 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 12px;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 500;
        transition: all 0.3s ease;
        padding: 0.5rem 1rem;
      }

      .stButton > button:hover {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.25) 0%, rgba(147, 51, 234, 0.25) 100%);
        border-color: rgba(59, 130, 246, 0.5);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
      }

      /* Spinner styling */
      .stSpinner > div {
        border-color: rgba(59, 130, 246, 0.3);
        border-top-color: #3b82f6;
      }

      /* Avatar images */
      [data-testid="stChatMessageAvatarUser"],
      [data-testid="stChatMessageAvatarAssistant"] {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
      }

      /* Error message styling */
      .stAlert {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 12px;
        color: #fca5a5;
      }

      /* Success message styling */
      .element-container:has(.stSuccess) {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 12px;
        padding: 0.5rem;
      }

      /* Hide Streamlit branding */
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      header {visibility: hidden;}

      /* Scrollbar styling */
      ::-webkit-scrollbar {
        width: 8px;
      }
      ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
      }
      ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 4px;
      }
      ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.3);
      }

      /* Typing indicator animation */
      @keyframes pulse {
        0%, 100% { opacity: 0.4; }
        50% { opacity: 1; }
      }
      
      .typing-indicator {
        display: inline-block;
        animation: pulse 1.5s ease-in-out infinite;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- Header Section ----
header_col1, header_col2, header_col3 = st.columns([1, 5, 1.5], vertical_alignment="center")

with header_col1:
    st.image("jb_avatar.jpeg", width=80)

with header_col2:
    st.markdown('<div class="jb-header" style="padding: 0.5rem 0; background: none; border: none; box-shadow: none;">', unsafe_allow_html=True)
    st.markdown('<div class="jb-title">John Bentley\'s Leadership Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="jb-subtitle">Ask anything about John Bentley\'s training, books, and leadership frameworks.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with header_col3:
    if st.button("🔄 New Chat", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": "Great to see you. What leadership question can I help with today?",
                "timestamp": datetime.now().strftime("%I:%M %p")
            }
        ]
        st.session_state.chat_count = st.session_state.get("chat_count", 0) + 1
        st.rerun()

st.markdown("---")

# ---- Health check ----
with st.spinner("Connecting to backend..."):
    try:
        r = requests.get(f"{API_BASE}/health", headers=HEADERS, timeout=5)
        r.raise_for_status()
    except Exception:
        st.error("🔌 Backend is not reachable right now. Please try again in a moment.")
        st.stop()

# ---- Session state ----
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Great to see you. What leadership question can I help with today?",
            "timestamp": datetime.now().strftime("%I:%M %p")
        }
    ]

if "chat_count" not in st.session_state:
    st.session_state.chat_count = 1

# ---- Chat container ----
st.markdown('<div class="jb-card">', unsafe_allow_html=True)

for m in st.session_state.messages:
    avatar = "jb_avatar.jpeg" if m["role"] == "assistant" else "🧑‍💼"
    
    with st.chat_message(m["role"], avatar=avatar):
        st.markdown(m["content"])
        # Show timestamp if available
        if "timestamp" in m:
            st.markdown(f'<div class="message-timestamp">{m["timestamp"]}</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---- Chat input ----
prompt = st.chat_input("Ask your question here...")

if prompt:
    # Add timestamp to user message
    user_timestamp = datetime.now().strftime("%I:%M %p")
    
    # Add user message
    st.session_state.messages.append({
        "role": "user", 
        "content": prompt,
        "timestamp": user_timestamp
    })
    
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(prompt)
        st.markdown(f'<div class="message-timestamp">{user_timestamp}</div>', unsafe_allow_html=True)

    # Generate assistant response
    with st.chat_message("assistant", avatar="jb_avatar.jpeg"):
        # Typing indicator with realistic delay
        with st.spinner("🤔 Thinking..."):
            time.sleep(0.8)  # Brief pause for realism
            
            try:
                resp = requests.post(
                    f"{API_BASE}/ask",
                    headers=HEADERS,
                    json={"question": prompt},
                    timeout=180
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data.get("answer", "Sorry — I couldn't generate a response.")
            except requests.exceptions.Timeout:
                answer = "⏱️ I'm taking longer than expected. Please try again, or ask a shorter question."
            except Exception as e:
                answer = "⚠️ Something went wrong on my side. Please try again."

        assistant_timestamp = datetime.now().strftime("%I:%M %p")
        st.markdown(answer)
        st.markdown(f'<div class="message-timestamp">{assistant_timestamp}</div>', unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer,
        "timestamp": assistant_timestamp
    })
    
    st.rerun()

# ---- Footer info (optional) ----
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.caption(f"💬 Chat session #{st.session_state.chat_count}")
with col2:
    st.caption(f"📝 {len(st.session_state.messages)} messages")
