import streamlit as st
import requests
import time

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="John Bentley AI Assistant", layout="centered")

API_BASE = st.secrets.get("API_BASE", "http://127.0.0.1:8000")
API_KEY = st.secrets.get("X_API_KEY", "")

HEADERS = {"Content-Type": "application/json"}
if API_KEY:
    HEADERS["x-api-key"] = API_KEY

# ---- CSS ----
st.markdown(
    """
    <style>
      /* Page background + spacing */
      .stApp {
        background: radial-gradient(1200px 600px at 50% 0%, #101826 0%, #070A0F 60%, #05070B 100%);
      }

      /* Narrower content column */
      .block-container {
        max-width: 980px;
        padding-top: 3.0rem;
      }

      /* Title styling */
      .jb-title {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: white;
        margin-bottom: 0.25rem;
      }
      .jb-subtitle {
        color: rgba(255,255,255,0.70);
        font-size: 1rem;
        margin-bottom: 1.75rem;
      }

      /* Chat container card */
      .jb-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 18px 18px 8px 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.35);
      }

      /* Make chat messages feel a bit cleaner */
      [data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 0.1rem 0.1rem;
      }

      /* Reduce extra vertical gap above chat input */
      [data-testid="stChatInput"] {
        margin-top: 0.75rem;
      }

      /* Hide Streamlit default footer / menu if you want */
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# ---- Header row: title left, reset button right ----
col1, col2 = st.columns([0.85, 0.15], vertical_alignment="center")
with col1:
    st.markdown('<div class="jb-title">John Bentley’s AI Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="jb-subtitle">Experience and leadership questions 24 hours a day. Try to stump him!</div>',
        unsafe_allow_html=True
    )
with col2:
    if st.button("↻", help="Reset chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---- Health check (fast) ----
try:
    r = requests.get(f"{API_BASE}/health", headers=HEADERS, timeout=5)
    r.raise_for_status()
except Exception:
    st.error("Backend is not reachable right now. Please try again in a moment.")
    st.stop()

# ---- Session state ----
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Great to see you. What leadership question can I help with today?"}
    ]

# ---- Chat "card" container ----
st.markdown('<div class="jb-card">', unsafe_allow_html=True)

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

st.markdown("</div>", unsafe_allow_html=True)

# ---- Single input (bottom) ----
prompt = st.chat_input("Ask your question here…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                resp = requests.post(
                    f"{API_BASE}/ask",
                    headers=HEADERS,
                    json={"question": prompt},
                    timeout=120
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data.get("answer", "Sorry — I couldn't generate a response.")
            except requests.exceptions.Timeout:
                answer = "I’m taking longer than expected. Please try again, or ask a shorter question."
            except Exception:
                answer = "Something went wrong on my side. Please try again."

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


    st.rerun()
