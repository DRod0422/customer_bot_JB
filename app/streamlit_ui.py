import streamlit as st
import requests
import time


# ----------------------------
# Page config
# ----------------------------
# st.set_page_config(page_title="John Bentley AI Assistant", layout="centered")

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
        padding: 0px 18px 8px 18px;
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
col1, col2 = st.columns([1, 6], vertical_alignment="center")
with col1:
    st.image("jb_avatar.jpeg", width=90)
with col2:
    st.title("John Bentley's Leadership Assistant")
    st.caption("Ask anything about John Bentley’s training, books, and leadership frameworks.")
    # st.rerun()

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
    if m["role"] == "assistant":
        avatar = "jb_avatar.jpeg"   # path relative to this file
    else:
        avatar = "🧑‍💼"  # or None

    with st.chat_message(m["role"], avatar=avatar):
        st.markdown(m["content"])

st.markdown("</div>", unsafe_allow_html=True)

# ---- Single input (bottom) ----
prompt = st.chat_input("Ask your question here…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar= "🧑‍💼" ):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="jb_avatar.jpeg"):
        with st.spinner("Thinking…"):
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
                answer = "I’m taking longer than expected. Please try again, or ask a shorter question."
            except Exception:
                answer = "Something went wrong on my side. Please try again."

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


    st.rerun()
