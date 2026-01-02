import streamlit as st
import requests
import time

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="JB Assistant", layout="centered")

API_BASE = st.secrets.get("API_BASE", "http://127.0.0.1:8000")
API_KEY = st.secrets.get("X_API_KEY", "")


# quick health check
# -------------------------------
# Backend health check
# -------------------------------
try:
    health_resp = requests.get(
        f"{API_BASE}/health",
        headers={"x-api-key": API_KEY} if API_KEY else {},
        timeout=3
    )
    health_resp.raise_for_status()
    health = health_resp.json()
    st.caption(f"Backend status: {health.get('status', 'unknown')}")
except Exception as e:
    st.error("Backend is not reachable. Please try again later.")
    st.stop()

# ----------------------------
# Minimal CSS to mimic the look
# ----------------------------
st.markdown("""
<style>
/* tighten page width + center */
.block-container { max-width: 820px; padding-top: 2rem; }

/* big title like screenshot */
h1 { font-size: 3.0rem !important; letter-spacing: -0.5px; }
.subtle { font-size: 1.1rem; opacity: 0.75; margin-top: -0.5rem; }

/* Chat "card" container */
.chat-card {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  padding: 22px;
  background: rgba(255,255,255,0.03);
  margin-top: 18px;
}

/* Header row inside card (avatar + optional button) */
.card-top {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 16px;
}
.avatar {
  width: 40px; height: 40px; border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.18);
  object-fit: cover;
}

/* message bubbles */
.bubble {
  padding: 14px 16px;
  border-radius: 14px;
  margin: 10px 0;
  line-height: 1.35;
  border: 1px solid rgba(255,255,255,0.10);
}
.user {
  background: rgba(66,133,244,0.12);
  margin-left: 18%;
}
.assistant {
  background: rgba(255,255,255,0.06);
  margin-right: 18%;
}

/* small helper text */
.helper { opacity: .7; font-size: .95rem; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Header (matches your screenshot vibe)
# ----------------------------
st.title("Customer Bot (Local RAG)")
st.markdown('<div class="subtle">Experience and marketing questions 24 hours a day. Try to stump him!</div>', unsafe_allow_html=True)

# ----------------------------
# Session state
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Great to see you. What leadership question can I help with today?"}
    ]

def reset_chat():
    st.session_state.messages = [
        {"role": "assistant", "content": "Great to see you. What leadership question can I help with today?"}
    ]

# ----------------------------
# Chat Card UI
# ----------------------------
st.markdown('<div class="chat-card">', unsafe_allow_html=True)

colA, colB = st.columns([6, 1])
with colA:
    # replace with your own hosted image if you want
    avatar_url = "https://i.imgur.com/0y0y0y0.png"  # placeholder
    st.markdown(f"""
      <div class="card-top">
        <img class="avatar" src="{avatar_url}" />
        <div></div>
      </div>
    """, unsafe_allow_html=True)

with colB:
    st.button("↻", on_click=reset_chat, help="Reset chat")

# render message history
for m in st.session_state.messages:
    css_class = "assistant" if m["role"] == "assistant" else "user"
    st.markdown(f'<div class="bubble {css_class}">{m["content"]}</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Input + API call
# ----------------------------
prompt = st.chat_input("Ask your question here...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    # optimistic render of user msg immediately
    st.rerun()

# If the last message is a user message, fetch assistant response
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    question = st.session_state.messages[-1]["content"]

    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["x-api-key"] = API_KEY

    with st.spinner("Thinking..."):
        try:
            resp = requests.post(
                f"{API_BASE}/ask",
                headers=headers,
                json={"question": question},
                timeout=120  # bump timeout a bit for streamlit cloud
            )

            # If backend error, show friendly msg
            if resp.status_code != 200:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "I’m taking longer than expected. Please try again, or ask a shorter question."
                })
            else:
                data = resp.json()
                answer = data.get("answer", "").strip() or "I’m not sure how to answer that yet."
                st.session_state.messages.append({"role": "assistant", "content": answer})

        except requests.exceptions.Timeout:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "I’m taking longer than expected to answer this. Please try again in a moment."
            })
        except Exception:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Something went sideways on my end. Please try again."
            })

    st.rerun()
