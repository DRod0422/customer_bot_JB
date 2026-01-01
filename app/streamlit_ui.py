import requests
import streamlit as st

st.set_page_config(page_title="Customer Bot (Local)", layout="wide")
st.title("Customer Bot (Local RAG)")

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

# -------------------------------
# Session state
# -------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask a question about the documents...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=90)

            if resp.status_code != 200:
                st.error(f"Sorry — the assistant hit an error ({resp.status_code}). Please try again.")
                with st.expander("Details (for admin/debug)"):
                    st.code(resp.text[:4000])
                st.stop()
            
            data = resp.json()


        st.markdown(data["answer"])

        with st.expander("Sources"):
            for s in data.get("sources", []):
                if s.get("page"):
                    st.write(f"- {s['source']} (p. {s['page']})")
                else:
                    st.write(f"- {s['source']} (chunk {s.get('chunk', '?')})")


    st.session_state.messages.append({"role": "assistant", "content": data["answer"]})
