import os
from fastapi import Header, HTTPException
from fastapi import FastAPI
from pydantic import BaseModel

import chromadb
from chromadb.config import Settings

from utils import load_config, get_embedding_ollama, chat_ollama

app = FastAPI(title="Local RAG Backend")
API_KEY = os.getenv("CUSTOMER_BOT_API_KEY","")
# -------------------------
# Load config & vector DB
# -------------------------
config = load_config("config.yaml")

BASE_URL = config["ollama_base_url"]
LLM_MODEL = config["llm_model"]
EMBED_MODEL = config["embedding_model"]
VECTOR_DIR = config["vector_store_dir"]
COLLECTION_NAME = config["collection_name"]
TOP_K = int(config.get("top_k", 3))

client = chromadb.PersistentClient(
    path=VECTOR_DIR,
    settings=Settings(allow_reset=False),
)
collection = client.get_collection(name=COLLECTION_NAME)

SYSTEM_PROMPT = """
You are a leadership and professional development AI assistant.

PRIMARY RULES:
- Use the provided document context when it is relevant.
- If a concept, book, framework, or program is described across multiple sections,
  summarize and explain it even if no single sentence defines it.
- If the documents do NOT contain the answer, you may answer using general leadership
  and business knowledge.
- Only cite documents when they actually support the answer.
- If answering from general knowledge, clearly state that the answer is based on
  general knowledge and not the provided documents.
- Do not fabricate citations.

Your goal is to be helpful, clear, and accurate — not overly restrictive.
"""

# -------------------------
# Request / Response models
# -------------------------
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: list

# -------------------------
# Routes
# -------------------------
@app.get("/health")
def health(x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"status": "ok"}


from fastapi import Header, HTTPException
import os
import requests

# --- tune these ---
TOP_K = int(os.getenv("RAG_TOP_K", "8"))         # more context helps “What is X?”
MIN_CHARS_CONTEXT = 200                         # if context too tiny, treat as “no docs”
OLLAMA_CHAT_URL = os.getenv("OLLAMA_CHAT_URL", "http://127.0.0.1:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")  # set to what you actually run

def _ollama_chat(messages: list[dict], model: str = OLLAMA_MODEL) -> str:
    payload = {"model": model, "messages": messages, "stream": False}
    r = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"]

def _format_sources(metadatas: list[dict], ids: list[str]) -> list[dict]:
    """
    Convert Chroma metadatas into a clean list of sources.
    Assumes each metadata includes: source, page, chunk (or similar).
    Falls back gracefully.
    """
    sources = []
    for i, md in enumerate(metadatas):
        src = md.get("source") or md.get("file") or md.get("filename") or "unknown"
        page = md.get("page")
        chunk = md.get("chunk")
        sources.append({
            "source": src,
            "page": page,
            "chunk": chunk,
            "id": ids[i] if i < len(ids) else None
        })
    return sources

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, x_api_key: str | None = Header(default=None)):
    # --- auth ---
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    question = (req.question or "").strip()
    if not question:
        return {"answer": "Please enter a question.", "sources": []}

    # --- retrieve from vector store ---
    # --- retrieve from vector store (CORRECT: query by embedding) ---
    try:
        q_emb = get_embedding_ollama(question)  # must be same function/model used during ingest
        results = collection.query(
            query_embeddings=[q_emb],
            n_results=TOP_K,
            include=["documents", "metadatas", "ids", "distances"]
        )
        docs = results.get("documents", [[]])[0] or []
        metadatas = results.get("metadatas", [[]])[0] or []
        ids = results.get("ids", [[]])[0] or []
        distances = results.get("distances", [[]])[0] or []
    except Exception:
        docs, metadatas, ids, distances = [], [], [], []


    # --- build context string ---
    context = ""
    if docs:
        context = "\n\n---\n\n".join(docs).strip()

    use_docs = bool(context) and len(context) >= MIN_CHARS_CONTEXT

    # --- user prompt construction ---
    if use_docs:
        user_prompt = f"""
Question:
{question}

Document Context (use this first):
{context}

Instructions:
- Answer using the document context above.
- If the concept is described across multiple sections, summarize it clearly.
- If the documents do not contain enough info, say what is missing and then provide a short general explanation.
- Provide a clean answer. Do NOT invent citations.
"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        answer = _ollama_chat(messages)

        # Return sources (document-based)
        sources = _format_sources(metadatas, ids)

        return {"answer": answer, "sources": sources}

    else:
        # --- fallback: general knowledge ---
        user_prompt = f"""
Question:
{question}

Instructions:
- Answer using general leadership and business knowledge.
- Be clear and helpful.
- Do not mention documents or citations unless you actually used them.
- If you are unsure, say so briefly and ask a clarifying question.
"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        answer = _ollama_chat(messages)

        # No sources when not using docs
        return {"answer": answer, "sources": []}
