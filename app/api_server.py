import requests
import os
from fastapi import Header, HTTPException
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from typing import Optional


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



# --- tune these ---
TOP_K = int(os.getenv("RAG_TOP_K", "8"))         # more context helps “What is X?”
MIN_CHARS_CONTEXT = 200                         # if context too tiny, treat as “no docs”
OLLAMA_CHAT_URL = os.getenv("OLLAMA_CHAT_URL", "http://127.0.0.1:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", LLM_MODEL)

def _ollama_chat(messages: list[dict], model: str = OLLAMA_MODEL) -> str:
    payload = {"model": model, "messages": messages, "stream": False, "options": {"num_predict":300, "temperature": 0.2}}
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
        chunk = md.get("chunk_index") or md.get("chunk")
        sources.append({
            "source": src,
            "page": page,
            "chunk": chunk,
            "id": ids[i] if i < len(ids) else None
        })
    return sources
    

def detect_doc_hint(text: str) -> Optional[str]:
    t = (text or "").lower()
    for hint in DOC_HINTS.keys():
        if hint in t:
            return hint
    return None

def cosine_top_chunks_for_sources(collection, q_emb: list[float], sources: list[str], top_n: int = 4):
    """
    Pull chunks for specific sources (via .get $eq), cosine rank locally, return top_n.
    Works even when Chroma doesn't support $contains in query filters.
    """
    q = np.array(q_emb, dtype=np.float32)
    qn = q / (np.linalg.norm(q) + 1e-9)

    candidates = []
    for src in sources:
        r = collection.get(
            where={"source": {"$eq": src}},
            include=["documents", "metadatas", "embeddings"],  # ids returned automatically
        )
        ids = r.get("ids", [])
        if not ids:
            continue

        E = np.array(r["embeddings"], dtype=np.float32)
        En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        sims = En @ qn

        idx = np.argsort(-sims)[: min(top_n, len(sims))]
        for j in idx:
            candidates.append({
                "id": ids[j],
                "doc": r["documents"][j],
                "md": r["metadatas"][j],
                "score": float(sims[j]),
            })

    candidates.sort(key=lambda x: x["score"], reverse=True)

    # dedupe by id, keep top_n
    seen = set()
    out = []
    for c in candidates:
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        out.append(c)
        if len(out) >= top_n:
            break
    return out

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
               # Always embed query with the SAME Ollama embedding model used during ingest
        q_emb = get_embedding_ollama(question, base_url=BASE_URL, model=EMBED_MODEL)
        
        # Simple global retrieval (FAST, production-safe)
        results = collection.query(
            query_embeddings=[q_emb],
            n_results=6,
            include=["documents", "metadatas", "distances"]
        )
        
        docs = results.get("documents", [[]])[0] or []
        metadatas = results.get("metadatas", [[]])[0] or []
        ids = results.get("ids", [[]])[0] or []
        
        # Hard cap what we send to the LLM (CPU speed)
        docs = docs[:4]
        metadatas = metadatas[:4]
        ids = ids[:4]

    
        # # 3) Merge (hinted first), dedupe by (source, chunk_index)
        # merged_docs = []
        # merged_mds = []
        # merged_ids = []
        # seen = set()
    
        # def k(md):
        #     return (md.get("source"), md.get("chunk_index"))
    
        # for d, md, _id in zip(hint_docs, hint_mds, hint_ids):
        #     kk = k(md)
        #     if kk not in seen:
        #         seen.add(kk)
        #         merged_docs.append(d)
        #         merged_mds.append(md)
        #         merged_ids.append(_id)
    
        # for d, md, _id in zip(docs, metadatas, ids):
        #     kk = k(md)
        #     if kk not in seen:
        #         seen.add(kk)
        #         merged_docs.append(d)
        #         merged_mds.append(md)
        #         merged_ids.append(_id)
    
        
        # overwrite outputs used downstream
        # Prioritize hinted docs (first) + a little global backup
        MAX_HINTED = 3
        MAX_GLOBAL = 1
        docs = merged_docs[: (MAX_HINTED + MAX_GLOBAL)]
        metadatas = merged_mds[: (MAX_HINTED + MAX_GLOBAL)]
        ids = merged_ids[: (MAX_HINTED + MAX_GLOBAL)]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {type(e).__name__}: {e}")





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

        return {"answer": answer, "sources": []}

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
