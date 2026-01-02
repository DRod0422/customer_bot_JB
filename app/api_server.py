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
You are a helpful, confident leadership training assistant speaking in John Bentley’s teaching style:
clear, practical, encouraging, and direct.

Behavior rules:
- Answer like a human coach/instructor, not like a document search tool.
- Use the provided document context to shape your answer when it is relevant.
- If the documents don’t fully cover something, fill gaps with general leadership/business knowledge.
- NEVER mention the documents, chunks, retrieval, citations, or “not explicitly mentioned.”
- Do not apologize or hedge (“might,” “perhaps,” “it can be inferred”) unless the user asks for uncertainty.
- Keep answers structured: short intro + 3–7 bullets + a practical next step.
- If the user asks about John Bentley specifically, speak in a professional third-person tone (no pretending to be him).
- If asked about ROI, provide a simple, practical ROI framework and example assumptions.
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


# -------------------------
# Intent helpers
# -------------------------
def is_reference_intent(q: str) -> bool:
    q = (q or "").lower()
    triggers = [
        "quote", "page", "cite", "source",
        "where does it say", "references",
        "bibliography", "list titles", "documents included"
    ]
    return any(t in q for t in triggers)

def is_ileadme_intent(q: str) -> bool:
    q = (q or "").lower()
    return ("i lead me" in q) or ("ileadme" in q)



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
        TEACHER_MODE = not is_reference_intent(question)

        q_emb = get_embedding_ollama(question, base_url=BASE_URL, model=EMBED_MODEL)
        
        if is_ileadme_intent(question):
            results = collection.query(
                query_embeddings=[q_emb],
                n_results=6,
                where={"source": {"$in": [
                    "I Lead Me (JB).pdf",
                    "I LEAD ME The 4 Self-Leadership Patterns.pdf"
                ]}},
                include=["documents", "metadatas"]
            )
        else:
            results = collection.query(
                query_embeddings=[q_emb],
                n_results=6,
                include=["documents", "metadatas"]
            )
        docs = results.get("documents", [[]])[0] or []
        metadatas = results.get("metadatas", [[]])[0] or []
        ids = results.get("ids", [[]])[0] or []

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval error: {type(e).__name__}: {e}"
        )
        # Fast hint: if the question mentions I Lead Me, prioritize those files
        q_low = question.lower()
        if "i lead me" in q_low or "ileadme" in q_low:
            preferred_sources = ["I Lead Me (JB).pdf", "I LEAD ME The 4 Self-Leadership Patterns.pdf"]
            for src in preferred_sources:
                r = collection.get(where={"source": {"$eq": src}}, include=["documents", "metadatas"])
                # take first ~2 chunks from each doc (simple + fast)
                docs += (r.get("documents") or [])[:2]
                metadatas += (r.get("metadatas") or [])[:2]
                ids += (r.get("ids") or [])[:2]
        
        # If still not enough context, add global results
        if len(docs) < 4:
            results = collection.query(
                query_embeddings=[q_emb],
                n_results=6,
                include=["documents","metadatas"]
            )
            docs += (results.get("documents", [[]])[0] or [])
            metadatas += (results.get("metadatas", [[]])[0] or [])
            ids += (results.get("ids", [[]])[0] or [])
        
        # Hard cap
        docs = docs[:4]
        metadatas = metadatas[:4]
        ids = ids[:4]

    
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

Context (use if helpful):
{context}

Answer instructions:
- Give the best possible answer in a confident, coach-like tone.
- Use the context above when it supports the answer, but do not mention it.
- If ROI is asked, give a simple ROI model with variables (turnover, productivity, rework, incidents) and a sample calculation.
- Keep it clean and practical.
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

Answer instructions:
- Answer confidently using leadership and business knowledge.
- Do not mention documents, sources, or citations.
- Keep it clean and practical.
"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        answer = _ollama_chat(messages)

        # No sources when not using docs
        return {"answer": answer, "sources": []}
