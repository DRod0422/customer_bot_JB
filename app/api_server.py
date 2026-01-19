import requests
import os
from datetime import datetime
from fastapi import Header, HTTPException
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from typing import Optional


import chromadb
from chromadb.config import Settings

from utils import load_config, get_embedding_ollama, chat_ollama
import json
import time
import psutil
from pathlib import Path
from sentence_transformers import SentenceTransformer

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

print("Loading MiniLM embedding model...")
minilm_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("✅ MiniLM loaded")

# Logging setup
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
USAGE_LOG = LOG_DIR / "usage.jsonl"
ERROR_LOG = LOG_DIR / "errors.jsonl"
HEALTH_LOG = LOG_DIR / "health.jsonl"

def log_query(question, response_time, sources_count, error=None):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question_length": len(question),
        "response_time_sec": round(response_time, 2),
        "sources_retrieved": sources_count,
        "error": error,
        "client": "john_bentley"
    }
    with open(USAGE_LOG, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    

def log_system_health():
    health = {
        "timestamp": datetime.now().isoformat(),
        "cpu_percent": psutil.cpu_percent(interval=1),
        "ram_percent": psutil.virtual_memory().percent,
        "ram_used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
        "disk_percent": psutil.disk_usage('/').percent
    }
    try:
        with open(HEALTH_LOG, "a") as f:
            f.write(json.dumps(health) + "\n")
    except Exception as e:
        print(f"Failed to log health: {e}")


SYSTEM_PROMPT = """
You are a helpful, confident leadership training assistant speaking in John Bentley’s teaching style:
clear, practical, encouraging, and direct.

FRAMEWORK COMPLETENESS RULE:

When a user asks about a named framework, model, system, methodology, or structured concept
(e.g., DISC, I Lead Me, leadership models, personality frameworks, training programs):

1. Assume the user expects a COMPLETE explanation of the full framework.
2. Do not base completeness on which components appear most frequently in retrieved documents.
3. Identify all core components of the framework and explain each one clearly and proportionally.
4. If document context emphasizes some components more than others, use general knowledge to
   fill gaps so the explanation remains balanced and accurate.
5. Never substitute or rename components unless explicitly asked to compare frameworks.
6. If the documents conflict or are incomplete, explain the framework as it is commonly taught
   and practiced, then align it back to the author’s intent and material where applicable.
7. Prioritize teaching clarity and consistency over literal document summarization.

The goal is to teach the framework the way a knowledgeable instructor would, not to mirror
document distribution or chunk frequency.


COMPLETENESS + NO TRAILING OFF RULE:

- Never end an answer mid-sentence or mid-bullet.
- Every bullet must be a complete thought: include a clear point + a brief explanation.
- If you are approaching the output limit, finish the current bullet fully, then stop.
- Prefer fewer complete bullets over more incomplete bullets.
- Use short paragraphs inside bullets if needed for clarity.

BULLET DEPTH RULE:

When responding in bullets, format each bullet as:
- Bold heading + 2–4 sentences of explanation + 1 practical example or action step.

Always end every response with the token <END>.


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
    
    log_system_health()  # ← Add this
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
TOP_K = int(os.getenv("RAG_TOP_K", "8"))
MIN_CHARS_CONTEXT = 200
OLLAMA_CHAT_URL = os.getenv("OLLAMA_CHAT_URL", "http://127.0.0.1:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", LLM_MODEL)

def _ollama_chat(messages: list[dict], model: str = OLLAMA_MODEL) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "num_predict": 2048,
            "temperature": 0.2,
            "num_ctx": 8192,
            "top_p": 0.9
        },
    }
    try:
        r = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=180)
        r.raise_for_status()
        return r.json()["message"]["content"]
    except requests.exceptions.Timeout:
        return "I'm taking longer than expected to answer this. Please try again, or ask a shorter question."
    except Exception as e:
        print(f"🔴 OLLAMA ERROR: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

def _ollama_chat_with_continuation(messages: list[dict], model: str = OLLAMA_MODEL, max_continuations: int = 2) -> str:
    """Chat with automatic continuation if response is cut off."""
    full_response = ""
    continuation_count = 0
    
    while continuation_count <= max_continuations:
        raw_answer = _ollama_chat(messages, model).strip()
        full_response += ("\n" if full_response else "") + raw_answer
        
        # Check if response completed
        if "<END>" in raw_answer:
            break
            
        # Check if we hit token limit (response ends mid-sentence)
        if raw_answer and not raw_answer.endswith((".", "!", "?", "<END>")):
            messages.append({"role": "assistant", "content": raw_answer})
            messages.append({
                "role": "user",
                "content": "Continue exactly where you left off. Finish any cut-off bullets. End with <END>."
            })
            continuation_count += 1
        else:
            break
    
    return full_response.replace("<END>", "").strip()


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
    start_time = time.time() 
    # --- auth ---
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    question = (req.question or "").strip()
    if not question:
        response_time = time.time() - start_time
        log_query(question, response_time, 0)
        return {"answer": "Please enter a question.", "sources": []}

    # --- retrieve from vector store ---
    docs: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    try:
        # Embed query using SAME embed model used during ingest
        q_emb = minilm_model.encode(question, convert_to_numpy=True).tolist()

        preferred_sources = [
            "I Lead Me (JB).pdf",
            "I LEAD ME The 4 Self-Leadership Patterns.pdf",
        ]

        # 1) Primary retrieval
        if is_ileadme_intent(question):
            results = collection.query(
                query_embeddings=[q_emb],
                n_results=6,
                where={"source": {"$in": preferred_sources}},
                include=["documents", "metadatas"],
            )
        else:
            results = collection.query(
                query_embeddings=[q_emb],
                n_results=6,
                include=["documents", "metadatas"],
            )

        docs = (results.get("documents") or [[]])[0] or []
        metadatas = (results.get("metadatas") or [[]])[0] or []
        ids = (results.get("ids") or [[]])[0] or []

        # 2) If asking about I Lead Me but retrieval came back thin
        if is_ileadme_intent(question) and len(docs) < 3:
            for src in preferred_sources:
                r = collection.get(
                    where={"source": {"$eq": src}},
                    include=["documents", "metadatas"],
                )
                docs += (r.get("documents") or [])[:2]
                metadatas += (r.get("metadatas") or [])[:2]
                ids += (r.get("ids") or [])[:2]

        # 3) If still thin, do a small global retrieval as backup
        if len(docs) < 4:
            results2 = collection.query(
                query_embeddings=[q_emb],
                n_results=4,
                include=["documents", "metadatas"],
            )
            docs += (results2.get("documents") or [[]])[0] or []
            metadatas += (results2.get("metadatas") or [[]])[0] or []
            ids += (results2.get("ids") or [[]])[0] or []

        # 4) Hard cap context size
        docs = docs[:4]
        metadatas = metadatas[:4]
        ids = ids[:4]

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval error: {type(e).__name__}: {e}",
        )

    # --- build context string ---
    MAX_CONTEXT_CHARS = 2500
    context = ""
    if docs:
        context = "\n\n---\n\n".join(docs)
        context = context[:MAX_CONTEXT_CHARS].strip()
    
    use_docs = len(context) >= MIN_CHARS_CONTEXT
    
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
- Keep it clean and practical.

Formatting:
- Use markdown bullets with "- " only.
- Do not use the "•" character.
"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        answer = _ollama_chat_with_continuation(messages)
        sources = _format_sources(metadatas, ids)
        
        # ADD LOGGING HERE (YOU WERE MISSING THIS!)
        response_time = time.time() - start_time
        log_query(question, response_time, len(sources))
        
        return {"answer": answer, "sources": sources}
    
    else:
        # --- fallback: general knowledge ---
        user_prompt = f"""
Question:
{question}

Answer instructions:
- Answer confidently using leadership and business knowledge.
- Do not mention documents, sources, or citations.
- Keep it clean and practical.

Formatting:
- Use markdown bullets with "- " only.
- Do not use the "•" character.
"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        answer = _ollama_chat_with_continuation(messages)
        sources = _format_sources(metadatas, ids)
        
        response_time = time.time() - start_time
        log_query(question, response_time, len(sources))
        
        return {"answer": answer, "sources": sources}
