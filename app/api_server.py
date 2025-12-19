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

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided context.\n"
    "If the answer is not clearly supported by the context, say you don't know.\n"
    "If multiple relevant ideas exist, you may summarize them concisely.\n"
)

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


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")



    # 1. Embed question
    q_emb = get_embedding_ollama(
        question,
        base_url=BASE_URL,
        model=EMBED_MODEL,
    )

    # 2. Retrieve context
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=TOP_K,
        include=["documents", "metadatas"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    context_blocks = []
    sources = []

    for doc, meta in zip(docs, metas):
        src = meta.get("source", "unknown")
        page = meta.get("page", None)
        idx = meta.get("chunk_index", -1)
        doc = doc[:1200]

        page_tag = f"p. {page}" if page else f"chunk {idx}"
        context_blocks.append(f"[Source: {src}, {page_tag}]\n{doc}")

        sources.append({
            "source": src,
            "page": page,
            "chunk": idx
        })

    context = "\n\n---\n\n".join(context_blocks)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"CONTEXT:\n{context}\n\n"
                f"QUESTION:\n{question}\n\n"
                "Answer using ONLY the context."
            ),
        },
    ]

    answer = chat_ollama(
        messages=messages,
        base_url=BASE_URL,
        model=LLM_MODEL,
    )

    return {
        "answer": answer,
        "sources": sources,
    }
