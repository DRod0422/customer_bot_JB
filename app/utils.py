import yaml
import requests
from typing import List
from pypdf import PdfReader


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_pages_from_pdf(pdf_path: str):
    reader = PdfReader(pdf_path)
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append((page_num, text))
    return pages



def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = " ".join(text.split())

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)

    return chunks


def get_embedding_ollama(text: str, base_url: str, model: str) -> List[float]:
    response = requests.post(
        f"{base_url}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["embedding"]


def chat_ollama(messages, base_url: str, model: str) -> str:
    response = requests.post(
        f"{base_url}/api/chat",
        json={"model": model, "messages": messages, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]
