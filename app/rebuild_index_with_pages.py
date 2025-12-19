import os
import uuid

import chromadb
from chromadb.config import Settings

from utils import load_config, chunk_text, get_embedding_ollama, extract_pages_from_pdf


def main():
    config = load_config("config.yaml")

    pdf_dir = config["pdf_dir"]
    vector_dir = config["vector_store_dir"]
    collection_name = config["collection_name"]

    chunk_size = int(config["chunk_size"])
    chunk_overlap = int(config["chunk_overlap"])

    base_url = config["ollama_base_url"]
    embed_model = config["embedding_model"]

    os.makedirs(vector_dir, exist_ok=True)

    client = chromadb.PersistentClient(
        path=vector_dir,
        settings=Settings(allow_reset=True),
    )

    # Rebuild from scratch:
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(name=collection_name)

    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    pdf_files.sort()

    total_added = 0

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        print(f"\nRebuilding: {pdf_file}")

        pages = extract_pages_from_pdf(pdf_path)

        file_added = 0
        for page_num, page_text in pages:
            if not page_text.strip():
                continue

            chunks = chunk_text(page_text, chunk_size=chunk_size, overlap=chunk_overlap)
            for i, chunk in enumerate(chunks):
                emb = get_embedding_ollama(chunk, base_url=base_url, model=embed_model)

                collection.add(
                    ids=[str(uuid.uuid4())],
                    documents=[chunk],
                    metadatas=[{
                        "source": pdf_file,
                        "page": page_num,
                        "chunk_index": i
                    }],
                    embeddings=[emb],
                )
                total_added += 1
                file_added += 1

        print(f"  Stored chunks: {file_added}")

    print(f"\nDONE. Total chunks stored: {total_added}")
    print(f"Vector store location: {vector_dir}")
    print(f"Collection name: {collection_name}")


if __name__ == "__main__":
    main()
