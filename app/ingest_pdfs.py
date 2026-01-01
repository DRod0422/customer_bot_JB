import os
import  uuid

import os
import uuid

import chromadb
from chromadb.config import Settings

from utils import load_config, extract_pages_from_pdf, chunk_text, get_embedding_ollama


def main():
    config = load_config("config.yaml")

    pdf_dir = config["pdf_dir"]
    vector_dir = config["vector_store_dir"]
    collection_name = config["collection_name"]

    chunk_size = int(config["chunk_size"])
    chunk_overlap = int(config["chunk_overlap"])

    base_url = config["ollama_base_url"]
    embed_model = config["embedding_model"]

    os.makedirs(vector_store_dir, exist_ok=True)

    client = chromadb.PersistentClient(
        path=vector_store_dir,
        settings=Settings(allow_reset=True),
    )
    collection = client.get_or_create_collection(name=collection_name)

    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDFs found in: {pdf_dir}")
        return

    total_added = 0

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        print(f"\nProcessing: {pdf_file}")
        

                # Skip PDFs already ingested
        existing = collection.get(
            where={"source": pdf_file},
            include=[]
        )
        if existing and existing.get("ids"):
            print(f"  Skipping (already ingested): {pdf_file}  (chunks: {len(existing['ids'])})")
            continue
            
        pages = extract_pages_from_pdf(pdf_path)
        # If pages returns tuples like (page_num, text) or (text, meta), grab the text part
        normalized_pages = []
        for p in pages:
            if p is None:
                continue
            if isinstance(p, tuple):
                # try common tuple shapes: (page_num, text) or (text, meta)
                # pick the first element that is a string
                text_part = next((x for x in p if isinstance(x, str)), "")
                normalized_pages.append(text_part)
            else:
                normalized_pages.append(str(p))
        
        text = "\n".join(t for t in normalized_pages if t and t.strip())
        
        chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            overlap=chunk_overlap
        )
        
        ids = []
        documents = []
        metadatas = []
        embeddings = []





        for i, chunk in enumerate(chunks):
            emb = get_embedding_ollama(chunk, base_url=base_url, model=embed_model)

            ids.append(str(uuid.uuid4()))
            documents.append(chunk)
            metadatas.append({"source": pdf_file, "chunk_index": i})
            embeddings.append(emb)

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        total_added += len(ids)
        print(f"  Stored: {len(ids)} chunks")

    print(f"\nDone. Total chunks stored: {total_added}")
    print(f"Vector store location: {vector_dir}")
    print(f"Collection name: {collection_name}")


if __name__ == "__main__":
    main()
