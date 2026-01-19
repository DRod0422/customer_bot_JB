import os
import uuid
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer  # NEW: Import MiniLM
from utils import load_config, extract_pages_from_pdf, chunk_text

def main():
    config = load_config("config.yaml")
    pdf_dir = config["pdf_dir"]
    vector_store_dir = config["vector_store_dir"]
    collection_name = config["collection_name"]
    chunk_size = int(config["chunk_size"])
    chunk_overlap = int(config["chunk_overlap"])
    
    # NEW: Load MiniLM model instead of using Ollama embeddings
    print("Loading MiniLM embedding model...")
    embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("✅ MiniLM model loaded\n")
    
    os.makedirs(vector_store_dir, exist_ok=True)
    
    client = chromadb.PersistentClient(
        path=vector_store_dir,
        settings=Settings(allow_reset=True),
    )
    
    # IMPORTANT: Use a NEW collection name or delete the old one
    # Option 1: Use new collection name
    collection_name_minilm = f"{collection_name}_minilm"
    print(f"Creating new collection: {collection_name_minilm}")
    
    # Option 2: Delete old collection and reuse name (commented out)
    # try:
    #     client.delete_collection(name=collection_name)
    #     print(f"Deleted old collection: {collection_name}")
    # except:
    #     pass
    
    collection = client.get_or_create_collection(name=collection_name_minilm)
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDFs found in: {pdf_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDFs to process\n")
    total_added = 0
    
    for pdf_idx, pdf_file in enumerate(pdf_files, 1):
        pdf_path = os.path.join(pdf_dir, pdf_file)
        print(f"[{pdf_idx}/{len(pdf_files)}] Processing: {pdf_file}")
        
        # Skip PDFs already ingested
        existing = collection.get(
            where={"source": pdf_file},
            include=[]
        )
        if existing and existing.get("ids"):
            print(f"  ⏭️  Skipping (already ingested): {pdf_file} ({len(existing['ids'])} chunks)")
            continue
        
        # Extract text from PDF
        pages = extract_pages_from_pdf(pdf_path)
        
        # Normalize pages
        normalized_pages = []
        for p in pages:
            if p is None:
                continue
            if isinstance(p, tuple):
                text_part = next((x for x in p if isinstance(x, str)), "")
                normalized_pages.append(text_part)
            else:
                normalized_pages.append(str(p))
        
        text = "\n".join(t for t in normalized_pages if t and t.strip())
        
        if not text.strip():
            print(f"  ⚠️  No text extracted from {pdf_file}, skipping")
            continue
        
        # Chunk the text
        chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            overlap=chunk_overlap
        )
        
        print(f"  📄 Extracted {len(chunks)} chunks")
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        embeddings = []
        
        # Generate embeddings with MiniLM
        print(f"  🔢 Generating embeddings...", end="", flush=True)
        for i, chunk in enumerate(chunks):
            # NEW: Use MiniLM instead of Ollama
            emb = embed_model.encode(chunk, convert_to_numpy=True).tolist()
            
            ids.append(str(uuid.uuid4()))
            documents.append(chunk)
            metadatas.append({"source": pdf_file, "chunk_index": i})
            embeddings.append(emb)
        
        print(" ✅")
        
        # Add to ChromaDB
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        
        total_added += len(ids)
        print(f"  ✅ Stored {len(ids)} chunks\n")
    
    print("=" * 60)
    print(f"✅ Ingestion Complete!")
    print(f"📊 Total chunks stored: {total_added}")
    print(f"📁 Vector store: {vector_store_dir}")
    print(f"🗃️  Collection: {collection_name_minilm}")
    print("=" * 60)
    print("\n⚠️  IMPORTANT: Update config.yaml collection name to:")
    print(f"   collection_name: {collection_name_minilm}")
    print()

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
