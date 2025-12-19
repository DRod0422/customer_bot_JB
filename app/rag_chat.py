import chromadb
from chromadb.config import Settings

from utils import load_config, get_embedding_ollama
from utils import chat_ollama  # <-- we will add this function next


def main():
    config = load_config("config.yaml")

    base_url = config["ollama_base_url"]
    llm_model = config["llm_model"]
    embed_model = config["embedding_model"]

    vector_dir = config["vector_store_dir"]
    collection_name = config["collection_name"]
    top_k = int(config.get("top_k", 3))
    print(f"DEBUG: top_k = {top_k}")



    client = chromadb.PersistentClient(
        path=vector_dir,
        settings=Settings(allow_reset=False),
    )
    collection = client.get_collection(name=collection_name)

    system_prompt = (
        "You are a helpful assistant that answers questions using ONLY the provided context.\n"
        "If the answer is not clearly supported by the context, say you don't know.\n"
        "Be concise.\n"
    )

    print("\nLocal RAG Bot (type 'exit' to quit)\n")

    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit", "q"):
            print("Bye.")
            break

        q_emb = get_embedding_ollama(q, base_url=base_url, model=embed_model)

        results = collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        context_blocks = []
        for doc, meta in zip(docs, metas):
            src = meta.get("source", "unknown")
            idx = meta.get("chunk_index", -1)
            context_blocks.append(f"[Source: {src}, chunk {idx}]\n{doc}")

        context = "\n\n---\n\n".join(context_blocks)

        user_prompt = (
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION:\n{q}\n\n"
            "Answer using ONLY the context. If not in context, say you don't know."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        answer = chat_ollama(messages=messages, base_url=base_url, model=llm_model)
        print("\nBot:", answer)

        print("\nSources used:")
        for meta in metas:
            print(f"- {meta.get('source')} (chunk {meta.get('chunk_index')})")
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
