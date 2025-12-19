import requests
import yaml

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

base_url = config["ollama_base_url"]
model = config["embedding_model"]

text = "This is a test sentence for embeddings."

response = requests.post(
    f"{base_url}/api/embeddings",
    json={
        "model": model,
        "prompt": text
    },
    timeout=30
)

response.raise_for_status()
data = response.json()

embedding = data.get("embedding", [])

print("Embedding length:", len(embedding))
print("First 5 values:", embedding[:5])
