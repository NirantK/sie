import os

import httpx
from dotenv import load_dotenv
from sie_sdk import SIEClient

load_dotenv()

endpoint = os.environ["API_ENDPOINT"]
api_key = os.environ["API_KEY"]

client = SIEClient(endpoint, api_key=api_key)

# Check server health first
print("=== Health ===")
resp = httpx.get(f"{endpoint}/health", headers={"Authorization": f"Bearer {api_key}"})
health = resp.json()
print(f"Status: {health['status']}, workers: {health['cluster']['worker_count']}, models: {health['models']}")

if health["status"] == "degraded" and health["cluster"]["worker_count"] == 0:
    print("\nServer is degraded with no workers. Calls will fail until workers spin up.")
    print("Re-run with wait_for_capacity=True (already set) once workers are available.")

MODEL_ENCODE = "bge-m3"
MODEL_SCORE = "bge-reranker-v2-m3"
MODEL_EXTRACT = "gliner-multi-v2.1"

# 1. Encode
print("\n=== Encode ===")
results = client.encode(
    MODEL_ENCODE,
    [{"text": "Hello world"}, {"text": "SIE SDK test"}],
    wait_for_capacity=True,
)
print(f"Got {len(results)} results")
for i, r in enumerate(results):
    dense = r.get("dense")
    if dense is not None:
        print(f"  [{i}] dense dim={len(dense)}")

# 2. Score
print("\n=== Score ===")
score_result = client.score(
    MODEL_SCORE,
    query={"text": "What is machine learning?"},
    items=[
        {"text": "Machine learning is a subset of AI."},
        {"text": "The weather is nice today."},
    ],
    wait_for_capacity=True,
)
print(f"Score result: {score_result}")

# 3. Extract
print("\n=== Extract ===")
extract_results = client.extract(
    MODEL_EXTRACT,
    [{"text": "Apple was founded by Steve Jobs in Cupertino."}],
    labels=["person", "organization", "location"],
    wait_for_capacity=True,
)
print(f"Got {len(extract_results)} extraction results")
for i, r in enumerate(extract_results):
    entities = r.get("entities", [])
    print(f"  [{i}] {len(entities)} entities")
    for e in entities:
        print(f"    - {e.get('text')} ({e.get('label')})")
