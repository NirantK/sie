import os
import time

from dotenv import load_dotenv
from loguru import logger
from sie_sdk import SIEClient

load_dotenv()

base_url = os.environ["SIE_BASE_URL"]
api_key = os.environ["SIE_API_KEY"]

def main():
    client = SIEClient(base_url, api_key=api_key, timeout_s=900)

    # Discover available models
    t0 = time.perf_counter()
    models = client.list_models()
    logger.info(f"list_models took {time.perf_counter() - t0:.3f}s — {len(models)} models")
    loaded = [m for m in models if m["loaded"]]
    logger.info(f"  {len(loaded)} loaded, {len(models) - len(loaded)} unloaded")
    for m in loaded:
        logger.info(f"  [loaded] {m['name']}")

    # Encode
    logger.info("--- Encode ---")
    t0 = time.perf_counter()
    result = client.encode(
        "mixedbread-ai/mxbai-colbert-large-v1",
        {"text": "Superlinked makes vector infra easy"},
        output_types=["multivector"],
        gpu="l4-spot",
        wait_for_capacity=True,
        provision_timeout_s=900,
    )
    elapsed = time.perf_counter() - t0
    logger.info(f"encode took {elapsed:.3f}s")
    if "id" in result:
        logger.info(f"  id={result['id']}")
    if "multivector" in result:
        logger.info(f"  multivector shape={result['multivector'].shape}")
    if "dense" in result:
        logger.info(f"  dense shape={result['dense'].shape}")

    # Score (rerank)
    logger.info("--- Score ---")
    t0 = time.perf_counter()
    score_result = client.score(
        "mixedbread-ai/mxbai-rerank-base-v2",
        query={"text": "best vector database"},
        items=[
            {"text": "Superlinked is a vector compute platform"},
            {"text": "The weather is nice today"},
            {"text": "Vector search enables semantic retrieval"},
        ],
        gpu="l4-spot",
        wait_for_capacity=True,
        provision_timeout_s=900,
    )
    elapsed = time.perf_counter() - t0
    logger.info(f"score took {elapsed:.3f}s")
    if "query_id" in score_result:
        logger.info(f"  query_id={score_result['query_id']}")
    for s in score_result["scores"]:
        logger.info(f"    rank={s['rank']}  score={s['score']:.4f}  item_id={s['item_id']}")

    # Extract (GLINer NER)
    logger.info("--- Extract ---")
    t0 = time.perf_counter()
    extract_result = client.extract(
        "urchade/gliner_multi-v2.1",
        {"text": "Superlinked is based in San Francisco and was founded by Daniel Svonava."},
        labels=["person", "organization", "location"],
        gpu="l4-spot",
        wait_for_capacity=True,
        provision_timeout_s=900,
    )
    elapsed = time.perf_counter() - t0
    logger.info(f"extract took {elapsed:.3f}s")
    if "id" in extract_result:
        logger.info(f"  id={extract_result['id']}")
    for e in extract_result["entities"]:
        logger.info(f"    text={e['text']}  label={e['label']}  score={e['score']:.4f}")


if __name__ == "__main__":
    main()
