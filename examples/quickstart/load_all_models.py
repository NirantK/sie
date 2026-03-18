"""Load all 90 SIE models and measure timing."""
import csv
import os
import time

from dotenv import load_dotenv
from loguru import logger
from sie_sdk import SIEClient

load_dotenv()

base_url = os.environ["SIE_BASE_URL"]
api_key = os.environ["SIE_API_KEY"]

# --- Model classification ---

RERANKER_PATTERNS = ["rerank", "cross-encoder", "gte-reranker"]

EXTRACT_PATTERNS = [
    "gliner", "glirel", "gliner-bert", "NuNER", "Florence-2", "donut",
    "gliclass", "zeroshot", "grounding-dino", "owlv2",
]

COLBERT_PATTERNS = ["colbert", "colpali", "colqwen", "GTE-ModernColBERT", "Reason-ModernColBERT"]
SPARSE_PATTERNS = ["splade", "sparse"]

# Vision encode models (need image input for encode)
VISION_ENCODE_PATTERNS = ["clip", "siglip", "CLIP"]


def classify_model(name: str) -> tuple[str, list[str] | None]:
    """Returns (endpoint, output_types) for a model."""
    lower = name.lower()

    # Score (reranker) models
    for p in RERANKER_PATTERNS:
        if p.lower() in lower:
            return "score", None

    # Extract models
    for p in EXTRACT_PATTERNS:
        if p.lower() in lower:
            return "extract", None

    # ColBERT-style multivector encode
    for p in COLBERT_PATTERNS:
        if p.lower() in lower:
            return "encode", ["multivector"]

    # Sparse encode
    for p in SPARSE_PATTERNS:
        if p.lower() in lower:
            return "encode", ["sparse"]

    # Vision encode models (CLIP, SigLIP) — skip for now, need image bytes
    for p in VISION_ENCODE_PATTERNS:
        if p.lower() in lower:
            return "vision_encode", ["dense"]

    # Default: dense encode
    return "encode", ["dense"]


def try_encode(client, model, output_types, gpu):
    return client.encode(
        model,
        {"text": "Superlinked makes vector infra easy"},
        output_types=output_types,
        gpu=gpu,
        wait_for_capacity=True,
        provision_timeout_s=900,
    )


def try_score(client, model, gpu):
    return client.score(
        model,
        query={"text": "best vector database"},
        items=[
            {"text": "Superlinked is a vector compute platform"},
            {"text": "The weather is nice today"},
        ],
        gpu=gpu,
        wait_for_capacity=True,
        provision_timeout_s=900,
    )


def try_extract(client, model, gpu):
    return client.extract(
        model,
        {"text": "Superlinked is based in San Francisco and was founded by Daniel Svonava."},
        labels=["person", "organization", "location"],
        gpu=gpu,
        wait_for_capacity=True,
        provision_timeout_s=900,
    )


def main():
    client = SIEClient(base_url, api_key=api_key, timeout_s=900)

    models = client.list_models()
    logger.info(f"Total models: {len(models)}")

    results = []
    overall_start = time.perf_counter()

    for i, m in enumerate(models):
        name = m["name"]
        bundle = m["bundles"][0] if m["bundles"] else "default"
        already_loaded = m["loaded"]
        endpoint, output_types = classify_model(name)

        gpu = "l4-spot"

        logger.info(f"[{i+1}/{len(models)}] {name} — endpoint={endpoint}, bundle={bundle}, loaded={already_loaded}")

        t0 = time.perf_counter()
        status = "ok"
        error_msg = ""

        try:
            if endpoint == "encode":
                try_encode(client, name, output_types, gpu)
            elif endpoint == "score":
                try_score(client, name, gpu)
            elif endpoint == "extract":
                try_extract(client, name, gpu)
            elif endpoint == "vision_encode":
                # Create a minimal 2x2 red PNG for vision models
                import struct
                import zlib
                def make_tiny_png():
                    width, height = 2, 2
                    raw = b""
                    for _ in range(height):
                        raw += b"\x00" + b"\xff\x00\x00" * width  # filter byte + RGB
                    def chunk(ctype, data):
                        c = ctype + data
                        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
                    return (
                        b"\x89PNG\r\n\x1a\n"
                        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                        + chunk(b"IDAT", zlib.compress(raw))
                        + chunk(b"IEND", b"")
                    )
                png_bytes = make_tiny_png()
                result = client.encode(
                    name,
                    {"images": [png_bytes]},
                    output_types=output_types,
                    gpu=gpu,
                    wait_for_capacity=True,
                    provision_timeout_s=900,
                )
        except Exception as e:
            status = "error"
            error_msg = str(e)[:200]
            logger.error(f"  FAILED: {error_msg}")

        elapsed = time.perf_counter() - t0
        logger.info(f"  {status} in {elapsed:.1f}s")

        results.append({
            "model": name,
            "bundle": bundle,
            "endpoint": endpoint,
            "was_loaded": already_loaded,
            "status": status,
            "time_s": round(elapsed, 2),
            "error": error_msg,
        })

    total_time = time.perf_counter() - overall_start
    logger.info(f"\n{'='*60}")
    logger.info(f"TOTAL TIME: {total_time:.1f}s ({total_time/60:.1f} min)")

    ok = [r for r in results if r["status"] == "ok"]
    fail = [r for r in results if r["status"] == "error"]
    logger.info(f"OK: {len(ok)}/{len(results)}, FAILED: {len(fail)}/{len(results)}")

    if fail:
        logger.info("\nFailed models:")
        for r in fail:
            logger.info(f"  {r['model']}: {r['error']}")

    # Write CSV
    csv_path = "model_load_times.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "bundle", "endpoint", "was_loaded", "status", "time_s", "error"])
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
