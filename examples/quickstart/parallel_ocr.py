"""Parallel OCR with Florence-2 via SIE extract.

Two levels of parallelism:
1. Batch: SIE extract accepts list[Item] — send N images per request
2. Thread: Multiple batch requests in flight via ThreadPoolExecutor

Results cached incrementally to cache/ocr_texts.json.
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import polars as pl
from dotenv import load_dotenv
from loguru import logger
from sie_sdk import SIEClient

load_dotenv()

SIE_BASE_URL = os.environ["SIE_BASE_URL"]
SIE_API_KEY = os.environ["SIE_API_KEY"]

GPU = "l4-spot"
PROVISION_TIMEOUT = 900
OCR_MODEL = "microsoft/Florence-2-base"
CACHE_DIR = Path("cache")
CACHE_FILE = CACHE_DIR / "ocr_texts.json"

BATCH_SIZE = 4      # Images per SIE extract request
N_WORKERS = 4       # Parallel threads (each sends a batch)
CHECKPOINT_EVERY = 50  # Save cache every N pages


def make_client():
    return SIEClient(SIE_BASE_URL, api_key=SIE_API_KEY, timeout_s=900)


def ocr_batch(batch_items):
    """OCR a batch of page images in one SIE extract call.

    Args:
        batch_items: list of (corpus_id, image_bytes, doc_id, page_number)

    Returns:
        list of (corpus_id, extracted_text)
    """
    client = make_client()

    # Build list[Item] — one Item per image
    items = []
    for corpus_id, image_bytes, doc_id, page_number in batch_items:
        if image_bytes:
            items.append({"images": [image_bytes]})
        else:
            items.append({"text": f"Page {page_number} of {doc_id}"})

    results = client.extract(
        OCR_MODEL,
        items,
        options={"task": "<OCR>"},
        gpu=GPU,
        wait_for_capacity=True,
        provision_timeout_s=PROVISION_TIMEOUT,
    )

    output = []
    for i, (corpus_id, image_bytes, doc_id, page_number) in enumerate(batch_items):
        r = results[i] if isinstance(results, list) else results
        page_text = ""
        if "entities" in r:
            page_text = " ".join(e.get("text", "") for e in r["entities"])
        if not page_text.strip():
            page_text = f"Page {page_number} of {doc_id}"
        output.append((str(corpus_id), page_text))

    return output


def main():
    CACHE_DIR.mkdir(exist_ok=True)

    # Load existing cache
    completed = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            completed = json.load(f)
        logger.info(f"Loaded {len(completed)} cached OCR results")

    # Load dataset images
    logger.info("Loading dataset images...")
    t0 = time.perf_counter()
    corpus = pl.read_parquet("hf://datasets/vidore/vidore_v3_finance_en/corpus/test-*.parquet")
    logger.info(f"Loaded {corpus.shape[0]} pages in {time.perf_counter() - t0:.1f}s")

    # Build work items (skip already completed)
    all_items = []
    all_corpus_ids = []
    for row in corpus.iter_rows(named=True):
        cid = str(row["corpus_id"])
        all_corpus_ids.append(cid)
        if cid in completed:
            continue
        img_data = row["image"]
        img_bytes = img_data["bytes"] if img_data else None
        all_items.append((row["corpus_id"], img_bytes, row["doc_id"], row["page_number_in_doc"]))

    total = len(all_corpus_ids)
    remaining = len(all_items)
    logger.info(f"Total: {total}, Cached: {len(completed)}, Remaining: {remaining}")

    if not all_items:
        logger.info("All pages already OCR'd!")
        _save_ordered(completed, all_corpus_ids)
        return

    # Split into batches
    batches = [all_items[i:i + BATCH_SIZE] for i in range(0, len(all_items), BATCH_SIZE)]
    logger.info(f"Split into {len(batches)} batches of {BATCH_SIZE} images each")

    # Warm up model with first batch (single-threaded)
    logger.info("Warming up Florence-2...")
    warm_results = ocr_batch(batches[0])
    for cid, text in warm_results:
        completed[cid] = text
    batches = batches[1:]
    logger.info(f"Warmup done ({len(warm_results)} pages). Starting {N_WORKERS}-thread parallel OCR...")

    # Parallel execution
    t0 = time.perf_counter()
    done_since_start = len(warm_results)
    cache_lock = Lock()

    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(ocr_batch, batch): batch for batch in batches}

        for future in as_completed(futures):
            batch_results = future.result()
            with cache_lock:
                for cid, text in batch_results:
                    completed[cid] = text
                done_since_start += len(batch_results)

                # Progress
                elapsed = time.perf_counter() - t0
                rate = done_since_start / elapsed
                remaining_pages = total - len(completed)
                eta = remaining_pages / rate if rate > 0 else 0
                logger.info(f"OCR: {len(completed)}/{total} "
                            f"({rate:.1f} pages/s, ETA: {eta / 60:.1f} min)")

                # Checkpoint
                if len(completed) % CHECKPOINT_EVERY < BATCH_SIZE:
                    with open(CACHE_FILE, "w") as f:
                        json.dump(completed, f)

    # Final save
    elapsed = time.perf_counter() - t0
    with open(CACHE_FILE, "w") as f:
        json.dump(completed, f)

    logger.info(f"\nOCR complete: {len(completed)} pages in {elapsed:.1f}s "
                f"({done_since_start / elapsed:.1f} pages/s, {N_WORKERS} workers, batch={BATCH_SIZE})")

    _save_ordered(completed, all_corpus_ids)


def _save_ordered(completed, all_corpus_ids):
    """Save OCR texts in corpus order for downstream pipeline."""
    ordered = [completed.get(cid, f"Page {i}") for i, cid in enumerate(all_corpus_ids)]
    with open(CACHE_DIR / "ocr_texts_ordered.json", "w") as f:
        json.dump(ordered, f)

    text_lens = [len(t) for t in ordered]
    logger.info(f"Saved ordered OCR texts ({len(ordered)} pages)")
    logger.info(f"Text lengths: min={min(text_lens)}, median={sorted(text_lens)[len(text_lens)//2]}, "
                f"max={max(text_lens)}, total={sum(text_lens):,} chars")


if __name__ == "__main__":
    main()
