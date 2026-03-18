"""Benchmark v3: Full OCR pipeline — Florence-2 -> GLINer NER -> Encode -> Turbopuffer.

Starts from raw page images (ignores pre-extracted markdown).

Pipeline:
1. Florence-2 OCR: image -> text (via SIE extract)
2. GLINer NER: text -> entities (company, person, location) for filtering
3. Encode: text -> dense vectors (via SIE encode)
4. Index: vectors + BM25 text + entity filters -> Turbopuffer
5. Search: BM25, Vector, RRF, RRF+Reranker
6. Evaluate: NDCG@10, MRR@10, Recall@10 vs qrels

Designed for scale: batch OCR, batch NER, batch encode.
"""
import csv
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import polars as pl
from dotenv import load_dotenv
from loguru import logger
from sie_sdk import SIEClient
from turbopuffer import Turbopuffer

load_dotenv()

SIE_BASE_URL = os.environ["SIE_BASE_URL"]
SIE_API_KEY = os.environ["SIE_API_KEY"]
TPUF_API_KEY = os.environ["TURBOPUFFER_API_KEY"]

GPU = "l4-spot"
PROVISION_TIMEOUT = 900
ENCODE_BATCH_SIZE = 64
TPUF_BATCH_SIZE = 500
TOP_K_RETRIEVE = 100
TOP_K_EVAL = 10
RRF_K = 60

# Models
OCR_MODEL = "microsoft/Florence-2-base"
NER_MODEL = "urchade/gliner_multi-v2.1"
NER_LABELS = ["company", "person", "financial_metric", "date", "location"]

ENCODER = "BAAI/bge-m3"  # Best from Phase 1

RERANKERS = [
    "mixedbread-ai/mxbai-rerank-base-v2",
    "BAAI/bge-reranker-v2-m3",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
]

NS_PREFIX = "vidore-v3-ocr"
CACHE_DIR = Path("cache")


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9-]", "-", name.lower()).strip("-")


# --- Metrics ---

def ndcg_at_k(ranked_ids, qrel_map, k=10):
    dcg = sum(qrel_map.get(cid, 0) / math.log2(i + 2) for i, cid in enumerate(ranked_ids[:k]))
    ideal_rels = sorted(qrel_map.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(ranked_ids, qrel_map, k=10):
    for i, cid in enumerate(ranked_ids[:k]):
        if qrel_map.get(cid, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked_ids, qrel_map, k=10):
    relevant = {cid for cid, s in qrel_map.items() if s > 0}
    return sum(1 for cid in ranked_ids[:k] if cid in relevant) / len(relevant) if relevant else 0.0


def evaluate(ranked_results, query_items, qrel_map, k=TOP_K_EVAL):
    ndcgs, mrrs, recalls = [], [], []
    for ranked, qi in zip(ranked_results, query_items):
        qrels = qrel_map.get(qi["query_id"], {})
        if not qrels:
            continue
        ndcgs.append(ndcg_at_k(ranked, qrels, k))
        mrrs.append(mrr_at_k(ranked, qrels, k))
        recalls.append(recall_at_k(ranked, qrels, k))
    return {"ndcg@10": np.mean(ndcgs), "mrr@10": np.mean(mrrs), "recall@10": np.mean(recalls), "n": len(ndcgs)}


def rrf_fuse(ranked_lists, k=RRF_K):
    scores = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


# --- Data Loading ---

def load_dataset():
    logger.info("Loading dataset...")
    t0 = time.perf_counter()
    corpus = pl.read_parquet("hf://datasets/vidore/vidore_v3_finance_en/corpus/test-*.parquet")
    queries = pl.read_parquet("hf://datasets/vidore/vidore_v3_finance_en/queries/test-*.parquet")
    qrels_df = pl.read_parquet("hf://datasets/vidore/vidore_v3_finance_en/qrels/test-*.parquet")
    logger.info(f"Loaded in {time.perf_counter() - t0:.1f}s")

    qrel_map = {}
    for row in qrels_df.iter_rows(named=True):
        qrel_map.setdefault(row["query_id"], {})[row["corpus_id"]] = row["score"]

    # Extract raw image bytes + metadata (NO markdown)
    corpus_items = []
    for row in corpus.iter_rows(named=True):
        img_data = row["image"]
        img_bytes = img_data["bytes"] if img_data else None
        corpus_items.append({
            "corpus_id": row["corpus_id"],
            "image_bytes": img_bytes,
            "doc_id": row["doc_id"],
            "page_number": row["page_number_in_doc"],
        })

    query_items = [{"query_id": r["query_id"], "text": r["query"]} for r in queries.iter_rows(named=True)]

    logger.info(f"Corpus: {len(corpus_items)} pages, Queries: {len(query_items)}, "
                f"Qrels: {sum(len(v) for v in qrel_map.values())}")
    return corpus_items, query_items, qrel_map


# --- Step 1: OCR with Florence-2 ---

def ocr_pages(sie: SIEClient, corpus_items: list[dict]) -> list[str]:
    """Run Florence-2 OCR on all page images. Returns list of extracted text."""
    cache_file = CACHE_DIR / "ocr_texts.json"
    if cache_file.exists():
        logger.info(f"Loading cached OCR from {cache_file}")
        with open(cache_file) as f:
            return json.load(f)

    logger.info(f"Running Florence-2 OCR on {len(corpus_items)} pages...")
    CACHE_DIR.mkdir(exist_ok=True)
    texts = []
    t0 = time.perf_counter()

    # Process one at a time (Florence-2 is image-based, batch=1 is safer)
    for i, item in enumerate(corpus_items):
        img_bytes = item["image_bytes"]
        if not img_bytes:
            texts.append(f"Page {item['page_number']} of {item['doc_id']}")
            continue

        result = sie.extract(
            OCR_MODEL,
            {"images": [img_bytes]},
            gpu=GPU,
            wait_for_capacity=True,
            provision_timeout_s=PROVISION_TIMEOUT,
        )

        # Collect all entity text from OCR result
        page_text = ""
        if "entities" in result:
            page_text = " ".join(e.get("text", "") for e in result["entities"])
        if not page_text.strip():
            page_text = f"Page {item['page_number']} of {item['doc_id']}"

        texts.append(page_text)

        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            eta = (len(corpus_items) - i - 1) / rate
            logger.info(f"  OCR {i + 1}/{len(corpus_items)} ({rate:.1f} pages/s, ETA: {eta / 60:.1f} min)")

        # Checkpoint every 500 pages
        if (i + 1) % 500 == 0:
            with open(cache_file, "w") as f:
                json.dump(texts, f)

    elapsed = time.perf_counter() - t0
    logger.info(f"OCR complete: {len(texts)} pages in {elapsed:.1f}s ({len(texts) / elapsed:.1f} pages/s)")

    with open(cache_file, "w") as f:
        json.dump(texts, f)
    return texts


# --- Step 2: NER with GLINer ---

def extract_entities(sie: SIEClient, texts: list[str]) -> list[dict]:
    """Run GLINer NER on OCR'd text. Returns list of {company: [...], person: [...], ...}."""
    cache_file = CACHE_DIR / "ner_results.json"
    if cache_file.exists():
        logger.info(f"Loading cached NER from {cache_file}")
        with open(cache_file) as f:
            return json.load(f)

    logger.info(f"Running GLINer NER on {len(texts)} pages...")
    CACHE_DIR.mkdir(exist_ok=True)
    all_entities = []
    t0 = time.perf_counter()

    # Batch NER - GLINer handles text, can batch
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        items = [{"text": t} for t in batch_texts]
        results = sie.extract(
            NER_MODEL,
            items,
            labels=NER_LABELS,
            gpu=GPU,
            wait_for_capacity=True,
            provision_timeout_s=PROVISION_TIMEOUT,
        )

        for r in results:
            entities_by_label = {label: [] for label in NER_LABELS}
            for e in r.get("entities", []):
                label = e.get("label", "")
                text = e.get("text", "").strip()
                if label in entities_by_label and text and e.get("score", 0) > 0.5:
                    entities_by_label[label].append(text)
            # Deduplicate
            for label in entities_by_label:
                entities_by_label[label] = list(set(entities_by_label[label]))
            all_entities.append(entities_by_label)

        if (i + batch_size) % 256 == 0 or i + batch_size >= len(texts):
            logger.info(f"  NER {min(i + batch_size, len(texts))}/{len(texts)}")

    elapsed = time.perf_counter() - t0
    logger.info(f"NER complete: {len(all_entities)} pages in {elapsed:.1f}s")

    with open(cache_file, "w") as f:
        json.dump(all_entities, f)
    return all_entities


# --- Step 3: Encode ---

def encode_batch(sie, model, texts, is_query=False):
    all_vectors = []
    for i in range(0, len(texts), ENCODE_BATCH_SIZE):
        batch = [{"text": t} for t in texts[i:i + ENCODE_BATCH_SIZE]]
        results = sie.encode(model, batch, output_types=["dense"], is_query=is_query,
                             gpu=GPU, wait_for_capacity=True, provision_timeout_s=PROVISION_TIMEOUT)
        all_vectors.extend(r["dense"] for r in results)
        if (i + ENCODE_BATCH_SIZE) % 512 == 0 or i + ENCODE_BATCH_SIZE >= len(texts):
            logger.info(f"  Encoded {min(i + ENCODE_BATCH_SIZE, len(texts))}/{len(texts)}")
    return all_vectors


# --- Step 4: Index in Turbopuffer ---

def index_all(tpuf, corpus_items, ocr_texts, entities, vectors):
    """Index everything: vectors + BM25 text + entity attributes for filtering."""
    ns_name = f"{NS_PREFIX}-{slugify(ENCODER)}"
    ns = tpuf.namespace(ns_name)
    logger.info(f"Indexing in {ns_name}...")
    t0 = time.perf_counter()

    for i in range(0, len(corpus_items), TPUF_BATCH_SIZE):
        end = min(i + TPUF_BATCH_SIZE, len(corpus_items))
        batch_items = corpus_items[i:end]
        batch_texts = ocr_texts[i:end]
        batch_entities = entities[i:end]
        batch_vecs = vectors[i:end]

        ns.write(
            distance_metric="cosine_distance",
            upsert_columns={
                "id": [str(item["corpus_id"]) for item in batch_items],
                "vector": [v.tolist() for v in batch_vecs],
                "text": batch_texts,
                "doc_id": [item["doc_id"] for item in batch_items],
                "page_number": [item["page_number"] for item in batch_items],
                # Entity attributes for filtering
                "companies": [", ".join(e.get("company", [])) for e in batch_entities],
                "persons": [", ".join(e.get("person", [])) for e in batch_entities],
                "locations": [", ".join(e.get("location", [])) for e in batch_entities],
            },
            schema={
                "text": {"type": "string", "full_text_search": {"tokenizer": "word_v2"}},
                "companies": {"type": "string", "full_text_search": {"tokenizer": "word_v2"}},
            },
        )
        if (i + TPUF_BATCH_SIZE) % 1000 == 0 or end >= len(corpus_items):
            logger.info(f"  Indexed {end}/{len(corpus_items)}")

    logger.info(f"Indexing done in {time.perf_counter() - t0:.1f}s")
    return ns_name


# --- Step 5: Search ---

def search_bm25(tpuf, ns_name, query_texts, top_k=TOP_K_RETRIEVE):
    ns = tpuf.namespace(ns_name)
    results = []
    for i, q in enumerate(query_texts):
        r = ns.query(rank_by=("text", "BM25", q), top_k=top_k, include_attributes=False)
        results.append([int(row.id) for row in r.rows])
        if (i + 1) % 200 == 0:
            logger.info(f"  BM25 {i + 1}/{len(query_texts)}")
    return results


def search_vector(tpuf, ns_name, query_vectors, top_k=TOP_K_RETRIEVE):
    ns = tpuf.namespace(ns_name)
    results = []
    for i, qv in enumerate(query_vectors):
        r = ns.query(rank_by=("vector", "ANN", qv.tolist()), top_k=top_k, include_attributes=False)
        results.append([int(row.id) for row in r.rows])
        if (i + 1) % 200 == 0:
            logger.info(f"  Vector {i + 1}/{len(query_vectors)}")
    return results


def rerank(sie, model, query_texts, candidate_ids_list, text_map):
    all_reranked = []
    for i, (q, cands) in enumerate(zip(query_texts, candidate_ids_list)):
        cands = cands[:TOP_K_RETRIEVE]
        if not cands:
            all_reranked.append([])
            continue
        items = [{"text": text_map.get(cid, "")} for cid in cands]
        result = sie.score(model, query={"text": q}, items=items,
                           gpu=GPU, wait_for_capacity=True, provision_timeout_s=PROVISION_TIMEOUT)
        scored = [(cands[s["rank"]], s["score"]) for s in result["scores"]]
        scored.sort(key=lambda x: x[1], reverse=True)
        all_reranked.append([cid for cid, _ in scored])
        if (i + 1) % 200 == 0:
            logger.info(f"  Reranked {i + 1}/{len(query_texts)}")
    return all_reranked


# --- Main ---

def main():
    sie = SIEClient(SIE_BASE_URL, api_key=SIE_API_KEY, timeout_s=900)
    tpuf = Turbopuffer(api_key=TPUF_API_KEY, region="aws-us-east-1")

    corpus_items, query_items, qrel_map = load_dataset()
    query_texts = [qi["text"] for qi in query_items]
    results_log = []

    # Step 1: OCR
    ocr_texts = ocr_pages(sie, corpus_items)
    text_map = {corpus_items[i]["corpus_id"]: ocr_texts[i] for i in range(len(corpus_items))}

    # Step 2: NER
    entities = extract_entities(sie, ocr_texts)

    # Log entity stats
    all_companies = set()
    for e in entities:
        all_companies.update(e.get("company", []))
    logger.info(f"Unique companies found: {len(all_companies)}")
    if all_companies:
        logger.info(f"Sample: {list(all_companies)[:10]}")

    # Step 3: Encode corpus
    logger.info(f"\nEncoding corpus with {ENCODER}...")
    t0 = time.perf_counter()
    corpus_vectors = encode_batch(sie, ENCODER, ocr_texts)
    logger.info(f"Corpus encoding: {time.perf_counter() - t0:.1f}s (dim={corpus_vectors[0].shape[0]})")

    # Step 4: Index
    ns_name = index_all(tpuf, corpus_items, ocr_texts, entities, corpus_vectors)

    # Step 5: Encode queries
    logger.info("Encoding queries...")
    t0 = time.perf_counter()
    query_vectors = encode_batch(sie, ENCODER, query_texts, is_query=True)
    logger.info(f"Query encoding: {time.perf_counter() - t0:.1f}s")

    # ========== Evaluation ==========

    # Condition 1: BM25 only
    logger.info("\n" + "=" * 60)
    logger.info("CONDITION 1: BM25 only (on OCR text)")
    t0 = time.perf_counter()
    bm25_results = search_bm25(tpuf, ns_name, query_texts)
    bm25_time = time.perf_counter() - t0
    bm25_m = evaluate(bm25_results, query_items, qrel_map)
    logger.info(f"BM25: NDCG@10={bm25_m['ndcg@10']:.4f}, MRR={bm25_m['mrr@10']:.4f}, "
                f"Recall={bm25_m['recall@10']:.4f} ({bm25_time:.1f}s)")
    results_log.append({"condition": "BM25", "encoder": "none", "reranker": "none", **bm25_m})

    # Condition 2: Vector only
    logger.info("\nCONDITION 2: Vector only")
    t0 = time.perf_counter()
    vec_results = search_vector(tpuf, ns_name, query_vectors)
    vec_time = time.perf_counter() - t0
    vec_m = evaluate(vec_results, query_items, qrel_map)
    logger.info(f"Vector: NDCG@10={vec_m['ndcg@10']:.4f}, MRR={vec_m['mrr@10']:.4f}, "
                f"Recall={vec_m['recall@10']:.4f} ({vec_time:.1f}s)")
    results_log.append({"condition": "Vector", "encoder": ENCODER, "reranker": "none", **vec_m})

    # Condition 3: RRF (BM25 + Vector)
    logger.info("\nCONDITION 3: RRF (BM25 + Vector)")
    rrf_results = [rrf_fuse([b, v]) for b, v in zip(bm25_results, vec_results)]
    rrf_m = evaluate(rrf_results, query_items, qrel_map)
    logger.info(f"RRF: NDCG@10={rrf_m['ndcg@10']:.4f}, MRR={rrf_m['mrr@10']:.4f}, "
                f"Recall={rrf_m['recall@10']:.4f}")
    results_log.append({"condition": "RRF", "encoder": ENCODER, "reranker": "none", **rrf_m})

    # Condition 4: RRF + Reranker
    for reranker_model in RERANKERS:
        logger.info(f"\nCONDITION 4: RRF + {reranker_model}")
        t0 = time.perf_counter()
        reranked = rerank(sie, reranker_model, query_texts, rrf_results, text_map)
        rerank_time = time.perf_counter() - t0
        rr_m = evaluate(reranked, query_items, qrel_map)
        logger.info(f"RRF+Rerank: NDCG@10={rr_m['ndcg@10']:.4f}, MRR={rr_m['mrr@10']:.4f}, "
                    f"Recall={rr_m['recall@10']:.4f} ({rerank_time:.1f}s)")
        results_log.append({"condition": "RRF+Rerank", "encoder": ENCODER, "reranker": reranker_model, **rr_m})

    # ========== Results ==========
    csv_path = "benchmark_v3_results.csv"
    all_keys = sorted({k for r in results_log for k in r})
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(results_log)

    logger.info(f"\n{'=' * 60}")
    logger.info("FINAL RESULTS (OCR Pipeline)")
    logger.info("=" * 60)
    logger.info(f"{'Condition':<15} {'Encoder':<20} {'Reranker':<40} {'NDCG@10':>8} {'MRR@10':>8} {'Recall@10':>10}")
    logger.info("-" * 105)
    for r in sorted(results_log, key=lambda x: x.get("ndcg@10", 0), reverse=True):
        logger.info(f"{r.get('condition', ''):<15} {r.get('encoder', ''):<20} {r.get('reranker', ''):<40} "
                    f"{r.get('ndcg@10', 0):>8.4f} {r.get('mrr@10', 0):>8.4f} {r.get('recall@10', 0):>10.4f}")
    logger.info(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
