"""Benchmark SIE embedding models on vidore_v3_finance_en using Turbopuffer."""
import csv
import math
import os
import re
import time

import numpy as np
import polars as pl
from dotenv import load_dotenv
from loguru import logger
from sie_sdk import SIEClient
from turbopuffer import Turbopuffer

load_dotenv()

# --- Config ---
SIE_BASE_URL = os.environ["SIE_BASE_URL"]
SIE_API_KEY = os.environ["SIE_API_KEY"]
TPUF_API_KEY = os.environ["TURBOPUFFER_API_KEY"]

GPU = "l4-spot"
PROVISION_TIMEOUT = 900
ENCODE_BATCH_SIZE = 64
TPUF_BATCH_SIZE = 500
TOP_K_RETRIEVE = 100  # retrieve 100, then rerank to 10
TOP_K_EVAL = 10

# Phase 1: 5 diverse encoders
PHASE1_ENCODERS = [
    "BAAI/bge-m3",
    "NovaSearch/stella_en_400M_v5",
    "sentence-transformers/all-MiniLM-L6-v2",
    "intfloat/e5-large-v2",
    "Qwen/Qwen3-Embedding-0.6B",
]

# Rerankers to test with top encoders
RERANKERS = [
    "mixedbread-ai/mxbai-rerank-base-v2",
    "BAAI/bge-reranker-v2-m3",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
]


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9-]", "-", name.lower()).strip("-")


def ndcg_at_k(ranked_ids: list[int], qrel_map: dict[int, int], k: int = 10) -> float:
    """Compute NDCG@k given ranked result IDs and {corpus_id: score} relevance map."""
    dcg = 0.0
    for i, cid in enumerate(ranked_ids[:k]):
        rel = qrel_map.get(cid, 0)
        dcg += rel / math.log2(i + 2)

    # Ideal DCG
    ideal_rels = sorted(qrel_map.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))

    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(ranked_ids: list[int], qrel_map: dict[int, int], k: int = 10) -> float:
    """Compute MRR@k."""
    for i, cid in enumerate(ranked_ids[:k]):
        if qrel_map.get(cid, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked_ids: list[int], qrel_map: dict[int, int], k: int = 10) -> float:
    """Compute Recall@k."""
    relevant = {cid for cid, score in qrel_map.items() if score > 0}
    if not relevant:
        return 0.0
    retrieved_relevant = sum(1 for cid in ranked_ids[:k] if cid in relevant)
    return retrieved_relevant / len(relevant)


def load_dataset():
    """Load corpus, queries, qrels from HuggingFace."""
    logger.info("Loading dataset from HuggingFace...")
    t0 = time.perf_counter()

    corpus = pl.read_parquet("hf://datasets/vidore/vidore_v3_finance_en/corpus/test-*.parquet")
    queries = pl.read_parquet("hf://datasets/vidore/vidore_v3_finance_en/queries/test-*.parquet")
    qrels = pl.read_parquet("hf://datasets/vidore/vidore_v3_finance_en/qrels/test-*.parquet")

    logger.info(f"Loaded in {time.perf_counter() - t0:.1f}s: "
                f"{corpus.shape[0]} pages, {queries.shape[0]} queries, {qrels.shape[0]} qrels")

    # Build qrel lookup: query_id -> {corpus_id: score}
    qrel_map = {}
    for row in qrels.iter_rows(named=True):
        qid = row["query_id"]
        if qid not in qrel_map:
            qrel_map[qid] = {}
        qrel_map[qid][row["corpus_id"]] = row["score"]

    # Corpus texts (filter empty)
    corpus_items = []
    for row in corpus.iter_rows(named=True):
        text = row["markdown"] or ""
        if len(text.strip()) < 10:
            text = f"Page {row['page_number_in_doc']} of {row['doc_id']}"
        corpus_items.append({
            "corpus_id": row["corpus_id"],
            "text": text,
            "doc_id": row["doc_id"],
            "page_number": row["page_number_in_doc"],
        })

    query_items = []
    for row in queries.iter_rows(named=True):
        query_items.append({
            "query_id": row["query_id"],
            "text": row["query"],
        })

    return corpus_items, query_items, qrel_map


def encode_batch(client: SIEClient, model: str, texts: list[str], is_query: bool = False) -> list[np.ndarray]:
    """Encode texts in batches, return list of dense vectors."""
    all_vectors = []
    for i in range(0, len(texts), ENCODE_BATCH_SIZE):
        batch = texts[i:i + ENCODE_BATCH_SIZE]
        items = [{"text": t} for t in batch]
        results = client.encode(
            model,
            items,
            output_types=["dense"],
            is_query=is_query,
            gpu=GPU,
            wait_for_capacity=True,
            provision_timeout_s=PROVISION_TIMEOUT,
        )
        for r in results:
            all_vectors.append(r["dense"])

        if (i + ENCODE_BATCH_SIZE) % 512 == 0 or i + ENCODE_BATCH_SIZE >= len(texts):
            logger.info(f"  Encoded {min(i + ENCODE_BATCH_SIZE, len(texts))}/{len(texts)}")

    return all_vectors


def index_in_turbopuffer(tpuf_client, namespace: str, corpus_items: list[dict], vectors: list[np.ndarray]):
    """Upsert corpus vectors into Turbopuffer namespace."""
    ns = tpuf_client.namespace(namespace)

    # Batch upsert using columnar format
    for i in range(0, len(corpus_items), TPUF_BATCH_SIZE):
        batch_items = corpus_items[i:i + TPUF_BATCH_SIZE]
        batch_vecs = vectors[i:i + TPUF_BATCH_SIZE]

        ns.write(
            distance_metric="cosine_distance",
            upsert_columns={
                "id": [str(item["corpus_id"]) for item in batch_items],
                "vector": [v.tolist() for v in batch_vecs],
                "doc_id": [item["doc_id"] for item in batch_items],
                "page_number": [item["page_number"] for item in batch_items],
            },
        )

        if (i + TPUF_BATCH_SIZE) % 1000 == 0 or i + TPUF_BATCH_SIZE >= len(corpus_items):
            logger.info(f"  Indexed {min(i + TPUF_BATCH_SIZE, len(corpus_items))}/{len(corpus_items)}")


def search_turbopuffer(tpuf_client, namespace: str, query_vectors: list[np.ndarray],
                       top_k: int = TOP_K_RETRIEVE) -> list[list[tuple[int, float]]]:
    """Search Turbopuffer for each query vector. Returns list of [(corpus_id, dist), ...]."""
    ns = tpuf_client.namespace(namespace)
    all_results = []

    for i, qvec in enumerate(query_vectors):
        result = ns.query(
            rank_by=("vector", "ANN", qvec.tolist()),
            top_k=top_k,
            include_attributes=False,
        )
        hits = [(int(row.id), row["$dist"]) for row in result.rows]
        all_results.append(hits)

        if (i + 1) % 200 == 0:
            logger.info(f"  Searched {i + 1}/{len(query_vectors)}")

    return all_results


def rerank_results(client: SIEClient, model: str, query_texts: list[str],
                   search_results: list[list[tuple[int, float]]],
                   corpus_text_map: dict[int, str]) -> list[list[int]]:
    """Rerank search results using SIE score endpoint."""
    all_reranked = []

    for i, (query_text, hits) in enumerate(zip(query_texts, search_results)):
        if not hits:
            all_reranked.append([])
            continue

        items = [{"text": corpus_text_map[cid]} for cid, _ in hits]
        score_result = client.score(
            model,
            query={"text": query_text},
            items=items,
            gpu=GPU,
            wait_for_capacity=True,
            provision_timeout_s=PROVISION_TIMEOUT,
        )

        # Map back to corpus IDs, sorted by score descending
        scored = []
        for s in score_result["scores"]:
            rank_idx = s["rank"]
            corpus_id = hits[rank_idx][0]
            scored.append((corpus_id, s["score"]))

        scored.sort(key=lambda x: x[1], reverse=True)
        all_reranked.append([cid for cid, _ in scored])

        if (i + 1) % 200 == 0:
            logger.info(f"  Reranked {i + 1}/{len(query_texts)}")

    return all_reranked


def evaluate(ranked_results: list[list[int]], query_items: list[dict],
             qrel_map: dict, k: int = TOP_K_EVAL) -> dict:
    """Compute NDCG@k, MRR@k, Recall@k."""
    ndcgs, mrrs, recalls = [], [], []

    for i, ranked in enumerate(ranked_results):
        qid = query_items[i]["query_id"]
        qrels = qrel_map.get(qid, {})
        if not qrels:
            continue

        ranked_ids = ranked if isinstance(ranked[0], int) else [cid for cid, _ in ranked]
        ndcgs.append(ndcg_at_k(ranked_ids, qrels, k))
        mrrs.append(mrr_at_k(ranked_ids, qrels, k))
        recalls.append(recall_at_k(ranked_ids, qrels, k))

    return {
        f"ndcg@{k}": np.mean(ndcgs),
        f"mrr@{k}": np.mean(mrrs),
        f"recall@{k}": np.mean(recalls),
        "n_queries": len(ndcgs),
    }


def main():
    # Initialize clients
    sie = SIEClient(SIE_BASE_URL, api_key=SIE_API_KEY, timeout_s=900)
    tpuf = Turbopuffer(api_key=TPUF_API_KEY, region="aws-us-east-1")

    # Load dataset
    corpus_items, query_items, qrel_map = load_dataset()

    # Build corpus text lookup
    corpus_text_map = {item["corpus_id"]: item["text"] for item in corpus_items}
    corpus_texts = [item["text"] for item in corpus_items]
    query_texts = [item["text"] for item in query_items]

    results_log = []

    # ========== Phase 1: Evaluate 5 encoders (no reranking) ==========
    logger.info("=" * 60)
    logger.info("PHASE 1: Evaluating encoders (retrieval only)")
    logger.info("=" * 60)

    encoder_scores = {}

    for model in PHASE1_ENCODERS:
        ns_name = f"vidore-{slugify(model)}"
        logger.info(f"\n--- {model} (namespace: {ns_name}) ---")

        try:
            # Encode corpus
            t0 = time.perf_counter()
            logger.info("Encoding corpus...")
            corpus_vectors = encode_batch(sie, model, corpus_texts)
            encode_time = time.perf_counter() - t0
            logger.info(f"Corpus encoding: {encode_time:.1f}s ({len(corpus_vectors)} vectors, dim={corpus_vectors[0].shape[0]})")

            # Index in Turbopuffer
            t0 = time.perf_counter()
            logger.info("Indexing in Turbopuffer...")
            index_in_turbopuffer(tpuf, ns_name, corpus_items, corpus_vectors)
            index_time = time.perf_counter() - t0
            logger.info(f"Indexing: {index_time:.1f}s")

            # Encode queries
            t0 = time.perf_counter()
            logger.info("Encoding queries...")
            query_vectors = encode_batch(sie, model, query_texts, is_query=True)
            query_encode_time = time.perf_counter() - t0
            logger.info(f"Query encoding: {query_encode_time:.1f}s")

            # Search
            t0 = time.perf_counter()
            logger.info("Searching...")
            search_results = search_turbopuffer(tpuf, ns_name, query_vectors, top_k=TOP_K_RETRIEVE)
            search_time = time.perf_counter() - t0
            logger.info(f"Search: {search_time:.1f}s")

            # Evaluate retrieval only (top 10 from vector search)
            retrieval_ranked = [[cid for cid, _ in hits] for hits in search_results]
            metrics = evaluate(retrieval_ranked, query_items, qrel_map)

            dim = corpus_vectors[0].shape[0]
            logger.info(f"RESULTS: NDCG@10={metrics['ndcg@10']:.4f}, "
                        f"MRR@10={metrics['mrr@10']:.4f}, "
                        f"Recall@10={metrics['recall@10']:.4f}")

            encoder_scores[model] = metrics["ndcg@10"]
            results_log.append({
                "model": model,
                "reranker": "none",
                "dim": dim,
                "ndcg@10": round(metrics["ndcg@10"], 4),
                "mrr@10": round(metrics["mrr@10"], 4),
                "recall@10": round(metrics["recall@10"], 4),
                "encode_corpus_s": round(encode_time, 1),
                "index_s": round(index_time, 1),
                "encode_queries_s": round(query_encode_time, 1),
                "search_s": round(search_time, 1),
                "n_queries": metrics["n_queries"],
            })

        except Exception as e:
            logger.error(f"FAILED {model}: {e}")
            encoder_scores[model] = 0.0
            results_log.append({
                "model": model,
                "reranker": "none",
                "dim": 0,
                "ndcg@10": 0,
                "mrr@10": 0,
                "recall@10": 0,
                "encode_corpus_s": 0,
                "index_s": 0,
                "encode_queries_s": 0,
                "search_s": 0,
                "error": str(e)[:200],
            })

    # ========== Phase 2: Top 3 encoders × rerankers ==========
    top3 = sorted(encoder_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    logger.info("\n" + "=" * 60)
    logger.info(f"PHASE 2: Reranking with top 3 encoders")
    logger.info(f"Top 3: {[m for m, s in top3]}")
    logger.info("=" * 60)

    for encoder_model, base_ndcg in top3:
        ns_name = f"vidore-{slugify(encoder_model)}"

        # Re-encode queries (or cache from phase 1 — for simplicity, re-encode)
        logger.info(f"\nEncoding queries for {encoder_model}...")
        query_vectors = encode_batch(sie, encoder_model, query_texts, is_query=True)

        # Get top-100 from Turbopuffer
        logger.info("Searching top-100 for reranking...")
        search_results = search_turbopuffer(tpuf, ns_name, query_vectors, top_k=TOP_K_RETRIEVE)

        for reranker in RERANKERS:
            logger.info(f"\n--- Reranking: {encoder_model} + {reranker} ---")
            try:
                t0 = time.perf_counter()
                reranked = rerank_results(sie, reranker, query_texts, search_results, corpus_text_map)
                rerank_time = time.perf_counter() - t0
                logger.info(f"Reranking: {rerank_time:.1f}s")

                metrics = evaluate(reranked, query_items, qrel_map)
                logger.info(f"RESULTS: NDCG@10={metrics['ndcg@10']:.4f} (base: {base_ndcg:.4f}), "
                            f"MRR@10={metrics['mrr@10']:.4f}, "
                            f"Recall@10={metrics['recall@10']:.4f}")

                results_log.append({
                    "model": encoder_model,
                    "reranker": reranker,
                    "ndcg@10": round(metrics["ndcg@10"], 4),
                    "mrr@10": round(metrics["mrr@10"], 4),
                    "recall@10": round(metrics["recall@10"], 4),
                    "rerank_s": round(rerank_time, 1),
                    "n_queries": metrics["n_queries"],
                })
            except Exception as e:
                logger.error(f"FAILED reranking {reranker}: {e}")
                results_log.append({
                    "model": encoder_model,
                    "reranker": reranker,
                    "ndcg@10": 0,
                    "error": str(e)[:200],
                })

    # ========== Save results ==========
    csv_path = "benchmark_results.csv"
    if results_log:
        all_keys = set()
        for r in results_log:
            all_keys.update(r.keys())
        all_keys = sorted(all_keys)

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(results_log)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Results saved to {csv_path}")
    logger.info(f"{'=' * 60}")

    # Print summary table
    logger.info("\n=== FINAL RANKINGS ===")
    sorted_results = sorted(results_log, key=lambda x: x.get("ndcg@10", 0), reverse=True)
    logger.info(f"{'Model':<45} {'Reranker':<40} {'NDCG@10':>8} {'MRR@10':>8} {'Recall@10':>10}")
    logger.info("-" * 115)
    for r in sorted_results:
        logger.info(f"{r.get('model', ''):<45} {r.get('reranker', ''):<40} "
                    f"{r.get('ndcg@10', 0):>8.4f} {r.get('mrr@10', 0):>8.4f} "
                    f"{r.get('recall@10', 0):>10.4f}")


if __name__ == "__main__":
    main()
