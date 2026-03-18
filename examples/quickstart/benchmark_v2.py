"""Benchmark v2: BM25 + Vector + RRF + Reranker on vidore_v3_finance_en.

Experimental design (ablation study):
  1. BM25-only baseline (Turbopuffer FTS)
  2. Vector-only (bge-m3)
  3. RRF(BM25 + Vector) — isolates hybrid value over either alone
  4. Reranker(Vector top-100) — head-to-head vs RRF
  5. Reranker(RRF top-100) — does better recall pool help reranker?

Cost tracking:
  - Ingestion: SIE encode wall-time, Turbopuffer index wall-time
  - Query: SIE encode query wall-time, Turbopuffer search wall-time, SIE rerank wall-time
  - Per-query latency computed for each condition

Pipeline starts from raw HTM 10-K filings downloaded from EDGAR.
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
from bs4 import BeautifulSoup
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
TOP_K_RETRIEVE = 100
TOP_K_EVAL = 10
RRF_K = 60

BEST_ENCODER = "BAAI/bge-m3"

RERANKERS = [
    "mixedbread-ai/mxbai-rerank-base-v2",
    "BAAI/bge-reranker-v2-m3",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
]

HTM_DIR = Path("pdfs")


# ==================== Metrics ====================

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
    relevant = {cid for cid, score in qrel_map.items() if score > 0}
    if not relevant:
        return 0.0
    return sum(1 for cid in ranked_ids[:k] if cid in relevant) / len(relevant)

def evaluate(ranked_results, query_items, qrel_map, k=TOP_K_EVAL):
    ndcgs, mrrs, recalls = [], [], []
    for ranked, q in zip(ranked_results, query_items):
        qrels = qrel_map.get(q["query_id"], {})
        if not qrels or not ranked:
            continue
        ndcgs.append(ndcg_at_k(ranked, qrels, k))
        mrrs.append(mrr_at_k(ranked, qrels, k))
        recalls.append(recall_at_k(ranked, qrels, k))
    return {
        "ndcg@5": round(float(np.mean([ndcg_at_k(r, qrel_map.get(q["query_id"], {}), 5)
                                        for r, q in zip(ranked_results, query_items)
                                        if qrel_map.get(q["query_id"]) and r])), 4),
        "ndcg@10": round(float(np.mean(ndcgs)), 4),
        "mrr@10": round(float(np.mean(mrrs)), 4),
        "recall@5": round(float(np.mean([recall_at_k(r, qrel_map.get(q["query_id"], {}), 5)
                                         for r, q in zip(ranked_results, query_items)
                                         if qrel_map.get(q["query_id"]) and r])), 4),
        "recall@10": round(float(np.mean(recalls)), 4),
        "recall@100": round(float(np.mean([recall_at_k(r, qrel_map.get(q["query_id"], {}), 100)
                                           for r, q in zip(ranked_results, query_items)
                                           if qrel_map.get(q["query_id"]) and r])), 4),
        "n_queries": len(ndcgs),
    }


# ==================== RRF ====================

def rrf_fuse(ranked_lists, k=RRF_K):
    scores = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


# ==================== Data Loading ====================

def parse_htm_files():
    """Parse raw HTM 10-K filings — log stats for the pipeline demo."""
    stats = {}
    for htm_file in sorted(HTM_DIR.glob("*.htm")):
        doc_id = htm_file.stem
        size_mb = htm_file.stat().st_size / 1_000_000
        t0 = time.perf_counter()
        with open(htm_file, "r", encoding="utf-8", errors="replace") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        full_text = soup.get_text(separator="\n", strip=True)
        parse_time = time.perf_counter() - t0
        stats[doc_id] = {
            "file_size_mb": round(size_mb, 1),
            "text_chars": len(full_text),
            "parse_time_s": round(parse_time, 2),
        }
        logger.info(f"  {doc_id}: {size_mb:.1f}MB HTM → {len(full_text):,} chars ({parse_time:.2f}s)")
    return stats


def load_dataset():
    logger.info("Loading dataset from HuggingFace...")
    t0 = time.perf_counter()
    corpus = pl.read_parquet("hf://datasets/vidore/vidore_v3_finance_en/corpus/test-*.parquet")
    queries = pl.read_parquet("hf://datasets/vidore/vidore_v3_finance_en/queries/test-*.parquet")
    qrels = pl.read_parquet("hf://datasets/vidore/vidore_v3_finance_en/qrels/test-*.parquet")
    load_time = time.perf_counter() - t0
    logger.info(f"Loaded in {load_time:.1f}s: {corpus.shape[0]} pages, {queries.shape[0]} queries, {qrels.shape[0]} qrels")

    qrel_map = {}
    for row in qrels.iter_rows(named=True):
        qrel_map.setdefault(row["query_id"], {})[row["corpus_id"]] = row["score"]

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

    query_items = [{"query_id": r["query_id"], "text": r["query"]}
                   for r in queries.iter_rows(named=True)]

    return corpus_items, query_items, qrel_map


# ==================== SIE Helpers ====================

def encode_batch(client, model, texts, is_query=False):
    all_vectors = []
    for i in range(0, len(texts), ENCODE_BATCH_SIZE):
        batch = [{"text": t} for t in texts[i:i + ENCODE_BATCH_SIZE]]
        results = client.encode(
            model, batch, output_types=["dense"], is_query=is_query,
            gpu=GPU, wait_for_capacity=True, provision_timeout_s=PROVISION_TIMEOUT,
        )
        all_vectors.extend(r["dense"] for r in results)
        if (i + ENCODE_BATCH_SIZE) % 512 == 0 or i + ENCODE_BATCH_SIZE >= len(texts):
            logger.info(f"  Encoded {min(i + ENCODE_BATCH_SIZE, len(texts))}/{len(texts)}")
    return all_vectors


# ==================== Turbopuffer Helpers ====================

def slugify(name):
    return re.sub(r"[^a-z0-9-]", "-", name.lower()).strip("-")


def index_with_fts(tpuf_client, namespace, corpus_items, vectors=None):
    ns = tpuf_client.namespace(namespace)
    for i in range(0, len(corpus_items), TPUF_BATCH_SIZE):
        batch = corpus_items[i:i + TPUF_BATCH_SIZE]
        row_data = {
            "id": [str(item["corpus_id"]) for item in batch],
            "text": [item["text"] for item in batch],
            "doc_id": [item["doc_id"] for item in batch],
            "page_number": [item["page_number"] for item in batch],
        }
        if vectors is not None:
            row_data["vector"] = [v.tolist() for v in vectors[i:i + TPUF_BATCH_SIZE]]

        write_kwargs = {
            "upsert_columns": row_data,
            "schema": {
                "text": {
                    "type": "string",
                    "full_text_search": {
                        "tokenizer": "word_v2",
                        "language": "english",
                    },
                },
            },
        }
        if vectors is not None:
            write_kwargs["distance_metric"] = "cosine_distance"

        ns.write(**write_kwargs)
        done = min(i + TPUF_BATCH_SIZE, len(corpus_items))
        if done % 1000 == 0 or done >= len(corpus_items):
            logger.info(f"  Indexed {done}/{len(corpus_items)}")
    return ns


def search_bm25(tpuf_client, namespace, query_texts, top_k=TOP_K_RETRIEVE):
    ns = tpuf_client.namespace(namespace)
    all_results = []
    for i, q in enumerate(query_texts):
        result = ns.query(rank_by=("text", "BM25", q), top_k=top_k, include_attributes=False)
        all_results.append([int(row.id) for row in result.rows])
        if (i + 1) % 500 == 0:
            logger.info(f"  BM25 searched {i + 1}/{len(query_texts)}")
    return all_results


def search_vector(tpuf_client, namespace, query_vectors, top_k=TOP_K_RETRIEVE):
    ns = tpuf_client.namespace(namespace)
    all_results = []
    for i, qvec in enumerate(query_vectors):
        result = ns.query(rank_by=("vector", "ANN", qvec.tolist()), top_k=top_k, include_attributes=False)
        all_results.append([int(row.id) for row in result.rows])
        if (i + 1) % 500 == 0:
            logger.info(f"  Vector searched {i + 1}/{len(query_vectors)}")
    return all_results


def rerank_results(client, model, query_texts, search_results, corpus_text_map):
    all_reranked = []
    for i, (q, hits) in enumerate(zip(query_texts, search_results)):
        if not hits:
            all_reranked.append([])
            continue
        items = [{"text": corpus_text_map.get(cid, "")} for cid in hits]
        score_result = client.score(
            model, query={"text": q}, items=items,
            gpu=GPU, wait_for_capacity=True, provision_timeout_s=PROVISION_TIMEOUT,
        )
        scored = [(hits[s["rank"]], s["score"]) for s in score_result["scores"]]
        scored.sort(key=lambda x: x[1], reverse=True)
        all_reranked.append([cid for cid, _ in scored])
        if (i + 1) % 200 == 0:
            logger.info(f"  Reranked {i + 1}/{len(query_texts)}")
    return all_reranked


# ==================== Main ====================

def main():
    sie = SIEClient(SIE_BASE_URL, api_key=SIE_API_KEY, timeout_s=900)
    tpuf = Turbopuffer(api_key=TPUF_API_KEY, region="aws-us-east-1")

    cost_log = {}  # Track all timing for cost analysis

    # --- Step 0: Parse raw HTM files ---
    logger.info("=" * 60)
    logger.info("STEP 0: Parse raw EDGAR HTM filings")
    logger.info("=" * 60)
    if HTM_DIR.exists() and list(HTM_DIR.glob("*.htm")):
        htm_stats = parse_htm_files()
        cost_log["htm_parse"] = htm_stats
    else:
        logger.info("No HTM files — skipping")

    # --- Load dataset ---
    corpus_items, query_items, qrel_map = load_dataset()
    corpus_text_map = {item["corpus_id"]: item["text"] for item in corpus_items}
    query_texts = [q["text"] for q in query_items]
    n_queries = len(query_texts)
    n_corpus = len(corpus_items)

    results_log = []

    # --- Step 1: Ingestion (encode + index) ---
    encoder = BEST_ENCODER
    ns_name = f"vidore-v2-{slugify(encoder)}"
    logger.info(f"\n{'=' * 60}")
    logger.info(f"STEP 1: Ingestion — encode({encoder}) + index(Turbopuffer FTS+Vector)")
    logger.info(f"{'=' * 60}")

    t0 = time.perf_counter()
    corpus_vectors = encode_batch(sie, encoder, [c["text"] for c in corpus_items])
    encode_corpus_s = time.perf_counter() - t0
    dim = corpus_vectors[0].shape[0]
    logger.info(f"Corpus encoding: {encode_corpus_s:.1f}s ({n_corpus} pages, dim={dim})")
    cost_log["encode_corpus_s"] = round(encode_corpus_s, 1)
    cost_log["encode_corpus_per_page_ms"] = round(encode_corpus_s / n_corpus * 1000, 1)

    t0 = time.perf_counter()
    index_with_fts(tpuf, ns_name, corpus_items, vectors=corpus_vectors)
    index_s = time.perf_counter() - t0
    logger.info(f"Indexing (FTS+Vector): {index_s:.1f}s")
    cost_log["index_s"] = round(index_s, 1)

    # --- Step 2: Query encoding ---
    t0 = time.perf_counter()
    query_vectors = encode_batch(sie, encoder, query_texts, is_query=True)
    encode_query_s = time.perf_counter() - t0
    logger.info(f"Query encoding: {encode_query_s:.1f}s ({n_queries} queries)")
    cost_log["encode_query_s"] = round(encode_query_s, 1)
    cost_log["encode_query_per_query_ms"] = round(encode_query_s / n_queries * 1000, 1)

    # --- Step 3: Search conditions ---

    # Condition 1: BM25-only
    logger.info(f"\n{'=' * 60}")
    logger.info("CONDITION 1: BM25-only")
    logger.info(f"{'=' * 60}")
    t0 = time.perf_counter()
    bm25_results = search_bm25(tpuf, ns_name, query_texts)
    bm25_search_s = time.perf_counter() - t0
    bm25_metrics = evaluate(bm25_results, query_items, qrel_map)
    logger.info(f"BM25: {bm25_search_s:.1f}s — NDCG@10={bm25_metrics['ndcg@10']:.4f}")
    results_log.append({
        "condition": "BM25-only",
        "encoder": "n/a",
        "reranker": "n/a",
        **bm25_metrics,
        "search_s": round(bm25_search_s, 1),
        "query_latency_ms": round(bm25_search_s / n_queries * 1000, 0),
        "ingestion_cost": "FTS index only",
        "query_cost": "BM25 search only",
    })

    # Condition 2: Vector-only
    logger.info(f"\n{'=' * 60}")
    logger.info(f"CONDITION 2: Vector-only ({encoder})")
    logger.info(f"{'=' * 60}")
    t0 = time.perf_counter()
    vector_results = search_vector(tpuf, ns_name, query_vectors)
    vector_search_s = time.perf_counter() - t0
    vector_metrics = evaluate(vector_results, query_items, qrel_map)
    logger.info(f"Vector: {vector_search_s:.1f}s — NDCG@10={vector_metrics['ndcg@10']:.4f}")
    results_log.append({
        "condition": "Vector-only",
        "encoder": encoder,
        "reranker": "n/a",
        **vector_metrics,
        "search_s": round(vector_search_s, 1),
        "query_latency_ms": round((encode_query_s + vector_search_s) / n_queries * 1000, 0),
        "ingestion_cost": f"Encode {n_corpus}p + Vector index",
        "query_cost": f"Encode 1q + ANN search",
    })

    # Condition 3: RRF(BM25 + Vector)
    logger.info(f"\n{'=' * 60}")
    logger.info("CONDITION 3: RRF(BM25 + Vector)")
    logger.info(f"{'=' * 60}")
    t0 = time.perf_counter()
    rrf_results = [rrf_fuse([bm25, vec]) for bm25, vec in zip(bm25_results, vector_results)]
    rrf_fuse_s = time.perf_counter() - t0
    rrf_metrics = evaluate(rrf_results, query_items, qrel_map)
    logger.info(f"RRF: {rrf_fuse_s:.2f}s — NDCG@10={rrf_metrics['ndcg@10']:.4f}")
    rrf_total_query_s = bm25_search_s + vector_search_s + encode_query_s + rrf_fuse_s
    results_log.append({
        "condition": "RRF(BM25+Vector)",
        "encoder": encoder,
        "reranker": "n/a",
        **rrf_metrics,
        "search_s": round(bm25_search_s + vector_search_s + rrf_fuse_s, 1),
        "query_latency_ms": round(rrf_total_query_s / n_queries * 1000, 0),
        "ingestion_cost": f"Encode {n_corpus}p + FTS+Vector index",
        "query_cost": "Encode 1q + BM25 + ANN + RRF fusion",
    })

    # Condition 4: Reranker(Vector top-100) — head-to-head vs RRF
    logger.info(f"\n{'=' * 60}")
    logger.info("CONDITION 4: Reranker(Vector top-100) — vs RRF")
    logger.info(f"{'=' * 60}")

    for reranker in RERANKERS:
        logger.info(f"\n--- {reranker} ---")
        t0 = time.perf_counter()
        reranked = rerank_results(sie, reranker, query_texts, vector_results, corpus_text_map)
        rerank_s = time.perf_counter() - t0
        rerank_metrics = evaluate(reranked, query_items, qrel_map)
        logger.info(f"Reranker: {rerank_s:.1f}s — NDCG@10={rerank_metrics['ndcg@10']:.4f}")
        rerank_total_s = encode_query_s + vector_search_s + rerank_s
        results_log.append({
            "condition": "Reranker(Vector)",
            "encoder": encoder,
            "reranker": reranker,
            **rerank_metrics,
            "search_s": round(vector_search_s + rerank_s, 1),
            "rerank_s": round(rerank_s, 1),
            "query_latency_ms": round(rerank_total_s / n_queries * 1000, 0),
            "ingestion_cost": f"Encode {n_corpus}p + Vector index",
            "query_cost": "Encode 1q + ANN + Rerank 100",
        })

    # Condition 5: Reranker(RRF top-100) — best recall pool + reranker
    logger.info(f"\n{'=' * 60}")
    logger.info("CONDITION 5: Reranker(RRF top-100)")
    logger.info(f"{'=' * 60}")

    for reranker in RERANKERS:
        logger.info(f"\n--- {reranker} over RRF ---")
        rrf_top100 = [r[:TOP_K_RETRIEVE] for r in rrf_results]
        t0 = time.perf_counter()
        reranked_rrf = rerank_results(sie, reranker, query_texts, rrf_top100, corpus_text_map)
        rerank_s = time.perf_counter() - t0
        rerank_rrf_metrics = evaluate(reranked_rrf, query_items, qrel_map)
        logger.info(f"RRF+Reranker: {rerank_s:.1f}s — NDCG@10={rerank_rrf_metrics['ndcg@10']:.4f}")
        total_s = encode_query_s + bm25_search_s + vector_search_s + rrf_fuse_s + rerank_s
        results_log.append({
            "condition": "Reranker(RRF)",
            "encoder": encoder,
            "reranker": reranker,
            **rerank_rrf_metrics,
            "search_s": round(bm25_search_s + vector_search_s + rrf_fuse_s + rerank_s, 1),
            "rerank_s": round(rerank_s, 1),
            "query_latency_ms": round(total_s / n_queries * 1000, 0),
            "ingestion_cost": f"Encode {n_corpus}p + FTS+Vector index",
            "query_cost": "Encode 1q + BM25 + ANN + RRF + Rerank 100",
        })

    # ========== Save Results ==========
    csv_path = "benchmark_v2_results.csv"
    all_keys = ["condition", "encoder", "reranker", "ndcg@5", "ndcg@10", "mrr@10",
                "recall@5", "recall@10", "recall@100", "n_queries",
                "search_s", "rerank_s", "query_latency_ms", "ingestion_cost", "query_cost"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results_log)

    # Save cost log
    cost_log["conditions"] = results_log
    with open("benchmark_v2_costs.json", "w") as f:
        json.dump(cost_log, f, indent=2, default=str)

    # ========== Print Ablation Tables ==========
    logger.info(f"\n{'=' * 80}")
    logger.info("ABLATION TABLE 1: Quality Metrics")
    logger.info(f"{'=' * 80}")
    header = f"{'Condition':<25} {'Reranker':<35} {'NDCG@5':>7} {'NDCG@10':>8} {'MRR@10':>8} {'R@5':>6} {'R@10':>7} {'R@100':>7}"
    logger.info(header)
    logger.info("-" * len(header))
    for r in sorted(results_log, key=lambda x: x.get("ndcg@10", 0), reverse=True):
        reranker = r.get("reranker", "n/a")
        if reranker != "n/a":
            reranker = reranker.split("/")[-1]  # Short name
        logger.info(f"{r['condition']:<25} {reranker:<35} "
                    f"{r.get('ndcg@5', 0):>7.4f} {r.get('ndcg@10', 0):>8.4f} "
                    f"{r.get('mrr@10', 0):>8.4f} {r.get('recall@5', 0):>6.4f} "
                    f"{r.get('recall@10', 0):>7.4f} {r.get('recall@100', 0):>7.4f}")

    logger.info(f"\n{'=' * 80}")
    logger.info("ABLATION TABLE 2: Latency & Cost")
    logger.info(f"{'=' * 80}")
    header2 = f"{'Condition':<25} {'Reranker':<35} {'Search(s)':>10} {'Rerank(s)':>10} {'Per-query(ms)':>14}"
    logger.info(header2)
    logger.info("-" * len(header2))
    for r in sorted(results_log, key=lambda x: x.get("query_latency_ms", 0)):
        reranker = r.get("reranker", "n/a")
        if reranker != "n/a":
            reranker = reranker.split("/")[-1]
        logger.info(f"{r['condition']:<25} {reranker:<35} "
                    f"{r.get('search_s', 0):>10.1f} {r.get('rerank_s', ''):>10} "
                    f"{r.get('query_latency_ms', 0):>14.0f}")

    logger.info(f"\n{'=' * 80}")
    logger.info("ABLATION TABLE 3: Ingestion Cost Breakdown")
    logger.info(f"{'=' * 80}")
    logger.info(f"  Corpus pages:           {n_corpus}")
    logger.info(f"  Queries:                {n_queries}")
    logger.info(f"  Embedding dim:          {dim}")
    logger.info(f"  Encoder:                {encoder}")
    logger.info(f"  Corpus encode time:     {cost_log.get('encode_corpus_s', 0)}s ({cost_log.get('encode_corpus_per_page_ms', 0)} ms/page)")
    logger.info(f"  Turbopuffer index time: {cost_log.get('index_s', 0)}s")
    logger.info(f"  Query encode time:      {cost_log.get('encode_query_s', 0)}s ({cost_log.get('encode_query_per_query_ms', 0)} ms/query)")

    logger.info(f"\n{'=' * 80}")
    logger.info("ABLATION TABLE 4: RRF vs Reranker (head-to-head)")
    logger.info(f"{'=' * 80}")
    rrf_row = next((r for r in results_log if r["condition"] == "RRF(BM25+Vector)"), None)
    if rrf_row:
        reranker_rows = [r for r in results_log if r["condition"] == "Reranker(Vector)"]
        logger.info(f"{'Method':<50} {'NDCG@10':>8} {'MRR@10':>8} {'R@100':>7} {'ms/query':>10}")
        logger.info("-" * 85)
        logger.info(f"{'RRF(BM25+Vector)':<50} {rrf_row['ndcg@10']:>8.4f} {rrf_row['mrr@10']:>8.4f} "
                    f"{rrf_row['recall@100']:>7.4f} {rrf_row['query_latency_ms']:>10.0f}")
        for r in reranker_rows:
            name = f"Reranker({r['reranker'].split('/')[-1]})"
            delta = r['ndcg@10'] - rrf_row['ndcg@10']
            logger.info(f"{name:<50} {r['ndcg@10']:>8.4f} {r['mrr@10']:>8.4f} "
                        f"{r['recall@100']:>7.4f} {r['query_latency_ms']:>10.0f}  (Δ={delta:+.4f})")

    logger.info(f"\nResults saved to {csv_path} and benchmark_v2_costs.json")


if __name__ == "__main__":
    main()
