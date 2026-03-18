"""Benchmark v2: BM25 + Vector + RRF + Reranker on vidore_v3_finance_en.

Experimental conditions:
1. BM25 only (Turbopuffer FTS)
2. Vector only (bge-m3, stella — already indexed from v1)
3. RRF (BM25 + Vector)
4. RRF + Reranker
"""
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

SIE_BASE_URL = os.environ["SIE_BASE_URL"]
SIE_API_KEY = os.environ["SIE_API_KEY"]
TPUF_API_KEY = os.environ["TURBOPUFFER_API_KEY"]

GPU = "l4-spot"
PROVISION_TIMEOUT = 900
ENCODE_BATCH_SIZE = 64
TPUF_BATCH_SIZE = 500
TOP_K_RETRIEVE = 100
TOP_K_EVAL = 10
RRF_K = 60  # standard RRF constant

ENCODERS = [
    "BAAI/bge-m3",
    "NovaSearch/stella_en_400M_v5",
]

RERANKERS = [
    "mixedbread-ai/mxbai-rerank-base-v2",
    "BAAI/bge-reranker-v2-m3",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
]

BM25_NAMESPACE = "vidore-bm25"


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9-]", "-", name.lower()).strip("-")


def ndcg_at_k(ranked_ids: list[int], qrel_map: dict[int, int], k: int = 10) -> float:
    dcg = sum(qrel_map.get(cid, 0) / math.log2(i + 2) for i, cid in enumerate(ranked_ids[:k]))
    ideal_rels = sorted(qrel_map.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(ranked_ids: list[int], qrel_map: dict[int, int], k: int = 10) -> float:
    for i, cid in enumerate(ranked_ids[:k]):
        if qrel_map.get(cid, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked_ids: list[int], qrel_map: dict[int, int], k: int = 10) -> float:
    relevant = {cid for cid, score in qrel_map.items() if score > 0}
    if not relevant:
        return 0.0
    return sum(1 for cid in ranked_ids[:k] if cid in relevant) / len(relevant)


def evaluate(ranked_results: list[list[int]], query_items: list[dict],
             qrel_map: dict, k: int = TOP_K_EVAL) -> dict:
    ndcgs, mrrs, recalls = [], [], []
    for ranked, qi in zip(ranked_results, query_items):
        qrels = qrel_map.get(qi["query_id"], {})
        if not qrels:
            continue
        ndcgs.append(ndcg_at_k(ranked, qrels, k))
        mrrs.append(mrr_at_k(ranked, qrels, k))
        recalls.append(recall_at_k(ranked, qrels, k))
    return {
        f"ndcg@{k}": np.mean(ndcgs),
        f"mrr@{k}": np.mean(mrrs),
        f"recall@{k}": np.mean(recalls),
        "n_queries": len(ndcgs),
    }


def rrf_fuse(ranked_lists: list[list[int]], k: int = RRF_K) -> list[int]:
    """Reciprocal Rank Fusion over multiple ranked lists."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def load_dataset():
    logger.info("Loading dataset from HuggingFace...")
    t0 = time.perf_counter()
    corpus = pl.read_parquet("hf://datasets/vidore/vidore_v3_finance_en/corpus/test-*.parquet")
    queries = pl.read_parquet("hf://datasets/vidore/vidore_v3_finance_en/queries/test-*.parquet")
    qrels = pl.read_parquet("hf://datasets/vidore/vidore_v3_finance_en/qrels/test-*.parquet")
    logger.info(f"Loaded in {time.perf_counter() - t0:.1f}s: "
                f"{corpus.shape[0]} pages, {queries.shape[0]} queries, {qrels.shape[0]} qrels")

    qrel_map = {}
    for row in qrels.iter_rows(named=True):
        qid = row["query_id"]
        if qid not in qrel_map:
            qrel_map[qid] = {}
        qrel_map[qid][row["corpus_id"]] = row["score"]

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

    query_items = [{"query_id": row["query_id"], "text": row["query"]}
                   for row in queries.iter_rows(named=True)]
    return corpus_items, query_items, qrel_map


def encode_batch(client, model, texts, is_query=False):
    all_vectors = []
    for i in range(0, len(texts), ENCODE_BATCH_SIZE):
        batch = [{"text": t} for t in texts[i:i + ENCODE_BATCH_SIZE]]
        results = client.encode(model, batch, output_types=["dense"], is_query=is_query,
                                gpu=GPU, wait_for_capacity=True, provision_timeout_s=PROVISION_TIMEOUT)
        all_vectors.extend(r["dense"] for r in results)
        if (i + ENCODE_BATCH_SIZE) % 512 == 0 or i + ENCODE_BATCH_SIZE >= len(texts):
            logger.info(f"  Encoded {min(i + ENCODE_BATCH_SIZE, len(texts))}/{len(texts)}")
    return all_vectors


def setup_bm25_namespace(tpuf, corpus_items):
    """Create a BM25-enabled namespace with markdown text."""
    ns = tpuf.namespace(BM25_NAMESPACE)
    logger.info("Setting up BM25 namespace...")
    t0 = time.perf_counter()

    for i in range(0, len(corpus_items), TPUF_BATCH_SIZE):
        batch = corpus_items[i:i + TPUF_BATCH_SIZE]
        ns.write(
            upsert_columns={
                "id": [str(item["corpus_id"]) for item in batch],
                "text": [item["text"] for item in batch],
                "doc_id": [item["doc_id"] for item in batch],
                "page_number": [item["page_number"] for item in batch],
            },
            schema={
                "text": {"type": "string", "full_text_search": {"tokenizer": "word_v2"}},
            },
        )
        if (i + TPUF_BATCH_SIZE) % 1000 == 0 or i + TPUF_BATCH_SIZE >= len(corpus_items):
            logger.info(f"  Indexed {min(i + TPUF_BATCH_SIZE, len(corpus_items))}/{len(corpus_items)}")

    logger.info(f"BM25 namespace ready in {time.perf_counter() - t0:.1f}s")


def search_bm25(tpuf, query_texts, top_k=TOP_K_RETRIEVE):
    """BM25 search for each query. Returns list of [corpus_id, ...]."""
    ns = tpuf.namespace(BM25_NAMESPACE)
    all_results = []
    for i, qtext in enumerate(query_texts):
        result = ns.query(
            rank_by=("text", "BM25", qtext),
            top_k=top_k,
            include_attributes=False,
        )
        hits = [int(row.id) for row in result.rows]
        all_results.append(hits)
        if (i + 1) % 200 == 0:
            logger.info(f"  BM25 searched {i + 1}/{len(query_texts)}")
    return all_results


def search_vector(tpuf, namespace, query_vectors, top_k=TOP_K_RETRIEVE):
    """Vector ANN search. Returns list of [corpus_id, ...]."""
    ns = tpuf.namespace(namespace)
    all_results = []
    for i, qvec in enumerate(query_vectors):
        result = ns.query(
            rank_by=("vector", "ANN", qvec.tolist()),
            top_k=top_k,
            include_attributes=False,
        )
        hits = [int(row.id) for row in result.rows]
        all_results.append(hits)
        if (i + 1) % 200 == 0:
            logger.info(f"  Vector searched {i + 1}/{len(query_vectors)}")
    return all_results


def index_vectors(tpuf, namespace, corpus_items, vectors):
    """Upsert corpus vectors into Turbopuffer namespace."""
    ns = tpuf.namespace(namespace)
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


def rerank_results(sie, model, query_texts, candidate_ids_list, corpus_text_map):
    """Rerank candidates using SIE score endpoint."""
    all_reranked = []
    for i, (qtext, cand_ids) in enumerate(zip(query_texts, candidate_ids_list)):
        if not cand_ids:
            all_reranked.append([])
            continue
        # Only rerank top 100
        cand_ids = cand_ids[:TOP_K_RETRIEVE]
        items = [{"text": corpus_text_map.get(cid, "")} for cid in cand_ids]
        score_result = sie.score(
            model, query={"text": qtext}, items=items,
            gpu=GPU, wait_for_capacity=True, provision_timeout_s=PROVISION_TIMEOUT,
        )
        scored = []
        for s in score_result["scores"]:
            corpus_id = cand_ids[s["rank"]]
            scored.append((corpus_id, s["score"]))
        scored.sort(key=lambda x: x[1], reverse=True)
        all_reranked.append([cid for cid, _ in scored])
        if (i + 1) % 200 == 0:
            logger.info(f"  Reranked {i + 1}/{len(query_texts)}")
    return all_reranked


def main():
    sie = SIEClient(SIE_BASE_URL, api_key=SIE_API_KEY, timeout_s=900)
    tpuf = Turbopuffer(api_key=TPUF_API_KEY, region="aws-us-east-1")

    corpus_items, query_items, qrel_map = load_dataset()
    corpus_text_map = {item["corpus_id"]: item["text"] for item in corpus_items}
    corpus_texts = [item["text"] for item in corpus_items]
    query_texts = [item["text"] for item in query_items]

    results_log = []

    # ========== Step 1: Setup BM25 namespace ==========
    setup_bm25_namespace(tpuf, corpus_items)

    # ========== Step 2: BM25 baseline ==========
    logger.info("\n" + "=" * 60)
    logger.info("CONDITION 1: BM25 only")
    logger.info("=" * 60)

    t0 = time.perf_counter()
    bm25_results = search_bm25(tpuf, query_texts)
    bm25_time = time.perf_counter() - t0
    logger.info(f"BM25 search: {bm25_time:.1f}s")

    bm25_metrics = evaluate(bm25_results, query_items, qrel_map)
    logger.info(f"BM25: NDCG@10={bm25_metrics['ndcg@10']:.4f}, "
                f"MRR@10={bm25_metrics['mrr@10']:.4f}, Recall@10={bm25_metrics['recall@10']:.4f}")
    results_log.append({"condition": "BM25 only", "encoder": "none", "reranker": "none",
                        **{k: round(v, 4) if isinstance(v, float) else v for k, v in bm25_metrics.items()},
                        "search_s": round(bm25_time, 1)})

    # ========== Step 3: Vector only + RRF for each encoder ==========
    for encoder in ENCODERS:
        ns_name = f"vidore-{slugify(encoder)}"
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Encoder: {encoder}")
        logger.info("=" * 60)

        # Encode corpus (check if namespace already exists)
        logger.info("Encoding corpus...")
        t0 = time.perf_counter()
        corpus_vectors = encode_batch(sie, encoder, corpus_texts)
        encode_corpus_time = time.perf_counter() - t0
        logger.info(f"Corpus encoding: {encode_corpus_time:.1f}s (dim={corpus_vectors[0].shape[0]})")

        logger.info("Indexing vectors...")
        t0 = time.perf_counter()
        index_vectors(tpuf, ns_name, corpus_items, corpus_vectors)
        index_time = time.perf_counter() - t0
        logger.info(f"Indexing: {index_time:.1f}s")

        # Encode queries
        logger.info("Encoding queries...")
        t0 = time.perf_counter()
        query_vectors = encode_batch(sie, encoder, query_texts, is_query=True)
        encode_query_time = time.perf_counter() - t0
        logger.info(f"Query encoding: {encode_query_time:.1f}s")

        # --- Condition 2: Vector only ---
        logger.info("\nCONDITION 2: Vector only")
        t0 = time.perf_counter()
        vector_results = search_vector(tpuf, ns_name, query_vectors)
        vector_time = time.perf_counter() - t0
        logger.info(f"Vector search: {vector_time:.1f}s")

        vec_metrics = evaluate(vector_results, query_items, qrel_map)
        logger.info(f"Vector: NDCG@10={vec_metrics['ndcg@10']:.4f}, "
                    f"MRR@10={vec_metrics['mrr@10']:.4f}, Recall@10={vec_metrics['recall@10']:.4f}")
        results_log.append({"condition": "Vector only", "encoder": encoder, "reranker": "none",
                            **{k: round(v, 4) if isinstance(v, float) else v for k, v in vec_metrics.items()},
                            "search_s": round(vector_time, 1),
                            "encode_corpus_s": round(encode_corpus_time, 1),
                            "encode_query_s": round(encode_query_time, 1)})

        # --- Condition 3: RRF (BM25 + Vector) ---
        logger.info("\nCONDITION 3: RRF (BM25 + Vector)")
        rrf_results = [rrf_fuse([bm25, vec]) for bm25, vec in zip(bm25_results, vector_results)]
        rrf_metrics = evaluate(rrf_results, query_items, qrel_map)
        logger.info(f"RRF: NDCG@10={rrf_metrics['ndcg@10']:.4f}, "
                    f"MRR@10={rrf_metrics['mrr@10']:.4f}, Recall@10={rrf_metrics['recall@10']:.4f}")
        results_log.append({"condition": "RRF (BM25+Vec)", "encoder": encoder, "reranker": "none",
                            **{k: round(v, 4) if isinstance(v, float) else v for k, v in rrf_metrics.items()}})

        # --- Condition 4: RRF + Reranker ---
        for reranker in RERANKERS:
            logger.info(f"\nCONDITION 4: RRF + {reranker}")
            t0 = time.perf_counter()
            reranked = rerank_results(sie, reranker, query_texts, rrf_results, corpus_text_map)
            rerank_time = time.perf_counter() - t0
            logger.info(f"Reranking: {rerank_time:.1f}s")

            rr_metrics = evaluate(reranked, query_items, qrel_map)
            logger.info(f"RRF+Rerank: NDCG@10={rr_metrics['ndcg@10']:.4f}, "
                        f"MRR@10={rr_metrics['mrr@10']:.4f}, Recall@10={rr_metrics['recall@10']:.4f}")
            results_log.append({"condition": "RRF+Reranker", "encoder": encoder, "reranker": reranker,
                                **{k: round(v, 4) if isinstance(v, float) else v for k, v in rr_metrics.items()},
                                "rerank_s": round(rerank_time, 1)})

    # ========== Save results ==========
    csv_path = "benchmark_v2_results.csv"
    all_keys = sorted({k for r in results_log for k in r})
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(results_log)

    logger.info(f"\n{'=' * 60}")
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(f"{'Condition':<20} {'Encoder':<35} {'Reranker':<35} {'NDCG@10':>8} {'MRR@10':>8} {'Recall@10':>10}")
    logger.info("-" * 120)
    for r in sorted(results_log, key=lambda x: x.get("ndcg@10", 0), reverse=True):
        logger.info(f"{r.get('condition', ''):<20} {r.get('encoder', ''):<35} {r.get('reranker', ''):<35} "
                    f"{r.get('ndcg@10', 0):>8.4f} {r.get('mrr@10', 0):>8.4f} {r.get('recall@10', 0):>10.4f}")
    logger.info(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
