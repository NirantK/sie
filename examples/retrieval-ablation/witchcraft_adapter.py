"""Witchcraft (Dropbox XTR-WARP) adapter for retrieval ablation benchmark.

Compares witchcraft's end-to-end stack (T5 128d + WARP + SQLite BM25)
against our pipeline (SIE + Turbopuffer + maxsim-cpu).

Prerequisites:
    git clone https://github.com/dropbox/witchcraft /tmp/witchcraft
    cd /tmp/witchcraft && make download && make warp-cli

Usage:
    uv run python witchcraft_adapter.py
    uv run python witchcraft_adapter.py --warp-cli /path/to/warp-cli
    uv run python witchcraft_adapter.py --hybrid  # include BM25 fusion
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")

WARP_CLI_DEFAULT = "/tmp/witchcraft/target/aarch64-apple-darwin/release/warp-cli"
ASSETS_DEFAULT = "/tmp/witchcraft/assets"
CACHE_DIR = Path(__file__).parent / "cache" / "witchcraft"


def load_dataset():
    from datasets import load_dataset as hf_load

    corpus = pl.from_arrow(hf_load("vidore/vidore_v3_finance_en", "corpus", split="test").data.table)
    queries = pl.from_arrow(hf_load("vidore/vidore_v3_finance_en", "queries", split="test").data.table)
    qrels_df = pl.from_arrow(hf_load("vidore/vidore_v3_finance_en", "qrels", split="test").data.table)

    corpus_items = []
    for row in corpus.iter_rows(named=True):
        text = row["markdown"] or ""
        if len(text.strip()) < 10:
            text = f"Page {row['page_number_in_doc']} of {row['doc_id']}"
        corpus_items.append({"corpus_id": row["corpus_id"], "text": text})

    query_items = []
    for row in queries.iter_rows(named=True):
        query_items.append({"query_id": row["query_id"], "text": row["query"]})

    qrel_map = {}
    for row in qrels_df.iter_rows(named=True):
        qrel_map.setdefault(row["query_id"], {})[row["corpus_id"]] = row["score"]

    return corpus_items, query_items, qrel_map


def export_corpus_tsv(corpus_items, output_path):
    """Write corpus to TSV in witchcraft's expected format: name<TAB>body"""
    with open(output_path, "w") as f:
        for item in corpus_items:
            # name = corpus_id (will be stored as metadata {"key": "<name>"})
            # body = document text (tabs/newlines replaced)
            name = str(item["corpus_id"])
            body = item["text"].replace("\t", " ").replace("\n", " ")
            f.write(f"{name}\t{body}\n")
    logger.info(f"Exported {len(corpus_items)} docs to {output_path}")


def export_queries_tsv(query_items, output_path):
    """Write queries to TSV: key<TAB>question"""
    with open(output_path, "w") as f:
        for item in query_items:
            key = str(item["query_id"])
            text = item["text"].replace("\t", " ").replace("\n", " ")
            f.write(f"{key}\t{text}\n")
    logger.info(f"Exported {len(query_items)} queries to {output_path}")


def run_cmd(args, label, env=None):
    """Run a subprocess and return (stdout, stderr, elapsed_s)."""
    logger.info(f"  [{label}] {' '.join(str(a) for a in args)}")
    t0 = time.perf_counter()
    result = subprocess.run(args, capture_output=True, text=True, env=env)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        logger.error(f"  [{label}] FAILED (exit {result.returncode})")
        logger.error(f"  stderr: {result.stderr[:500]}")
        raise RuntimeError(f"{label} failed: {result.stderr[:200]}")
    logger.info(f"  [{label}] done ({elapsed:.1f}s)")
    return result.stdout, result.stderr, elapsed


def run_witchcraft_pipeline(warp_cli, assets, db_path, corpus_tsv, queries_tsv, results_path, use_hybrid=False):
    """Orchestrate: readcsv → embed → index → querycsv/hybridcsv"""
    env = {**os.environ, "WITCHCRAFT_DB": str(db_path), "WITCHCRAFT_ASSETS": str(assets)}
    timings = {}

    # Step 1: Import corpus
    _, _, t = run_cmd([warp_cli, "readcsv", str(corpus_tsv)], "readcsv", env)
    timings["readcsv"] = t

    # Step 2: Embed with T5
    _, _, t = run_cmd([warp_cli, "embed"], "embed", env)
    timings["embed"] = t

    # Step 3: Build WARP index
    _, _, t = run_cmd([warp_cli, "index"], "index", env)
    timings["index"] = t

    # Step 4: Batch search
    cmd_name = "hybridcsv" if use_hybrid else "querycsv"
    stdout, _, t = run_cmd([warp_cli, cmd_name, str(queries_tsv), str(results_path)], cmd_name, env)
    timings[cmd_name] = t

    # Parse p95 latency from stdout
    for line in stdout.split("\n"):
        if "p95" in line:
            logger.info(f"  {line.strip()}")

    return timings


def parse_results(results_path, query_items):
    """Parse witchcraft output: key<TAB>result1,result2,...,"""
    results = {}
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            key = parts[0]
            if len(parts) > 1:
                # Results are comma-separated, trailing comma
                result_keys = [r for r in parts[1].split(",") if r.strip()]
                # Convert metadata keys back to corpus_ids (they're stored as-is)
                results[key] = [int(r) if r.isdigit() else r for r in result_keys]
            else:
                results[key] = []

    # Align with query_items order
    ranked_results = []
    for qi in query_items:
        qid = str(qi["query_id"])
        ranked_results.append(results.get(qid, []))
    return ranked_results


def ndcg_at_k(ranked_ids, qrels, k=10):
    dcg = sum(qrels.get(cid, 0) / math.log2(i + 2) for i, cid in enumerate(ranked_ids[:k]))
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_ids, qrels, k=10):
    rel = {c for c, s in qrels.items() if s > 0}
    return sum(1 for c in ranked_ids[:k] if c in rel) / len(rel) if rel else 0.0


def evaluate(ranked_results, query_items, qrel_map, label):
    ndcgs, recs = [], []
    for ranked, qi in zip(ranked_results, query_items):
        qrels = qrel_map.get(qi["query_id"], {})
        if not qrels:
            continue
        ndcgs.append(ndcg_at_k(ranked, qrels))
        recs.append(recall_at_k(ranked, qrels))
    ndcg = np.mean(ndcgs)
    recall = np.mean(recs)
    print(f"{label:<50s} NDCG@10={ndcg:.4f}  R@10={recall:.4f}  (n={len(ndcgs)})")
    return ndcg, recall


def main():
    parser = argparse.ArgumentParser(description="Witchcraft (XTR-WARP) benchmark adapter")
    parser.add_argument("--warp-cli", default=WARP_CLI_DEFAULT, help="Path to warp-cli binary")
    parser.add_argument("--assets", default=ASSETS_DEFAULT, help="Path to witchcraft assets dir")
    parser.add_argument("--hybrid", action="store_true", help="Use hybrid (semantic + BM25) search")
    args = parser.parse_args()

    if not Path(args.warp_cli).exists():
        logger.error(f"warp-cli not found at {args.warp_cli}")
        logger.error("Build it: cd /tmp/witchcraft && make download && make warp-cli")
        sys.exit(1)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset...")
    corpus_items, query_items, qrel_map = load_dataset()
    logger.info(f"Corpus: {len(corpus_items)}, Queries: {len(query_items)}")

    # Export data
    corpus_tsv = CACHE_DIR / "corpus.tsv"
    queries_tsv = CACHE_DIR / "queries.tsv"
    export_corpus_tsv(corpus_items, corpus_tsv)
    export_queries_tsv(query_items, queries_tsv)

    # Run witchcraft pipeline
    db_path = CACHE_DIR / "benchmark.sqlite"
    results_path = CACHE_DIR / "results.txt"
    mode = "hybrid" if args.hybrid else "semantic"

    logger.info(f"\n=== Witchcraft ({mode}) ===")
    timings = run_witchcraft_pipeline(
        args.warp_cli, args.assets, db_path, corpus_tsv, queries_tsv, results_path, use_hybrid=args.hybrid,
    )

    logger.info(f"\nTimings:")
    for step, t in timings.items():
        logger.info(f"  {step}: {t:.1f}s")
    logger.info(f"  total: {sum(timings.values()):.1f}s")

    # Parse and evaluate
    ranked_results = parse_results(results_path, query_items)
    empty = sum(1 for r in ranked_results if not r)
    logger.info(f"Parsed {len(ranked_results)} results ({empty} empty)")

    print()
    print("=== RESULTS ===")
    evaluate(ranked_results, query_items, qrel_map, f"Witchcraft ({mode}, T5 128d)")

    # Compare with our cached baselines if available
    ablation_cache = Path(__file__).parent / "cache" / "ablation"
    print()
    print("=== COMPARISON (our pipeline) ===")
    for name, path in [
        ("BM25 (Turbopuffer)", ablation_cache / "bm25_top50.json"),
        ("Vector (bge-m3 dense)", ablation_cache / "vector_top50.json"),
        ("CE large (mxbai-rerank-large)", ablation_cache / "autoresearch_ce_mixedbread-ai-mxbai-rerank-large-v2.json"),
    ]:
        if path.exists():
            data = json.load(open(path))
            evaluate(data, query_items, qrel_map, name)


if __name__ == "__main__":
    main()
