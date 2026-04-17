"""Pool C: MV-bge200 + MV-jina200 → CE rerank. New client per request to avoid session death."""

import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from dotenv import load_dotenv
from sie_sdk import SIEAsyncClient

load_dotenv(Path(__file__).parent / ".env")

CACHE = Path(__file__).parent / "cache" / "ablation"
MV_K = 100  # candidates per MV model (total ~151 after dedup)
SLUG = f"pool-mv-bge{MV_K}-jina{MV_K}-mxbai-large"
CACHE_PATH = CACHE / f"autoresearch_ce_{SLUG}.json"
CHECKPOINT_PATH = CACHE / f"autoresearch_ce_{SLUG}.partial.json"
MODEL = "mixedbread-ai/mxbai-rerank-large-v2"
GPU = "l4-spot"
SIE_URL = os.environ["SIE_BASE_URL"]
SIE_KEY = os.environ["SIE_API_KEY"]


def load_data():
    from datasets import load_dataset as hf_load

    corpus = pl.from_arrow(hf_load("vidore/vidore_v3_finance_en", "corpus", split="test").data.table)
    queries = pl.from_arrow(hf_load("vidore/vidore_v3_finance_en", "queries", split="test").data.table)
    qrels_df = pl.from_arrow(hf_load("vidore/vidore_v3_finance_en", "qrels", split="test").data.table)

    text_map = {}
    for row in corpus.iter_rows(named=True):
        text_map[row["corpus_id"]] = row["markdown"] or f"Page {row['page_number_in_doc']}"

    query_texts = [r["query"] for r in queries.iter_rows(named=True)]
    query_items = [{"query_id": r["query_id"]} for r in queries.iter_rows(named=True)]
    qrel_map = {}
    for row in qrels_df.iter_rows(named=True):
        qrel_map.setdefault(row["query_id"], {})[row["corpus_id"]] = row["score"]

    mv_bge = json.load(open(CACHE / "retrieve_mv_baai-bge-m3.json"))
    mv_jina = json.load(open(CACHE / "retrieve_mv_jinaai-jina-colbert-v2.json"))
    pools = [list(dict.fromkeys(m[:MV_K] + j[:MV_K])) for m, j in zip(mv_bge, mv_jina)]

    return query_texts, query_items, qrel_map, text_map, pools


async def score_one(query_text, candidates, text_map):
    """Score one query with a fresh SIE client — no session reuse."""
    items = [{"text": text_map.get(cid, "")} for cid in candidates]
    for attempt in range(5):
        try:
            async with SIEAsyncClient(SIE_URL, api_key=SIE_KEY, timeout_s=300, max_connections=1) as sie:
                result = await sie.score(
                    MODEL,
                    query={"text": query_text},
                    items=items,
                    gpu=GPU,
                    wait_for_capacity=True,
                    provision_timeout_s=1200,
                )
                scored = sorted(
                    [(candidates[int(s["item_id"].split("-")[1])], s["score"]) for s in result["scores"]],
                    key=lambda x: x[1],
                    reverse=True,
                )
                return [cid for cid, _ in scored]
        except Exception as e:
            if attempt < 4:
                backoff = (2**attempt) + 1
                print(f"  retry {attempt + 1}/5 ({type(e).__name__}), backoff {backoff}s", file=sys.stderr)
                await asyncio.sleep(backoff)
            else:
                raise


async def main():
    print("Loading data...", file=sys.stderr)
    query_texts, query_items, qrel_map, text_map, pools = load_data()
    n = len(query_texts)
    print(f"Pool: ~{np.mean([len(p) for p in pools]):.0f} candidates/query, {n} queries", file=sys.stderr)

    # Resume from checkpoint
    results = [None] * n
    done = 0
    if CHECKPOINT_PATH.exists():
        partial = json.load(open(CHECKPOINT_PATH))
        for i, r in enumerate(partial):
            if r is not None:
                results[i] = r
        done = sum(1 for r in results if r is not None)
        print(f"Resumed {done}/{n} from checkpoint", file=sys.stderr)

    t0 = time.perf_counter()
    for i in range(n):
        if results[i] is not None:
            continue
        results[i] = await score_one(query_texts[i], pools[i], text_map)
        done += 1
        if done % 50 == 0:
            elapsed = time.perf_counter() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n - done) / rate / 60 if rate > 0 else 0
            print(f"  {done}/{n} ({elapsed:.0f}s, {rate:.1f} q/s, ETA {eta:.0f}m)", file=sys.stderr)
        if done % 100 == 0:
            with open(CHECKPOINT_PATH, "w") as f:
                json.dump(results, f)

    with open(CACHE_PATH, "w") as f:
        json.dump(results, f)
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    # Evaluate
    ndcgs, recs = [], []
    for ranked, qi in zip(results, query_items):
        qrels = qrel_map.get(qi["query_id"], {})
        if not qrels:
            continue
        dcg = sum(qrels.get(c, 0) / math.log2(j + 2) for j, c in enumerate(ranked[:10]))
        ideal = sorted(qrels.values(), reverse=True)[:10]
        idcg = sum(r / math.log2(j + 2) for j, r in enumerate(ideal))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        rel = {c for c, s in qrels.items() if s > 0}
        recs.append(sum(1 for c in ranked[:10] if c in rel) / len(rel) if rel else 0.0)

    print(f"Pool C (mv-bge200+jina200 → CE): NDCG={np.mean(ndcgs):.4f}, R@10={np.mean(recs):.4f}")


asyncio.run(main())
