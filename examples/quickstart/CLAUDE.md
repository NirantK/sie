# Quickstart Benchmarks

## benchmark_ablation.py

Retrieval ablation on `vidore_v3_finance_en` (2942 pages, 1854 queries, 6 bank 10-Ks).

### Run

```bash
# Full run (all 6 conditions)
uv run python benchmark_ablation.py --gpu l4-spot

# CE rerank only (conditions 1-4)
uv run python benchmark_ablation.py --gpu l4-spot --skip-conditions 5,6

# MV direct only (condition 6 — CPU-bound, uses cached MV encodings)
uv run python benchmark_ablation.py --gpu l4-spot --skip-conditions 1,2,3,4,5

# Let router pick GPU
uv run python benchmark_ablation.py --gpu none
```

### Caching

All expensive operations cache to `cache/ablation/`. Re-runs skip completed steps.

| Cache file | What | Size |
|-----------|------|------|
| `dense_corpus.npz` | bge-m3 dense vectors (2942 docs) | 11MB |
| `dense_query.npz` | bge-m3 dense vectors (1854 queries) | 7MB |
| `mv_corpus_*.npz` | Multivector corpus embeddings | 286MB-11GB |
| `mv_query_*.npz` | Multivector query embeddings | 34MB-203MB |
| `bm25_top25.json` | BM25 search results | ~200KB |
| `vector_top25.json` | Vector ANN search results | ~200KB |
| `rerank_ce_*.json` | CE reranked results (with partial checkpoints) | ~60KB |

CE reranking saves `.partial.json` checkpoints every 100 queries. Interrupted runs resume.

### Known issues

- SIE router sends all requests to workers that already have models loaded. New workers from KEDA autoscaling sit idle (superlinked/sie#168).
- Server drops connections under concurrent load (superlinked/sie#169). Script retries transport errors 3 times.
- `GPU_CONCURRENCY=10` keeps concurrent GPU requests low. SDK retries 503 internally; higher concurrency causes thundering herd.
- `sie.score()` returns `rank` as output position (not original index). Use `item_id` ("item-N") to map back to input order.

### Environment

Requires `.env` with:
```
SIE_BASE_URL=http://...
SIE_API_KEY=SL-...
TURBOPUFFER_API_KEY=tpuf_...
```

### Dependencies

- `sie_sdk` (SIEAsyncClient for encode/score)
- `turbopuffer` (AsyncTurbopuffer for BM25/vector search)
- `maxsim-cpu` (optimized MaxSim scoring from mixedbread)
- `datasets` (HuggingFace dataset loading with local cache)
- `polars` (DataFrame handling)
- `loguru` (structured logging)
