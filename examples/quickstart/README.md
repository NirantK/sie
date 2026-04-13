# Retrieval Ablation Benchmark

Benchmark comparing 6 retrieval strategies on financial 10-K filings (vidore_v3_finance_en: 2942 pages, 1854 queries).

## Results

| # | Condition | Model | NDCG@10 | MRR@10 | Recall@10 |
|---|-----------|-------|---------|--------|-----------|
| 4 | CE Rerank (hybrid) | mxbai-rerank-base-v2 | **0.5098** | 0.6228 | 0.5587 |
| 4 | CE Rerank (hybrid) | bge-reranker-v2-m3 | 0.5069 | 0.6321 | 0.5558 |
| 6 | MV Direct | bge-m3 (1024d) | 0.4354 | 0.581 | 0.4815 |
| 5 | MV Rerank (hybrid) | bge-m3 (1024d) | 0.433 | 0.5808 | 0.4737 |
| 2 | Vector (dense) | bge-m3 | 0.3962 | 0.5317 | 0.4377 |
| 3 | RRF(BM25+Vec) | - | 0.3583 | 0.4505 | 0.4337 |
| 5 | MV Rerank (hybrid) | mxbai-colbert (128d) | 0.2415 | 0.3101 | 0.3034 |
| 1 | BM25-only | - | 0.1849 | 0.2115 | 0.2386 |
| 6 | MV Direct | mxbai-colbert (128d) | 0.1768 | 0.2392 | 0.2110 |

See [RESULTS.md](RESULTS.md) for methodology and detailed findings.

## Quick Start

### Prerequisites

Create `.env` in this directory:

```
SIE_BASE_URL=http://your-sie-endpoint:8080
SIE_API_KEY=SL-...
TURBOPUFFER_API_KEY=tpuf_...
```

### Install

```bash
uv sync
```

### Run

```bash
# Validate config without GPU
uv run python benchmark_ablation.py --dry-run

# Full benchmark (all 6 conditions, all models)
uv run python benchmark_ablation.py --gpu l4-spot

# CE reranking only (conditions 1-4)
uv run python benchmark_ablation.py --gpu l4-spot --skip-conditions 5,6

# MV direct only (condition 6, CPU-bound after encoding)
uv run python benchmark_ablation.py --gpu l4-spot --skip-conditions 1,2,3,4,5

# Select specific models
uv run python benchmark_ablation.py --ce-models mxbai-rerank --mv-models bge-m3

# Parallel on separate GPUs
uv run python benchmark_ablation.py --gpu l4-spot --skip-conditions 5,6 &
uv run python benchmark_ablation.py --gpu rtx6000-spot --skip-conditions 1,2,3,4 &
```

### Caching

All expensive operations cache to `cache/ablation/`. Re-runs skip completed steps.
CE reranking saves `.partial.json` checkpoints every 100 queries for crash recovery.

Results written incrementally to `ablation_results.csv` after each condition completes.

## Dependencies

- [SIE SDK](https://github.com/superlinked/sie) - async encode/score
- [Turbopuffer](https://turbopuffer.com) - BM25 + vector search
- [maxsim-cpu](https://github.com/mixedbread-ai/maxsim-cpu) - optimized MaxSim scoring
- [datasets](https://huggingface.co/docs/datasets) - HuggingFace dataset loading (cached locally)
