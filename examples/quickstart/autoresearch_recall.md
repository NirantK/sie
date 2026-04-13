# Autoresearch: Maximize Recall@10

Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch). 
An autonomous agent iteratively modifies retrieval config to maximize Recall@10.

## Setup

The benchmark script `benchmark_ablation.py` is the experiment harness.
All MV encodings are pre-cached (expensive GPU work done).
Each experiment iteration is CPU-only (MaxSim scoring) — runs in ~2 min.

## What you CAN modify

Config knobs per experiment:
- `TOP_K_RETRIEVE` — 10, 25, 50, 100 (more candidates = higher recall ceiling)
- Hybrid pool construction — ratio of BM25 vs Vector (currently 50/50)
- RRF k parameter (currently 60)
- Candidate pool strategy — union vs interleave vs RRF-scored
- Combine MV rerank with CE rerank (two-stage)
- Use MV scores to filter before CE scoring
- Weighted fusion of MV + dense scores

## What you CANNOT modify

- Model weights (already encoded and cached)
- Evaluation code (ndcg_at_k, mrr_at_k, recall_at_k)
- Dataset (vidore_v3_finance_en, fixed)

## The goal

**Maximize Recall@10.** Current best: 0.5587 (CE rerank mxbai over hybrid-25 pool).

Secondary: maintain NDCG@10 > 0.45 (don't sacrifice precision for recall).

## Experiment loop

Each experiment:
1. Modify pool construction or scoring strategy in benchmark_ablation.py
2. Run: `uv run python benchmark_ablation.py --skip-conditions 1,2,3 --gpu none`
3. Check ablation_results.csv for Recall@10
4. If Recall@10 improved and NDCG@10 > 0.45: keep
5. If worse: revert
6. Log to results.tsv and iterate

## Baseline results

| Strategy | NDCG@10 | Recall@10 |
|----------|---------|-----------|
| CE Rerank mxbai (hybrid-25) | 0.5098 | 0.5587 |
| CE Rerank bge-reranker (hybrid-25) | 0.5069 | 0.5558 |
| MV Rerank jina-colbert (hybrid-25) | 0.431 | 0.4937 |
| MV Direct bge-m3 | 0.4354 | 0.4815 |
| MV Direct jina-colbert | 0.4187 | 0.486 |

## Ideas to try

1. Increase TOP_K to 50 or 100 — more candidates in pool = higher recall ceiling
2. Two-stage: MV rerank top-100 → CE rerank top-20 from MV output
3. Score fusion: combine dense + MV scores before final ranking
4. Asymmetric pool: 40 vector + 10 BM25 (vector is stronger on this dataset)
5. RRF with different k values (lower k = more weight to top results)
