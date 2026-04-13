# Retrieval Ablation Benchmark

## Intent

Answer one question with maximum rigor: **What is the best retrieval pipeline for page-level document search on financial 10-K filings?**

Specifically, isolate the contribution of each pipeline stage (keyword, semantic, hybrid fusion, cross-encoder reranking, multi-vector reranking) so we can make an informed cost/quality tradeoff.

**Non-goals:** OCR quality comparison (separate experiment), multi-modal retrieval, production latency optimization.

---

## Dataset

**vidore_v3_finance_en** — 6 bank 10-K filings from SEC EDGAR.

| Property | Value |
|----------|-------|
| Pages | 2,942 |
| Queries | 1,854 |
| Relevance judgments | 8,766 (1=relevant, 2=highly relevant) |
| Avg relevant docs/query | 4.7 |
| Median page text | 3,809 chars (~950 tokens) |
| Text source | Pre-extracted markdown (not OCR) |

---

## Ablation Design

| # | Condition | Retriever | Reranker | What it isolates |
|---|-----------|-----------|----------|------------------|
| 1 | BM25-only | Turbopuffer FTS (word_v2) | none | Keyword baseline |
| 2 | Vector-only | bge-m3 dense ANN | none | Semantic search baseline |
| 3 | RRF(BM25+Vector) | Fused (k=60) | none | Does hybrid beat vector-only? |
| 4 | Cross-Encoder Rerank | RRF top-100 | mxbai-rerank, bge-reranker | Cross-encoder value over retrieval |
| 5 | Multi-Vector Rerank | RRF top-100 | bge-m3 MV, mxbai-colbert | Multi-vector vs cross-encoder |
| 6 | Multi-Vector Direct | Brute-force MaxSim (full corpus) | bge-m3 MV, mxbai-colbert | MV as standalone retriever vs reranker |

**Key controls:**
- Conditions 4 and 5 rerank the **same** RRF top-100 candidate pool — only the reranker changes
- Condition 6 uses the **same** MV encodings as condition 5, but searches the full corpus (no pre-filtering)
- bge-m3 appears as dense encoder (condition 2), MV reranker (condition 5), and MV retriever (condition 6) — isolates representation type AND retrieval strategy
- All metrics computed identically across conditions (same eval code, same qrels)

**Condition 6 approach:**
- Brute-force MaxSim: for each query, compute MaxSim against all 2,942 corpus multivectors
- Rank all docs by MaxSim score, take top-10 for evaluation
- Reuses cached MV encodings from condition 5 (no extra GPU work)
- Only 2,942 docs — brute-force is feasible (~5 min per model on CPU)
- Answers: "Is multi-vector good enough as standalone retriever, or only valuable as reranker?"

---

## Models

### Dense Encoder
| Model | Dim | Max tokens | Why |
|-------|-----|------------|-----|
| BAAI/bge-m3 | 1024 | 8192 | Best from Phase 1 (NDCG@10=0.3972) |

### Cross-Encoder Rerankers (Condition 4)
| Model | Scoring | Notes |
|-------|---------|-------|
| mixedbread-ai/mxbai-rerank-base-v2 | Server-side `sie.score()` | Strong general-purpose |
| BAAI/bge-reranker-v2-m3 | Server-side `sie.score()` | Matches our encoder family |

### Multi-Vector Rerankers (Condition 5)
| Model | Dim | Max tokens | Scoring | Notes |
|-------|-----|------------|---------|-------|
| BAAI/bge-m3 | 1024 | 8192 | Client-side MaxSim | Same model as encoder, MV output |
| mixedbread-ai/mxbai-colbert-large-v1 | 128 | 512 | Client-side MaxSim | Dedicated ColBERT, truncation risk |

---

## Infrastructure

| Component | Details |
|-----------|---------|
| SIE cluster | Self-hosted, RTX6000 GPUs, KEDA autoscaling |
| Endpoint | From `SIE_BASE_URL` env var |
| Turbopuffer | aws-us-east-1, cosine_distance, word_v2 FTS tokenizer |
| GPU type | RTX6000 |

---

## Metrics

| Metric | Description | k |
|--------|-------------|---|
| NDCG@10 | Normalized Discounted Cumulative Gain | 10 |
| MRR@10 | Mean Reciprocal Rank | 10 |
| Recall@10 | Fraction of relevant docs in top 10 | 10 |

All computed against official qrels (score 1=relevant, 2=highly relevant).

---

## Results

**Sorted by NDCG@10 (hybrid pool = union of top-K BM25 + top-K Vector):**

| # | Condition | Model | Dim | Tokens | NDCG@10 | MRR@10 | R@10 |
|---|-----------|-------|-----|--------|---------|--------|------|
| 4 | **CE Rerank** | **mxbai-rerank-base-v2** | — | — | **0.5098** | **0.6228** | **0.5587** |
| 4 | CE Rerank | bge-reranker-v2-m3 | — | — | 0.5069 | 0.6321 | 0.5558 |
| 6 | MV Direct | bge-m3 | 1024 | 8192 | 0.4354 | 0.581 | 0.4815 |
| 5 | MV Rerank | bge-m3 | 1024 | 8192 | 0.433 | 0.5808 | 0.4737 |
| 5 | MV Rerank | **jina-colbert-v2** | 128 | 8192 | **0.431** | 0.548 | 0.4937 |
| 6 | MV Direct | **jina-colbert-v2** | 128 | 8192 | **0.4187** | 0.5322 | 0.486 |
| 2 | Vector | bge-m3 dense | 1024 | — | 0.3962 | 0.5317 | 0.4377 |
| 3 | RRF | — | — | — | 0.3583 | 0.4505 | 0.4337 |
| 5 | MV Rerank | GTE-ModernColBERT | 128 | 8192 | 0.3439 | 0.4188 | 0.4241 |
| 6 | MV Direct | GTE-ModernColBERT | 128 | 8192 | 0.2853 | 0.355 | 0.3437 |
| 5 | MV Rerank | mxbai-colbert | 128 | 512 | 0.2415 | 0.3101 | 0.3034 |
| 5 | MV Rerank | colbertv2.0 | 128 | 512 | 0.2146 | 0.2626 | 0.2816 |
| 1 | BM25 | — | — | — | 0.1849 | 0.2115 | 0.2386 |
| 6 | MV Direct | mxbai-colbert | 128 | 512 | 0.1768 | 0.2392 | 0.2110 |
| 6 | MV Direct | colbertv2.0 | 128 | 512 | 0.1667 | 0.2139 | 0.206 |

---

## Key Findings

1. **Cross-encoder reranking wins**: mxbai (0.5098) ≈ bge-reranker (0.5069) — both +28% over dense vector
2. **jina-colbert-v2 = best cost/quality tradeoff**: 96% of bge-m3 MV quality at 12.5% storage (128d vs 1024d)
3. **bge-m3 multivector direct** = strong second (0.4354) — +10% over dense, no GPU at inference
4. **Token limit matters more than architecture**: 8192-token models (jina, GTE) >> 512-token models (colbertv2, mxbai-colbert)
5. **Score fusion can't beat cross-encoder**: best MV+vector fusion = 0.4413, CE cross-attention gap is fundamental
6. **Pool optimization**: TOP_K=50, 50/50 hybrid union is optimal. Pool ceiling = 0.77 recall; reranker quality is the bottleneck
7. **RRF hurts** (0.3583 < 0.3962) — BM25 dilutes strong vector signal on this dataset

---

## Reproducibility

```bash
# Full run (all conditions, all models)
uv run python benchmark_ablation.py --gpu l4-spot

# Dry run (validate config)
uv run python benchmark_ablation.py --dry-run

# Selective models
uv run python benchmark_ablation.py --ce-models mxbai-rerank --mv-models bge-m3
```

- All intermediate artifacts cached in `cache/ablation/`
- Re-runs skip completed steps automatically
- Random seed: 42
- RRF k parameter: 60
- Retrieval depth: top-25
- Evaluation depth: top-10
- MaxSim scoring: `maxsim-cpu` (mixedbread C library)
