# SIE + Turbopuffer Retrieval Benchmark Proposal

**Dataset**: vidore_v3_finance_en (6 bank 10-K filings, 2,942 pages, 1,854 queries with relevance judgments)

**Goal**: Find the optimal retrieval pipeline (model selection, search strategy, reranking) that maximizes quality at acceptable cost/latency.

---

## 1. Experimental Design

### 1.1 Pipeline Architecture

```
Raw PDF pages (images)
    │
    ├── Florence-2 OCR (SIE extract) ──→ page text
    │
    ├── GLINer NER (SIE extract) ──→ entities (company, person, location)
    │                                  → Turbopuffer filter attributes
    │
    ├── Embedding (SIE encode) ──→ dense vectors → Turbopuffer ANN index
    │
    └── FTS tokenization ──→ Turbopuffer BM25 index (word_v2 tokenizer)

Query Pipeline:
    Query → Encode → { BM25 | Vector ANN | RRF(BM25+Vector) }
              │
              └── Optional: Rerank top-100 (SIE score)
              └── Evaluate: NDCG@k, MRR@k, Recall@k vs qrels
```

### 1.2 Ablation Conditions

| # | Condition | Retriever | Reranker | What it isolates |
|---|-----------|-----------|----------|------------------|
| 1 | BM25-only | Turbopuffer FTS | none | Keyword baseline |
| 2 | Vector-only | bge-m3 ANN | none | Semantic search baseline |
| 3 | RRF(BM25+Vector) | Fused | none | Does hybrid beat vector-only? |
| 4 | Reranker(Vector) | bge-m3 ANN | mxbai-rerank / bge-reranker / MiniLM | Reranker value over vector |
| 5 | Reranker(RRF) | Fused | same 3 rerankers | Does better recall pool help reranker? |

### 1.3 Encoder Selection (Phase 1)

Shortlist 5 diverse encoders, evaluate vector-only retrieval, take top 3 for reranking.

| Model | Dim | Max tokens | Bundle |
|-------|-----|------------|--------|
| BAAI/bge-m3 | 1024 | 8192 | default |
| NovaSearch/stella_en_400M_v5 | 1024 | 8192 | default |
| sentence-transformers/all-MiniLM-L6-v2 | 384 | 512 | default |
| intfloat/e5-large-v2 | 1024 | 512 | default |
| Qwen/Qwen3-Embedding-0.6B | 1024 | 8192 | default |

### 1.4 Reranker Selection

| Model | Type | Bundle |
|-------|------|--------|
| mixedbread-ai/mxbai-rerank-base-v2 | Cross-encoder | default |
| BAAI/bge-reranker-v2-m3 | Cross-encoder | default |
| cross-encoder/ms-marco-MiniLM-L-12-v2 | Cross-encoder | default |

### 1.5 Cost Tracking

Three cost categories:

| Category | Components | When incurred |
|----------|-----------|---------------|
| **Ingestion** | OCR (Florence-2) + NER (GLINer) + Encode (bge-m3) + Turbopuffer write | Once per corpus |
| **Query** | Encode query + Turbopuffer search (BM25/ANN) | Per query |
| **Rerank** | SIE score on top-100 candidates | Per query (optional) |

Metrics: wall-clock time, per-page/per-query latency (ms), throughput (pages/s or queries/s).

SIE is self-hosted (no per-token pricing) — GPU time is the cost proxy. Turbopuffer pricing is based on storage + queries.

---

## 2. Dataset Characteristics

| Property | Value |
|----------|-------|
| Documents | 6 bank 10-K filings (JPMorgan, Goldman Sachs, Citigroup, Wells Fargo, Morgan Stanley, Bank of America) |
| Pages | 2,942 |
| Queries | 1,854 |
| Relevance judgments | 8,766 (scores: 1=relevant, 2=highly relevant) |
| Avg relevant docs/query | 4.7 |
| Page text length | median 3,809 chars, max 9,835 chars |
| Query types | boolean, compare-contrast, open-ended |
| Language | English |

### Source files

5/6 raw HTM filings downloaded from SEC EDGAR (Bank of America EDGAR rate-limited).

---

## 3. Key Questions

1. **Does BM25 add value over vector search?** (RRF vs Vector-only)
2. **Are rerankers worth the latency?** (Reranker vs no-reranker, with cost/latency tradeoff)
3. **Does a better recall pool help rerankers?** (Reranker over RRF vs Reranker over Vector)
4. **Which encoder gives best quality/cost ratio?** (Phase 1 encoder selection)
5. **How does OCR text quality compare to ground-truth markdown?** (v3 OCR pipeline comparison)

---

## 4. Infrastructure

| Component | Details |
|-----------|---------|
| SIE cluster | Self-hosted, L4-spot GPUs, KEDA autoscaling |
| Cold start | 5-7 min (GPU provisioning + model download) |
| Warm inference | milliseconds for encode/score, ~20s/page for Florence-2 OCR |
| Turbopuffer | aws-us-east-1, cosine_distance, word_v2 FTS tokenizer |
| Models available | 90 total, 67 loadable on L4-spot (7 sglang models need A100) |

---

## Appendix A: Phase 1 Results — Encoder Selection

*5 encoders, vector-only retrieval, no reranking. Dataset markdown text.*

| Encoder | Dim | NDCG@10 | MRR@10 | Recall@10 | Encode time (2942 pages) |
|---------|-----|---------|--------|-----------|--------------------------|
| **BAAI/bge-m3** | 1024 | **0.3972** | **0.5333** | **0.4380** | 231s |
| NovaSearch/stella_en_400M_v5 | 1024 | 0.3649 | 0.4834 | 0.4167 | 113s |
| intfloat/e5-large-v2 | 1024 | 0.2393 | 0.3241 | 0.2851 | 84s |
| sentence-transformers/all-MiniLM-L6-v2 | 384 | 0.1368 | 0.1873 | 0.1642 | 171s |
| Qwen/Qwen3-Embedding-0.6B | 1024 | 0.0024 | 0.0025 | 0.0048 | 625s |

**Winner**: bge-m3 — best quality across all metrics. stella is a strong runner-up at 2x encoding speed.

**Observations**:
- MiniLM (512 token limit) loses badly on 3.8K char pages — truncation destroys context
- Qwen3-Embedding near zero — likely needs instruction prefix or different input format
- bge-m3's 8192 token context window is critical for page-level retrieval

## Appendix B: Phase 2 Intermediate Results — BM25 + RRF

*Using bge-m3 encoder, Turbopuffer FTS (word_v2 tokenizer) for BM25.*

| Condition | NDCG@10 | Notes |
|-----------|---------|-------|
| BM25-only | 0.1849 | Keyword search baseline |
| Vector-only (bge-m3) | 0.3972 | Semantic search — 2.1x better than BM25 |
| RRF(BM25 + Vector) | 0.3712 | Hybrid — *worse* than vector-only |
| Reranker(Vector) | *running...* | 3 rerankers being evaluated |
| Reranker(RRF) | *running...* | 3 rerankers being evaluated |

**Early finding**: RRF *hurts* quality here (0.3972 → 0.3712). BM25 is so much weaker that fusing it dilutes the vector signal. This is specific to this dataset — financial 10-K filings with dense, specialized vocabulary where semantic search dominates keyword matching.

## Appendix C: Ingestion Cost Breakdown

| Step | Time | Per-unit | Notes |
|------|------|----------|-------|
| Corpus encode (bge-m3, 2942 pages) | 254s | 86 ms/page | Warm cluster |
| Turbopuffer index (FTS + Vector) | 8.4s | 2.9 ms/page | Batch upsert |
| Query encode (1854 queries) | 16.8s | 9.1 ms/query | |
| BM25 search (1854 queries, top-100) | 690s | 372 ms/query | |
| Vector ANN search (1854 queries, top-100) | 572s | 308 ms/query | |
| Florence-2 OCR (2942 pages) | *~8h estimated* | *~10s/page* | Single GPU, bottleneck |

## Appendix D: Model Loading Times (90 models)

| Category | Count | Median load time | Notes |
|----------|-------|------------------|-------|
| Default bundle (worked) | 67 | 11-30s | After warm cluster |
| Sglang bundle | 7 | ALL TIMEOUT (900s) | Need A100 or longer timeout |
| Vision models (need images) | 8 | 11-27s | Require image input |
| Server errors | 8 | instant | Missing preprocessors, wrong output types |
| **Total** | **90** | | **67/90 loadable on L4-spot** |

Cold start (0 → 1 worker): 5-7 min (GPU provision + image pull + model download).
