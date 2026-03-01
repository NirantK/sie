<!-- TODO: Replace with actual logo/banner image -->

<h1 align="center">⚡ SIE</h1>
<h4 align="center">Superlinked Inference Engine</h4>
<p align="center"><strong>Open-source inference server and production cluster for embeddings, reranking, and extraction.</strong></p>
<p align="center">85+ models. Three functions. From laptop to Kubernetes. All Apache 2.0.</p>

<!-- TODO: Add social preview / OG image (1280x640) -->
<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square" alt="License"></a>
  <a href="https://pypi.org/project/sie-sdk/"><img src="https://img.shields.io/pypi/v/sie-sdk?style=flat-square" alt="PyPI"></a>
  <a href="https://github.com/superlinked/sie/stargazers"><img src="https://img.shields.io/github/stars/superlinked/sie?style=flat-square" alt="GitHub stars"></a>
</p>

<!--
  Badges to add when ready:
  - CI: <a href="https://github.com/superlinked/sie/actions"><img src="https://img.shields.io/github/actions/workflow/status/superlinked/sie/ci.yml?style=flat-square" alt="CI"></a>  (needs ci.yml workflow)
  - Downloads: <a href="https://pypi.org/project/sie-sdk/"><img src="https://img.shields.io/pypi/dm/sie-sdk?style=flat-square" alt="Downloads"></a>  (needs sie-sdk published)
  - Discord: <a href="https://discord.gg/superlinked"><img src="https://img.shields.io/discord/GUILD_ID?style=flat-square&label=Discord" alt="Discord"></a>  (needs real GUILD_ID)
-->

<p align="center">
  <a href="https://sie.dev/docs">Docs</a> · <a href="https://sie.dev/docs/quickstart">Quickstart</a> · <a href="https://sie.dev/docs/reference/sdk">API Reference</a> · <a href="https://sie.dev/docs/reference/models">Models</a>
</p>

---

### Quickstart

Or try it in your browser, no install needed: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/superlinked/sie/blob/main/notebooks/quickstart.ipynb)

**1. Start the server**

```bash
pip install sie-server
sie-server serve            # auto-detects CUDA / Apple Silicon / CPU
```

Or with Docker:

```bash
docker run -p 8080:8080 ghcr.io/superlinked/sie:latest             # CPU
docker run --gpus all -p 8080:8080 ghcr.io/superlinked/sie:latest  # GPU
```

**2. Install the SDK and go**

```bash
pip install sie-sdk
```

The entire API is three functions: `encode`, `score`, `extract`.

```python
from sie_sdk import SIEClient
from sie_sdk.types import Item

client = SIEClient("http://localhost:8080")

# Encode: dense embeddings, 400M-parameter model
result = client.encode("NovaSearch/stella_en_400M_v5", Item(text="Hello world"))
print(result["dense"].shape)  # (1024,)

# Score: rerank documents by relevance
scores = client.score(
    "BAAI/bge-reranker-v2-m3",
    Item(text="What is machine learning?"),
    [Item(text="ML learns from data."), Item(text="The weather is sunny.")]
)
print(scores["scores"])
# [{'item_id': 'item-0', 'score': 0.998, 'rank': 0},
#  {'item_id': 'item-1', 'score': 0.012, 'rank': 1}]

# Extract: zero-shot named entity recognition, no training data
result = client.extract(
    "urchade/gliner_multi-v2.1",
    Item(text="Tim Cook is the CEO of Apple."),
    labels=["person", "organization"]
)
print(result["entities"])
# [{'text': 'Tim Cook', 'label': 'person', 'score': 0.96},
#  {'text': 'Apple', 'label': 'organization', 'score': 0.91}]
```

TypeScript: `pnpm add @sie/sdk` - [TypeScript docs ->](https://sie.dev/docs/reference/typescript-sdk)

[Full quickstart guide ->](https://sie.dev/docs/quickstart) · [API reference ->](https://sie.dev/docs/reference/sdk)

---

### Production

The same code works against a production cluster. SIE ships a load-balancing router, KEDA autoscaling (scale to zero), Grafana dashboards, and Terraform modules for GKE and EKS. Not just the server, the whole stack. All Apache 2.0.

```bash
helm install sie deploy/helm/sie-cluster/ -f values.yaml
```

[Deployment guide ->](https://sie.dev/docs/deployment/docker)

---

### Explore

[**85+ models**](https://sie.dev/docs/reference/models) - `Stella v5` · `BGE-M3` · `SPLADE v3` · `SigLIP` · `ColQwen2.5` · `BGE-reranker` · `GLiNER` · `Florence-2` · [and more ->](https://sie.dev/docs/reference/models)
Dense, sparse, multi-vector, vision, rerankers, extractors. All pre-configured. All quality-verified against MTEB in CI.

[**Integrations**](https://sie.dev/docs/integrations/) - LangChain · LlamaIndex · Haystack · DSPy · CrewAI · Chroma

[**Notebooks**](notebooks/) - Quickstarts and walkthroughs *(in progress)*

[**Examples**](examples/) - End-to-end project gallery. [Add yours ->](examples/)

---

<p align="center">
  If SIE saves you time, a ⭐ helps others find it.
</p>

<p align="center">
  <a href="https://sie.dev/docs"><strong>sie.dev/docs</strong></a> · Apache 2.0
</p>
