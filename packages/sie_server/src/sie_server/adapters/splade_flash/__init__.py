"""SPLADE Flash Attention adapter using flash_attn_varlen_func.

This adapter uses Flash Attention 2's variable-length attention to process
sequences without padding, eliminating padding waste and improving throughput.

SPLADE models produce sparse lexical representations by:
1. Running BERT encoder with MLM head
2. Computing weights: log(1 + ReLU(logits))
3. Max-aggregating over tokens per vocabulary term

Supports SPLADE-based models like:
- naver/splade-v3, splade-cocondenser-selfdistil
- opensearch-project/opensearch-neural-sparse-*
- prithivida/Splade_PP_en_v2

Key features:
- Uses flash_attn_varlen_func with cu_seqlens for packed sequences
- No padding tokens = no wasted compute
- Sparse output format: {indices: int32[], values: float32[]}

See: https://github.com/Dao-AILab/flash-attention
"""

from __future__ import annotations

import gc
import logging
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch

from sie_server.adapters.base import ModelAdapter, ModelCapabilities, ModelDims
from sie_server.adapters.peft_lora_mixin import PEFTLoRAMixin
from sie_server.core.inference_output import EncodeOutput, SparseVector
from sie_server.core.preprocessor import CharCountPreprocessor
from sie_server.types.inputs import Item

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

ComputePrecision = Literal["float16", "bfloat16", "float32"]

_ERR_NOT_LOADED = "Model not loaded. Call load() first."
_ERR_REQUIRES_TEXT = "SPLADEFlashAdapter requires text input"
_ERR_CPU_NOT_SUPPORTED = "SPLADEFlashAdapter requires CUDA. Use sentence_transformer adapter for CPU."


class SPLADEFlashAdapter(PEFTLoRAMixin, ModelAdapter):
    """SPLADE adapter using Flash Attention 2 with variable-length sequences.

    This adapter eliminates padding waste by packing sequences and using
    flash_attn_varlen_func. Achieves higher throughput than library-based adapters.

    SPLADE produces sparse lexical representations using masked language modeling:
    - weights = log(1 + ReLU(MLM_logits))
    - max-pool over tokens to get per-term weights
    """

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        max_seq_length: int = 512,
        compute_precision: ComputePrecision = "float16",
        query_template: str | None = None,
        doc_template: str | None = None,
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            max_seq_length: Maximum sequence length.
            compute_precision: Compute precision (float16 recommended for flash).
            query_template: Optional template for queries.
            doc_template: Optional template for documents.
            trust_remote_code: Whether to trust remote code in model files.
            **kwargs: Additional arguments (ignored, for compatibility).
        """
        _ = kwargs
        self._model_name_or_path = str(model_name_or_path)
        self._max_seq_length = max_seq_length
        self._compute_precision = compute_precision
        self._query_template = query_template
        self._doc_template = doc_template
        self._trust_remote_code = trust_remote_code

        self._model: Any = None  # BertForMaskedLM / DistilBertForMaskedLM
        self._tokenizer: PreTrainedTokenizerFast | None = None
        self._device: str | None = None
        self._vocab_size: int | None = None
        self._arch: str | None = None  # "bert", "roberta", or "distilbert"

    @classmethod
    def create_for_device(cls, device: str, **kwargs: Any) -> ModelAdapter:
        """Factory method that returns the appropriate adapter for the device.

        SPLADE requires CUDA and has no CPU fallback. SPLADE is a specialized sparse
        encoding method that requires MLM head, and no generic fallback adapter exists.

        Args:
            device: Device string (e.g., "cuda:0", "mps", "cpu").
            **kwargs: Adapter initialization parameters.

        Returns:
            SPLADEFlashAdapter for CUDA with flash-attn.

        Raises:
            RuntimeError: If device is not CUDA or flash-attn is unavailable.
        """
        if not device.startswith("cuda"):
            msg = (
                f"SPLADEFlashAdapter requires CUDA, got device='{device}'. "
                "SPLADE is a specialized sparse encoding method with no CPU fallback adapter. "
                "Use a CUDA-enabled device for SPLADE models."
            )
            raise RuntimeError(msg)

        try:
            import flash_attn

            return cls(**kwargs)
        except ImportError:
            msg = (
                f"flash-attn not installed for device '{device}'. "
                "SPLADEFlashAdapter requires flash-attn. Install with: uv sync --extra flash-attn"
            )
            raise RuntimeError(msg) from None

    @property
    def capabilities(self) -> ModelCapabilities:
        """Return model capabilities."""
        return ModelCapabilities(
            inputs=["text"],
            outputs=["sparse"],
        )

    @property
    def dims(self) -> ModelDims:
        """Return model dimensions."""
        if self._vocab_size is None:
            raise RuntimeError(_ERR_NOT_LOADED)
        return ModelDims(sparse=self._vocab_size)

    def load(self, device: str) -> None:
        """Load the model onto the specified device.

        Args:
            device: Device string (must be "cuda" or "cuda:X").

        Raises:
            RuntimeError: If device is not CUDA (flash attention requires GPU).
        """
        if not device.startswith("cuda"):
            raise RuntimeError(_ERR_CPU_NOT_SUPPORTED)

        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self._device = device
        dtype = self._resolve_dtype()

        logger.info(
            "Loading %s on device=%s with dtype=%s, attn=flash_varlen",
            self._model_name_or_path,
            device,
            dtype,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name_or_path,
            trust_remote_code=self._trust_remote_code,
        )

        # Load model with eager attention - we'll run our own flash attention
        self._model = AutoModelForMaskedLM.from_pretrained(
            self._model_name_or_path,
            torch_dtype=dtype,
            attn_implementation="eager",  # We handle attention manually
            trust_remote_code=self._trust_remote_code,
        )
        self._model.to(device)
        self._model.eval()

        self._vocab_size = self._model.config.vocab_size

        # Detect model architecture
        if hasattr(self._model, "bert"):
            self._arch = "bert"
        elif hasattr(self._model, "roberta"):
            self._arch = "roberta"
        elif hasattr(self._model, "distilbert"):
            self._arch = "distilbert"
        else:
            msg = f"Unsupported model architecture: {type(self._model).__name__}"
            raise ValueError(msg)

        logger.info(
            "Loaded SPLADE: arch=%s, vocab_size=%d, hidden_size=%d",
            self._arch,
            self._vocab_size,
            self._model.config.hidden_size,
        )

    def _resolve_dtype(self) -> torch.dtype:
        """Resolve compute dtype."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self._compute_precision, torch.float16)

    def unload(self) -> None:
        """Unload the model and free resources."""
        device = self._device

        if self._model is not None:
            del self._model
            self._model = None

        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        self._device = None
        self._vocab_size = None
        self._arch = None

        gc.collect()
        if device and device.startswith("cuda"):
            torch.cuda.empty_cache()

    def encode(
        self,
        items: list[Item],
        output_types: list[str],
        *,
        instruction: str | None = None,
        is_query: bool = False,
        prepared_items: Any = None,
        options: dict[str, Any] | None = None,
    ) -> EncodeOutput:
        """Run inference returning standardized batched output.

        Args:
            items: List of items to encode.
            output_types: Which outputs to compute (only "sparse" supported).
            instruction: Optional instruction prefix.
            is_query: Whether items are queries (affects template selection).
            prepared_items: Not used by this adapter.

        Returns:
            EncodeOutput with sparse embeddings.
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(_ERR_NOT_LOADED)

        self._validate_output_types(output_types)

        # Resolve runtime options (config defaults -> profile -> request overrides)
        opts = options or {}
        query_template = opts.get("query_template", self._query_template)
        doc_template = opts.get("doc_template", self._doc_template)

        texts = self._extract_texts(
            items,
            instruction,
            is_query=is_query,
            query_template=query_template,
            doc_template=doc_template,
        )

        # Tokenize each sequence individually (no padding)
        encodings = [
            self._tokenizer(
                text,
                max_length=self._max_seq_length,
                truncation=True,
                return_tensors="pt",
            )
            for text in texts
        ]

        # Build packed representation
        seq_lengths = [enc["input_ids"].shape[1] for enc in encodings]
        total_tokens = sum(seq_lengths)
        max_seqlen = max(seq_lengths)

        # Pack input_ids
        input_ids_packed = torch.cat([enc["input_ids"].squeeze(0) for enc in encodings]).to(self._device)

        # Build cu_seqlens (cumulative sequence lengths)
        cu_seqlens = torch.zeros(len(texts) + 1, dtype=torch.int32, device=self._device)
        for i, length in enumerate(seq_lengths):
            cu_seqlens[i + 1] = cu_seqlens[i] + length

        with torch.inference_mode():
            # Build BERT-style position IDs (start at 0)
            position_ids_packed = self._build_position_ids(cu_seqlens, len(texts))

            # Run embeddings
            hidden = self._run_embeddings(input_ids_packed, position_ids_packed)

            # Run transformer layers with flash attention
            hidden = self._run_transformer_flash(hidden, cu_seqlens, max_seqlen, total_tokens)

            # Run MLM head to get logits
            logits = self._run_mlm_head(hidden)  # [total_tokens, vocab_size]

            # Compute SPLADE weights: log(1 + ReLU(logits))
            weights = torch.log1p(torch.relu(logits))

            # Max-pool over tokens per sequence to get sparse vectors
            sparse_results = self._aggregate_sparse(weights, input_ids_packed, cu_seqlens, seq_lengths)

        return self._to_inference_output(sparse_results, len(items), is_query)

    def _build_position_ids(self, cu_seqlens: torch.Tensor, num_seqs: int) -> torch.Tensor:
        """Build BERT-style position IDs for packed sequences."""
        pos_list = []
        for i in range(num_seqs):
            seq_len = cu_seqlens[i + 1].item() - cu_seqlens[i].item()
            pos_list.append(torch.arange(0, seq_len, device=self._device))
        return torch.cat(pos_list)

    def _get_base_model(self) -> Any:
        """Get the base transformer model (bert, roberta, or distilbert)."""
        if self._arch == "bert":
            return self._model.bert
        if self._arch == "roberta":
            return self._model.roberta
        # distilbert
        return self._model.distilbert

    def _run_mlm_head(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run the MLM head to get vocabulary logits."""
        if self._arch == "distilbert":
            # DistilBERT MLM: vocab_transform -> activation -> vocab_layer_norm -> vocab_projector
            hidden = self._model.vocab_transform(hidden)
            hidden = self._model.activation(hidden)
            hidden = self._model.vocab_layer_norm(hidden)
            return self._model.vocab_projector(hidden)
        if self._arch == "roberta":
            # RoBERTa: lm_head module
            return self._model.lm_head(hidden)
        # BERT: cls module
        return self._model.cls(hidden)

    def _run_embeddings(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Compute embeddings for packed input."""
        base_model = self._get_base_model()
        embeddings = base_model.embeddings

        word_emb = embeddings.word_embeddings(input_ids)
        pos_emb = embeddings.position_embeddings(position_ids)

        # BERT has token_type_embeddings, DistilBERT doesn't
        if hasattr(embeddings, "token_type_embeddings"):
            token_type_emb = embeddings.token_type_embeddings(torch.zeros_like(input_ids))
            hidden = word_emb + pos_emb + token_type_emb
        else:
            hidden = word_emb + pos_emb

        hidden = embeddings.LayerNorm(hidden)
        return embeddings.dropout(hidden)

    def _run_transformer_flash(
        self,
        hidden: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        total_tokens: int,
    ) -> torch.Tensor:
        """Run transformer layers using flash_attn_varlen_func."""
        from flash_attn import flash_attn_varlen_func

        base_model = self._get_base_model()
        num_heads = self._model.config.num_attention_heads
        hidden_size = self._model.config.hidden_size
        head_dim = hidden_size // num_heads
        softmax_scale = 1.0 / (head_dim**0.5)

        # Get encoder/transformer layers based on architecture
        if self._arch == "distilbert":
            layers = base_model.transformer.layer
        else:
            layers = base_model.encoder.layer

        for layer in layers:
            if self._arch == "distilbert":
                # DistilBERT attention structure: layer.attention (MultiHeadSelfAttention)
                attention = layer.attention
                query = attention.q_lin(hidden).view(total_tokens, num_heads, head_dim)
                key = attention.k_lin(hidden).view(total_tokens, num_heads, head_dim)
                value = attention.v_lin(hidden).view(total_tokens, num_heads, head_dim)
            else:
                # BERT/RoBERTa: layer.attention.self
                attention = layer.attention.self
                query = attention.query(hidden).view(total_tokens, num_heads, head_dim)
                key = attention.key(hidden).view(total_tokens, num_heads, head_dim)
                value = attention.value(hidden).view(total_tokens, num_heads, head_dim)

            # Flash attention with variable-length sequences
            attn_out = flash_attn_varlen_func(
                query,
                key,
                value,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=False,
                softmax_scale=softmax_scale,
            )
            attn_out = attn_out.reshape(total_tokens, hidden_size)

            if self._arch == "distilbert":
                # DistilBERT: out_lin, then sa_layer_norm (post-attention)
                attn_out = attention.out_lin(attn_out)
                attn_out = attention.dropout(attn_out)
                hidden = layer.sa_layer_norm(attn_out + hidden)

                # FFN: ffn (Linear) -> activation -> ffn (Linear)
                inter = layer.ffn.lin1(hidden)
                inter = layer.ffn.activation(inter)
                inter = layer.ffn.dropout(inter)
                out = layer.ffn.lin2(inter)
                out = layer.ffn.dropout(out)
                hidden = layer.output_layer_norm(out + hidden)
            else:
                # BERT/RoBERTa structure
                attn_out = layer.attention.output.dense(attn_out)
                attn_out = layer.attention.output.dropout(attn_out)
                hidden = layer.attention.output.LayerNorm(attn_out + hidden)

                # FFN
                inter = layer.intermediate.dense(hidden)
                inter = layer.intermediate.intermediate_act_fn(inter)
                out = layer.output.dense(inter)
                out = layer.output.dropout(out)
                hidden = layer.output.LayerNorm(out + hidden)

        return hidden

    def _aggregate_sparse(
        self,
        weights: torch.Tensor,
        input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        seq_lengths: list[int],
    ) -> list[dict[int, float]]:
        """Aggregate token weights to sparse vectors via max-pooling.

        For each sequence, max-pool weights over positions for each vocab term.
        Exclude special tokens (CLS, SEP, PAD).

        Args:
            weights: SPLADE weights [total_tokens, vocab_size].
            input_ids: Packed input token IDs.
            cu_seqlens: Cumulative sequence lengths.
            seq_lengths: Length of each sequence.

        Returns:
            List of sparse dicts mapping token_id -> weight.
        """
        # Special tokens to exclude
        special_tokens = {
            self._tokenizer.cls_token_id,
            self._tokenizer.sep_token_id,
            self._tokenizer.pad_token_id,
            self._tokenizer.unk_token_id,
        }
        special_tokens.discard(None)  # Remove None if any token ID is not set

        results = []
        num_seqs = len(seq_lengths)

        for i in range(num_seqs):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()

            # Get weights and input_ids for this sequence
            seq_weights = weights[start:end]  # [seq_len, vocab_size]
            input_ids[start:end]  # [seq_len]

            # Max-pool over positions for each vocab term
            # For efficiency, use scatter_reduce
            max_weights, _ = seq_weights.max(dim=0)  # [vocab_size]

            # Get non-zero indices and values
            nonzero_mask = max_weights > 0
            indices = torch.where(nonzero_mask)[0]
            values = max_weights[nonzero_mask]

            # Filter out special tokens and build sparse dict
            sparse_dict: dict[int, float] = {}
            for idx, val in zip(indices.cpu().numpy(), values.cpu().numpy(), strict=True):
                if idx not in special_tokens:
                    sparse_dict[int(idx)] = float(val)

            results.append(sparse_dict)

        return results

    def _validate_output_types(self, output_types: list[str]) -> None:
        """Validate that output types are supported."""
        unsupported = set(output_types) - {"sparse"}
        if unsupported:
            msg = f"Unsupported output types: {unsupported}. SPLADEFlashAdapter only supports 'sparse'."
            raise ValueError(msg)

    def _extract_texts(
        self,
        items: list[Item],
        instruction: str | None,
        *,
        is_query: bool,
        query_template: str | None = None,
        doc_template: str | None = None,
    ) -> list[str]:
        """Extract texts from items, applying templates if configured."""
        query_template = query_template if query_template is not None else self._query_template
        doc_template = doc_template if doc_template is not None else self._doc_template
        texts = []
        for item in items:
            if item.get("text") is None:
                raise ValueError(_ERR_REQUIRES_TEXT)

            text = item["text"]

            # Apply template based on query/document mode
            template = query_template if is_query else doc_template
            if template:
                text = template.format(text=text, instruction=instruction or "")
            elif instruction:
                text = f"{instruction} {text}"

            texts.append(text)
        return texts

    def _to_inference_output(
        self,
        sparse_results: list[dict[int, float]],
        batch_size: int,
        is_query: bool,
    ) -> EncodeOutput:
        """Convert sparse dicts to EncodeOutput."""
        sparse_list = []
        for sparse_dict in sparse_results:
            if sparse_dict:
                indices = np.array(list(sparse_dict.keys()), dtype=np.int32)
                values = np.array(list(sparse_dict.values()), dtype=np.float32)
            else:
                indices = np.array([], dtype=np.int32)
                values = np.array([], dtype=np.float32)
            sparse_list.append(SparseVector(indices=indices, values=values))
        return EncodeOutput(sparse=sparse_list, batch_size=batch_size, is_query=is_query)

    def get_preprocessor(self) -> CharCountPreprocessor:
        """Return CharCountPreprocessor for cost estimation without tokenization overhead."""
        return CharCountPreprocessor(model_name=self._model_name_or_path)
