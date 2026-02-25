"""GTE Sparse adapter for NewForMaskedLM architecture with Flash Attention.

This adapter supports sparse embedding models based on the NewForMaskedLM architecture
from Alibaba-NLP, such as opensearch-neural-sparse-encoding-doc-v3-gte.

Key architecture details:
- POST-norm architecture (LayerNorm AFTER residual connection)
- Rotary Position Embeddings (RoPE) applied in attention layers
- Fused QKV projection via attention.qkv_proj
- Uses flash_attn_varlen_func for efficient batched inference on GPU

Produces SPLADE-style sparse vectors:
- weights = log(1 + ReLU(MLM_logits))
- max-pool over tokens to get per-term weights
"""

from __future__ import annotations

import gc
import logging
from typing import TYPE_CHECKING, Any, Literal, cast

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
_ERR_REQUIRES_TEXT = "GTESparseFlashAdapter requires text input"
_ERR_WRONG_ARCH = "GTESparseFlashAdapter requires NewForMaskedLM architecture (model.new attribute)"


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Position Embedding to query and key tensors.

    Args:
        q: Query tensor [total_tokens, num_heads, head_dim].
        k: Key tensor [total_tokens, num_heads, head_dim].
        cos: Cosine part [total_tokens, head_dim].
        sin: Sine part [total_tokens, head_dim].

    Returns:
        Rotated query and key tensors.
    """
    cos = cos.unsqueeze(1).to(q.dtype)
    sin = sin.unsqueeze(1).to(q.dtype)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GTESparseFlashAdapter(PEFTLoRAMixin, ModelAdapter):
    """GTE sparse flash adapter for NewForMaskedLM architecture.

    This adapter uses Flash Attention 2's variable-length attention for efficient
    batched inference without padding waste on GPU. Falls back to native forward
    on CPU.

    Architecture: POST-norm (LayerNorm after residual, not before sublayer).

    Produces SPLADE-style sparse lexical representations using masked language modeling.

    Supports LoRA adapters via PEFTLoRAMixin.
    """

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        max_seq_length: int = 512,
        compute_precision: ComputePrecision = "float16",
        query_template: str | None = None,
        doc_template: str | None = None,
        trust_remote_code: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            max_seq_length: Maximum sequence length.
            compute_precision: Compute precision (float16/bfloat16/float32).
            query_template: Optional template for queries.
            doc_template: Optional template for documents.
            trust_remote_code: Whether to trust remote code (required for NewForMaskedLM).
            **kwargs: Additional arguments (ignored, for compatibility).
        """
        _ = kwargs
        self._model_name_or_path = str(model_name_or_path)
        self._max_seq_length = max_seq_length
        self._compute_precision = compute_precision
        self._query_template = query_template
        self._doc_template = doc_template
        self._trust_remote_code = trust_remote_code

        self._model: Any = None
        self._tokenizer: PreTrainedTokenizerFast | None = None
        self._device: str | None = None
        self._vocab_size: int | None = None
        self._num_heads: int | None = None
        self._head_dim: int | None = None
        self._hidden_size: int | None = None
        self._use_flash: bool = False

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
            device: Device string ("cuda", "cuda:X", or "cpu").

        Raises:
            ValueError: If model is not NewForMaskedLM architecture.
        """
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self._device = device
        self._use_flash = device.startswith("cuda")
        dtype = self._resolve_dtype()

        attn_mode = "flash_varlen" if self._use_flash else "native"
        logger.info(
            "Loading %s on device=%s with dtype=%s, attn=%s (GTE sparse)",
            self._model_name_or_path,
            device,
            dtype,
            attn_mode,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name_or_path,
            trust_remote_code=self._trust_remote_code,
        )

        self._model = AutoModelForMaskedLM.from_pretrained(
            self._model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=self._trust_remote_code,
        )
        self._model.to(device)
        self._model.eval()

        # Verify this is NewForMaskedLM architecture
        if not hasattr(self._model, "new"):
            raise ValueError(_ERR_WRONG_ARCH)

        self._vocab_size = self._model.config.vocab_size
        self._num_heads = cast("int", self._model.config.num_attention_heads)
        self._hidden_size = cast("int", self._model.config.hidden_size)
        self._head_dim = self._hidden_size // self._num_heads

        logger.info(
            "Loaded GTE sparse: vocab_size=%d, hidden_size=%d, num_heads=%d, head_dim=%d",
            self._vocab_size,
            self._hidden_size,
            self._num_heads,
            self._head_dim,
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
        self._num_heads = None
        self._head_dim = None
        self._hidden_size = None
        self._use_flash = False

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

        Uses flash_attn_varlen_func on GPU for efficient batched processing.
        Falls back to native forward on CPU.

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

        if self._use_flash:
            return self._encode_flash(texts, is_query)
        return self._encode_native(texts, is_query)

    def _encode_native(self, texts: list[str], is_query: bool) -> EncodeOutput:
        """Encode using native forward pass (for CPU or fallback)."""
        if self._tokenizer is None:
            raise RuntimeError(_ERR_NOT_LOADED)
        inputs = self._tokenizer(
            texts,
            max_length=self._max_seq_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self._model(**inputs)
            logits = outputs.logits
            weights = torch.log1p(torch.relu(logits))
            attention_mask = inputs.get("attention_mask")
            sparse_results = self._aggregate_sparse_padded(weights, attention_mask)

        return self._to_inference_output(sparse_results, len(texts), is_query)

    def _encode_flash(self, texts: list[str], is_query: bool) -> EncodeOutput:
        """Encode using flash attention with packed sequences."""
        from flash_attn import flash_attn_varlen_func  # ty: ignore[unresolved-import]

        if self._tokenizer is None:
            raise RuntimeError(_ERR_NOT_LOADED)

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

        seq_lengths = [enc["input_ids"].shape[1] for enc in encodings]
        total_tokens = sum(seq_lengths)
        max_seqlen = max(seq_lengths)

        # Pack input_ids
        input_ids_packed = torch.cat([enc["input_ids"].squeeze(0) for enc in encodings]).to(self._device)

        # Build cu_seqlens
        cu_seqlens = torch.zeros(len(texts) + 1, dtype=torch.int32, device=self._device)
        for i, length in enumerate(seq_lengths):
            cu_seqlens[i + 1] = cu_seqlens[i] + length

        with torch.inference_mode():
            # Build position IDs for RoPE
            position_ids = self._build_position_ids(cu_seqlens, len(texts))

            # Compute RoPE cos/sin
            cos, sin = self._compute_rope(position_ids, max_seqlen)

            # Run embeddings
            hidden = self._run_embeddings(input_ids_packed)

            # Run transformer with flash attention (POST-norm architecture)
            hidden = self._run_transformer_flash(
                hidden, cu_seqlens, max_seqlen, total_tokens, cos, sin, flash_attn_varlen_func
            )

            # Run MLM head
            logits = self._model.lm_head(hidden)

            # SPLADE weights
            weights = torch.log1p(torch.relu(logits))

            # Aggregate sparse vectors
            sparse_results = self._aggregate_sparse_packed(weights, cu_seqlens, seq_lengths)

        return self._to_inference_output(sparse_results, len(texts), is_query)

    def _build_position_ids(self, cu_seqlens: torch.Tensor, num_seqs: int) -> torch.Tensor:
        """Build position IDs for packed sequences (each starts from 0)."""
        pos_list = []
        for i in range(num_seqs):
            seq_len = cu_seqlens[i + 1].item() - cu_seqlens[i].item()
            pos_list.append(torch.arange(0, seq_len, device=self._device))
        return torch.cat(pos_list)

    def _compute_rope(
        self,
        position_ids: torch.Tensor,
        max_seqlen: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute RoPE cos/sin values for packed positions."""
        rotary_emb = self._model.new.embeddings.rotary_emb
        dtype = self._resolve_dtype()

        # Ensure cache is large enough
        dummy_x = torch.zeros(1, max_seqlen, 1, device=self._device, dtype=dtype)
        _ = rotary_emb(dummy_x, seq_len=max_seqlen)

        # Index into cached values
        cos = rotary_emb.cos_cached[position_ids].to(dtype)
        sin = rotary_emb.sin_cached[position_ids].to(dtype)

        return cos, sin

    def _run_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get token embeddings for packed input."""
        embeddings = self._model.new.embeddings
        hidden = embeddings.word_embeddings(input_ids)
        hidden = embeddings.LayerNorm(hidden)
        hidden = embeddings.dropout(hidden)
        return hidden

    def _run_transformer_flash(
        self,
        hidden: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        total_tokens: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        flash_attn_varlen_func: Any,
    ) -> torch.Tensor:
        """Run transformer layers with flash attention.

        NewForMaskedLM uses POST-norm architecture:
        - hidden = hidden + attention(hidden)
        - hidden = attn_ln(hidden)  # POST-norm
        - hidden = hidden + mlp(hidden)
        - hidden = mlp_ln(hidden)  # POST-norm
        """
        if self._head_dim is None:
            raise RuntimeError(_ERR_NOT_LOADED)
        softmax_scale = 1.0 / (self._head_dim**0.5)

        for layer in self._model.new.encoder.layer:
            attn = layer.attention

            # QKV projection (no pre-norm in this architecture)
            qkv = attn.qkv_proj(hidden)
            qkv = qkv.view(total_tokens, 3, self._num_heads, self._head_dim)
            query = qkv[:, 0]
            key = qkv[:, 1]
            value = qkv[:, 2]

            # Apply RoPE
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

            # Flash attention
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
            attn_out = attn_out.reshape(total_tokens, self._hidden_size)

            # Output projection + dropout
            attn_out = attn.o_proj(attn_out)
            attn_out = layer.hidden_dropout(attn_out)

            # Residual + POST-norm
            hidden = hidden + attn_out
            hidden = layer.attn_ln(hidden)

            # MLP (no pre-norm)
            mlp_out = layer.mlp(hidden)
            mlp_out = layer.hidden_dropout(mlp_out)

            # Residual + POST-norm
            hidden = hidden + mlp_out
            hidden = layer.mlp_ln(hidden)

        return hidden

    def _aggregate_sparse_padded(
        self,
        weights: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> list[dict[int, float]]:
        """Aggregate sparse vectors from padded input."""
        batch_size = weights.shape[0]
        special_tokens = self._get_special_tokens()
        results = []

        for i in range(batch_size):
            seq_weights = weights[i]
            if attention_mask is not None:
                mask = attention_mask[i].unsqueeze(-1)
                seq_weights = seq_weights * mask

            max_weights, _ = seq_weights.max(dim=0)
            nonzero_mask = max_weights > 0
            indices = torch.where(nonzero_mask)[0]
            values = max_weights[nonzero_mask]

            sparse_dict: dict[int, float] = {}
            for idx, val in zip(indices.cpu().numpy(), values.cpu().numpy(), strict=True):
                if idx not in special_tokens:
                    sparse_dict[int(idx)] = float(val)
            results.append(sparse_dict)

        return results

    def _aggregate_sparse_packed(
        self,
        weights: torch.Tensor,
        cu_seqlens: torch.Tensor,
        seq_lengths: list[int],
    ) -> list[dict[int, float]]:
        """Aggregate sparse vectors from packed input."""
        special_tokens = self._get_special_tokens()
        results = []

        for i in range(len(seq_lengths)):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            seq_weights = weights[start:end]

            max_weights, _ = seq_weights.max(dim=0)
            nonzero_mask = max_weights > 0
            indices = torch.where(nonzero_mask)[0]
            values = max_weights[nonzero_mask]

            sparse_dict: dict[int, float] = {}
            for idx, val in zip(indices.cpu().numpy(), values.cpu().numpy(), strict=True):
                if idx not in special_tokens:
                    sparse_dict[int(idx)] = float(val)
            results.append(sparse_dict)

        return results

    def _get_special_tokens(self) -> set[int]:
        """Get set of special token IDs to exclude from sparse output."""
        if self._tokenizer is None:
            raise RuntimeError(_ERR_NOT_LOADED)
        special_tokens = {
            self._tokenizer.cls_token_id,
            self._tokenizer.sep_token_id,
            self._tokenizer.pad_token_id,
            self._tokenizer.unk_token_id,
        }
        special_tokens.discard(None)
        return special_tokens

    def _validate_output_types(self, output_types: list[str]) -> None:
        """Validate that output types are supported."""
        unsupported = set(output_types) - {"sparse"}
        if unsupported:
            msg = f"Unsupported output types: {unsupported}. GTESparseFlashAdapter only supports 'sparse'."
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
