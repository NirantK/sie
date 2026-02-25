"""Tests for model adapters."""

from __future__ import annotations

from typing import ClassVar
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sie_server.adapters.base import ModelAdapter, ModelCapabilities, ModelDims
from sie_server.adapters.bge_m3 import BGEM3Adapter
from sie_server.adapters.bge_m3_flag import BGEM3FlagAdapter
from sie_server.adapters.clip import CLIPAdapter
from sie_server.adapters.colbert import ColBERTAdapter
from sie_server.adapters.gte_sparse_flash import GTESparseFlashAdapter
from sie_server.adapters.sentence_transformer import (
    SentenceTransformerDenseAdapter,
    SentenceTransformerSparseAdapter,
)
from sie_server.adapters.siglip import SiglipAdapter
from sie_server.types.inputs import Item

# Create a random generator for tests
_RNG = np.random.default_rng(42)


class TestModelCapabilities:
    """Tests for ModelCapabilities."""

    def test_valid_capabilities(self) -> None:
        """Can create capabilities with valid inputs."""
        caps = ModelCapabilities(
            inputs=["text", "image"],
            outputs=["dense", "sparse"],
        )
        assert caps.inputs == ["text", "image"]
        assert caps.outputs == ["dense", "sparse"]


class TestModelDims:
    """Tests for ModelDims."""

    def test_valid_dims(self) -> None:
        """Can create dims with valid values."""
        dims = ModelDims(dense=1024, sparse=30522, multivector=128)
        assert dims.dense == 1024
        assert dims.sparse == 30522
        assert dims.multivector == 128

    def test_optional_dims(self) -> None:
        """Dims can be None."""
        dims = ModelDims(dense=768)
        assert dims.dense == 768
        assert dims.sparse is None
        assert dims.multivector is None


class TestSentenceTransformerDenseAdapter:
    """Tests for SentenceTransformerDenseAdapter with mocked model."""

    @pytest.fixture
    def mock_st_model(self) -> MagicMock:
        """Create a mock SentenceTransformer model."""
        mock = MagicMock()
        mock.get_sentence_embedding_dimension.return_value = 384

        # Return correct batch size based on input
        def mock_encode(texts, **kwargs):
            return _RNG.standard_normal((len(texts), 384)).astype(np.float32)

        mock.encode.side_effect = mock_encode
        return mock

    @pytest.fixture
    def adapter(self) -> SentenceTransformerDenseAdapter:
        """Create an adapter instance."""
        return SentenceTransformerDenseAdapter(
            "test-model",
            normalize=True,
            max_seq_length=512,
        )

    def test_capabilities(self, adapter: SentenceTransformerDenseAdapter) -> None:
        """Adapter reports correct capabilities."""
        caps = adapter.capabilities
        assert caps.inputs == ["text"]
        assert caps.outputs == ["dense"]

    def test_dims_before_load_raises(self, adapter: SentenceTransformerDenseAdapter) -> None:
        """Accessing dims before load raises error."""
        with pytest.raises(RuntimeError, match="Model not loaded"):
            _ = adapter.dims

    @patch("sie_server.adapters.sentence_transformer.SentenceTransformer")
    def test_load(
        self,
        mock_st_class: MagicMock,
        adapter: SentenceTransformerDenseAdapter,
        mock_st_model: MagicMock,
    ) -> None:
        """Load initializes the model."""
        mock_st_class.return_value = mock_st_model

        adapter.load("cpu")

        mock_st_class.assert_called_once_with(
            "test-model",
            device="cpu",
            trust_remote_code=False,
            config_kwargs=None,
        )
        assert adapter.dims.dense == 384

    @patch("sie_server.adapters.sentence_transformer.SentenceTransformer")
    def test_encode(
        self,
        mock_st_class: MagicMock,
        adapter: SentenceTransformerDenseAdapter,
        mock_st_model: MagicMock,
    ) -> None:
        """Encode returns dense embeddings."""
        mock_st_class.return_value = mock_st_model
        adapter.load("cpu")

        items = [Item(text="hello"), Item(text="world")]
        output = adapter.encode(items, output_types=["dense"])

        assert output.batch_size == 2
        assert output.dense is not None
        assert output.dense[0].shape == (384,)

        # First call is warmup during load, second is actual encode
        assert mock_st_model.encode.call_count == 2
        call_args = mock_st_model.encode.call_args
        assert call_args[0][0] == ["hello", "world"]

    @patch("sie_server.adapters.sentence_transformer.SentenceTransformer")
    def test_encode_with_instruction(
        self,
        mock_st_class: MagicMock,
        adapter: SentenceTransformerDenseAdapter,
        mock_st_model: MagicMock,
    ) -> None:
        """Encode prepends instruction to text."""
        mock_st_class.return_value = mock_st_model
        adapter.load("cpu")

        items = [Item(text="query")]
        adapter.encode(items, output_types=["dense"], instruction="search:")

        call_args = mock_st_model.encode.call_args
        assert call_args[0][0] == ["search: query"]

    @patch("sie_server.adapters.sentence_transformer.SentenceTransformer")
    def test_encode_unsupported_output_type(
        self,
        mock_st_class: MagicMock,
        adapter: SentenceTransformerDenseAdapter,
        mock_st_model: MagicMock,
    ) -> None:
        """Encode raises for unsupported output types."""
        mock_st_class.return_value = mock_st_model
        adapter.load("cpu")

        items = [Item(text="hello")]
        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode(items, output_types=["sparse"])

    @patch("sie_server.adapters.sentence_transformer.SentenceTransformer")
    def test_encode_without_text_raises(
        self,
        mock_st_class: MagicMock,
        adapter: SentenceTransformerDenseAdapter,
        mock_st_model: MagicMock,
    ) -> None:
        """Encode raises if item has no text."""
        mock_st_class.return_value = mock_st_model
        adapter.load("cpu")

        items = [Item()]  # No text
        with pytest.raises(ValueError, match="require text input"):
            adapter.encode(items, output_types=["dense"])

    def test_encode_before_load_raises(self, adapter: SentenceTransformerDenseAdapter) -> None:
        """Encode before load raises error."""
        items = [Item(text="hello")]
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.encode(items, output_types=["dense"])

    @patch("gc.collect")
    @patch("sie_server.adapters.sentence_transformer.SentenceTransformer")
    @patch("sie_server.adapters.sentence_transformer.torch")
    def test_unload(
        self,
        mock_torch: MagicMock,
        mock_st_class: MagicMock,
        mock_gc: MagicMock,
        adapter: SentenceTransformerDenseAdapter,
        mock_st_model: MagicMock,
    ) -> None:
        """Unload clears the model."""
        mock_st_class.return_value = mock_st_model

        adapter.load("cpu")
        adapter.unload()

        with pytest.raises(RuntimeError, match="Model not loaded"):
            _ = adapter.dims


class TestSentenceTransformerSparseAdapter:
    """Tests for SentenceTransformerSparseAdapter with mocked model."""

    @pytest.fixture
    def mock_sparse_model(self) -> MagicMock:
        """Create a mock SparseEncoder model."""
        import torch

        mock = MagicMock()
        mock.get_sentence_embedding_dimension.return_value = 30522

        # Create sparse COO tensor output
        # 2 rows, vocab_size columns, few non-zero values
        # indices: [row_indices, col_indices], values: weights
        row_indices = torch.tensor([0, 0, 0, 1, 1])
        col_indices = torch.tensor([100, 500, 1000, 200, 800])
        indices = torch.stack([row_indices, col_indices])
        values = torch.tensor([0.5, 0.3, 0.8, 0.4, 0.6], dtype=torch.float32)
        sparse_result = torch.sparse_coo_tensor(indices, values, size=(2, 30522))

        mock.encode_query.return_value = sparse_result
        mock.encode_document.return_value = sparse_result
        return mock

    @pytest.fixture
    def adapter(self) -> SentenceTransformerSparseAdapter:
        """Create an adapter instance."""
        return SentenceTransformerSparseAdapter("test-sparse-model")

    def test_capabilities(self, adapter: SentenceTransformerSparseAdapter) -> None:
        """Adapter reports correct capabilities."""
        caps = adapter.capabilities
        assert caps.inputs == ["text"]
        assert caps.outputs == ["sparse"]

    def test_dims_before_load_raises(self, adapter: SentenceTransformerSparseAdapter) -> None:
        """Accessing dims before load raises error."""
        with pytest.raises(RuntimeError, match="Model not loaded"):
            _ = adapter.dims

    @patch("sie_server.adapters.sentence_transformer.SparseEncoder")
    def test_load(
        self,
        mock_sparse_class: MagicMock,
        adapter: SentenceTransformerSparseAdapter,
        mock_sparse_model: MagicMock,
    ) -> None:
        """Load initializes sparse model."""
        mock_sparse_class.return_value = mock_sparse_model

        adapter.load("cpu")

        mock_sparse_class.assert_called_once()
        assert adapter.dims.sparse == 30522
        assert adapter.dims.dense is None

    @patch("sie_server.adapters.sentence_transformer.SparseEncoder")
    def test_encode_document(
        self,
        mock_sparse_class: MagicMock,
        adapter: SentenceTransformerSparseAdapter,
        mock_sparse_model: MagicMock,
    ) -> None:
        """Encode returns sparse embeddings for documents."""
        mock_sparse_class.return_value = mock_sparse_model
        adapter.load("cpu")

        items = [Item(text="hello"), Item(text="world")]
        output = adapter.encode(items, output_types=["sparse"], is_query=False)

        assert output.batch_size == 2
        assert output.sparse is not None
        assert len(output.sparse) == 2
        # SparseVector has indices and values attributes
        assert hasattr(output.sparse[0], "indices")
        assert hasattr(output.sparse[0], "values")

        mock_sparse_model.encode_document.assert_called_once()

    @patch("sie_server.adapters.sentence_transformer.SparseEncoder")
    def test_encode_query(
        self,
        mock_sparse_class: MagicMock,
        adapter: SentenceTransformerSparseAdapter,
        mock_sparse_model: MagicMock,
    ) -> None:
        """Encode uses encode_query for queries."""
        mock_sparse_class.return_value = mock_sparse_model
        adapter.load("cpu")

        items = [Item(text="query")]
        adapter.encode(items, output_types=["sparse"], is_query=True)

        mock_sparse_model.encode_query.assert_called_once()

    @patch("sie_server.adapters.sentence_transformer.SparseEncoder")
    def test_rejects_dense_output(
        self,
        mock_sparse_class: MagicMock,
        adapter: SentenceTransformerSparseAdapter,
        mock_sparse_model: MagicMock,
    ) -> None:
        """Sparse model rejects dense output type."""
        mock_sparse_class.return_value = mock_sparse_model
        adapter.load("cpu")

        items = [Item(text="hello")]
        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode(items, output_types=["dense"])

    @patch("sie_server.adapters.sentence_transformer.SparseEncoder")
    def test_encode_without_text_raises(
        self,
        mock_sparse_class: MagicMock,
        adapter: SentenceTransformerSparseAdapter,
        mock_sparse_model: MagicMock,
    ) -> None:
        """Encode raises if item has no text."""
        mock_sparse_class.return_value = mock_sparse_model
        adapter.load("cpu")

        items = [Item()]  # No text
        with pytest.raises(ValueError, match="require text input"):
            adapter.encode(items, output_types=["sparse"])

    def test_encode_before_load_raises(self, adapter: SentenceTransformerSparseAdapter) -> None:
        """Encode before load raises error."""
        items = [Item(text="hello")]
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.encode(items, output_types=["sparse"])

    @patch("gc.collect")
    @patch("sie_server.adapters.sentence_transformer.SparseEncoder")
    @patch("sie_server.adapters.sentence_transformer.torch")
    def test_unload(
        self,
        mock_torch: MagicMock,
        mock_sparse_class: MagicMock,
        mock_gc: MagicMock,
        adapter: SentenceTransformerSparseAdapter,
        mock_sparse_model: MagicMock,
    ) -> None:
        """Unload clears the model."""
        mock_sparse_class.return_value = mock_sparse_model

        adapter.load("cpu")
        adapter.unload()

        with pytest.raises(RuntimeError, match="Model not loaded"):
            _ = adapter.dims


class TestBGEM3FlagAdapter:
    """Tests for BGEM3FlagAdapter with mocked model."""

    @pytest.fixture
    def mock_bgem3_model(self) -> MagicMock:
        """Create a mock BGEM3FlagModel."""
        mock = MagicMock()
        mock.encode.return_value = {
            "dense_vecs": _RNG.standard_normal((2, 1024)).astype(np.float32),
            "lexical_weights": [
                {1: 0.5, 100: 0.3, 500: 0.8},
                {2: 0.4, 200: 0.6},
            ],
            "colbert_vecs": [
                _RNG.standard_normal((10, 1024)).astype(np.float32),
                _RNG.standard_normal((8, 1024)).astype(np.float32),
            ],
        }
        return mock

    @pytest.fixture
    def adapter(self) -> BGEM3FlagAdapter:
        """Create an adapter instance."""
        return BGEM3FlagAdapter(
            "BAAI/bge-m3",
            normalize=True,
            max_seq_length=8192,
        )

    def test_capabilities(self, adapter: BGEM3FlagAdapter) -> None:
        """Adapter reports correct capabilities."""
        caps = adapter.capabilities
        assert caps.inputs == ["text"]
        assert caps.outputs == ["dense", "sparse", "multivector"]

    def test_dims(self, adapter: BGEM3FlagAdapter) -> None:
        """Adapter reports correct dimensions."""
        dims = adapter.dims
        assert dims.dense == 1024
        assert dims.sparse == 250002
        assert dims.multivector == 1024

    @patch("FlagEmbedding.BGEM3FlagModel")
    def test_load(
        self,
        mock_bgem3_class: MagicMock,
        adapter: BGEM3FlagAdapter,
    ) -> None:
        """Load initializes the model with resolved path."""
        adapter.load("cuda:0")

        # Model should be loaded (path may be cached or downloaded)
        mock_bgem3_class.assert_called_once()
        call_kwargs = mock_bgem3_class.call_args
        # Should use fp16 on CUDA
        assert call_kwargs.kwargs["use_fp16"] is True
        assert call_kwargs.kwargs["device"] == "cuda:0"
        # First positional arg is the model path (local or HF ID)
        model_path = call_kwargs.args[0]
        assert isinstance(model_path, str)
        assert len(model_path) > 0

    @patch("FlagEmbedding.BGEM3FlagModel")
    def test_encode_dense_only(
        self,
        mock_bgem3_class: MagicMock,
        adapter: BGEM3FlagAdapter,
        mock_bgem3_model: MagicMock,
    ) -> None:
        """Encode can return dense embeddings only."""
        mock_bgem3_class.return_value = mock_bgem3_model
        adapter.load("cuda:0")

        items = [Item(text="hello"), Item(text="world")]
        output = adapter.encode(items, output_types=["dense"])

        assert output.batch_size == 2
        assert output.dense is not None
        assert output.dense[0].shape == (1024,)
        # Should not have sparse or multivector
        assert output.sparse is None
        assert output.multivector is None

    @patch("FlagEmbedding.BGEM3FlagModel")
    def test_encode_all_outputs(
        self,
        mock_bgem3_class: MagicMock,
        adapter: BGEM3FlagAdapter,
        mock_bgem3_model: MagicMock,
    ) -> None:
        """Encode can return all output types."""
        mock_bgem3_class.return_value = mock_bgem3_model
        adapter.load("cuda:0")

        items = [Item(text="hello"), Item(text="world")]
        output = adapter.encode(items, output_types=["dense", "sparse", "multivector"])

        assert output.batch_size == 2

        # Check dense
        assert output.dense is not None
        assert output.dense[0].shape == (1024,)

        # Check sparse
        assert output.sparse is not None
        assert len(output.sparse) == 2
        assert len(output.sparse[0].indices) == 3  # 3 tokens in mock

        # Check multivector
        assert output.multivector is not None
        assert output.multivector[0].shape == (10, 1024)

    @patch("FlagEmbedding.BGEM3FlagModel")
    def test_encode_sparse_only(
        self,
        mock_bgem3_class: MagicMock,
        adapter: BGEM3FlagAdapter,
    ) -> None:
        """Encode can return sparse embeddings only."""
        # Create mock with single item outputs
        mock = MagicMock()
        mock.encode.return_value = {
            "dense_vecs": _RNG.standard_normal((1, 1024)).astype(np.float32),
            "lexical_weights": [{1: 0.5, 100: 0.3}],
            "colbert_vecs": [_RNG.standard_normal((5, 1024)).astype(np.float32)],
        }
        mock_bgem3_class.return_value = mock
        adapter.load("cuda:0")

        items = [Item(text="hello")]
        output = adapter.encode(items, output_types=["sparse"])

        assert output.batch_size == 1
        assert output.sparse is not None
        assert output.dense is None

    def test_encode_before_load_raises(self, adapter: BGEM3FlagAdapter) -> None:
        """Encode before load raises error."""
        items = [Item(text="hello")]
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.encode(items, output_types=["dense"])

    @patch("FlagEmbedding.BGEM3FlagModel")
    def test_encode_without_text_raises(
        self,
        mock_bgem3_class: MagicMock,
        adapter: BGEM3FlagAdapter,
        mock_bgem3_model: MagicMock,
    ) -> None:
        """Encode raises if item has no text."""
        mock_bgem3_class.return_value = mock_bgem3_model
        adapter.load("cuda:0")

        items = [Item()]  # No text
        with pytest.raises(ValueError, match="requires text input"):
            adapter.encode(items, output_types=["dense"])

    @patch("gc.collect")
    @patch("FlagEmbedding.BGEM3FlagModel")
    @patch("sie_server.adapters.bge_m3_flag.torch")
    def test_unload(
        self,
        mock_torch: MagicMock,
        mock_bgem3_class: MagicMock,
        mock_gc: MagicMock,
        adapter: BGEM3FlagAdapter,
        mock_bgem3_model: MagicMock,
    ) -> None:
        """Unload clears the model."""
        mock_bgem3_class.return_value = mock_bgem3_model

        adapter.load("cuda:0")
        adapter.unload()

        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.encode([Item(text="test")], output_types=["dense"])


class TestBGEM3Adapter:
    """Tests for native BGEM3Adapter with mocked transformers."""

    @pytest.fixture
    def adapter(self) -> BGEM3Adapter:
        """Create an adapter instance."""
        return BGEM3Adapter(
            "BAAI/bge-m3",
            normalize=True,
            max_seq_length=8192,
        )

    def test_capabilities(self, adapter: BGEM3Adapter) -> None:
        """Adapter reports correct capabilities."""
        caps = adapter.capabilities
        assert caps.inputs == ["text"]
        assert caps.outputs == ["dense", "sparse", "multivector"]

    def test_dims(self, adapter: BGEM3Adapter) -> None:
        """Adapter reports correct dimensions."""
        dims = adapter.dims
        assert dims.dense == 1024
        assert dims.sparse == 250002
        assert dims.multivector == 1024

    def test_encode_before_load_raises(self, adapter: BGEM3Adapter) -> None:
        """Encode before load raises error."""
        items = [Item(text="hello")]
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.encode(items, output_types=["dense"])


class TestModelAdapterABC:
    """Tests for ModelAdapter abstract base class."""

    def test_cannot_instantiate_directly(self) -> None:
        """Cannot instantiate ModelAdapter directly."""
        with pytest.raises(TypeError, match="abstract"):
            ModelAdapter()  # type: ignore[abstract]

    def test_default_methods_raise(self) -> None:
        """Default encode/score/extract raise NotImplementedError."""

        class MinimalAdapter(ModelAdapter):
            @property
            def capabilities(self) -> ModelCapabilities:
                return ModelCapabilities(inputs=["text"], outputs=["dense"])

            @property
            def dims(self) -> ModelDims:
                return ModelDims(dense=768)

            def load(self, device: str) -> None:
                pass

            def unload(self) -> None:
                pass

        adapter = MinimalAdapter()

        with pytest.raises(NotImplementedError, match="does not implement encode"):
            adapter.encode([Item(text="test")], output_types=["dense"])

        with pytest.raises(NotImplementedError, match="does not support score"):
            adapter.score(Item(text="query"), [Item(text="doc")])

        with pytest.raises(NotImplementedError, match="does not support extract"):
            adapter.extract([Item(text="test")])


class TestColBERTAdapter:
    """Tests for ColBERTAdapter with mocked model."""

    @pytest.fixture
    def adapter(self) -> ColBERTAdapter:
        """Create an adapter instance."""
        return ColBERTAdapter(
            "test-colbert-model",
            token_dim=128,
            normalize=True,
            max_seq_length=512,
            query_max_length=32,
        )

    def test_capabilities(self, adapter: ColBERTAdapter) -> None:
        """Adapter reports correct capabilities."""
        caps = adapter.capabilities
        assert caps.inputs == ["text"]
        assert caps.outputs == ["multivector", "score"]

    def test_dims_before_load_raises(self, adapter: ColBERTAdapter) -> None:
        """Accessing dims before load raises error."""
        with pytest.raises(RuntimeError, match="Model not loaded"):
            _ = adapter.dims

    def test_encode_before_load_raises(self, adapter: ColBERTAdapter) -> None:
        """Encode before load raises error."""
        items = [Item(text="hello")]
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.encode(items, output_types=["multivector"])

    def test_score_before_load_raises(self, adapter: ColBERTAdapter) -> None:
        """Score before load raises error."""
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.score(Item(text="query"), [Item(text="doc")])

    def test_cpu_not_supported(self, adapter: ColBERTAdapter) -> None:
        """CPU device raises error (flash attention requires CUDA)."""
        with pytest.raises(RuntimeError, match="requires CUDA"):
            adapter.load("cpu")

    def test_validate_output_types(self, adapter: ColBERTAdapter) -> None:
        """Only multivector output type is supported."""
        # This tests the validation logic without loading
        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter._validate_output_types(["dense"])

        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter._validate_output_types(["sparse"])

        # Should not raise for multivector
        adapter._validate_output_types(["multivector"])

    def test_extract_texts_with_prefix(self, adapter: ColBERTAdapter) -> None:
        """Text extraction applies query/doc prefixes."""
        adapter._query_prefix = "[Q] "
        adapter._doc_prefix = "[D] "

        items = [Item(text="hello"), Item(text="world")]

        # Query mode
        texts = adapter._extract_texts(items, instruction=None, is_query=True)
        assert texts == ["[Q] hello", "[Q] world"]

        # Document mode
        texts = adapter._extract_texts(items, instruction=None, is_query=False)
        assert texts == ["[D] hello", "[D] world"]

    def test_extract_texts_with_instruction(self, adapter: ColBERTAdapter) -> None:
        """Text extraction handles instruction."""
        items = [Item(text="hello")]

        texts = adapter._extract_texts(items, instruction="search:", is_query=True)
        assert texts == ["search: hello"]

    def test_extract_texts_without_text_raises(self, adapter: ColBERTAdapter) -> None:
        """Text extraction raises if item has no text."""
        items = [Item()]  # No text
        with pytest.raises(ValueError, match="requires text input"):
            adapter._extract_texts(items, instruction=None, is_query=False)

    def test_run_embeddings_bert_architecture(self, adapter: ColBERTAdapter) -> None:
        """Test _run_embeddings with BERT-style embeddings (word_embeddings)."""
        from unittest.mock import MagicMock

        import torch

        # Mock BERT-style embeddings
        mock_embeddings = MagicMock()
        mock_embeddings.word_embeddings = MagicMock(return_value=torch.randn(3, 768))
        mock_embeddings.position_embeddings = MagicMock(return_value=torch.randn(3, 768))
        mock_embeddings.token_type_embeddings = MagicMock(return_value=torch.randn(3, 768))
        mock_embeddings.LayerNorm = MagicMock(side_effect=lambda x: x)
        mock_embeddings.dropout = MagicMock(side_effect=lambda x: x)

        mock_model = MagicMock()
        mock_model.embeddings = mock_embeddings

        adapter._model = mock_model
        adapter._device = "cpu"

        input_ids = torch.tensor([1, 2, 3])
        position_ids = torch.tensor([0, 1, 2])

        result = adapter._run_embeddings(input_ids, position_ids)

        # Verify BERT path was taken
        mock_embeddings.word_embeddings.assert_called_once()
        mock_embeddings.position_embeddings.assert_called_once()
        assert result.shape == (3, 768)

    def test_run_embeddings_modernbert_architecture(self, adapter: ColBERTAdapter) -> None:
        """Test _run_embeddings handles ModernBERT architecture (tok_embeddings).

        ModernBERT uses different embedding attribute names:
        - tok_embeddings instead of word_embeddings
        - norm instead of LayerNorm
        - drop instead of dropout
        - No position embeddings (uses RoPE in attention)

        This test verifies the fallback path when ColBERTAdapter is used
        as a fallback for ColBERTModernBERTFlashAdapter on non-CUDA devices.
        """
        from unittest.mock import MagicMock

        import torch

        # Mock ModernBERT-style embeddings (no word_embeddings, no position_embeddings)
        mock_embeddings = MagicMock(spec=[])  # Empty spec to avoid auto-attributes

        # Add only ModernBERT attributes
        mock_embeddings.tok_embeddings = MagicMock(return_value=torch.randn(3, 768))
        mock_embeddings.norm = MagicMock(side_effect=lambda x: x)
        mock_embeddings.drop = MagicMock(side_effect=lambda x: x)

        mock_model = MagicMock()
        mock_model.embeddings = mock_embeddings

        adapter._model = mock_model
        adapter._device = "cpu"

        input_ids = torch.tensor([1, 2, 3])
        position_ids = torch.tensor([0, 1, 2])

        result = adapter._run_embeddings(input_ids, position_ids)

        # Verify ModernBERT path was taken
        mock_embeddings.tok_embeddings.assert_called_once()
        mock_embeddings.norm.assert_called_once()
        mock_embeddings.drop.assert_called_once()
        assert result.shape == (3, 768)

    def test_run_embeddings_unsupported_architecture_raises(self, adapter: ColBERTAdapter) -> None:
        """Test _run_embeddings raises for unknown embedding architecture."""
        from unittest.mock import MagicMock

        import torch

        # Mock embeddings without word_embeddings or tok_embeddings
        mock_embeddings = MagicMock(spec=[])  # No embedding methods

        mock_model = MagicMock()
        mock_model.embeddings = mock_embeddings

        adapter._model = mock_model
        adapter._device = "cpu"

        input_ids = torch.tensor([1, 2, 3])
        position_ids = torch.tensor([0, 1, 2])

        with pytest.raises(AttributeError, match=r"word_embeddings.*tok_embeddings"):
            adapter._run_embeddings(input_ids, position_ids)


class TestColBERTAdapterMultivectorOutput:
    """Tests for ColBERT multivector output format."""

    def test_multivector_shape_validation(self) -> None:
        """Multivector output should have shape [num_tokens, token_dim]."""
        # Create sample multivector output
        num_tokens = 10
        token_dim = 128
        multivector = _RNG.standard_normal((num_tokens, token_dim)).astype(np.float32)

        # Verify shape
        assert multivector.ndim == 2
        assert multivector.shape[0] == num_tokens
        assert multivector.shape[1] == token_dim

    def test_multivector_is_normalized(self) -> None:
        """Multivector tokens should be L2 normalized."""
        import torch
        from torch.nn import functional

        # Create random vectors
        num_tokens = 5
        token_dim = 128
        raw = torch.randn(num_tokens, token_dim)

        # Normalize
        normalized = functional.normalize(raw, p=2, dim=-1)

        # Check L2 norm is 1 for each token
        norms = torch.norm(normalized, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_maxsim_computation(self) -> None:
        """MaxSim should compute correctly."""
        import torch

        # Create query and doc multivectors
        query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # 2 query tokens
        doc = torch.tensor([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]])  # 3 doc tokens

        # Normalize
        query = torch.nn.functional.normalize(query, p=2, dim=-1)
        doc = torch.nn.functional.normalize(doc, p=2, dim=-1)

        # Compute MaxSim: sim[i,j] = cosine(query[i], doc[j])
        sim = torch.matmul(query, doc.T)

        # For each query token, find max similarity with any doc token
        max_sims, _ = sim.max(dim=-1)  # [num_query_tokens]

        # Sum over query tokens
        maxsim_score = max_sims.sum().item()

        # Query token 0: [1, 0] -> max sim with doc token 1 [1, 0] = 1.0
        # Query token 1: [0, 1] -> max sim with doc token 2 [0, 1] = 1.0
        # Total MaxSim = 2.0
        assert abs(maxsim_score - 2.0) < 1e-5


class TestColBERTScoreMethod:
    """Tests for ColBERT adapter's score() method (MaxSim scoring)."""

    def test_score_returns_correct_number_of_scores(self) -> None:
        """Score returns one score per document."""
        from unittest.mock import MagicMock, patch

        from sie_server.core.inference_output import EncodeOutput

        adapter = ColBERTAdapter("test-model")

        # Mock the encode method to return known multivectors via EncodeOutput
        query_mv = _RNG.standard_normal((5, 128)).astype(np.float32)  # 5 query tokens
        doc1_mv = _RNG.standard_normal((10, 128)).astype(np.float32)  # 10 doc tokens
        doc2_mv = _RNG.standard_normal((8, 128)).astype(np.float32)  # 8 doc tokens

        with patch.object(adapter, "encode") as mock_encode:
            # First call returns query multivector, second returns doc multivectors
            mock_encode.side_effect = [
                EncodeOutput(multivector=[query_mv], batch_size=1, is_query=True),
                EncodeOutput(multivector=[doc1_mv, doc2_mv], batch_size=2, is_query=False),
            ]

            # Mock that model is loaded
            adapter._model = MagicMock()
            adapter._device = "cpu"

            scores = adapter.score(
                Item(text="query"),
                [Item(text="doc1"), Item(text="doc2")],
            )

        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)

    def test_maxsim_score_increases_with_similarity(self) -> None:
        """More similar documents get higher MaxSim scores."""
        from unittest.mock import MagicMock, patch

        from sie_server.core.inference_output import EncodeOutput

        adapter = ColBERTAdapter("test-model")

        # Create query with specific pattern
        query_mv = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        # Doc 1: similar to query (should get high score)
        doc1_mv = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        # Doc 2: orthogonal to query (should get lower score)
        doc2_mv = np.array([[0.707, 0.707], [-0.707, 0.707]], dtype=np.float32)

        with patch.object(adapter, "encode") as mock_encode:
            mock_encode.side_effect = [
                EncodeOutput(multivector=[query_mv], batch_size=1, is_query=True),
                EncodeOutput(multivector=[doc1_mv, doc2_mv], batch_size=2, is_query=False),
            ]

            adapter._model = MagicMock()
            adapter._device = "cpu"

            scores = adapter.score(
                Item(text="query"),
                [Item(text="similar"), Item(text="different")],
            )

        # Doc 1 (similar) should have higher score than Doc 2 (orthogonal)
        assert scores[0] > scores[1]


class TestCLIPAdapter:
    """Tests for CLIPAdapter with mocked model."""

    @pytest.fixture
    def mock_clip_model(self) -> MagicMock:
        """Create a mock CLIPModel."""
        mock = MagicMock()
        # Mock config with projection_dim
        mock.config.projection_dim = 512
        # Mock get_text_features and get_image_features
        mock.get_text_features.return_value = MagicMock(
            __getitem__=lambda self, idx: MagicMock(
                float=lambda: MagicMock(
                    cpu=lambda: MagicMock(numpy=lambda: _RNG.standard_normal(512).astype(np.float32))
                )
            )
        )
        mock.get_image_features.return_value = MagicMock(
            mean=lambda dim, keepdim: MagicMock(
                __getitem__=lambda self, idx: MagicMock(
                    float=lambda: MagicMock(
                        cpu=lambda: MagicMock(numpy=lambda: _RNG.standard_normal(512).astype(np.float32))
                    )
                )
            )
        )
        return mock

    @pytest.fixture
    def mock_clip_processor(self) -> MagicMock:
        """Create a mock CLIPProcessor."""
        mock = MagicMock()
        # Return dict-like object for processor outputs
        mock.return_value = {"pixel_values": MagicMock(), "input_ids": MagicMock()}
        return mock

    @pytest.fixture
    def adapter(self) -> CLIPAdapter:
        """Create an adapter instance."""
        return CLIPAdapter(
            "openai/clip-vit-base-patch32",
            normalize=True,
            compute_precision="float16",
        )

    def test_capabilities(self, adapter: CLIPAdapter) -> None:
        """Adapter reports correct capabilities."""
        caps = adapter.capabilities
        assert caps.inputs == ["text", "image"]
        assert caps.outputs == ["dense"]

    def test_dims_before_load_raises(self, adapter: CLIPAdapter) -> None:
        """Accessing dims before load raises error."""
        with pytest.raises(RuntimeError, match="Model not loaded"):
            _ = adapter.dims

    def test_encode_before_load_raises(self, adapter: CLIPAdapter) -> None:
        """Encode before load raises error."""
        items = [Item(text="hello")]
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.encode(items, output_types=["dense"])

    def test_encode_without_input_raises(self, adapter: CLIPAdapter) -> None:
        """Encode raises if item has no text or images."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()

        items = [Item()]  # No text or images
        with pytest.raises(ValueError, match="requires either text or images"):
            adapter.encode(items, output_types=["dense"])

    def test_validate_output_types(self, adapter: CLIPAdapter) -> None:
        """Only dense output type is supported."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()

        items = [Item(text="test")]
        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode(items, output_types=["sparse"])

        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode(items, output_types=["multivector"])

    @patch("transformers.CLIPModel")
    @patch("transformers.CLIPProcessor")
    def test_load(
        self,
        mock_processor_class: MagicMock,
        mock_model_class: MagicMock,
        adapter: CLIPAdapter,
        mock_clip_model: MagicMock,
        mock_clip_processor: MagicMock,
    ) -> None:
        """Load initializes the model."""
        mock_model_class.from_pretrained.return_value = mock_clip_model
        mock_processor_class.from_pretrained.return_value = mock_clip_processor

        adapter.load("cpu")

        mock_model_class.from_pretrained.assert_called_once()
        mock_processor_class.from_pretrained.assert_called_once()
        assert adapter.dims.dense == 512

    @patch("sie_server.adapters.clip.torch")
    def test_unload(self, mock_torch: MagicMock, adapter: CLIPAdapter) -> None:
        """Unload clears the model."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()
        adapter._device = "cpu"
        adapter._dense_dim = 512

        adapter.unload()

        with pytest.raises(RuntimeError, match="Model not loaded"):
            _ = adapter.dims


class TestFlorence2Adapter:
    """Tests for Florence2Adapter with mocked model."""

    @pytest.fixture
    def mock_florence2_model(self) -> MagicMock:
        """Create a mock Florence2 model."""
        mock = MagicMock()
        # Mock generate method
        mock.generate.return_value = MagicMock()
        mock.dtype = MagicMock()
        return mock

    @pytest.fixture
    def mock_florence2_processor(self) -> MagicMock:
        """Create a mock Florence2 processor."""
        mock = MagicMock()
        # Return dict-like object for processor outputs
        mock.return_value = {
            "pixel_values": MagicMock(),
            "input_ids": MagicMock(),
        }
        # Mock batch_decode
        mock.batch_decode.return_value = ["<s><OCR_WITH_REGION>text</s>"]
        # Mock post_process_generation
        mock.post_process_generation.return_value = {
            "<OCR_WITH_REGION>": {
                "quad_boxes": [[10.0, 10.0, 100.0, 10.0, 100.0, 50.0, 10.0, 50.0]],
                "labels": ["Hello World"],
            }
        }
        return mock

    @pytest.fixture
    def adapter(self) -> Florence2Adapter:
        """Create an adapter instance."""
        from sie_server.adapters.florence2 import Florence2Adapter

        return Florence2Adapter(
            "microsoft/Florence-2-base",
            default_task="<OCR_WITH_REGION>",
            compute_precision="float16",
        )

    def test_capabilities(self, adapter: Florence2Adapter) -> None:
        """Adapter reports correct capabilities."""
        caps = adapter.capabilities
        assert caps.inputs == ["image"]
        assert caps.outputs == ["json"]

    def test_dims(self, adapter: Florence2Adapter) -> None:
        """Adapter reports empty dimensions (extraction model)."""
        dims = adapter.dims
        assert dims.dense is None
        assert dims.sparse is None
        assert dims.multivector is None

    def test_encode_raises_not_implemented(self, adapter: Florence2Adapter) -> None:
        """Encode raises NotImplementedError."""
        items = [Item(text="hello")]
        with pytest.raises(NotImplementedError, match="does not support encode"):
            adapter.encode(items, output_types=["dense"])

    def test_extract_before_load_raises(self, adapter: Florence2Adapter) -> None:
        """Extract before load raises error."""
        from sie_server.types.inputs import ImageInput

        items = [Item(images=[ImageInput(data=b"fake", format="jpeg")])]
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.extract(items)

    def test_quad_to_bbox_conversion(self, adapter: Florence2Adapter) -> None:
        """Quad box is correctly converted to bbox."""
        # Quad box: 4 corners (x1,y1,x2,y2,x3,y3,x4,y4)
        quad_box = [10.0, 10.0, 100.0, 10.0, 100.0, 50.0, 10.0, 50.0]
        image_size = (200, 100)

        bbox = adapter._quad_to_bbox(quad_box, image_size)

        # Expected: [10/200, 10/100, 100/200, 50/100] = [0.05, 0.1, 0.5, 0.5]
        assert len(bbox) == 4
        assert abs(bbox[0] - 0.05) < 1e-6
        assert abs(bbox[1] - 0.1) < 1e-6
        assert abs(bbox[2] - 0.5) < 1e-6
        assert abs(bbox[3] - 0.5) < 1e-6

    def test_normalize_bbox(self, adapter: Florence2Adapter) -> None:
        """Bbox is correctly normalized."""
        bbox = [50.0, 25.0, 150.0, 75.0]
        image_size = (200, 100)

        norm_bbox = adapter._normalize_bbox(bbox, image_size)

        assert len(norm_bbox) == 4
        assert abs(norm_bbox[0] - 0.25) < 1e-6
        assert abs(norm_bbox[1] - 0.25) < 1e-6
        assert abs(norm_bbox[2] - 0.75) < 1e-6
        assert abs(norm_bbox[3] - 0.75) < 1e-6

    def test_build_prompt_basic_task(self, adapter: Florence2Adapter) -> None:
        """Build prompt returns task token for basic tasks."""
        prompt = adapter._build_prompt("<OCR>", labels=None, instruction=None)
        assert prompt == "<OCR>"

    def test_build_prompt_with_instruction(self, adapter: Florence2Adapter) -> None:
        """Build prompt appends instruction."""
        prompt = adapter._build_prompt("<OCR>", labels=None, instruction="Extract all text")
        assert prompt == "<OCR>Extract all text"

    def test_build_prompt_phrase_grounding_with_labels(self, adapter: Florence2Adapter) -> None:
        """Build prompt appends labels for phrase grounding."""
        prompt = adapter._build_prompt(
            "<CAPTION_TO_PHRASE_GROUNDING>",
            labels=["person", "car"],
            instruction=None,
        )
        assert prompt == "<CAPTION_TO_PHRASE_GROUNDING>person, car"

    def test_convert_output_ocr_with_region(self, adapter: Florence2Adapter) -> None:
        """Convert output handles OCR_WITH_REGION format."""
        parsed = {
            "<OCR_WITH_REGION>": {
                "quad_boxes": [[0.0, 0.0, 100.0, 0.0, 100.0, 50.0, 0.0, 50.0]],
                "labels": ["Hello"],
            }
        }
        image_size = (100, 100)

        entities = adapter._convert_output(parsed, "<OCR_WITH_REGION>", image_size)

        assert len(entities) == 1
        assert entities[0]["text"] == "Hello"
        assert entities[0]["label"] == "text"
        assert entities[0]["score"] == 1.0
        assert entities[0]["bbox"] is not None

    def test_convert_output_object_detection(self, adapter: Florence2Adapter) -> None:
        """Convert output handles OD format."""
        parsed = {
            "<OD>": {
                "bboxes": [[10.0, 20.0, 80.0, 90.0]],
                "labels": ["car"],
            }
        }
        image_size = (100, 100)

        entities = adapter._convert_output(parsed, "<OD>", image_size)

        assert len(entities) == 1
        assert entities[0]["text"] == "car"  # Detected class is in text field
        assert entities[0]["label"] == "object"  # Entity type is "object"
        assert entities[0]["score"] == 1.0
        assert entities[0]["bbox"] is not None


class TestDonutAdapter:
    """Tests for DonutAdapter with mocked model."""

    @pytest.fixture
    def mock_donut_model(self) -> MagicMock:
        """Create a mock Donut model."""
        mock = MagicMock()
        # Mock generate method - returns ModelOutput-like object
        mock_output = MagicMock()
        mock_output.sequences = MagicMock()
        mock.generate.return_value = mock_output
        mock.dtype = MagicMock()
        mock.decoder.config.max_position_embeddings = 2048
        return mock

    @pytest.fixture
    def mock_donut_processor(self) -> MagicMock:
        """Create a mock Donut processor."""
        mock = MagicMock()
        # Mock image processing - returns dict with pixel_values
        mock_image_result = MagicMock()
        mock_image_result.pixel_values = MagicMock()
        mock.return_value = mock_image_result
        # Mock tokenizer
        mock.tokenizer.return_value = MagicMock()
        mock.tokenizer.pad_token_id = 0
        mock.tokenizer.eos_token_id = 2
        mock.tokenizer.unk_token_id = 3
        mock.tokenizer.eos_token = "</s>"  # noqa: S105 — test mock tokenizer config
        mock.tokenizer.pad_token = "<pad>"  # noqa: S105 — test mock tokenizer config
        # Mock batch_decode
        mock.batch_decode.return_value = ["<s_cord-v2><s_menu><nm>Coffee</nm><price>5.00</price></s_menu></s>"]
        # Mock token2json
        mock.token2json.return_value = {"menu": {"nm": "Coffee", "price": "5.00"}}
        return mock

    @pytest.fixture
    def adapter(self) -> DonutAdapter:
        """Create an adapter instance."""
        from sie_server.adapters.donut import DonutAdapter

        return DonutAdapter(
            "naver-clova-ix/donut-base-finetuned-cord-v2",
            default_task="<s_cord-v2>",
            compute_precision="float16",
        )

    def test_capabilities(self, adapter: DonutAdapter) -> None:
        """Adapter reports correct capabilities."""
        caps = adapter.capabilities
        assert caps.inputs == ["image"]
        assert caps.outputs == ["json"]

    def test_dims(self, adapter: DonutAdapter) -> None:
        """Adapter reports empty dimensions (extraction model)."""
        dims = adapter.dims
        assert dims.dense is None
        assert dims.sparse is None
        assert dims.multivector is None

    def test_encode_raises_not_implemented(self, adapter: DonutAdapter) -> None:
        """Encode raises NotImplementedError."""
        items = [Item(text="hello")]
        with pytest.raises(NotImplementedError, match="does not support encode"):
            adapter.encode(items, output_types=["dense"])

    def test_extract_before_load_raises(self, adapter: DonutAdapter) -> None:
        """Extract before load raises error."""
        from sie_server.types.inputs import ImageInput

        items = [Item(images=[ImageInput(data=b"fake", format="jpeg")])]
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.extract(items)

    def test_build_prompt_basic_task(self, adapter: DonutAdapter) -> None:
        """Build prompt returns task token for basic tasks."""
        prompt = adapter._build_prompt("<s_cord-v2>", instruction=None)
        assert prompt == "<s_cord-v2>"

    def test_build_prompt_docvqa_with_question(self, adapter: DonutAdapter) -> None:
        """Build prompt formats DocVQA question correctly."""
        prompt = adapter._build_prompt("<s_docvqa>", instruction="What is the total?")
        assert prompt == "<s_docvqa><s_question>What is the total?</s_question><s_answer>"

    def test_try_parse_json_valid(self, adapter: DonutAdapter) -> None:
        """Try parse JSON handles valid JSON."""
        result = adapter._try_parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_try_parse_json_invalid(self, adapter: DonutAdapter) -> None:
        """Try parse JSON handles invalid JSON gracefully."""
        result = adapter._try_parse_json("not json at all")
        assert result == {"raw": "not json at all"}

    def test_try_parse_json_with_special_tokens(self, adapter: DonutAdapter) -> None:
        """Try parse JSON strips special tokens."""
        result = adapter._try_parse_json('<s_menu>{"key": "value"}</s_menu>')
        # After stripping tags: {"key": "value"}
        assert result == {"key": "value"}

    def test_convert_output_cord(self, adapter: DonutAdapter) -> None:
        """Convert output handles CORD format."""
        from sie_server.adapters.donut import TASK_CORD

        parsed = {"menu": {"nm": "Coffee", "price": "5.00"}}
        raw_text = "<s_menu><nm>Coffee</nm><price>5.00</price></s_menu>"

        entities = adapter._convert_output(parsed, TASK_CORD, raw_text)

        # Check entities extracted from nested structure
        assert len(entities) >= 2
        entity_labels = [e["label"] for e in entities]
        assert "menu.nm" in entity_labels
        assert "menu.price" in entity_labels

    def test_convert_output_docvqa(self, adapter: DonutAdapter) -> None:
        """Convert output handles DocVQA format."""
        from sie_server.adapters.donut import TASK_DOCVQA

        parsed = {"answer": "The total is $10.00"}
        raw_text = "<s_answer>The total is $10.00</s_answer>"

        entities = adapter._convert_output(parsed, TASK_DOCVQA, raw_text)

        assert len(entities) == 1
        assert entities[0]["text"] == "The total is $10.00"
        assert entities[0]["label"] == "answer"
        assert entities[0]["score"] == 1.0

    def test_convert_output_rvlcdip(self, adapter: DonutAdapter) -> None:
        """Convert output handles RVLCDIP format."""
        from sie_server.adapters.donut import TASK_RVLCDIP

        parsed = {"class": "invoice"}
        raw_text = "<s_class>invoice</s_class>"

        entities = adapter._convert_output(parsed, TASK_RVLCDIP, raw_text)

        assert len(entities) == 1
        assert entities[0]["text"] == "invoice"
        assert entities[0]["label"] == "document_class"

    def test_extract_cord_entities_nested(self, adapter: DonutAdapter) -> None:
        """Extract CORD entities handles nested structures."""
        parsed = {
            "menu": [
                {"nm": "Coffee", "price": "5.00"},
                {"nm": "Tea", "price": "3.00"},
            ],
            "total": {"total_price": "8.00"},
        }

        entities = adapter._extract_cord_entities(parsed)

        # Should have 5 entities: 2 items × 2 fields + 1 total
        assert len(entities) == 5
        # Check labels have correct prefixes
        labels = [e["label"] for e in entities]
        assert "menu[0].nm" in labels
        assert "menu[0].price" in labels
        assert "menu[1].nm" in labels
        assert "total.total_price" in labels


class TestSiglipAdapter:
    """Tests for SiglipAdapter with mocked model."""

    @pytest.fixture
    def mock_siglip_model(self) -> MagicMock:
        """Create a mock SiglipModel."""
        mock = MagicMock()
        # Mock config with vision_config.hidden_size (SigLIP uses hidden_size, not projection_dim)
        mock.config.vision_config.hidden_size = 1152
        return mock

    @pytest.fixture
    def mock_siglip_processor(self) -> MagicMock:
        """Create a mock SiglipProcessor."""
        mock = MagicMock()
        mock.return_value = {"pixel_values": MagicMock(), "input_ids": MagicMock()}
        return mock

    @pytest.fixture
    def adapter(self) -> SiglipAdapter:
        """Create an adapter instance."""
        return SiglipAdapter(
            "google/siglip-so400m-patch14-384",
            normalize=True,
            compute_precision="float16",
        )

    def test_capabilities(self, adapter: SiglipAdapter) -> None:
        """Adapter reports correct capabilities."""
        caps = adapter.capabilities
        assert caps.inputs == ["text", "image"]
        assert caps.outputs == ["dense"]

    def test_dims_before_load_raises(self, adapter: SiglipAdapter) -> None:
        """Accessing dims before load raises error."""
        with pytest.raises(RuntimeError, match="Model not loaded"):
            _ = adapter.dims

    def test_encode_before_load_raises(self, adapter: SiglipAdapter) -> None:
        """Encode before load raises error."""
        items = [Item(text="hello")]
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.encode(items, output_types=["dense"])

    def test_encode_without_input_raises(self, adapter: SiglipAdapter) -> None:
        """Encode raises if item has no text or images."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()

        items = [Item()]  # No text or images
        with pytest.raises(ValueError, match="requires either text or images"):
            adapter.encode(items, output_types=["dense"])

    def test_validate_output_types(self, adapter: SiglipAdapter) -> None:
        """Only dense output type is supported."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()

        items = [Item(text="test")]
        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode(items, output_types=["sparse"])

    @patch("transformers.SiglipModel")
    @patch("transformers.SiglipProcessor")
    def test_load(
        self,
        mock_processor_class: MagicMock,
        mock_model_class: MagicMock,
        adapter: SiglipAdapter,
        mock_siglip_model: MagicMock,
        mock_siglip_processor: MagicMock,
    ) -> None:
        """Load initializes the model."""
        mock_model_class.from_pretrained.return_value = mock_siglip_model
        mock_processor_class.from_pretrained.return_value = mock_siglip_processor

        adapter.load("cpu")

        mock_model_class.from_pretrained.assert_called_once()
        mock_processor_class.from_pretrained.assert_called_once()
        assert adapter.dims.dense == 1152

    @patch("sie_server.adapters.siglip.torch")
    def test_unload(self, mock_torch: MagicMock, adapter: SiglipAdapter) -> None:
        """Unload clears the model."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()
        adapter._device = "cpu"
        adapter._dense_dim = 1152

        adapter.unload()

        with pytest.raises(RuntimeError, match="Model not loaded"):
            _ = adapter.dims


class TestGTESparseFlashAdapter:
    """Tests for GTESparseFlashAdapter with mocked model."""

    @pytest.fixture
    def adapter(self) -> GTESparseFlashAdapter:
        """Create an adapter instance."""
        return GTESparseFlashAdapter(
            "test-gte-sparse-model",
            max_seq_length=512,
            compute_precision="float16",
            trust_remote_code=True,
        )

    def test_capabilities(self, adapter: GTESparseFlashAdapter) -> None:
        """Adapter reports correct capabilities."""
        caps = adapter.capabilities
        assert caps.inputs == ["text"]
        assert caps.outputs == ["sparse"]

    def test_dims_before_load_raises(self, adapter: GTESparseFlashAdapter) -> None:
        """Accessing dims before load raises error."""
        with pytest.raises(RuntimeError, match="Model not loaded"):
            _ = adapter.dims

    def test_encode_before_load_raises(self, adapter: GTESparseFlashAdapter) -> None:
        """Encode before load raises error."""
        items = [Item(text="hello")]
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.encode(items, output_types=["sparse"])

    def test_validate_output_types(self, adapter: GTESparseFlashAdapter) -> None:
        """Only sparse output type is supported."""
        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter._validate_output_types(["dense"])

        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter._validate_output_types(["multivector"])

        # Should not raise for sparse
        adapter._validate_output_types(["sparse"])

    def test_extract_texts_basic(self, adapter: GTESparseFlashAdapter) -> None:
        """Text extraction works without templates."""
        items = [Item(text="hello"), Item(text="world")]
        texts = adapter._extract_texts(items, instruction=None, is_query=False)
        assert texts == ["hello", "world"]

    def test_extract_texts_with_instruction(self, adapter: GTESparseFlashAdapter) -> None:
        """Text extraction handles instruction."""
        items = [Item(text="hello")]
        texts = adapter._extract_texts(items, instruction="search:", is_query=True)
        assert texts == ["search: hello"]

    def test_extract_texts_with_query_template(self, adapter: GTESparseFlashAdapter) -> None:
        """Text extraction uses query template when is_query=True."""
        adapter._query_template = "Query: {text}"
        items = [Item(text="hello")]
        texts = adapter._extract_texts(items, instruction=None, is_query=True)
        assert texts == ["Query: hello"]

    def test_extract_texts_with_doc_template(self, adapter: GTESparseFlashAdapter) -> None:
        """Text extraction uses doc template when is_query=False."""
        adapter._doc_template = "Document: {text}"
        items = [Item(text="hello")]
        texts = adapter._extract_texts(items, instruction=None, is_query=False)
        assert texts == ["Document: hello"]

    def test_extract_texts_without_text_raises(self, adapter: GTESparseFlashAdapter) -> None:
        """Text extraction raises if item has no text."""
        items = [Item()]  # No text
        with pytest.raises(ValueError, match="requires text input"):
            adapter._extract_texts(items, instruction=None, is_query=False)

    @patch("sie_server.adapters.gte_sparse_flash.torch")
    def test_unload(self, mock_torch: MagicMock, adapter: GTESparseFlashAdapter) -> None:
        """Unload clears the model."""
        adapter._model = MagicMock()
        adapter._tokenizer = MagicMock()
        adapter._device = "cuda:0"
        adapter._vocab_size = 30522

        adapter.unload()

        with pytest.raises(RuntimeError, match="Model not loaded"):
            _ = adapter.dims

    def test_to_inference_output_empty_sparse(self, adapter: GTESparseFlashAdapter) -> None:
        """Converting empty sparse dict produces valid output."""
        sparse_results = [{}]  # Empty sparse dict
        output = adapter._to_inference_output(sparse_results, batch_size=1, is_query=False)

        assert output.batch_size == 1
        assert output.sparse is not None
        assert len(output.sparse) == 1
        assert len(output.sparse[0].indices) == 0
        assert len(output.sparse[0].values) == 0

    def test_to_inference_output_with_values(self, adapter: GTESparseFlashAdapter) -> None:
        """Converting sparse dict with values produces valid output."""
        sparse_results = [{100: 0.5, 500: 0.3, 1000: 0.8}]
        output = adapter._to_inference_output(sparse_results, batch_size=1, is_query=True)

        assert output.batch_size == 1
        assert output.is_query is True
        assert output.sparse is not None
        assert len(output.sparse) == 1
        assert len(output.sparse[0].indices) == 3
        assert len(output.sparse[0].values) == 3

    @patch("transformers.AutoModelForMaskedLM")
    @patch("transformers.AutoTokenizer")
    def test_load_wrong_architecture_raises(
        self,
        mock_tokenizer_class: MagicMock,
        mock_model_class: MagicMock,
        adapter: GTESparseFlashAdapter,
    ) -> None:
        """Load raises error for non-NewForMaskedLM architecture."""
        # Mock model without 'new' attribute (e.g., BERT)
        mock_model = MagicMock()
        mock_model.config.vocab_size = 30522
        # MagicMock has 'new' by default via __getattr__, so we need spec
        mock_model = MagicMock(spec=["config", "to", "eval"])
        mock_model.config.vocab_size = 30522
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = MagicMock()

        with pytest.raises(ValueError, match="NewForMaskedLM"):
            adapter.load("cuda:0")


class TestGLiRELAdapterExtractEntities:
    """Tests for GLiRELAdapter._extract_entities with TypedDict items.

    Item is a TypedDict — at runtime it's a plain dict. The adapter must use
    dict access (.get()) rather than attribute access (.metadata).
    """

    @pytest.fixture
    def adapter(self) -> GLiRELAdapter:
        from sie_server.adapters.glirel import GLiRELAdapter

        return GLiRELAdapter(
            "jackboyla/glirel-large-v0",
            compute_precision="float32",
        )

    def test_extract_entities_with_metadata(self, adapter: GLiRELAdapter) -> None:
        """Entities are extracted from item metadata dict."""
        item: Item = {"text": "test", "metadata": {"entities": [{"text": "Alice", "label": "PER"}]}}
        result = adapter._extract_entities(item)
        assert result == [{"text": "Alice", "label": "PER"}]

    def test_extract_entities_no_metadata(self, adapter: GLiRELAdapter) -> None:
        """Returns empty list when metadata is absent."""
        item: Item = {"text": "test"}
        result = adapter._extract_entities(item)
        assert result == []

    def test_extract_entities_metadata_none(self, adapter: GLiRELAdapter) -> None:
        """Returns empty list when metadata is explicitly None."""
        item: Item = {"text": "test", "metadata": None}  # type: ignore[typeddict-item]
        result = adapter._extract_entities(item)
        assert result == []

    def test_extract_entities_metadata_no_entities_key(self, adapter: GLiRELAdapter) -> None:
        """Returns empty list when metadata has no 'entities' key."""
        item: Item = {"text": "test", "metadata": {"other": "data"}}
        result = adapter._extract_entities(item)
        assert result == []


class TestAdapterEncodeAcceptsOptions:
    """Verify all adapter encode() methods accept the 'options' keyword parameter.

    The worker pipeline passes options= to adapter.encode(). If any adapter
    is missing the parameter, it will get a TypeError at runtime.
    """

    def test_all_adapters_encode_accepts_options(self) -> None:
        """Every concrete adapter's encode() accepts options kwarg."""
        import inspect

        from sie_server.adapters.base import ModelAdapter

        # Discover all adapter classes that override encode()
        adapter_modules = [
            "sie_server.adapters.sentence_transformer",
            "sie_server.adapters.bge_m3",
            "sie_server.adapters.bge_m3_flag",
            "sie_server.adapters.colbert",
            "sie_server.adapters.colpali",
            "sie_server.adapters.colqwen2",
            "sie_server.adapters.nemo_colembed",
            "sie_server.adapters.clip",
            "sie_server.adapters.siglip",
            "sie_server.adapters.florence2",
            "sie_server.adapters.donut",
            "sie_server.adapters.owlv2",
            "sie_server.adapters.grounding_dino",
            "sie_server.adapters.pytorch_embedding",
        ]

        missing = []
        for mod_name in adapter_modules:
            try:
                mod = __import__(mod_name, fromlist=["__name__"])
            except ImportError:
                continue  # Skip if module can't be imported (missing deps)

            for name in dir(mod):
                obj = getattr(mod, name)
                if (
                    isinstance(obj, type)
                    and issubclass(obj, ModelAdapter)
                    and obj is not ModelAdapter
                    and hasattr(obj, "encode")
                ):
                    sig = inspect.signature(obj.encode)
                    if "options" not in sig.parameters:
                        missing.append(f"{mod_name}.{name}")

        assert missing == [], f"Adapters missing 'options' param in encode(): {missing}"


class TestRuntimeOptionsConsumption:
    """Verify adapters consume runtime options from the options dict.

    These tests exercise the wiring from options -> _format_texts/_extract_texts
    and options -> _pool_embeddings/_apply_pooling, without loading models.
    """

    # --- Template override tests (PyTorchEmbeddingAdapter) ---

    def test_format_texts_uses_query_template_from_options(self) -> None:
        """query_template from options overrides the loadtime default."""
        from sie_server.adapters.pytorch_embedding import PyTorchEmbeddingAdapter

        adapter = PyTorchEmbeddingAdapter(
            "test-model",
            query_template="default_query: {text}",
            doc_template="default_doc: {text}",
        )
        items: list[Item] = [Item(text="hello")]

        # Use options override
        texts = adapter._format_texts(
            items,
            None,
            is_query=True,
            query_template="custom_query: {text}",
        )
        assert texts == ["custom_query: hello"]

    def test_format_texts_uses_doc_template_from_options(self) -> None:
        """doc_template from options overrides the loadtime default."""
        from sie_server.adapters.pytorch_embedding import PyTorchEmbeddingAdapter

        adapter = PyTorchEmbeddingAdapter(
            "test-model",
            query_template="default_query: {text}",
            doc_template="default_doc: {text}",
        )
        items: list[Item] = [Item(text="hello")]

        texts = adapter._format_texts(
            items,
            None,
            is_query=False,
            doc_template="custom_doc: {text}",
        )
        assert texts == ["custom_doc: hello"]

    def test_format_texts_uses_default_instruction_from_options(self) -> None:
        """default_instruction from options overrides the loadtime default."""
        from sie_server.adapters.pytorch_embedding import PyTorchEmbeddingAdapter

        adapter = PyTorchEmbeddingAdapter(
            "test-model",
            query_template="Instruct: {instruction}\nQuery: {text}",
            default_instruction="default_instr",
        )
        items: list[Item] = [Item(text="hello")]

        # Override default_instruction via options
        texts = adapter._format_texts(
            items,
            None,
            is_query=True,
            default_instruction="custom_instr",
        )
        assert texts == ["Instruct: custom_instr\nQuery: hello"]

    def test_format_texts_falls_back_to_loadtime_defaults(self) -> None:
        """Empty options dict falls back to loadtime defaults."""
        from sie_server.adapters.pytorch_embedding import PyTorchEmbeddingAdapter

        adapter = PyTorchEmbeddingAdapter(
            "test-model",
            query_template="loadtime: {text}",
        )
        items: list[Item] = [Item(text="hello")]

        # No options override — should use loadtime template
        texts = adapter._format_texts(items, None, is_query=True)
        assert texts == ["loadtime: hello"]

    def test_format_texts_none_options_uses_loadtime_defaults(self) -> None:
        """None passed for template params falls back to loadtime defaults."""
        from sie_server.adapters.pytorch_embedding import PyTorchEmbeddingAdapter

        adapter = PyTorchEmbeddingAdapter(
            "test-model",
            query_template="loadtime: {text}",
        )
        items: list[Item] = [Item(text="hello")]

        # Explicitly pass None — should use loadtime template
        texts = adapter._format_texts(
            items,
            None,
            is_query=True,
            query_template=None,
        )
        assert texts == ["loadtime: hello"]

    # --- Template override tests (flash adapter _extract_texts) ---

    def test_extract_texts_uses_template_from_options(self) -> None:
        """Flash adapter _extract_texts uses template params over self._* defaults."""
        from sie_server.adapters.bert_flash import BertFlashAdapter

        adapter = BertFlashAdapter(
            "test-model",
            query_template="default_query: {text}",
            doc_template="default_doc: {text}",
        )
        items: list[Item] = [Item(text="hello")]

        texts = adapter._extract_texts(
            items,
            None,
            is_query=True,
            query_template="custom: {text}",
        )
        assert texts == ["custom: hello"]

    def test_extract_texts_falls_back_to_loadtime(self) -> None:
        """Flash adapter _extract_texts falls back to self._* when no params given."""
        from sie_server.adapters.bert_flash import BertFlashAdapter

        adapter = BertFlashAdapter(
            "test-model",
            query_template="loadtime: {text}",
        )
        items: list[Item] = [Item(text="hello")]

        texts = adapter._extract_texts(items, None, is_query=True)
        assert texts == ["loadtime: hello"]

    # --- Pooling override tests (PyTorchEmbeddingAdapter) ---

    def test_apply_pooling_cls_override(self) -> None:
        """Pooling can be overridden to 'cls' at runtime."""
        import torch
        from sie_server.adapters.pytorch_embedding import PyTorchEmbeddingAdapter

        adapter = PyTorchEmbeddingAdapter("test-model", pooling="mean")
        hidden = torch.randn(2, 5, 10)
        mask = torch.ones(2, 5)

        result = adapter._apply_pooling(hidden, mask, pooling="cls")
        # CLS pooling takes position 0
        expected = hidden[:, 0]
        assert torch.equal(result, expected)

    def test_apply_pooling_mean_override(self) -> None:
        """Pooling can be overridden to 'mean' at runtime."""
        import torch
        from sie_server.adapters.pytorch_embedding import PyTorchEmbeddingAdapter

        adapter = PyTorchEmbeddingAdapter("test-model", pooling="cls")
        hidden = torch.randn(2, 5, 10)
        mask = torch.ones(2, 5)

        result = adapter._apply_pooling(hidden, mask, pooling="mean")
        # Mean pooling averages all tokens
        expected = hidden.mean(dim=1)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_apply_pooling_falls_back_to_loadtime(self) -> None:
        """Pooling falls back to loadtime config when not overridden."""
        import torch
        from sie_server.adapters.pytorch_embedding import PyTorchEmbeddingAdapter

        adapter = PyTorchEmbeddingAdapter("test-model", pooling="cls")
        hidden = torch.randn(2, 5, 10)
        mask = torch.ones(2, 5)

        result = adapter._apply_pooling(hidden, mask)
        expected = hidden[:, 0]
        assert torch.equal(result, expected)

    # --- Pooling safety validation ---

    def test_last_token_pooling_rejected_at_runtime_when_not_loaded(self) -> None:
        """Requesting last_token pooling at runtime when loaded with mean raises ValueError."""
        from sie_server.adapters.pytorch_embedding import PyTorchEmbeddingAdapter

        adapter = PyTorchEmbeddingAdapter("test-model", pooling="mean")
        # Pretend model is loaded
        adapter._model = MagicMock()
        adapter._tokenizer = MagicMock()
        adapter._device = "cpu"
        adapter._dense_dim = 10

        items: list[Item] = [Item(text="hello")]
        with pytest.raises(ValueError, match="Cannot use 'last_token' pooling at runtime"):
            adapter.encode(
                items,
                ["dense"],
                options={"pooling": "last_token"},
            )

    # --- SGLang adapter does NOT wire pooling ---

    def test_sglang_format_texts_uses_options(self) -> None:
        """SGLang adapter _format_texts accepts template overrides."""
        from sie_server.adapters.sglang import SGLangEmbeddingAdapter

        adapter = SGLangEmbeddingAdapter(
            "test-model",
            query_template="default: {text}",
        )
        items: list[Item] = [Item(text="hello")]

        texts = adapter._format_texts(
            items,
            None,
            is_query=True,
            query_template="custom: {text}",
        )
        assert texts == ["custom: hello"]

    # --- Normalize override in encode() flow ---

    def test_encode_respects_normalize_false_override(self) -> None:
        """options={"normalize": False} disables normalization even when loadtime is True."""
        import torch
        from sie_server.adapters.pytorch_embedding import PyTorchEmbeddingAdapter

        adapter = PyTorchEmbeddingAdapter("test-model", normalize=True, pooling="cls")

        # Set up minimal mocks for encode() to run
        mock_model = MagicMock()
        hidden = torch.randn(1, 5, 10)
        mock_model.return_value = MagicMock(last_hidden_state=hidden)
        adapter._model = mock_model
        adapter._device = "cpu"
        adapter._dense_dim = 10

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = MagicMock(
            __getitem__=lambda self, key: torch.ones(1, 5) if key == "attention_mask" else None,
        )
        mock_tokenizer.return_value.to = MagicMock(return_value=mock_tokenizer.return_value)
        mock_tokenizer.return_value.__getitem__ = lambda self, key: torch.ones(1, 5)
        adapter._tokenizer = mock_tokenizer
        adapter._forward_kwargs = {}

        items: list[Item] = [Item(text="hello")]
        result = adapter.encode(items, ["dense"], options={"normalize": False})

        # With normalize=False, the output should NOT be unit-length
        embeddings = result.dense
        norms = np.linalg.norm(embeddings, axis=-1)
        # CLS token from random hidden state will NOT have unit norm
        assert not np.allclose(norms, 1.0, atol=0.01)


class TestRuntimeOptionsWiringRegression:
    """Structural regression tests ensuring runtime options wiring is preserved.

    These tests inspect adapter method signatures and source code to catch
    accidental removal of options wiring during refactoring. If a test here
    fails, it means someone removed or renamed a runtime-options parameter
    that adapters must accept for per-request overrides to work.
    """

    # ---- Helper ----

    @staticmethod
    def _get_adapter_class(module_name: str, class_name: str) -> type:
        """Import and return an adapter class by module and class name."""
        import importlib

        mod = importlib.import_module(f"sie_server.adapters.{module_name}")
        return getattr(mod, class_name)

    @staticmethod
    def _get_param_names(adapter_cls: type, method_name: str) -> set[str]:
        """Return the set of parameter names for a method on a class."""
        import inspect

        method = getattr(adapter_cls, method_name)
        sig = inspect.signature(method)
        return set(sig.parameters.keys())

    # ---- Group A dense: flash adapters with _extract_texts + _pool_embeddings ----

    _DENSE_FLASH_ADAPTERS: ClassVar[list[tuple[str, str]]] = [
        ("bert_flash", "BertFlashAdapter"),
        ("nomic_flash", "NomicFlashAdapter"),
        ("qwen2_flash", "Qwen2FlashAdapter"),
        ("rope_flash", "RoPEFlashAdapter"),
        ("xlm_roberta_flash", "XLMRobertaFlashAdapter"),
    ]

    @pytest.mark.parametrize(
        ("module_name", "class_name"),
        _DENSE_FLASH_ADAPTERS,
        ids=[m for m, _ in _DENSE_FLASH_ADAPTERS],
    )
    def test_dense_flash_extract_texts_accepts_template_params(
        self,
        module_name: str,
        class_name: str,
    ) -> None:
        """Dense flash adapters' _extract_texts must accept query_template and doc_template."""
        cls = self._get_adapter_class(module_name, class_name)
        params = self._get_param_names(cls, "_extract_texts")
        assert "query_template" in params, f"{class_name}._extract_texts missing query_template"
        assert "doc_template" in params, f"{class_name}._extract_texts missing doc_template"

    @pytest.mark.parametrize(
        ("module_name", "class_name"),
        _DENSE_FLASH_ADAPTERS,
        ids=[m for m, _ in _DENSE_FLASH_ADAPTERS],
    )
    def test_dense_flash_pool_embeddings_accepts_runtime_params(
        self,
        module_name: str,
        class_name: str,
    ) -> None:
        """Dense flash adapters' _pool_embeddings must accept normalize and pooling."""
        cls = self._get_adapter_class(module_name, class_name)
        params = self._get_param_names(cls, "_pool_embeddings")
        assert "normalize" in params, f"{class_name}._pool_embeddings missing normalize"
        assert "pooling" in params, f"{class_name}._pool_embeddings missing pooling"

    # ---- Group A sparse: flash adapters with _extract_texts only ----

    _SPARSE_FLASH_ADAPTERS: ClassVar[list[tuple[str, str]]] = [
        ("splade_flash", "SPLADEFlashAdapter"),
        ("gte_sparse_flash", "GTESparseFlashAdapter"),
    ]

    @pytest.mark.parametrize(
        ("module_name", "class_name"),
        _SPARSE_FLASH_ADAPTERS,
        ids=[m for m, _ in _SPARSE_FLASH_ADAPTERS],
    )
    def test_sparse_flash_extract_texts_accepts_template_params(
        self,
        module_name: str,
        class_name: str,
    ) -> None:
        """Sparse flash adapters' _extract_texts must accept query_template and doc_template."""
        cls = self._get_adapter_class(module_name, class_name)
        params = self._get_param_names(cls, "_extract_texts")
        assert "query_template" in params, f"{class_name}._extract_texts missing query_template"
        assert "doc_template" in params, f"{class_name}._extract_texts missing doc_template"

    # ---- Group B: PyTorch and SGLang ----

    def test_pytorch_format_texts_accepts_all_template_params(self) -> None:
        """PyTorchEmbeddingAdapter._format_texts must accept template and instruction params."""
        cls = self._get_adapter_class("pytorch_embedding", "PyTorchEmbeddingAdapter")
        params = self._get_param_names(cls, "_format_texts")
        assert "query_template" in params
        assert "doc_template" in params
        assert "default_instruction" in params

    def test_pytorch_apply_pooling_accepts_pooling_param(self) -> None:
        """PyTorchEmbeddingAdapter._apply_pooling must accept pooling param."""
        cls = self._get_adapter_class("pytorch_embedding", "PyTorchEmbeddingAdapter")
        params = self._get_param_names(cls, "_apply_pooling")
        assert "pooling" in params

    def test_sglang_format_texts_accepts_all_template_params(self) -> None:
        """SGLangEmbeddingAdapter._format_texts must accept template and instruction params."""
        cls = self._get_adapter_class("sglang", "SGLangEmbeddingAdapter")
        params = self._get_param_names(cls, "_format_texts")
        assert "query_template" in params
        assert "doc_template" in params
        assert "default_instruction" in params

    # ---- Group C: BGE-M3 variants (normalize only) ----

    def test_bge_m3_flash_compute_embeddings_accepts_normalize(self) -> None:
        """BGEM3FlashAdapter._compute_embeddings must accept normalize param."""
        cls = self._get_adapter_class("bge_m3_flash", "BGEM3FlashAdapter")
        params = self._get_param_names(cls, "_compute_embeddings")
        assert "normalize" in params

    def test_bge_m3_compute_embeddings_accepts_normalize(self) -> None:
        """BGEM3Adapter._compute_embeddings must accept normalize param."""
        cls = self._get_adapter_class("bge_m3", "BGEM3Adapter")
        params = self._get_param_names(cls, "_compute_embeddings")
        assert "normalize" in params

    # ---- Cross-cutting: encode() must resolve options dict ----

    _ALL_ENCODE_ADAPTERS: ClassVar[list[tuple[str, str]]] = [
        ("bert_flash", "BertFlashAdapter"),
        ("nomic_flash", "NomicFlashAdapter"),
        ("qwen2_flash", "Qwen2FlashAdapter"),
        ("rope_flash", "RoPEFlashAdapter"),
        ("xlm_roberta_flash", "XLMRobertaFlashAdapter"),
        ("splade_flash", "SPLADEFlashAdapter"),
        ("gte_sparse_flash", "GTESparseFlashAdapter"),
        ("pytorch_embedding", "PyTorchEmbeddingAdapter"),
        ("sglang", "SGLangEmbeddingAdapter"),
        ("bge_m3_flash", "BGEM3FlashAdapter"),
        ("bge_m3", "BGEM3Adapter"),
    ]

    @pytest.mark.parametrize(
        ("module_name", "class_name"),
        _ALL_ENCODE_ADAPTERS,
        ids=[m for m, _ in _ALL_ENCODE_ADAPTERS],
    )
    def test_encode_accepts_options_parameter(
        self,
        module_name: str,
        class_name: str,
    ) -> None:
        """All encoding adapters' encode() must accept an options parameter."""
        cls = self._get_adapter_class(module_name, class_name)
        params = self._get_param_names(cls, "encode")
        assert "options" in params, f"{class_name}.encode() missing 'options' parameter"

    @pytest.mark.parametrize(
        ("module_name", "class_name"),
        _ALL_ENCODE_ADAPTERS,
        ids=[m for m, _ in _ALL_ENCODE_ADAPTERS],
    )
    def test_encode_resolves_options_dict(
        self,
        module_name: str,
        class_name: str,
    ) -> None:
        """All encoding adapters' encode() must resolve the options dict (not ignore it)."""
        import inspect

        cls = self._get_adapter_class(module_name, class_name)
        source = inspect.getsource(cls.encode)
        assert "options or {}" in source or "options or dict()" in source, (
            f"{class_name}.encode() does not resolve options dict — "
            "expected 'options or {{}}' pattern. Runtime options will be silently ignored."
        )


class TestScoreRuntimeOptions:
    """Tests for score runtime options wiring.

    Verifies that all cross-encoder adapters accept options in score_pairs(),
    and that flash CE adapters consume max_length from options.
    """

    # ---- Helper ----

    @staticmethod
    def _get_adapter_class(module_name: str, class_name: str) -> type:
        """Import and return an adapter class by module and class name."""
        import importlib

        mod = importlib.import_module(f"sie_server.adapters.{module_name}")
        return getattr(mod, class_name)

    @staticmethod
    def _get_param_names(adapter_cls: type, method_name: str) -> set[str]:
        """Return the set of parameter names for a method on a class."""
        import inspect

        method = getattr(adapter_cls, method_name)
        return set(inspect.signature(method).parameters.keys())

    # ---- All CE adapters accept options ----

    _ALL_CE_ADAPTERS: ClassVar[list[tuple[str, str]]] = [
        ("cross_encoder", "CrossEncoderAdapter"),
        ("bert_flash_cross_encoder", "BertFlashCrossEncoderAdapter"),
        ("jina_flash_cross_encoder", "JinaFlashCrossEncoderAdapter"),
        ("modernbert_flash_cross_encoder", "ModernBertFlashCrossEncoderAdapter"),
        ("qwen2_flash_cross_encoder", "Qwen2FlashCrossEncoderAdapter"),
    ]

    @pytest.mark.parametrize(
        ("module_name", "class_name"),
        _ALL_CE_ADAPTERS,
        ids=[m for m, _ in _ALL_CE_ADAPTERS],
    )
    def test_score_pairs_accepts_options(
        self,
        module_name: str,
        class_name: str,
    ) -> None:
        """All cross-encoder adapters' score_pairs() must accept options kwarg."""
        cls = self._get_adapter_class(module_name, class_name)
        params = self._get_param_names(cls, "score_pairs")
        assert "options" in params, f"{class_name}.score_pairs() missing 'options' parameter"

    # ---- Flash CE adapters resolve options and consume max_length ----

    _FLASH_CE_ADAPTERS: ClassVar[list[tuple[str, str]]] = [
        ("bert_flash_cross_encoder", "BertFlashCrossEncoderAdapter"),
        ("jina_flash_cross_encoder", "JinaFlashCrossEncoderAdapter"),
        ("modernbert_flash_cross_encoder", "ModernBertFlashCrossEncoderAdapter"),
        ("qwen2_flash_cross_encoder", "Qwen2FlashCrossEncoderAdapter"),
    ]

    @pytest.mark.parametrize(
        ("module_name", "class_name"),
        _FLASH_CE_ADAPTERS,
        ids=[m for m, _ in _FLASH_CE_ADAPTERS],
    )
    def test_flash_ce_score_pairs_resolves_options(
        self,
        module_name: str,
        class_name: str,
    ) -> None:
        """Flash CE adapters' score_pairs() must resolve the options dict."""
        import inspect

        cls = self._get_adapter_class(module_name, class_name)
        source = inspect.getsource(cls.score_pairs)
        assert "options or {}" in source or "options or dict()" in source, (
            f"{class_name}.score_pairs() does not resolve options dict — expected 'options or {{{{}}}}' pattern."
        )

    @pytest.mark.parametrize(
        ("module_name", "class_name"),
        _FLASH_CE_ADAPTERS,
        ids=[m for m, _ in _FLASH_CE_ADAPTERS],
    )
    def test_flash_ce_consumes_max_length_from_options(
        self,
        module_name: str,
        class_name: str,
    ) -> None:
        """Flash CE adapters' score_pairs() must read max_length from options."""
        import inspect

        cls = self._get_adapter_class(module_name, class_name)
        source = inspect.getsource(cls.score_pairs)
        assert 'opts.get("max_length"' in source or "opts.get('max_length'" in source, (
            f"{class_name}.score_pairs() does not read max_length from options — "
            "per-request max_length override will not work."
        )

    # ---- Base class score()/score_pairs() accept options ----

    def test_base_class_score_accepts_options(self) -> None:
        """ModelAdapter.score() must accept options parameter."""
        from sie_server.adapters.base import ModelAdapter

        params = self._get_param_names(ModelAdapter, "score")
        assert "options" in params, "ModelAdapter.score() missing 'options' parameter"

    def test_base_class_score_pairs_accepts_options(self) -> None:
        """ModelAdapter.score_pairs() must accept options parameter."""
        from sie_server.adapters.base import ModelAdapter

        params = self._get_param_names(ModelAdapter, "score_pairs")
        assert "options" in params, "ModelAdapter.score_pairs() missing 'options' parameter"


class TestExtractRuntimeOptions:
    """Tests for extract runtime options wiring.

    Verifies that GLiNER reads merge_adjacent_entities from options, and that
    extract adapters properly wire runtime options.
    """

    def test_gliner_reads_merge_adjacent_entities_from_options(self) -> None:
        """GLiNER adapter reads merge_adjacent_entities from options dict."""
        import inspect

        from sie_server.adapters.gliner import GLiNERAdapter

        source = inspect.getsource(GLiNERAdapter.extract)
        assert 'opts.get("merge_adjacent_entities"' in source or "opts.get('merge_adjacent_entities'" in source, (
            "GLiNERAdapter.extract() does not read merge_adjacent_entities from options — "
            "NuNER_Zero runtime override will not work."
        )

    def test_gliner_reads_threshold_from_options(self) -> None:
        """GLiNER adapter reads threshold from options dict."""
        import inspect

        from sie_server.adapters.gliner import GLiNERAdapter

        source = inspect.getsource(GLiNERAdapter.extract)
        assert 'opts.get("threshold"' in source or "opts.get('threshold'" in source, (
            "GLiNERAdapter.extract() does not read threshold from options."
        )

    def test_gliner_reads_flat_ner_from_options(self) -> None:
        """GLiNER adapter reads flat_ner from options dict."""
        import inspect

        from sie_server.adapters.gliner import GLiNERAdapter

        source = inspect.getsource(GLiNERAdapter.extract)
        assert 'opts.get("flat_ner"' in source or "opts.get('flat_ner'" in source, (
            "GLiNERAdapter.extract() does not read flat_ner from options."
        )

    def test_gliner_constructor_default_merge_adjacent_false(self) -> None:
        """GLiNER merge_adjacent_entities defaults to False (safe default)."""
        from sie_server.adapters.gliner import GLiNERAdapter

        adapter = GLiNERAdapter("test-model")
        assert adapter._merge_adjacent_entities is False

    def test_gliner_constructor_accepts_merge_adjacent(self) -> None:
        """GLiNER constructor accepts merge_adjacent_entities=True."""
        from sie_server.adapters.gliner import GLiNERAdapter

        adapter = GLiNERAdapter("test-model", merge_adjacent_entities=True)
        assert adapter._merge_adjacent_entities is True

    def test_gliclass_reads_threshold_from_options(self) -> None:
        """GLiClass adapter reads threshold from options dict."""
        import inspect

        from sie_server.adapters.gliclass import GLiClassAdapter

        source = inspect.getsource(GLiClassAdapter.extract)
        assert 'opts.get("threshold"' in source or "opts.get('threshold'" in source, (
            "GLiClassAdapter.extract() does not read threshold from options."
        )

    def test_grounding_dino_reads_thresholds_from_options(self) -> None:
        """GroundingDINO adapter reads box_threshold and text_threshold from options."""
        import inspect

        from sie_server.adapters.grounding_dino import GroundingDINOAdapter

        source = inspect.getsource(GroundingDINOAdapter.extract)
        assert 'opts.get("box_threshold"' in source, (
            "GroundingDINOAdapter.extract() does not read box_threshold from options."
        )
        assert 'opts.get("text_threshold"' in source, (
            "GroundingDINOAdapter.extract() does not read text_threshold from options."
        )

    def test_owlv2_reads_score_threshold_from_options(self) -> None:
        """OWLv2 adapter reads score_threshold from options."""
        import inspect

        from sie_server.adapters.owlv2 import Owlv2Adapter

        source = inspect.getsource(Owlv2Adapter.extract)
        assert 'opts.get("score_threshold"' in source, (
            "Owlv2Adapter.extract() does not read score_threshold from options."
        )

    def test_florence2_reads_task_from_options(self) -> None:
        """Florence2 adapter reads task, max_new_tokens, num_beams from options."""
        import inspect

        from sie_server.adapters.florence2 import Florence2Adapter

        source = inspect.getsource(Florence2Adapter.extract)
        assert 'options.get("task"' in source or "options.get('task'" in source, (
            "Florence2Adapter.extract() does not read task from options."
        )
        assert 'options.get("max_new_tokens"' in source or "options.get('max_new_tokens'" in source, (
            "Florence2Adapter.extract() does not read max_new_tokens from options."
        )
        assert 'options.get("num_beams"' in source or "options.get('num_beams'" in source, (
            "Florence2Adapter.extract() does not read num_beams from options."
        )
