"""Tests for SIE SDK SIEClient and SIEAsyncClient."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import msgpack
import msgpack_numpy as m
import numpy as np
import pytest
from sie_sdk import RequestError, ServerError, SIEAsyncClient, SIEClient, SIEConnectionError

# Patch msgpack for numpy support
m.patch()


@pytest.fixture
def mock_httpx_client() -> MagicMock:
    """Create a mock httpx.Client."""
    return MagicMock(spec=httpx.Client)


class TestSIEClientInit:
    """Tests for SIEClient initialization."""

    def test_default_initialization(self) -> None:
        """Client initializes with default settings."""
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            client = SIEClient("http://localhost:8080")
            mock_client.assert_called_once()
            call_kwargs = mock_client.call_args.kwargs
            assert call_kwargs["base_url"] == "http://localhost:8080"
            assert call_kwargs["timeout"] == 30.0
            assert call_kwargs["headers"]["Content-Type"] == "application/msgpack"
            client.close()

    def test_custom_timeout(self) -> None:
        """Client respects custom timeout."""
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            client = SIEClient("http://localhost:8080", timeout_s=60.0)
            call_kwargs = mock_client.call_args.kwargs
            assert call_kwargs["timeout"] == 60.0
            client.close()

    def test_api_key_sets_auth_header(self) -> None:
        """Client sets Authorization header when api_key provided."""
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            client = SIEClient("http://localhost:8080", api_key="secret")
            call_kwargs = mock_client.call_args.kwargs
            assert call_kwargs["headers"]["Authorization"] == "Bearer secret"
            client.close()

    def test_trailing_slash_removed(self) -> None:
        """Base URL trailing slash is removed."""
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            client = SIEClient("http://localhost:8080/")
            call_kwargs = mock_client.call_args.kwargs
            assert call_kwargs["base_url"] == "http://localhost:8080"
            client.close()

    def test_gpu_stored(self) -> None:
        """Client stores gpu for use in requests."""
        with patch("sie_sdk.client.sync.httpx.Client"):
            client = SIEClient("http://localhost:8080", gpu="l4")
            assert client._default_gpu == "l4"
            client.close()

    def test_options_stored(self) -> None:
        """Client stores options for use in requests."""
        with patch("sie_sdk.client.sync.httpx.Client"):
            client = SIEClient("http://localhost:8080", options={"key": "value"})
            assert client._default_options == {"key": "value"}
            client.close()

    def test_resolve_gpu_uses_default(self) -> None:
        """_resolve_gpu returns default when gpu is None."""
        with patch("sie_sdk.client.sync.httpx.Client"):
            client = SIEClient("http://localhost:8080", gpu="a100")
            assert client._resolve_gpu(None) == "a100"
            assert client._resolve_gpu("l4") == "l4"  # Override
            client.close()

    def test_resolve_options_merges_defaults(self) -> None:
        """_resolve_options merges defaults with per-call options."""
        with patch("sie_sdk.client.sync.httpx.Client"):
            client = SIEClient("http://localhost:8080", options={"a": 1, "b": 2})
            # No per-call options returns defaults
            assert client._resolve_options(None) == {"a": 1, "b": 2}
            # Per-call options override defaults
            assert client._resolve_options({"b": 3, "c": 4}) == {"a": 1, "b": 3, "c": 4}
            client.close()

    def test_base_url_property(self) -> None:
        """Client exposes base_url as a public property."""
        with patch("sie_sdk.client.sync.httpx.Client"):
            client = SIEClient("http://localhost:8080")
            assert client.base_url == "http://localhost:8080"
            client.close()

    def test_base_url_property_strips_trailing_slash(self) -> None:
        """base_url property returns normalized URL without trailing slash."""
        with patch("sie_sdk.client.sync.httpx.Client"):
            client = SIEClient("http://localhost:8080/")
            assert client.base_url == "http://localhost:8080"
            client.close()


class TestEncode:
    """Tests for encode() method."""

    def test_encode_single_item_returns_single_result(self) -> None:
        """Single item input returns single result (not list)."""
        # Mock response with dense embedding
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = msgpack.packb(
            {
                "model": "bge-m3",
                "items": [
                    {
                        "id": "doc-1",
                        "dense": {
                            "dims": 4,
                            "dtype": "float32",
                            "values": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                        },
                    }
                ],
            },
            use_bin_type=True,
        )

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = mock_response
            client = SIEClient("http://localhost:8080")
            result = client.encode("bge-m3", {"id": "doc-1", "text": "hello"})

            # Should be single result, not list
            assert isinstance(result, dict)
            assert result["id"] == "doc-1"
            assert isinstance(result["dense"], np.ndarray)
            assert result["dense"].shape == (4,)
            client.close()

    def test_encode_list_returns_list(self) -> None:
        """List of items input returns list of results."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = msgpack.packb(
            {
                "model": "bge-m3",
                "items": [
                    {"dense": {"dims": 4, "dtype": "float32", "values": np.array([1.0, 2.0, 3.0, 4.0])}},
                    {"dense": {"dims": 4, "dtype": "float32", "values": np.array([5.0, 6.0, 7.0, 8.0])}},
                ],
            },
            use_bin_type=True,
        )

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = mock_response
            client = SIEClient("http://localhost:8080")
            results = client.encode("bge-m3", [{"text": "hello"}, {"text": "world"}])

            assert isinstance(results, list)
            assert len(results) == 2
            client.close()

    def test_encode_with_output_types(self) -> None:
        """Output types are passed correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        dense_values = np.array([1, 2, 3, 4], dtype=np.float32)
        mock_response.content = msgpack.packb(
            {"model": "bge-m3", "items": [{"dense": {"dims": 4, "dtype": "float32", "values": dense_values}}]},
            use_bin_type=True,
        )

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = mock_response
            client = SIEClient("http://localhost:8080")
            client.encode("bge-m3", {"text": "hello"}, output_types=["dense", "sparse"])

            # Check request body
            call_args = mock_client.return_value.post.call_args
            request_body = msgpack.unpackb(call_args.kwargs["content"], raw=False)
            assert request_body["params"]["output_types"] == ["dense", "sparse"]
            client.close()

    def test_encode_with_instruction(self) -> None:
        """Instruction is passed correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        dense_values = np.array([1, 2, 3, 4], dtype=np.float32)
        mock_response.content = msgpack.packb(
            {"model": "bge-m3", "items": [{"dense": {"dims": 4, "dtype": "float32", "values": dense_values}}]},
            use_bin_type=True,
        )

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = mock_response
            client = SIEClient("http://localhost:8080")
            client.encode(
                "gte-qwen2-7b",
                {"text": "What is ML?"},
                instruction="Retrieve passages that answer this question",
                options={"is_query": True},
            )

            call_args = mock_client.return_value.post.call_args
            request_body = msgpack.unpackb(call_args.kwargs["content"], raw=False)
            assert request_body["params"]["instruction"] == "Retrieve passages that answer this question"
            assert request_body["params"]["options"]["is_query"] is True
            client.close()

    def test_encode_parses_sparse_result(self) -> None:
        """Sparse results are parsed correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = msgpack.packb(
            {
                "model": "bge-m3",
                "items": [
                    {
                        "sparse": {
                            "dims": 30000,
                            "dtype": "float32",
                            "indices": np.array([100, 200, 300], dtype=np.int32),
                            "values": np.array([0.5, 0.3, 0.2], dtype=np.float32),
                        }
                    }
                ],
            },
            use_bin_type=True,
        )

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = mock_response
            client = SIEClient("http://localhost:8080")
            result = client.encode("bge-m3", {"text": "hello"})

            assert "sparse" in result
            assert isinstance(result["sparse"]["indices"], np.ndarray)
            assert isinstance(result["sparse"]["values"], np.ndarray)
            np.testing.assert_array_equal(result["sparse"]["indices"], [100, 200, 300])
            client.close()

    def test_encode_parses_multivector_result(self) -> None:
        """Multivector results are parsed correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = msgpack.packb(
            {
                "model": "bge-m3",
                "items": [
                    {
                        "multivector": {
                            "token_dims": 128,
                            "num_tokens": 3,
                            "dtype": "float32",
                            "values": np.random.default_rng().random((3, 128), dtype=np.float32),
                        }
                    }
                ],
            },
            use_bin_type=True,
        )

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = mock_response
            client = SIEClient("http://localhost:8080")
            result = client.encode("bge-m3", {"text": "hello"})

            assert "multivector" in result
            assert isinstance(result["multivector"], np.ndarray)
            assert result["multivector"].shape == (3, 128)
            client.close()


class TestErrorHandling:
    """Tests for error handling."""

    def test_connection_error(self) -> None:
        """Connection errors are wrapped as SIEConnectionError."""
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.side_effect = httpx.ConnectError("Connection refused")
            client = SIEClient("http://localhost:8080")

            with pytest.raises(SIEConnectionError, match="Failed to connect"):
                client.encode("bge-m3", {"text": "hello"})
            client.close()

    def test_timeout_error(self) -> None:
        """Timeout errors are wrapped as SIEConnectionError."""
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.side_effect = httpx.TimeoutException("Timeout")
            client = SIEClient("http://localhost:8080")

            with pytest.raises(SIEConnectionError, match="timed out"):
                client.encode("bge-m3", {"text": "hello"})
            client.close()

    def test_request_error_400(self) -> None:
        """400 errors are raised as RequestError."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"detail": {"code": "INVALID_INPUT", "message": "Invalid text"}}

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = mock_response
            client = SIEClient("http://localhost:8080")

            with pytest.raises(RequestError) as exc_info:
                client.encode("bge-m3", {"text": "hello"})

            assert exc_info.value.status_code == 400
            assert exc_info.value.code == "INVALID_INPUT"
            client.close()

    def test_server_error_500(self) -> None:
        """500 errors are raised as ServerError."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"error": {"code": "INFERENCE_ERROR", "message": "Model failed"}}

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = mock_response
            client = SIEClient("http://localhost:8080")

            with pytest.raises(ServerError) as exc_info:
                client.encode("bge-m3", {"text": "hello"})

            assert exc_info.value.status_code == 500
            assert exc_info.value.code == "INFERENCE_ERROR"
            client.close()

    def test_server_error_string_format(self) -> None:
        """500 errors with string error format are handled correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.headers = {"content-type": "application/json"}
        # Server may return error as a simple string instead of dict
        mock_response.json.return_value = {"error": "Internal server error"}

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = mock_response
            client = SIEClient("http://localhost:8080")

            with pytest.raises(ServerError) as exc_info:
                client.encode("bge-m3", {"text": "hello"})

            assert exc_info.value.status_code == 500
            assert exc_info.value.code is None  # No code in string format
            assert "Internal server error" in str(exc_info.value)
            client.close()

    def test_model_not_found_404(self) -> None:
        """404 errors are raised as RequestError."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"detail": {"code": "MODEL_NOT_FOUND", "message": "Model not found"}}

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = mock_response
            client = SIEClient("http://localhost:8080")

            with pytest.raises(RequestError) as exc_info:
                client.encode("unknown-model", {"text": "hello"})

            assert exc_info.value.status_code == 404
            client.close()


class TestListModels:
    """Tests for list_models() method."""

    def test_list_models_returns_list(self) -> None:
        """list_models returns a list of ModelInfo."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        # Match server's flat ModelInfo structure (no nested capabilities)
        mock_response.json.return_value = {
            "models": [
                {
                    "name": "bge-m3",
                    "loaded": True,
                    "inputs": ["text"],
                    "outputs": ["dense", "sparse", "multivector"],
                    "dims": {"dense": 1024, "sparse": 250002, "multivector": 1024},
                    "max_sequence_length": 8192,
                }
            ]
        }

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.get.return_value = mock_response
            client = SIEClient("http://localhost:8080")
            models = client.list_models()

            assert len(models) == 1
            assert models[0]["name"] == "bge-m3"
            assert models[0]["loaded"] is True
            assert "dense" in models[0]["outputs"]
            client.close()


class TestContextManager:
    """Tests for context manager protocol."""

    def test_context_manager_closes_client(self) -> None:
        """Context manager calls close() on exit."""
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            with SIEClient("http://localhost:8080") as client:
                assert client is not None
            mock_client.return_value.close.assert_called_once()


class TestResourceCleanup:
    """Tests for resource cleanup via weakref.finalize safety net."""

    def test_finalizer_closes_transport_on_gc(self) -> None:
        """GC closes httpx transport if close() was never called."""
        import gc

        with patch("sie_sdk.client.sync.httpx.Client") as mock_httpx:
            mock_transport = MagicMock()
            mock_httpx.return_value = mock_transport

            client = SIEClient("http://localhost:8080")
            # Simulate forgetting to close — drop all references
            del client
            gc.collect()

        mock_transport.close.assert_called_once()

    def test_close_detaches_finalizer(self) -> None:
        """Explicit close() prevents double-close from GC finalizer."""
        import gc

        with patch("sie_sdk.client.sync.httpx.Client") as mock_httpx:
            mock_transport = MagicMock()
            mock_httpx.return_value = mock_transport

            client = SIEClient("http://localhost:8080")
            client.close()
            mock_transport.close.assert_called_once()

            # Reset and verify finalizer doesn't fire on GC
            mock_transport.reset_mock()
            del client
            gc.collect()

        mock_transport.close.assert_not_called()

    def test_double_close_is_safe(self) -> None:
        """Calling close() twice does not raise."""
        with patch("sie_sdk.client.sync.httpx.Client") as mock_httpx:
            mock_transport = MagicMock()
            mock_httpx.return_value = mock_transport

            client = SIEClient("http://localhost:8080")
            client.close()
            client.close()  # Should not raise


class TestScore:
    """Tests for score() method."""

    def test_score_returns_score_result(self) -> None:
        """score() returns a ScoreResult with sorted scores."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = msgpack.packb(
            {
                "model": "bge-reranker-v2",
                "scores": [
                    {"item_id": "doc-1", "score": 0.95, "rank": 0},
                    {"item_id": "doc-2", "score": 0.72, "rank": 1},
                    {"item_id": "doc-3", "score": 0.31, "rank": 2},
                ],
            },
            use_bin_type=True,
        )

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = mock_response
            client = SIEClient("http://localhost:8080")
            result = client.score(
                "bge-reranker-v2",
                query={"text": "What is machine learning?"},
                items=[
                    {"id": "doc-1", "text": "ML is a subset of AI..."},
                    {"id": "doc-2", "text": "Python is a language..."},
                    {"id": "doc-3", "text": "Cooking recipes..."},
                ],
            )

            assert result["model"] == "bge-reranker-v2"
            assert len(result["scores"]) == 3
            # Scores should be sorted by rank
            assert result["scores"][0]["rank"] == 0
            assert result["scores"][0]["item_id"] == "doc-1"
            assert result["scores"][0]["score"] == 0.95
            client.close()

    def test_score_with_instruction(self) -> None:
        """score() passes instruction correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = msgpack.packb(
            {
                "model": "bge-reranker-v2",
                "scores": [{"item_id": "0", "score": 0.8, "rank": 0}],
            },
            use_bin_type=True,
        )

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = mock_response
            client = SIEClient("http://localhost:8080")
            client.score(
                "bge-reranker-v2",
                query={"text": "What is ML?"},
                items=[{"text": "ML info"}],
                instruction="Rank by relevance to the query",
            )

            call_args = mock_client.return_value.post.call_args
            request_body = msgpack.unpackb(call_args.kwargs["content"], raw=False)
            assert request_body["instruction"] == "Rank by relevance to the query"
            client.close()

    def test_score_with_query_id(self) -> None:
        """score() preserves query_id in result."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = msgpack.packb(
            {
                "model": "bge-reranker-v2",
                "query_id": "q-123",
                "scores": [{"item_id": "0", "score": 0.8, "rank": 0}],
            },
            use_bin_type=True,
        )

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = mock_response
            client = SIEClient("http://localhost:8080")
            result = client.score(
                "bge-reranker-v2",
                query={"id": "q-123", "text": "What is ML?"},
                items=[{"text": "ML info"}],
            )

            assert result.get("query_id") == "q-123"
            client.close()


class TestExtract:
    """Tests for extract() method."""

    def test_extract_single_item_returns_single_result(self) -> None:
        """Single item input returns single result (not list)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = msgpack.packb(
            {
                "model": "gliner-multi-v2.1",
                "items": [
                    {
                        "entities": [
                            {"text": "Apple", "label": "organization", "score": 0.98, "start": 0, "end": 5},
                            {"text": "Steve Jobs", "label": "person", "score": 0.97, "start": 22, "end": 32},
                        ]
                    }
                ],
            },
            use_bin_type=True,
        )

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = mock_response
            client = SIEClient("http://localhost:8080")
            result = client.extract(
                "gliner-multi-v2.1",
                {"text": "Apple was founded by Steve Jobs."},
                labels=["person", "organization"],
            )

            # Should be single result, not list
            assert isinstance(result, dict)
            assert "entities" in result
            assert len(result["entities"]) == 2
            assert result["entities"][0]["text"] == "Apple"
            assert result["entities"][0]["label"] == "organization"
            client.close()

    def test_extract_list_returns_list(self) -> None:
        """List of items input returns list of results."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = msgpack.packb(
            {
                "model": "gliner-multi-v2.1",
                "items": [
                    {"entities": [{"text": "Apple", "label": "org", "score": 0.9, "start": 0, "end": 5}]},
                    {"entities": [{"text": "Tesla", "label": "org", "score": 0.95, "start": 0, "end": 5}]},
                ],
            },
            use_bin_type=True,
        )

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = mock_response
            client = SIEClient("http://localhost:8080")
            results = client.extract(
                "gliner-multi-v2.1",
                [{"text": "Apple info"}, {"text": "Tesla info"}],
                labels=["org"],
            )

            assert isinstance(results, list)
            assert len(results) == 2
            client.close()

    def test_extract_with_labels(self) -> None:
        """extract() passes labels correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = msgpack.packb(
            {"model": "gliner", "items": [{"entities": []}]},
            use_bin_type=True,
        )

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = mock_response
            client = SIEClient("http://localhost:8080")
            client.extract(
                "gliner",
                {"text": "Test"},
                labels=["person", "organization", "location"],
            )

            call_args = mock_client.return_value.post.call_args
            request_body = msgpack.unpackb(call_args.kwargs["content"], raw=False)
            assert request_body["params"]["labels"] == ["person", "organization", "location"]
            client.close()

    def test_extract_preserves_item_id(self) -> None:
        """extract() preserves item IDs in results."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = msgpack.packb(
            {
                "model": "gliner",
                "items": [{"id": "doc-123", "entities": []}],
            },
            use_bin_type=True,
        )

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = mock_response
            client = SIEClient("http://localhost:8080")
            result = client.extract("gliner", {"id": "doc-123", "text": "Test"})

            assert result.get("id") == "doc-123"
            client.close()


# =============================================================================
# SIEAsyncClient Tests
# =============================================================================


class TestSIEAsyncClientInit:
    """Tests for SIEAsyncClient initialization."""

    @pytest.mark.asyncio
    async def test_default_initialization(self) -> None:
        """Async client initializes with default settings."""
        with patch("sie_sdk.client.async_.httpx.AsyncClient") as mock_client:
            mock_client.return_value.aclose = AsyncMock()
            client = SIEAsyncClient("http://localhost:8080")
            mock_client.assert_called_once()
            call_kwargs = mock_client.call_args.kwargs
            assert call_kwargs["base_url"] == "http://localhost:8080"
            assert call_kwargs["timeout"] == 30.0
            assert call_kwargs["headers"]["Content-Type"] == "application/msgpack"
            await client.close()

    @pytest.mark.asyncio
    async def test_custom_timeout(self) -> None:
        """Async client respects custom timeout."""
        with patch("sie_sdk.client.async_.httpx.AsyncClient") as mock_client:
            mock_client.return_value.aclose = AsyncMock()
            client = SIEAsyncClient("http://localhost:8080", timeout_s=60.0)
            call_kwargs = mock_client.call_args.kwargs
            assert call_kwargs["timeout"] == 60.0
            await client.close()

    @pytest.mark.asyncio
    async def test_api_key_sets_auth_header(self) -> None:
        """Async client sets Authorization header when api_key provided."""
        with patch("sie_sdk.client.async_.httpx.AsyncClient") as mock_client:
            mock_client.return_value.aclose = AsyncMock()
            client = SIEAsyncClient("http://localhost:8080", api_key="secret")
            call_kwargs = mock_client.call_args.kwargs
            assert call_kwargs["headers"]["Authorization"] == "Bearer secret"
            await client.close()

    @pytest.mark.asyncio
    async def test_base_url_property(self) -> None:
        """Async client exposes base_url as a public property."""
        with patch("sie_sdk.client.async_.httpx.AsyncClient") as mock_client:
            mock_client.return_value.aclose = AsyncMock()
            client = SIEAsyncClient("http://localhost:8080")
            assert client.base_url == "http://localhost:8080"
            await client.close()


class TestAsyncEncode:
    """Tests for async encode() method."""

    @pytest.mark.asyncio
    async def test_encode_single_item_returns_single_result(self) -> None:
        """Single item input returns single result (not list)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = msgpack.packb(
            {
                "model": "bge-m3",
                "items": [
                    {
                        "id": "doc-1",
                        "dense": {
                            "dims": 4,
                            "dtype": "float32",
                            "values": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                        },
                    }
                ],
            },
            use_bin_type=True,
        )

        with patch("sie_sdk.client.async_.httpx.AsyncClient") as mock_client:
            mock_client.return_value.post = AsyncMock(return_value=mock_response)
            mock_client.return_value.aclose = AsyncMock()
            client = SIEAsyncClient("http://localhost:8080")
            result = await client.encode("bge-m3", {"id": "doc-1", "text": "hello"})

            # Should be single result, not list
            assert isinstance(result, dict)
            assert result["id"] == "doc-1"
            assert isinstance(result["dense"], np.ndarray)
            assert result["dense"].shape == (4,)
            await client.close()

    @pytest.mark.asyncio
    async def test_encode_list_returns_list(self) -> None:
        """List of items input returns list of results."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = msgpack.packb(
            {
                "model": "bge-m3",
                "items": [
                    {"dense": {"dims": 4, "dtype": "float32", "values": np.array([1.0, 2.0, 3.0, 4.0])}},
                    {"dense": {"dims": 4, "dtype": "float32", "values": np.array([5.0, 6.0, 7.0, 8.0])}},
                ],
            },
            use_bin_type=True,
        )

        with patch("sie_sdk.client.async_.httpx.AsyncClient") as mock_client:
            mock_client.return_value.post = AsyncMock(return_value=mock_response)
            mock_client.return_value.aclose = AsyncMock()
            client = SIEAsyncClient("http://localhost:8080")
            results = await client.encode("bge-m3", [{"text": "hello"}, {"text": "world"}])

            assert isinstance(results, list)
            assert len(results) == 2
            await client.close()


class TestAsyncListModels:
    """Tests for async list_models() method."""

    @pytest.mark.asyncio
    async def test_list_models_returns_list(self) -> None:
        """list_models returns a list of ModelInfo."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {
                    "name": "bge-m3",
                    "loaded": True,
                    "capabilities": {
                        "inputs": ["text"],
                        "outputs": ["dense", "sparse", "multivector"],
                    },
                    "dims": {"dense": 1024, "sparse": 250002, "multivector": 1024},
                }
            ]
        }

        with patch("sie_sdk.client.async_.httpx.AsyncClient") as mock_client:
            mock_client.return_value.get = AsyncMock(return_value=mock_response)
            mock_client.return_value.aclose = AsyncMock()
            client = SIEAsyncClient("http://localhost:8080")
            models = await client.list_models()

            assert len(models) == 1
            assert models[0]["name"] == "bge-m3"
            assert models[0]["loaded"] is True
            await client.close()


class TestAsyncScore:
    """Tests for async score() method."""

    @pytest.mark.asyncio
    async def test_score_returns_score_result(self) -> None:
        """score() returns a ScoreResult with sorted scores."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = msgpack.packb(
            {
                "model": "bge-reranker-v2",
                "scores": [
                    {"item_id": "doc-1", "score": 0.95, "rank": 0},
                    {"item_id": "doc-2", "score": 0.72, "rank": 1},
                ],
            },
            use_bin_type=True,
        )

        with patch("sie_sdk.client.async_.httpx.AsyncClient") as mock_client:
            mock_client.return_value.post = AsyncMock(return_value=mock_response)
            mock_client.return_value.aclose = AsyncMock()
            client = SIEAsyncClient("http://localhost:8080")
            result = await client.score(
                "bge-reranker-v2",
                query={"text": "What is ML?"},
                items=[{"id": "doc-1", "text": "ML info"}, {"id": "doc-2", "text": "Other"}],
            )

            assert result["model"] == "bge-reranker-v2"
            assert len(result["scores"]) == 2
            assert result["scores"][0]["item_id"] == "doc-1"
            await client.close()


class TestAsyncExtract:
    """Tests for async extract() method."""

    @pytest.mark.asyncio
    async def test_extract_single_item_returns_single_result(self) -> None:
        """Single item input returns single result (not list)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = msgpack.packb(
            {
                "model": "gliner",
                "items": [
                    {
                        "entities": [
                            {"text": "Apple", "label": "org", "score": 0.98, "start": 0, "end": 5},
                        ]
                    }
                ],
            },
            use_bin_type=True,
        )

        with patch("sie_sdk.client.async_.httpx.AsyncClient") as mock_client:
            mock_client.return_value.post = AsyncMock(return_value=mock_response)
            mock_client.return_value.aclose = AsyncMock()
            client = SIEAsyncClient("http://localhost:8080")
            result = await client.extract(
                "gliner",
                {"text": "Apple founded by Steve Jobs."},
                labels=["org", "person"],
            )

            assert isinstance(result, dict)
            assert "entities" in result
            assert len(result["entities"]) == 1
            assert result["entities"][0]["text"] == "Apple"
            await client.close()

    @pytest.mark.asyncio
    async def test_extract_list_returns_list(self) -> None:
        """List of items input returns list of results."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = msgpack.packb(
            {
                "model": "gliner",
                "items": [
                    {"entities": [{"text": "Apple", "label": "org", "score": 0.9, "start": 0, "end": 5}]},
                    {"entities": [{"text": "Tesla", "label": "org", "score": 0.95, "start": 0, "end": 5}]},
                ],
            },
            use_bin_type=True,
        )

        with patch("sie_sdk.client.async_.httpx.AsyncClient") as mock_client:
            mock_client.return_value.post = AsyncMock(return_value=mock_response)
            mock_client.return_value.aclose = AsyncMock()
            client = SIEAsyncClient("http://localhost:8080")
            results = await client.extract(
                "gliner",
                [{"text": "Apple info"}, {"text": "Tesla info"}],
                labels=["org"],
            )

            assert isinstance(results, list)
            assert len(results) == 2
            await client.close()


class TestAsyncContextManager:
    """Tests for async context manager protocol."""

    @pytest.mark.asyncio
    async def test_async_context_manager_closes_client(self) -> None:
        """Async context manager calls close() on exit."""
        with patch("sie_sdk.client.async_.httpx.AsyncClient") as mock_client:
            mock_client.return_value.aclose = AsyncMock()
            async with SIEAsyncClient("http://localhost:8080") as client:
                assert client is not None
            mock_client.return_value.aclose.assert_called_once()


class TestAsyncResourceCleanup:
    """Tests for async client resource cleanup warnings."""

    def test_unclosed_async_client_warns(self) -> None:
        """Unclosed SIEAsyncClient emits ResourceWarning on GC."""
        import gc
        import warnings

        with patch("sie_sdk.client.async_.httpx.AsyncClient"):
            client = SIEAsyncClient("http://localhost:8080")

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                del client
                gc.collect()

            resource_warnings = [x for x in w if issubclass(x.category, ResourceWarning)]
            assert len(resource_warnings) == 1
            assert "Unclosed" in str(resource_warnings[0].message)

    @pytest.mark.asyncio
    async def test_closed_async_client_no_warning(self) -> None:
        """Properly closed SIEAsyncClient does not warn."""
        import gc
        import warnings

        with patch("sie_sdk.client.async_.httpx.AsyncClient") as mock_client:
            mock_client.return_value.aclose = AsyncMock()
            client = SIEAsyncClient("http://localhost:8080")
            await client.close()

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                del client
                gc.collect()

            resource_warnings = [x for x in w if issubclass(x.category, ResourceWarning)]
            assert len(resource_warnings) == 0


class TestAsyncErrorHandling:
    """Tests for async error handling."""

    @pytest.mark.asyncio
    async def test_connection_error(self) -> None:
        """Connection errors are wrapped as SIEConnectionError."""
        with patch("sie_sdk.client.async_.httpx.AsyncClient") as mock_client:
            mock_client.return_value.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_client.return_value.aclose = AsyncMock()
            client = SIEAsyncClient("http://localhost:8080")

            with pytest.raises(SIEConnectionError, match="Failed to connect"):
                await client.encode("bge-m3", {"text": "hello"})
            await client.close()

    @pytest.mark.asyncio
    async def test_timeout_error(self) -> None:
        """Timeout errors are wrapped as SIEConnectionError."""
        with patch("sie_sdk.client.async_.httpx.AsyncClient") as mock_client:
            mock_client.return_value.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
            mock_client.return_value.aclose = AsyncMock()
            client = SIEAsyncClient("http://localhost:8080")

            with pytest.raises(SIEConnectionError, match="timed out"):
                await client.encode("bge-m3", {"text": "hello"})
            await client.close()

    @pytest.mark.asyncio
    async def test_request_error_400(self) -> None:
        """400 errors are raised as RequestError."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"detail": {"code": "INVALID_INPUT", "message": "Invalid text"}}

        with patch("sie_sdk.client.async_.httpx.AsyncClient") as mock_client:
            mock_client.return_value.post = AsyncMock(return_value=mock_response)
            mock_client.return_value.aclose = AsyncMock()
            client = SIEAsyncClient("http://localhost:8080")

            with pytest.raises(RequestError) as exc_info:
                await client.encode("bge-m3", {"text": "hello"})

            assert exc_info.value.status_code == 400
            assert exc_info.value.code == "INVALID_INPUT"
            await client.close()

    @pytest.mark.asyncio
    async def test_server_error_500(self) -> None:
        """500 errors are raised as ServerError."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"error": {"code": "INFERENCE_ERROR", "message": "Model failed"}}

        with patch("sie_sdk.client.async_.httpx.AsyncClient") as mock_client:
            mock_client.return_value.post = AsyncMock(return_value=mock_response)
            mock_client.return_value.aclose = AsyncMock()
            client = SIEAsyncClient("http://localhost:8080")

            with pytest.raises(ServerError) as exc_info:
                await client.encode("bge-m3", {"text": "hello"})

            assert exc_info.value.status_code == 500
            assert exc_info.value.code == "INFERENCE_ERROR"
            await client.close()


class TestProvisioningRetry:
    """Tests for 202 provisioning retry functionality."""

    def test_machine_profile_header_sent_when_gpu_specified(self) -> None:
        """X-SIE-MACHINE-PROFILE header is sent when gpu parameter provided."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/msgpack"}
        mock_response.content = msgpack.packb({"items": [{"dense": {"values": np.zeros(1024)}}]}, use_bin_type=True)

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(return_value=mock_response)
            client = SIEClient("http://localhost:8080")

            client.encode("bge-m3", {"text": "hello"}, gpu="l4")

            # Check that X-SIE-MACHINE-PROFILE header was sent
            call_args = mock_client.return_value.post.call_args
            assert call_args.kwargs["headers"]["X-SIE-MACHINE-PROFILE"] == "l4"
            client.close()

    def test_202_raises_provisioning_error_without_wait(self) -> None:
        """202 response raises ProvisioningError when wait_for_capacity=False."""
        from sie_sdk import ProvisioningError

        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_response.headers = {"Retry-After": "30"}

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(return_value=mock_response)
            client = SIEClient("http://localhost:8080")

            with pytest.raises(ProvisioningError) as exc_info:
                client.encode("bge-m3", {"text": "hello"}, gpu="l4")

            assert exc_info.value.gpu == "l4"
            assert exc_info.value.retry_after == 30.0
            client.close()

    def test_202_retries_with_wait_for_capacity(self) -> None:
        """202 response is retried when wait_for_capacity=True."""
        # First response is 202, second is 200
        mock_response_202 = MagicMock()
        mock_response_202.status_code = 202
        mock_response_202.headers = {"Retry-After": "0.01"}  # Short delay for test

        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        mock_response_200.headers = {"content-type": "application/msgpack"}
        mock_response_200.content = msgpack.packb({"items": [{"dense": {"values": np.zeros(1024)}}]}, use_bin_type=True)

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(side_effect=[mock_response_202, mock_response_200])
            client = SIEClient("http://localhost:8080")

            result = client.encode(
                "bge-m3",
                {"text": "hello"},
                gpu="l4",
                wait_for_capacity=True,
                provision_timeout_s=0.2,
            )

            # Should have retried and succeeded
            assert "dense" in result
            assert mock_client.return_value.post.call_count == 2
            client.close()

    def test_202_timeout_raises_provisioning_error(self) -> None:
        """Provisioning timeout raises ProvisioningError."""
        from sie_sdk import ProvisioningError

        mock_response_202 = MagicMock()
        mock_response_202.status_code = 202
        mock_response_202.headers = {"Retry-After": "0.01"}

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            # Always return 202
            mock_client.return_value.post = MagicMock(return_value=mock_response_202)
            client = SIEClient("http://localhost:8080")

            with pytest.raises(ProvisioningError) as exc_info:
                client.encode(
                    "bge-m3",
                    {"text": "hello"},
                    gpu="l4",
                    wait_for_capacity=True,
                    provision_timeout_s=0.05,  # Very short timeout
                )

            assert "timeout" in str(exc_info.value).lower()
            client.close()

    def test_retry_after_header_parsed(self) -> None:
        """Retry-After header is correctly parsed."""
        from sie_sdk import ProvisioningError

        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_response.headers = {"Retry-After": "60"}

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(return_value=mock_response)
            client = SIEClient("http://localhost:8080")

            with pytest.raises(ProvisioningError) as exc_info:
                client.encode("bge-m3", {"text": "hello"}, gpu="l4")

            assert exc_info.value.retry_after == 60.0
            client.close()

    def test_missing_retry_after_uses_default(self) -> None:
        """Missing Retry-After header uses None."""
        from sie_sdk import ProvisioningError

        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_response.headers = {}  # No Retry-After

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(return_value=mock_response)
            client = SIEClient("http://localhost:8080")

            with pytest.raises(ProvisioningError) as exc_info:
                client.encode("bge-m3", {"text": "hello"}, gpu="l4")

            assert exc_info.value.retry_after is None
            client.close()


class TestModelLoadingRetry:
    """Tests for 503 MODEL_LOADING retry functionality."""

    def test_503_model_loading_retries_until_success(self) -> None:
        """503 with MODEL_LOADING error code is retried until model is loaded."""
        from sie_sdk.client._shared import MODEL_LOADING_ERROR_CODE

        # First response is 503 MODEL_LOADING, second is 200
        mock_response_503 = MagicMock()
        mock_response_503.status_code = 503
        mock_response_503.headers = {"Retry-After": "0.01", "content-type": "application/json"}
        mock_response_503.json.return_value = {"detail": {"code": MODEL_LOADING_ERROR_CODE, "message": "Model loading"}}

        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        mock_response_200.headers = {"content-type": "application/msgpack"}
        mock_response_200.content = msgpack.packb(
            {"items": [{"dense": {"dims": 4, "values": np.zeros(4)}}]}, use_bin_type=True
        )

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(side_effect=[mock_response_503, mock_response_200])
            client = SIEClient("http://localhost:8080")

            result = client.encode("bge-m3", {"text": "hello"})

            # Should have retried and succeeded
            assert "dense" in result
            assert mock_client.return_value.post.call_count == 2
            client.close()

    def test_503_model_loading_timeout_raises_error(self) -> None:
        """MODEL_LOADING retry timeout raises timeout error after provision_timeout_s exceeded.

        Either ModelLoadingError (timeout during retry) or ProvisioningError
        (pre-request timeout check) are valid - both indicate the timeout was enforced.
        """
        from sie_sdk import ModelLoadingError, ProvisioningError
        from sie_sdk.client._shared import MODEL_LOADING_ERROR_CODE

        mock_response_503 = MagicMock()
        mock_response_503.status_code = 503
        mock_response_503.headers = {"Retry-After": "0.01", "content-type": "application/json"}
        mock_response_503.json.return_value = {"detail": {"code": MODEL_LOADING_ERROR_CODE, "message": "Model loading"}}

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            # Always return 503 MODEL_LOADING
            mock_client.return_value.post = MagicMock(return_value=mock_response_503)
            client = SIEClient("http://localhost:8080")

            # Use a very short provision_timeout_s to trigger timeout quickly
            with pytest.raises((ModelLoadingError, ProvisioningError)) as exc_info:
                client.encode("bge-m3", {"text": "hello"}, provision_timeout_s=0.05)

            assert "timeout" in str(exc_info.value).lower()
            client.close()

    def test_503_non_model_loading_not_retried(self) -> None:
        """503 without MODEL_LOADING code raises ServerError immediately."""
        mock_response_503 = MagicMock()
        mock_response_503.status_code = 503
        mock_response_503.headers = {"content-type": "application/json"}
        mock_response_503.json.return_value = {"error": {"code": "OVERLOADED", "message": "Server overloaded"}}

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(return_value=mock_response_503)
            client = SIEClient("http://localhost:8080")

            with pytest.raises(ServerError) as exc_info:
                client.encode("bge-m3", {"text": "hello"})

            assert exc_info.value.status_code == 503
            assert exc_info.value.code == "OVERLOADED"
            # Should not have retried
            assert mock_client.return_value.post.call_count == 1
            client.close()

    def test_model_loading_retry_respects_retry_after_header(self) -> None:
        """MODEL_LOADING retry uses Retry-After header value."""
        from sie_sdk.client._shared import MODEL_LOADING_ERROR_CODE

        mock_response_503 = MagicMock()
        mock_response_503.status_code = 503
        mock_response_503.headers = {"Retry-After": "0.05", "content-type": "application/json"}
        mock_response_503.json.return_value = {"detail": {"code": MODEL_LOADING_ERROR_CODE, "message": "Model loading"}}

        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        mock_response_200.headers = {"content-type": "application/msgpack"}
        mock_response_200.content = msgpack.packb(
            {"items": [{"dense": {"dims": 4, "values": np.zeros(4)}}]}, use_bin_type=True
        )

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(side_effect=[mock_response_503, mock_response_200])
            with patch("sie_sdk.client.sync.time.sleep") as mock_sleep:
                client = SIEClient("http://localhost:8080")
                client.encode("bge-m3", {"text": "hello"})

                # Should have slept with the Retry-After value
                mock_sleep.assert_called_with(0.05)
            client.close()


class TestAsyncModelLoadingRetry:
    """Tests for async 503 MODEL_LOADING retry functionality."""

    @pytest.mark.asyncio
    async def test_503_model_loading_retries_until_success(self) -> None:
        """503 with MODEL_LOADING error code is retried until model is loaded."""
        from sie_sdk.client._shared import MODEL_LOADING_ERROR_CODE

        mock_response_503 = MagicMock()
        mock_response_503.status_code = 503
        mock_response_503.headers = {"Retry-After": "0.01", "content-type": "application/json"}
        mock_response_503.json.return_value = {"detail": {"code": MODEL_LOADING_ERROR_CODE, "message": "Model loading"}}

        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        mock_response_200.headers = {"content-type": "application/msgpack"}
        mock_response_200.content = msgpack.packb(
            {"items": [{"dense": {"dims": 4, "values": np.zeros(4)}}]}, use_bin_type=True
        )

        with patch("sie_sdk.client.async_.httpx.AsyncClient") as mock_client:
            mock_client.return_value.post = AsyncMock(side_effect=[mock_response_503, mock_response_200])
            mock_client.return_value.aclose = AsyncMock()
            client = SIEAsyncClient("http://localhost:8080")

            result = await client.encode("bge-m3", {"text": "hello"})

            # Should have retried and succeeded
            assert "dense" in result
            assert mock_client.return_value.post.call_count == 2
            await client.close()

    @pytest.mark.asyncio
    async def test_503_model_loading_timeout_raises_error(self) -> None:
        """MODEL_LOADING retry timeout raises timeout error after provision_timeout_s exceeded.

        Either ModelLoadingError (timeout during retry) or ProvisioningError
        (pre-request timeout check) are valid - both indicate the timeout was enforced.
        """
        from sie_sdk import ModelLoadingError, ProvisioningError
        from sie_sdk.client._shared import MODEL_LOADING_ERROR_CODE

        mock_response_503 = MagicMock()
        mock_response_503.status_code = 503
        mock_response_503.headers = {"Retry-After": "0.01", "content-type": "application/json"}
        mock_response_503.json.return_value = {"detail": {"code": MODEL_LOADING_ERROR_CODE, "message": "Model loading"}}

        with patch("sie_sdk.client.async_.httpx.AsyncClient") as mock_client:
            mock_client.return_value.post = AsyncMock(return_value=mock_response_503)
            mock_client.return_value.aclose = AsyncMock()
            client = SIEAsyncClient("http://localhost:8080")

            # Use a very short provision_timeout_s to trigger timeout quickly
            with pytest.raises((ModelLoadingError, ProvisioningError)) as exc_info:
                await client.encode("bge-m3", {"text": "hello"}, provision_timeout_s=0.05)

            assert "timeout" in str(exc_info.value).lower()
            await client.close()


class TestTimeoutEnforcement:
    """Tests for timeout enforcement safeguards.

    These tests verify that the provision_timeout_s is enforced even when
    individual HTTP requests might block for longer than expected.
    """

    def test_per_request_timeout_capped_to_remaining_provision_time(self) -> None:
        """Per-request timeout is capped to remaining provision time.

        This prevents a single hanging request from exceeding the overall
        provision_timeout_s. The safeguard was added to prevent scenarios where
        httpx timeout > provision_timeout, causing requests to block indefinitely.
        """
        from sie_sdk.client._shared import MODEL_LOADING_ERROR_CODE

        mock_response_503 = MagicMock()
        mock_response_503.status_code = 503
        mock_response_503.headers = {"Retry-After": "0.01", "content-type": "application/json"}
        mock_response_503.json.return_value = {"detail": {"code": MODEL_LOADING_ERROR_CODE, "message": "Model loading"}}

        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        mock_response_200.headers = {"content-type": "application/msgpack"}
        mock_response_200.content = msgpack.packb(
            {"items": [{"dense": {"dims": 4, "values": np.zeros(4)}}]}, use_bin_type=True
        )

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(side_effect=[mock_response_503, mock_response_200])

            # Create client with LONG httpx timeout (300s)
            # but SHORT provision_timeout (0.5s)
            client = SIEClient("http://localhost:8080", timeout_s=300.0)
            client.encode("bge-m3", {"text": "hello"}, provision_timeout_s=0.5)

            # Verify that per-request timeout was passed to httpx
            # (should be capped to remaining provision time, not 300s)
            calls = mock_client.return_value.post.call_args_list
            for call in calls:
                # The timeout kwarg should be present and <= provision_timeout_s
                timeout_used = call.kwargs.get("timeout")
                assert timeout_used is not None, "Per-request timeout should be set"
                assert timeout_used <= 0.5, f"Request timeout {timeout_used}s should be <= provision_timeout 0.5s"

            client.close()

    def test_provision_timeout_enforced_across_retries(self) -> None:
        """Provision timeout is enforced across multiple retries.

        Even if individual requests complete quickly (returning 503 MODEL_LOADING),
        the cumulative wall-clock time is tracked and the operation times out
        after provision_timeout_s.
        """
        import time

        from sie_sdk import ModelLoadingError, ProvisioningError
        from sie_sdk.client._shared import MODEL_LOADING_ERROR_CODE

        mock_response_503 = MagicMock()
        mock_response_503.status_code = 503
        mock_response_503.headers = {"Retry-After": "0.01", "content-type": "application/json"}
        mock_response_503.json.return_value = {"detail": {"code": MODEL_LOADING_ERROR_CODE, "message": "Model loading"}}

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            # Always return 503 MODEL_LOADING (model never finishes loading)
            mock_client.return_value.post = MagicMock(return_value=mock_response_503)
            client = SIEClient("http://localhost:8080")

            start_time = time.monotonic()
            provision_timeout = 0.01  # 100ms - enough for several retries

            # Either ModelLoadingError (timeout during retry) or ProvisioningError
            # (timeout check before request) are valid timeout behaviors
            with pytest.raises((ModelLoadingError, ProvisioningError)):
                client.encode("bge-m3", {"text": "hello"}, provision_timeout_s=provision_timeout)

            elapsed = time.monotonic() - start_time

            # The operation should complete within a reasonable margin of the timeout
            # Allow some overhead for test execution
            assert elapsed < provision_timeout + 0.15, (
                f"Operation took {elapsed:.2f}s, should timeout around {provision_timeout}s"
            )

            client.close()

    def test_httpx_timeout_exception_respects_provision_timeout(self) -> None:
        """httpx.TimeoutException is retried but respects provision_timeout_s.

        When httpx times out (e.g., server hanging), the SDK retries but still
        enforces the overall provision_timeout_s.
        """
        import time

        from sie_sdk import SIEConnectionError

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            # Always raise httpx.TimeoutException
            mock_client.return_value.post = MagicMock(side_effect=httpx.TimeoutException("Request timed out"))
            client = SIEClient("http://localhost:8080")

            start_time = time.monotonic()
            provision_timeout = 0.15

            # Without wait_for_capacity, timeout is not retried
            with pytest.raises(SIEConnectionError, match="timed out"):
                client.encode("bge-m3", {"text": "hello"}, provision_timeout_s=provision_timeout)

            elapsed = time.monotonic() - start_time

            # Should fail quickly on first timeout (no retry without wait_for_capacity)
            assert elapsed < 0.5, f"Should fail quickly, took {elapsed:.2f}s"

            client.close()

    def test_httpx_timeout_retried_with_wait_for_capacity(self) -> None:
        """httpx.TimeoutException is retried when wait_for_capacity=True.

        The SDK retries on httpx timeout but enforces provision_timeout_s.
        """
        import time

        from sie_sdk import ProvisioningError

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            # Always raise httpx.TimeoutException
            mock_client.return_value.post = MagicMock(side_effect=httpx.TimeoutException("Request timed out"))
            client = SIEClient("http://localhost:8080")

            start_time = time.monotonic()
            provision_timeout = 0.05

            # With wait_for_capacity, timeout is retried until provision_timeout
            with pytest.raises(ProvisioningError, match="exceeded"):
                client.encode(
                    "bge-m3",
                    {"text": "hello"},
                    wait_for_capacity=True,
                    provision_timeout_s=provision_timeout,
                )

            elapsed = time.monotonic() - start_time

            # Should timeout around provision_timeout (with some overhead)
            assert elapsed < provision_timeout + 0.1, (
                f"Operation took {elapsed:.2f}s, should timeout around {provision_timeout}s"
            )
