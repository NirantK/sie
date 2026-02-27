from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import msgpack
import numpy as np
import pytest
from sie_sdk import RequestError, ServerError, SIEAsyncClient, SIEConnectionError


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
