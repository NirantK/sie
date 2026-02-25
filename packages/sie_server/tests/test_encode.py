import json
import math
from typing import Any
from unittest.mock import MagicMock

import msgpack
import msgpack_numpy as m
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_server.api.encode import router as encode_router
from sie_server.config.model import (
    AdapterOptions,
    EmbeddingDim,
    EncodeTask,
    ModelConfig,
    ProfileConfig,
    Tasks,
)
from sie_server.core.registry import ModelRegistry

# Patch msgpack for numpy support
m.patch()

# Header for JSON responses (msgpack is default)
JSON_HEADERS = {"Accept": "application/json"}


def _mock_encode_impl(items: list[Any], output_types: list[str], **kwargs: Any) -> Any:
    """Implementation for mock encode - returns EncodeOutput."""
    from sie_server.core.inference_output import EncodeOutput, SparseVector

    batch_size = len(items)

    # Build dense embeddings if requested
    dense = None
    if "dense" in output_types:
        dense = np.array([[0.1, 0.2, 0.3]] * batch_size, dtype=np.float32)

    # Build sparse embeddings if requested
    sparse = None
    if "sparse" in output_types:
        sparse = [
            SparseVector(
                indices=np.array([1, 5, 10]),
                values=np.array([0.5, 0.3, 0.2], dtype=np.float32),
            )
            for _ in range(batch_size)
        ]

    # Build multivector if requested
    multivector = None
    if "multivector" in output_types:
        rng = np.random.default_rng(42)
        multivector = [rng.standard_normal((5, 128)).astype(np.float32) for _ in range(batch_size)]

    return EncodeOutput(
        dense=dense,
        sparse=sparse,
        multivector=multivector,
        batch_size=batch_size,
        dense_dim=3 if dense is not None else None,
        multivector_token_dim=128 if multivector is not None else None,
    )


@pytest.fixture
def mock_adapter() -> MagicMock:
    """Create a mock adapter that returns test embeddings."""
    adapter = MagicMock()
    adapter.encode = MagicMock(side_effect=_mock_encode_impl)
    return adapter


@pytest.fixture
def mock_registry(mock_adapter: MagicMock) -> MagicMock:
    """Create a mock registry."""
    from concurrent.futures import ThreadPoolExecutor

    from sie_server.core.postprocessor_registry import PostprocessorRegistry

    registry = MagicMock(spec=ModelRegistry)
    registry.has_model.return_value = True
    registry.is_loaded.return_value = True
    registry.is_loading.return_value = False
    registry.is_unloading.return_value = False
    registry.get.return_value = mock_adapter
    registry.get_config.return_value = ModelConfig(
        sie_id="test-model",
        hf_id="org/test",
        tasks=Tasks(
            encode=EncodeTask(
                dense=EmbeddingDim(dim=3),
                sparse=EmbeddingDim(dim=30522),
                multivector=EmbeddingDim(dim=128),
            ),
        ),
        profiles={"default": ProfileConfig(adapter_path="test:TestAdapter", max_batch_tokens=8192)},
    )
    registry.model_names = ["test-model"]
    registry.device = "cpu"
    # Mock preprocessor_registry to NOT have a tokenizer (use direct adapter path)
    preprocessor_registry = MagicMock()
    preprocessor_registry.has_tokenizer.return_value = False
    preprocessor_registry.has_preprocessor.return_value = False
    registry.preprocessor_registry = preprocessor_registry
    # Use real postprocessor_registry for quantization
    cpu_pool = ThreadPoolExecutor(max_workers=1)
    registry.postprocessor_registry = PostprocessorRegistry(cpu_pool)
    return registry


@pytest.fixture
def client(mock_registry: MagicMock) -> TestClient:
    """Create test client with mocked registry."""
    app = FastAPI()
    app.include_router(encode_router)
    app.state.registry = mock_registry
    return TestClient(app)


class TestEncodeEndpoint:
    """Tests for POST /v1/encode/{model}."""

    def test_encode_basic_json(self, client: TestClient) -> None:
        """Basic encode request returns JSON when Accept header set."""
        response = client.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "Hello world"}]},
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "test-model"
        assert len(data["items"]) == 1
        assert data["items"][0]["dense"] is not None
        assert data["items"][0]["dense"]["dims"] == 3
        assert len(data["items"][0]["dense"]["values"]) == 3

    def test_encode_basic_msgpack(self, client: TestClient) -> None:
        """Basic encode request returns msgpack by default."""
        response = client.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "Hello world"}]},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/msgpack"

        # Deserialize msgpack
        data = msgpack.unpackb(response.content, raw=False)
        assert data["model"] == "test-model"
        assert len(data["items"]) == 1
        # Values come back as numpy arrays with msgpack-numpy
        assert isinstance(data["items"][0]["dense"]["values"], np.ndarray)

    def test_encode_with_id(self, client: TestClient) -> None:
        """Item IDs are preserved in response."""
        response = client.post(
            "/v1/encode/test-model",
            json={"items": [{"id": "doc-1", "text": "Hello"}]},
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["items"][0]["id"] == "doc-1"

    def test_encode_multiple_items(self, client: TestClient) -> None:
        """Can encode multiple items at once."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [
                    {"text": "Hello"},
                    {"text": "World"},
                    {"text": "Test"},
                ]
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 3

    def test_encode_sparse_output(self, client: TestClient) -> None:
        """Can request sparse output type."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"output_types": ["sparse"]},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["items"][0]["sparse"] is not None
        assert "indices" in data["items"][0]["sparse"]
        assert "values" in data["items"][0]["sparse"]

    def test_encode_multiple_output_types(self, client: TestClient) -> None:
        """Can request multiple output types."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"output_types": ["dense", "sparse"]},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["items"][0]["dense"] is not None
        assert data["items"][0]["sparse"] is not None

    def test_encode_model_not_found(self, client: TestClient, mock_registry: MagicMock) -> None:
        """Returns 404 for unknown model."""
        mock_registry.has_model.return_value = False
        response = client.post(
            "/v1/encode/unknown-model",
            json={"items": [{"text": "Hello"}]},
        )
        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["code"] == "MODEL_NOT_FOUND"

    def test_encode_model_load_failure(self, client: TestClient, mock_registry: MagicMock) -> None:
        """Returns 503 MODEL_LOADING when model is not loaded (non-blocking load)."""
        mock_registry.is_loaded.return_value = False
        mock_registry.is_loading.return_value = False

        async def start_load_async_success(*args: Any, **kwargs: Any) -> bool:
            return True

        mock_registry.start_load_async = MagicMock(side_effect=start_load_async_success)
        response = client.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "Hello"}]},
        )
        # Non-blocking loading returns 503 + MODEL_LOADING immediately
        assert response.status_code == 503
        data = response.json()
        assert data["detail"]["code"] == "MODEL_LOADING"
        assert "loading" in data["detail"]["message"].lower()
        mock_registry.start_load_async.assert_called_once()

    def test_encode_lazy_loads_model(
        self, client: TestClient, mock_registry: MagicMock, mock_adapter: MagicMock
    ) -> None:
        """Model triggers background load on first request if not loaded."""
        mock_registry.is_loaded.return_value = False
        mock_registry.is_loading.return_value = False

        async def start_load_async_success(*args: Any, **kwargs: Any) -> bool:
            return True

        mock_registry.start_load_async = MagicMock(side_effect=start_load_async_success)
        response = client.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "Hello"}]},
            headers=JSON_HEADERS,
        )
        # Non-blocking loading returns 503 + MODEL_LOADING immediately
        assert response.status_code == 503
        mock_registry.start_load_async.assert_called_once_with("test-model", device="cpu")

    def test_encode_unsupported_output_type(self, client: TestClient, mock_registry: MagicMock) -> None:
        """Returns 400 for unsupported output type."""
        mock_registry.get_config.return_value = ModelConfig(
            sie_id="test-model",
            hf_id="org/test",
            tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=3))),
            profiles={"default": ProfileConfig(adapter_path="test:TestAdapter", max_batch_tokens=8192)},
        )
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"output_types": ["sparse"]},  # Request unsupported type
            },
        )
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["code"] == "INVALID_INPUT"
        assert "sparse" in data["detail"]["message"]

    def test_encode_with_instruction(self, client: TestClient, mock_adapter: MagicMock) -> None:
        """Instruction parameter is passed to adapter."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"instruction": "Search for documents"},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        # Verify instruction was passed
        mock_adapter.encode.assert_called_once()
        call_kwargs = mock_adapter.encode.call_args
        assert call_kwargs.kwargs["instruction"] == "Search for documents"

    def test_encode_is_query_param(self, client: TestClient, mock_adapter: MagicMock) -> None:
        """is_query option is passed to adapter via options dict."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"options": {"is_query": True}},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        call_kwargs = mock_adapter.encode.call_args
        assert call_kwargs.kwargs["is_query"] is True

    def test_encode_empty_items_rejected(self, client: TestClient) -> None:
        """Empty items list is rejected."""
        response = client.post(
            "/v1/encode/test-model",
            json={"items": []},
        )
        assert response.status_code == 400  # Custom validation error (not Pydantic)


class TestMsgpackRequests:
    """Tests for msgpack request body handling (DESIGN.md Section 4.3)."""

    def test_msgpack_request_basic(self, client: TestClient) -> None:
        """Msgpack request body is parsed correctly."""
        request_data = {"items": [{"text": "Hello world"}]}
        msgpack_body = msgpack.packb(request_data, use_bin_type=True)

        response = client.post(
            "/v1/encode/test-model",
            content=msgpack_body,
            headers={"Content-Type": "application/msgpack"},
        )
        assert response.status_code == 200
        # Response is also msgpack by default
        assert response.headers["content-type"] == "application/msgpack"
        data = msgpack.unpackb(response.content, raw=False)
        assert data["model"] == "test-model"
        assert len(data["items"]) == 1

    def test_msgpack_request_with_json_response(self, client: TestClient) -> None:
        """Msgpack request can get JSON response with Accept header."""
        request_data = {"items": [{"text": "Hello"}]}
        msgpack_body = msgpack.packb(request_data, use_bin_type=True)

        response = client.post(
            "/v1/encode/test-model",
            content=msgpack_body,
            headers={
                "Content-Type": "application/msgpack",
                "Accept": "application/json",
            },
        )
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]
        data = response.json()
        assert data["model"] == "test-model"

    def test_msgpack_request_with_params(self, client: TestClient) -> None:
        """Msgpack request with params is parsed correctly."""
        request_data = {
            "items": [{"id": "doc-1", "text": "Hello"}],
            "params": {"output_types": ["dense", "sparse"], "options": {"is_query": True}},
        }
        msgpack_body = msgpack.packb(request_data, use_bin_type=True)

        response = client.post(
            "/v1/encode/test-model",
            content=msgpack_body,
            headers={
                "Content-Type": "application/msgpack",
                "Accept": "application/json",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["items"][0]["id"] == "doc-1"
        assert data["items"][0]["dense"] is not None
        assert data["items"][0]["sparse"] is not None

    def test_msgpack_request_with_binary_image(self, client: TestClient) -> None:
        """Msgpack request can include binary image data."""
        # Simulate image bytes (would be actual image in real usage)
        fake_image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        request_data = {
            "items": [
                {
                    "text": "Describe this image",
                    "images": [{"data": fake_image_bytes, "format": "png"}],
                }
            ]
        }
        msgpack_body = msgpack.packb(request_data, use_bin_type=True)

        response = client.post(
            "/v1/encode/test-model",
            content=msgpack_body,
            headers={
                "Content-Type": "application/msgpack",
                "Accept": "application/json",
            },
        )
        # Should parse successfully (adapter will handle the image)
        assert response.status_code == 200

    def test_msgpack_request_roundtrip(self, client: TestClient) -> None:
        """Full msgpack request/response roundtrip preserves numpy arrays."""
        request_data = {"items": [{"text": "Test embedding"}]}
        msgpack_body = msgpack.packb(request_data, use_bin_type=True)

        response = client.post(
            "/v1/encode/test-model",
            content=msgpack_body,
            headers={"Content-Type": "application/msgpack"},
        )
        assert response.status_code == 200

        # Deserialize response
        data = msgpack.unpackb(response.content, raw=False)
        dense_values = data["items"][0]["dense"]["values"]

        # Values should be numpy array (not list)
        assert isinstance(dense_values, np.ndarray)
        assert dense_values.dtype == np.float32

    def test_msgpack_request_invalid_body(self, client: TestClient) -> None:
        """Invalid msgpack body returns 400."""
        response = client.post(
            "/v1/encode/test-model",
            content=b"not valid msgpack",
            headers={"Content-Type": "application/msgpack"},
        )
        assert response.status_code == 400

    def test_msgpack_request_validation_error(self, client: TestClient) -> None:
        """Msgpack request with invalid schema returns 400."""
        request_data = {"items": []}  # Empty items should fail validation
        msgpack_body = msgpack.packb(request_data, use_bin_type=True)

        response = client.post(
            "/v1/encode/test-model",
            content=msgpack_body,
            headers={"Content-Type": "application/msgpack"},
        )
        assert response.status_code == 400  # Custom validation error (not Pydantic)

    def test_x_msgpack_content_type(self, client: TestClient) -> None:
        """Alternative x-msgpack content type is also accepted."""
        request_data = {"items": [{"text": "Hello"}]}
        msgpack_body = msgpack.packb(request_data, use_bin_type=True)

        response = client.post(
            "/v1/encode/test-model",
            content=msgpack_body,
            headers={"Content-Type": "application/x-msgpack"},
        )
        assert response.status_code == 200


class TestTimingHeaders:
    """Tests for timing headers in encode responses (DESIGN.md Section 5.7)."""

    @pytest.fixture
    def mock_adapter_with_timing(self) -> MagicMock:
        """Create a mock adapter that returns test embeddings."""
        adapter = MagicMock()
        adapter.encode = MagicMock(side_effect=_mock_encode_impl)
        return adapter

    @pytest.fixture
    def mock_registry_with_worker(self, mock_adapter_with_timing: MagicMock) -> MagicMock:
        """Create a mock registry that uses the worker path with timing."""
        from unittest.mock import AsyncMock

        from sie_server.core.timing import RequestTiming
        from sie_server.core.worker import WorkerResult

        registry = MagicMock(spec=ModelRegistry)
        registry.has_model.return_value = True
        registry.is_loaded.return_value = True
        registry.is_loading.return_value = False
        registry.is_unloading.return_value = False
        registry.get.return_value = mock_adapter_with_timing
        registry.get_config.return_value = ModelConfig(
            sie_id="test-model",
            hf_id="org/test",
            tasks=Tasks(
                encode=EncodeTask(
                    dense=EmbeddingDim(dim=3),
                    sparse=EmbeddingDim(dim=30522),
                    multivector=EmbeddingDim(dim=128),
                ),
            ),
            profiles={"default": ProfileConfig(adapter_path="test:TestAdapter", max_batch_tokens=8192)},
        )
        registry.model_names = ["test-model"]
        registry.device = "cpu"

        # Set up preprocessor_registry to trigger worker path
        prepared_batch = MagicMock()
        prepared_item = MagicMock()
        prepared_item.cost = 5
        prepared_item.original_index = 0
        prepared_batch.items = [prepared_item]

        preprocessor_registry = MagicMock()
        # has_preprocessor returns True for "text", False for "image"
        preprocessor_registry.has_preprocessor.side_effect = lambda model, modality: modality == "text"
        preprocessor_registry.prepare = AsyncMock(return_value=prepared_batch)
        registry.preprocessor_registry = preprocessor_registry

        # Mock worker with timing
        timing = RequestTiming()
        timing.start_tokenization()
        timing.end_tokenization()
        timing.start_queue()
        timing.start_inference()
        timing.end_inference()
        timing.finish()

        from sie_server.core.inference_output import EncodeOutput

        worker_result = WorkerResult(
            output=EncodeOutput(
                dense=np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
                batch_size=1,
                dense_dim=3,
            ),
            timing=timing,
        )

        worker = MagicMock()
        worker.submit = AsyncMock(return_value=AsyncMock(return_value=worker_result)())
        registry.start_worker = AsyncMock(return_value=worker)

        return registry

    @pytest.fixture
    def client_with_worker(self, mock_registry_with_worker: MagicMock) -> TestClient:
        """Create test client with worker path enabled."""
        app = FastAPI()
        app.include_router(encode_router)
        app.state.registry = mock_registry_with_worker
        return TestClient(app)

    def test_timing_headers_present_json(self, client_with_worker: TestClient) -> None:
        """Timing headers are present in JSON responses."""
        response = client_with_worker.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "Hello world"}]},
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200

        # Check timing headers present
        assert "x-queue-time" in response.headers
        assert "x-tokenization-time" in response.headers
        assert "x-inference-time" in response.headers
        assert "x-total-time" in response.headers

    def test_timing_headers_present_msgpack(self, client_with_worker: TestClient) -> None:
        """Timing headers are present in msgpack responses."""
        response = client_with_worker.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "Hello world"}]},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/msgpack"

        # Check timing headers present
        assert "x-queue-time" in response.headers
        assert "x-tokenization-time" in response.headers
        assert "x-inference-time" in response.headers
        assert "x-total-time" in response.headers

    def test_timing_header_values_format(self, client_with_worker: TestClient) -> None:
        """Timing header values are formatted with 2 decimal places."""
        response = client_with_worker.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "Hello"}]},
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200

        # All timing values should be floats with 2 decimal places
        for header in ["x-queue-time", "x-tokenization-time", "x-inference-time", "x-total-time"]:
            value = response.headers[header]
            # Should be parseable as float
            float_val = float(value)
            assert float_val >= 0.0
            # Should have format X.XX (2 decimal places)
            parts = value.split(".")
            assert len(parts) == 2
            assert len(parts[1]) == 2

    def test_no_timing_headers_without_worker(self, client: TestClient) -> None:
        """No timing headers when using direct adapter path (no worker)."""
        response = client.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "Hello"}]},
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200

        # Timing headers should NOT be present (direct adapter path has no timing)
        # Note: Headers may be absent or have value "0.00" depending on implementation
        # The current implementation doesn't add headers when timing is None
        assert "x-queue-time" not in response.headers or response.headers.get("x-queue-time") == "0.00"


class TestOutputDtype:
    """Tests for output_dtype parameter (DESIGN.md Section 5.9)."""

    def test_default_dtype_is_float32(self, client: TestClient) -> None:
        """Default output dtype is float32."""
        response = client.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "Hello"}]},
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["items"][0]["dense"]["dtype"] == "float32"

    def test_explicit_float32_dtype(self, client: TestClient) -> None:
        """Explicit float32 dtype works."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"output_dtype": "float32"},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["items"][0]["dense"]["dtype"] == "float32"

    def test_float16_dtype(self, client: TestClient) -> None:
        """float16 dtype casting works."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"output_dtype": "float16"},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["items"][0]["dense"]["dtype"] == "float16"

    def test_int8_dtype(self, client: TestClient) -> None:
        """int8 dtype casting works for dense embeddings."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"output_dtype": "int8"},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["items"][0]["dense"]["dtype"] == "int8"

    def test_binary_dtype(self, client: TestClient) -> None:
        """Binary dtype casting works for dense embeddings."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"output_dtype": "binary"},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["items"][0]["dense"]["dtype"] == "binary"
        # Binary packing: 3 dims -> ceil(3/8) = 1 byte
        # But dims should still report original dimension
        assert data["items"][0]["dense"]["dims"] == 3

    def test_sparse_with_float16_dtype(self, client: TestClient) -> None:
        """Sparse embeddings support float16 dtype."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"output_types": ["sparse"], "output_dtype": "float16"},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["items"][0]["sparse"]["dtype"] == "float16"

    def test_sparse_int8_falls_back_to_float32(self, client: TestClient) -> None:
        """Sparse embeddings fall back to float32 for int8/binary."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"output_types": ["sparse"], "output_dtype": "int8"},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        # Sparse falls back to float32 for int8
        assert data["items"][0]["sparse"]["dtype"] == "float32"

    def test_sparse_binary_falls_back_to_float32(self, client: TestClient) -> None:
        """Sparse embeddings fall back to float32 for binary."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"output_types": ["sparse"], "output_dtype": "binary"},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        # Sparse falls back to float32 for binary
        assert data["items"][0]["sparse"]["dtype"] == "float32"

    def test_multivector_with_float16_dtype(self, client: TestClient) -> None:
        """Multivector embeddings support float16 dtype."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"output_types": ["multivector"], "output_dtype": "float16"},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["items"][0]["multivector"]["dtype"] == "float16"

    def test_multivector_int8_quantization(self, client: TestClient) -> None:
        """Multivector embeddings support int8 quantization."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"output_types": ["multivector"], "output_dtype": "int8"},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        # Multivector supports int8 quantization (per-token, like ColBERTv2)
        assert data["items"][0]["multivector"]["dtype"] == "int8"

    def test_multivector_binary_quantization(self, client: TestClient) -> None:
        """Multivector embeddings support binary quantization."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"output_types": ["multivector"], "output_dtype": "binary"},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        # Multivector supports binary quantization (per-token)
        assert data["items"][0]["multivector"]["dtype"] == "binary"

    def test_multiple_output_types_with_dtype(self, client: TestClient) -> None:
        """Multiple output types all respect output_dtype."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"output_types": ["dense", "sparse"], "output_dtype": "float16"},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        # Both should be float16
        assert data["items"][0]["dense"]["dtype"] == "float16"
        assert data["items"][0]["sparse"]["dtype"] == "float16"

    def test_dtype_with_msgpack_response(self, client: TestClient) -> None:
        """Output dtype works with msgpack responses."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"output_dtype": "float16"},
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/msgpack"

        data = msgpack.unpackb(response.content, raw=False)
        assert data["items"][0]["dense"]["dtype"] == "float16"
        # Values should be numpy array with float16 dtype
        values = data["items"][0]["dense"]["values"]
        assert isinstance(values, np.ndarray)
        assert values.dtype == np.float16

    def test_int8_values_are_integers(self, client: TestClient) -> None:
        """int8 dtype returns integer values."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"output_dtype": "int8"},
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/msgpack"

        data = msgpack.unpackb(response.content, raw=False)
        values = data["items"][0]["dense"]["values"]
        assert isinstance(values, np.ndarray)
        assert values.dtype == np.int8

    def test_binary_values_are_packed(self, client: TestClient) -> None:
        """Binary dtype returns packed uint8 values."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"output_dtype": "binary"},
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/msgpack"

        data = msgpack.unpackb(response.content, raw=False)
        values = data["items"][0]["dense"]["values"]
        assert isinstance(values, np.ndarray)
        assert values.dtype == np.uint8
        # Original dims = 3, packed into ceil(3/8) = 1 byte
        assert len(values) == 1


class TestProfileOutputDtype:
    """Tests for profile-based output_dtype (profile > request > default)."""

    @pytest.fixture
    def mock_registry_with_profile(self, mock_adapter: MagicMock) -> MagicMock:
        """Registry with a model that has a quantized profile."""
        from concurrent.futures import ThreadPoolExecutor

        from sie_server.core.postprocessor_registry import PostprocessorRegistry

        registry = MagicMock(spec=ModelRegistry)
        registry.has_model.return_value = True
        registry.is_loaded.return_value = True
        registry.is_loading.return_value = False
        registry.is_unloading.return_value = False
        registry.get.return_value = mock_adapter
        registry.get_config.return_value = ModelConfig(
            sie_id="test-model",
            hf_id="org/test",
            tasks=Tasks(
                encode=EncodeTask(
                    dense=EmbeddingDim(dim=3),
                    sparse=EmbeddingDim(dim=30522),
                    multivector=EmbeddingDim(dim=128),
                ),
            ),
            profiles={
                "default": ProfileConfig(adapter_path="test:TestAdapter", max_batch_tokens=8192),
                "quantized": ProfileConfig(
                    adapter_path="test:TestAdapter",
                    max_batch_tokens=8192,
                    adapter_options=AdapterOptions(runtime={"output_dtype": "int8"}),
                ),
            },
        )
        registry.model_names = ["test-model"]
        registry.device = "cpu"
        preprocessor_registry = MagicMock()
        preprocessor_registry.has_tokenizer.return_value = False
        preprocessor_registry.has_preprocessor.return_value = False
        registry.preprocessor_registry = preprocessor_registry
        # Use real postprocessor_registry for quantization
        cpu_pool = ThreadPoolExecutor(max_workers=1)
        registry.postprocessor_registry = PostprocessorRegistry(cpu_pool)
        return registry

    @pytest.fixture
    def client_with_profile(self, mock_registry_with_profile: MagicMock) -> TestClient:
        """Client with profile-aware registry."""
        app = FastAPI()
        app.include_router(encode_router)
        app.state.registry = mock_registry_with_profile
        return TestClient(app)

    def test_default_profile_uses_float32(self, client_with_profile: TestClient) -> None:
        """Default profile with no output_dtype uses float32."""
        response = client_with_profile.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "Hello"}]},
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["items"][0]["dense"]["dtype"] == "float32"

    def test_quantized_profile_uses_int8(self, client_with_profile: TestClient) -> None:
        """Quantized profile with output_dtype=int8 returns int8."""
        response = client_with_profile.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"options": {"profile": "quantized"}},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["items"][0]["dense"]["dtype"] == "int8"

    def test_request_overrides_profile(self, client_with_profile: TestClient) -> None:
        """Request output_dtype overrides profile output_dtype."""
        response = client_with_profile.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {
                    "options": {"profile": "quantized"},
                    "output_dtype": "float16",
                },
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        # Request float16 overrides profile int8
        assert data["items"][0]["dense"]["dtype"] == "float16"


class TestMachineProfileValidation:
    """Tests for X-SIE-MACHINE-PROFILE header validation."""

    def test_no_profile_header_succeeds(self, client: TestClient) -> None:
        """Request without X-SIE-MACHINE-PROFILE header proceeds normally."""
        response = client.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "Hello world"}]},
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200

    def test_profile_header_with_no_gpu_fails(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """Request with X-SIE-MACHINE-PROFILE when worker has no identity returns 400."""
        # Mock get_worker_gpu_type to return None (no GPU)
        monkeypatch.setattr(
            "sie_server.api.validation.get_worker_gpu_type",
            lambda: None,
        )
        # No SIE_MACHINE_PROFILE env var
        monkeypatch.delenv("SIE_MACHINE_PROFILE", raising=False)

        response = client.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "Hello world"}]},
            headers={**JSON_HEADERS, "X-SIE-MACHINE-PROFILE": "l4"},
        )
        assert response.status_code == 400
        data = response.json()
        assert "no GPU" in data["detail"]["message"]
        assert data["detail"]["requested_profile"] == "l4"
        assert data["detail"]["worker_identity"] is None

    def test_profile_header_mismatch_fails(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """Request with mismatched X-SIE-MACHINE-PROFILE returns 400."""
        # Mock get_worker_gpu_type to return different GPU
        monkeypatch.setattr(
            "sie_server.api.validation.get_worker_gpu_type",
            lambda: "a100-80gb",
        )
        # No SIE_MACHINE_PROFILE env var, so identity comes from detected GPU
        monkeypatch.delenv("SIE_MACHINE_PROFILE", raising=False)

        response = client.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "Hello world"}]},
            headers={**JSON_HEADERS, "X-SIE-MACHINE-PROFILE": "l4"},
        )
        assert response.status_code == 400
        data = response.json()
        assert "routing error" in data["detail"]["message"]
        assert data["detail"]["requested_profile"] == "l4"
        assert data["detail"]["worker_identity"] == "a100-80gb"

    def test_profile_header_matches_succeeds(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """Request with matching X-SIE-MACHINE-PROFILE proceeds normally."""
        # Mock get_worker_gpu_type to return matching GPU
        monkeypatch.setattr(
            "sie_server.api.validation.get_worker_gpu_type",
            lambda: "l4",
        )
        # No SIE_MACHINE_PROFILE env var, so identity comes from detected GPU
        monkeypatch.delenv("SIE_MACHINE_PROFILE", raising=False)

        response = client.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "Hello world"}]},
            headers={**JSON_HEADERS, "X-SIE-MACHINE-PROFILE": "l4"},
        )
        assert response.status_code == 200

    def test_profile_header_case_insensitive(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """Machine profile header comparison is case-insensitive."""
        # Mock get_worker_gpu_type to return lowercase GPU
        monkeypatch.setattr(
            "sie_server.api.validation.get_worker_gpu_type",
            lambda: "a100-80gb",
        )
        # No SIE_MACHINE_PROFILE env var
        monkeypatch.delenv("SIE_MACHINE_PROFILE", raising=False)

        response = client.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "Hello world"}]},
            headers={**JSON_HEADERS, "X-SIE-MACHINE-PROFILE": "A100-80GB"},
        )
        assert response.status_code == 200

    def test_env_var_overrides_detected_gpu(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """SIE_MACHINE_PROFILE env var takes precedence over detected GPU."""
        # Mock get_worker_gpu_type to return l4
        monkeypatch.setattr(
            "sie_server.api.validation.get_worker_gpu_type",
            lambda: "l4",
        )
        # Set SIE_MACHINE_PROFILE to l4-spot (K8s scenario)
        monkeypatch.setenv("SIE_MACHINE_PROFILE", "l4-spot")

        # Request for l4 should fail (worker identity is l4-spot, not l4)
        response = client.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "Hello world"}]},
            headers={**JSON_HEADERS, "X-SIE-MACHINE-PROFILE": "l4"},
        )
        assert response.status_code == 400

        # Request for l4-spot should succeed
        response = client.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "Hello world"}]},
            headers={**JSON_HEADERS, "X-SIE-MACHINE-PROFILE": "l4-spot"},
        )
        assert response.status_code == 200


class TestJsonResponseSchema:
    """Tests for JSON response schema validation and human-readability.

    Verifies that:
    - JSON responses match documented OpenAPI schemas
    - Embeddings are human-readable float lists (not binary blobs)
    - All documented fields are present
    """

    def test_json_encode_response_matches_openapi_schema(self, client: TestClient) -> None:
        """JSON response structure matches EncodeResponseModel schema."""
        response = client.post(
            "/v1/encode/test-model",
            json={"items": [{"id": "doc-1", "text": "Hello world"}]},
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()

        # Top-level required fields per EncodeResponseModel
        assert "model" in data
        assert "items" in data
        assert isinstance(data["model"], str)
        assert isinstance(data["items"], list)

        # Per-item structure per EncodeResultModel
        item = data["items"][0]
        assert "id" in item or item.get("id") is None  # Optional field
        # Dense embedding per DenseVectorModel
        assert "dense" in item
        dense = item["dense"]
        assert "dims" in dense
        assert "dtype" in dense
        assert "values" in dense
        assert isinstance(dense["dims"], int)
        assert isinstance(dense["dtype"], str)
        assert dense["dtype"] == "float32"

    def test_json_dense_values_are_human_readable_floats(self, client: TestClient) -> None:
        """Dense embedding values in JSON are human-readable float lists."""
        response = client.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "Hello world"}]},
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()

        values = data["items"][0]["dense"]["values"]
        # Must be a plain list (not binary blob, not base64)
        assert isinstance(values, list)
        # Each element must be a number
        for val in values:
            assert isinstance(val, int | float)
            # Should be readable (not NaN, not inf)
            assert not math.isnan(val)
            assert not math.isinf(val)

    def test_json_sparse_values_are_human_readable(self, client: TestClient) -> None:
        """Sparse embedding indices/values in JSON are human-readable lists."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello world"}],
                "params": {"output_types": ["sparse"]},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()

        sparse = data["items"][0]["sparse"]
        # Check SparseVectorModel schema
        assert "indices" in sparse
        assert "values" in sparse
        assert "dtype" in sparse

        # Indices must be a list of integers
        assert isinstance(sparse["indices"], list)
        for idx in sparse["indices"]:
            assert isinstance(idx, int)
            assert idx >= 0

        # Values must be a list of floats
        assert isinstance(sparse["values"], list)
        for val in sparse["values"]:
            assert isinstance(val, int | float)

    def test_json_multivector_values_are_human_readable(self, client: TestClient) -> None:
        """Multivector embedding values in JSON are human-readable nested lists."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello world"}],
                "params": {"output_types": ["multivector"]},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()

        mv = data["items"][0]["multivector"]
        # Check MultiVectorModel schema
        assert "token_dims" in mv
        assert "num_tokens" in mv
        assert "dtype" in mv
        assert "values" in mv

        assert isinstance(mv["token_dims"], int)
        assert isinstance(mv["num_tokens"], int)
        assert isinstance(mv["values"], list)

        # Values is list[list[float]] - nested structure
        for token_embedding in mv["values"]:
            assert isinstance(token_embedding, list)
            for val in token_embedding:
                assert isinstance(val, int | float)

    def test_json_response_is_valid_json_string(self, client: TestClient) -> None:
        """Response can be parsed as JSON and re-serialized."""
        response = client.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "Test"}]},
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200

        # Should not raise - content is valid JSON
        data = json.loads(response.text)
        # Re-serialization should work
        json_str = json.dumps(data, indent=2)
        assert len(json_str) > 0
        # Round-trip should preserve data
        assert json.loads(json_str) == data

    def test_json_float16_values_serialized_as_floats(self, client: TestClient) -> None:
        """float16 dtype values are serialized as readable floats in JSON."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Test"}],
                "params": {"output_dtype": "float16"},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()

        assert data["items"][0]["dense"]["dtype"] == "float16"
        values = data["items"][0]["dense"]["values"]
        # Values should still be readable numbers in JSON (not binary)
        assert isinstance(values, list)
        for val in values:
            assert isinstance(val, int | float)

    def test_json_int8_values_serialized_as_integers(self, client: TestClient) -> None:
        """int8 dtype values are serialized as readable integers in JSON."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Test"}],
                "params": {"output_dtype": "int8"},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()

        assert data["items"][0]["dense"]["dtype"] == "int8"
        values = data["items"][0]["dense"]["values"]
        # Values should be integers
        assert isinstance(values, list)
        for val in values:
            assert isinstance(val, int)
            assert -128 <= val <= 127

    def test_json_contains_all_documented_fields(self, client: TestClient) -> None:
        """JSON response contains all fields documented in OpenAPI schema."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"id": "doc-1", "text": "Hello"}],
                "params": {"output_types": ["dense", "sparse"]},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()

        # Response level - EncodeResponseModel
        assert "model" in data
        assert "items" in data
        # timing is optional

        # Item level - EncodeResultModel
        item = data["items"][0]
        assert item.get("id") == "doc-1"  # ID preserved from request

        # Dense - DenseVectorModel
        dense = item["dense"]
        assert set(dense.keys()) >= {"dims", "dtype", "values"}

        # Sparse - SparseVectorModel
        sparse = item["sparse"]
        assert set(sparse.keys()) >= {"dtype", "indices", "values"}
