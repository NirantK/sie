"""Tests for the router proxy module.

Tests cover:
- Request routing logic (GPU, pool, fallback)
- 202 provisioning responses for no capacity
- 503 responses for unconfigured GPUs and no workers
- Pool-aware routing
- Request forwarding
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_router.pools import AssignedWorker, Pool, PoolManager, PoolSpec, PoolState, PoolStatus
from sie_router.proxy import (
    _filter_headers,
    _make_provisioning_response,
    _make_unconfigured_gpu_response,
    _mask_token,
    audit_logger,
    router,
)
from sie_router.registry import WorkerRegistry
from sie_router.types import WorkerState

# =============================================================================
# Fixtures
# =============================================================================


@dataclass
class MockWorker:
    """Mock worker for testing."""

    name: str
    url: str
    gpu: str = "l4"
    machine_profile: str = "l4"  # Usually same as gpu, but can differ (e.g., "l4-spot")
    bundle: str = "default"
    models: set[str] = field(default_factory=set)
    healthy: bool = True
    qps: float = 0.0


@pytest.fixture
def app() -> FastAPI:
    """Create a test FastAPI app with proxy router."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def registry() -> MagicMock:
    """Create a mock worker registry."""
    mock = MagicMock(spec=WorkerRegistry)
    mock.healthy_workers = []
    mock.select_worker.return_value = None
    mock.get_models.return_value = {}
    return mock


@pytest.fixture
def pool_manager() -> MagicMock:
    """Create a mock pool manager."""
    mock = MagicMock(spec=PoolManager)
    mock.pools = {}
    mock.get_pool.return_value = None
    return mock


@pytest.fixture
def client(app: FastAPI, registry: MagicMock, pool_manager: MagicMock) -> TestClient:
    """Create test client with mocked dependencies."""
    app.state.registry = registry
    app.state.pool_manager = pool_manager
    return TestClient(app)


# =============================================================================
# Helper function tests
# =============================================================================


class TestFilterHeaders:
    """Tests for _filter_headers."""

    def test_filters_hop_by_hop_headers(self) -> None:
        """Should remove hop-by-hop headers."""
        headers = {
            "content-type": "application/json",
            "connection": "keep-alive",
            "keep-alive": "timeout=5",
            "host": "example.com",
            "x-custom": "value",
        }
        result = _filter_headers(headers)

        assert "content-type" in result
        assert "x-custom" in result
        assert "connection" not in result
        assert "keep-alive" not in result
        assert "host" not in result

    def test_preserves_normal_headers(self) -> None:
        """Should preserve non-hop-by-hop headers."""
        headers = {
            "content-type": "application/json",
            "authorization": "Bearer token",
            "x-sie-gpu": "l4",
        }
        result = _filter_headers(headers)

        assert result == headers


class TestMakeProvisioningResponse:
    """Tests for _make_provisioning_response."""

    def test_returns_202_status(self) -> None:
        """Should return 202 Accepted."""
        response = _make_provisioning_response("l4")
        assert response.status_code == 202

    def test_includes_retry_after_header(self) -> None:
        """Should include Retry-After header."""
        response = _make_provisioning_response("l4")
        assert "retry-after" in response.headers

    def test_includes_gpu_in_body(self) -> None:
        """Should include GPU type in response body."""
        response = _make_provisioning_response("a100-40gb")
        import json

        body = json.loads(response.body)
        assert body["gpu"] == "a100-40gb"
        assert body["status"] == "provisioning"


class TestMakeUnconfiguredGpuResponse:
    """Tests for _make_unconfigured_gpu_response."""

    def test_returns_503_status(self) -> None:
        """Should return 503 Service Unavailable."""
        response = _make_unconfigured_gpu_response("h100", ["l4", "a100-40gb"])
        assert response.status_code == 503

    def test_includes_gpu_and_configured_types(self) -> None:
        """Should include requested GPU and available types."""
        import json

        response = _make_unconfigured_gpu_response("h100", ["l4", "a100-40gb"])
        body = json.loads(response.body)

        assert body["gpu"] == "h100"
        assert body["configured_gpu_types"] == ["l4", "a100-40gb"]
        assert body["status"] == "gpu_not_configured"


# =============================================================================
# Proxy routing tests - no workers available scenarios
# =============================================================================


class TestProxyNoWorkers:
    """Tests for proxy behavior when no workers are available."""

    def test_no_workers_no_gpu_returns_503(self, client: TestClient, registry: MagicMock) -> None:
        """No workers + no GPU specified → 503."""
        registry.select_worker.return_value = None

        response = client.post(
            "/v1/encode/test-model",
            json={"text": "hello"},
        )

        assert response.status_code == 503
        assert "No healthy workers" in response.json()["detail"]["message"]

    def test_no_workers_with_gpu_returns_202(self, client: TestClient, registry: MagicMock) -> None:
        """No workers + GPU specified → 202 provisioning."""
        registry.select_worker.return_value = None

        with patch("sie_router.proxy.record_pending_demand") as mock_demand:
            with patch("sie_router.proxy.CONFIGURED_GPU_TYPES", ["l4"]):
                response = client.post(
                    "/v1/encode/test-model",
                    headers={"X-SIE-MACHINE-PROFILE": "l4"},
                    json={"text": "hello"},
                )

        assert response.status_code == 202
        assert response.json()["gpu"] == "l4"
        assert response.json()["status"] == "provisioning"
        # record_pending_demand now takes (machine_profile, bundle, pool_name) for KEDA scaling
        mock_demand.assert_called_once_with("l4", "default", "default")

    def test_no_workers_with_pool_extracts_gpu_returns_202(
        self, client: TestClient, registry: MagicMock, pool_manager: MagicMock
    ) -> None:
        """No workers + pool specified (no explicit GPU) → extracts GPU from pool → 202.

        This is the bug that was fixed: when a pool is specified without an explicit
        GPU header, the router should extract the GPU from the pool spec and return
        202 (not 503), so KEDA can scale up workers.
        """
        registry.select_worker.return_value = None

        # Create a pool with GPU spec
        pool = Pool(
            spec=PoolSpec(name="eval-l4-0", gpus={"l4": 1}),
            status=PoolStatus(state=PoolState.PENDING),
        )
        pool_manager.get_pool.return_value = pool

        with patch("sie_router.proxy.record_pending_demand") as mock_demand:
            with patch("sie_router.proxy.CONFIGURED_GPU_TYPES", ["l4"]):
                response = client.post(
                    "/v1/encode/test-model",
                    headers={"X-SIE-Pool": "eval-l4-0"},
                    json={"text": "hello"},
                )

        assert response.status_code == 202
        assert response.json()["gpu"] == "l4"
        # record_pending_demand now takes (machine_profile, bundle, pool_name) for KEDA scaling
        # The machine_profile is extracted from pool spec (l4 in this case)
        mock_demand.assert_called_once_with("l4", "default", "eval-l4-0")

    def test_no_workers_pool_no_gpu_spec_returns_503(
        self, client: TestClient, registry: MagicMock, pool_manager: MagicMock
    ) -> None:
        """No workers + pool with empty GPU spec → 503."""
        registry.select_worker.return_value = None

        # Create a pool without GPU spec
        pool = Pool(
            spec=PoolSpec(name="empty-pool", gpus={}),
            status=PoolStatus(state=PoolState.PENDING),
        )
        pool_manager.get_pool.return_value = pool

        response = client.post(
            "/v1/encode/test-model",
            headers={"X-SIE-Pool": "empty-pool"},
            json={"text": "hello"},
        )

        assert response.status_code == 503

    def test_unconfigured_gpu_returns_503(self, client: TestClient, registry: MagicMock) -> None:
        """Request for unconfigured GPU type → 503 with specific error."""
        with patch("sie_router.proxy.CONFIGURED_GPU_TYPES", ["l4", "a100-40gb"]):
            response = client.post(
                "/v1/encode/test-model",
                headers={"X-SIE-MACHINE-PROFILE": "h100"},
                json={"text": "hello"},
            )

        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "gpu_not_configured"
        assert body["gpu"] == "h100"
        assert "l4" in body["configured_gpu_types"]


# =============================================================================
# Pool routing tests
# =============================================================================


class TestPoolRouting:
    """Tests for pool-aware routing."""

    def test_pool_header_routing(self, client: TestClient, registry: MagicMock, pool_manager: MagicMock) -> None:
        """X-SIE-Pool header routes to pool workers."""
        # Set up active pool with assigned worker (MockWorker created for context)
        pool = Pool(
            spec=PoolSpec(name="my-pool", gpus={"l4": 1}),
            status=PoolStatus(
                state=PoolState.ACTIVE,
                assigned_workers=[AssignedWorker(name="worker-0", url="http://worker-0:8080", gpu="l4")],
            ),
        )
        pool_manager.get_pool.return_value = pool

        # Registry should be called with pool worker URLs
        registry.select_worker.return_value = WorkerState(
            name="worker-0", url="http://worker-0:8080", machine_profile="l4"
        )

        with patch("sie_router.proxy._forward_request", new_callable=AsyncMock) as mock_forward:
            from fastapi import Response

            mock_forward.return_value = Response(content=b'{"embeddings": []}', status_code=200)

            response = client.post(
                "/v1/encode/test-model",
                headers={"X-SIE-Pool": "my-pool"},
                json={"text": "hello"},
            )

        assert response.status_code == 200
        # Verify registry was called with pool worker URLs constraint
        registry.select_worker.assert_called()
        call_kwargs = registry.select_worker.call_args[1]
        assert call_kwargs.get("worker_urls") == {"http://worker-0:8080"}

    def test_gpu_param_with_pool_prefix(self, client: TestClient, registry: MagicMock, pool_manager: MagicMock) -> None:
        """GPU param with pool prefix (e.g., 'pool-name/l4') extracts pool and GPU."""
        pool = Pool(
            spec=PoolSpec(name="eval-pool", gpus={"l4": 1}),
            status=PoolStatus(
                state=PoolState.ACTIVE,
                assigned_workers=[AssignedWorker(name="worker-0", url="http://worker-0:8080", gpu="l4")],
            ),
        )
        pool_manager.get_pool.return_value = pool

        registry.select_worker.return_value = WorkerState(
            name="worker-0", url="http://worker-0:8080", machine_profile="l4"
        )

        with patch("sie_router.proxy._forward_request", new_callable=AsyncMock) as mock_forward:
            from fastapi import Response

            mock_forward.return_value = Response(content=b'{"embeddings": []}', status_code=200)

            response = client.post(
                "/v1/encode/test-model",
                headers={"X-SIE-MACHINE-PROFILE": "eval-pool/l4"},
                json={"text": "hello"},
            )

        assert response.status_code == 200
        pool_manager.get_pool.assert_called_with("eval-pool")

    def test_inactive_pool_falls_through(
        self, client: TestClient, registry: MagicMock, pool_manager: MagicMock
    ) -> None:
        """Inactive pool doesn't route to pool workers."""
        pool = Pool(
            spec=PoolSpec(name="inactive-pool", gpus={"l4": 1}),
            status=PoolStatus(state=PoolState.PENDING),
        )
        pool_manager.get_pool.return_value = pool
        registry.select_worker.return_value = None

        with patch("sie_router.proxy.record_pending_demand"):
            with patch("sie_router.proxy.CONFIGURED_GPU_TYPES", ["l4"]):
                response = client.post(
                    "/v1/encode/test-model",
                    headers={"X-SIE-Pool": "inactive-pool"},
                    json={"text": "hello"},
                )

        # Should return 202 (provisioning) since pool has GPU spec but is inactive
        assert response.status_code == 202


# =============================================================================
# Worker forwarding tests
# =============================================================================


class TestWorkerForwarding:
    """Tests for request forwarding to workers."""

    def test_successful_forward(self, client: TestClient, registry: MagicMock) -> None:
        """Successful request forwarding."""
        registry.select_worker.return_value = WorkerState(
            name="worker-0", url="http://worker-0:8080", machine_profile="l4"
        )

        with patch("sie_router.proxy._forward_request", new_callable=AsyncMock) as mock_forward:
            from fastapi import Response

            mock_forward.return_value = Response(
                content=b'{"embeddings": [[0.1, 0.2]]}',
                status_code=200,
                media_type="application/json",
            )

            response = client.post(
                "/v1/encode/test-model",
                json={"text": "hello"},
            )

        assert response.status_code == 200
        registry.record_request.assert_called_once_with("http://worker-0:8080")

    def test_timeout_returns_504(self, client: TestClient, registry: MagicMock) -> None:
        """Worker timeout returns 504."""
        registry.select_worker.return_value = WorkerState(
            name="worker-0", url="http://worker-0:8080", machine_profile="l4"
        )

        with patch("sie_router.proxy._forward_request", new_callable=AsyncMock) as mock_forward:
            mock_forward.side_effect = httpx.TimeoutException("timeout")

            response = client.post(
                "/v1/encode/test-model",
                json={"text": "hello"},
            )

        assert response.status_code == 504
        assert "timed out" in response.json()["detail"]["message"]

    def test_connection_error_returns_502_and_marks_unhealthy(self, client: TestClient, registry: MagicMock) -> None:
        """Worker connection error returns 502 and marks worker unhealthy."""
        registry.select_worker.return_value = WorkerState(
            name="worker-0", url="http://worker-0:8080", machine_profile="l4"
        )
        registry.mark_unhealthy = AsyncMock()

        with patch("sie_router.proxy._forward_request", new_callable=AsyncMock) as mock_forward:
            mock_forward.side_effect = httpx.RequestError("connection refused")

            response = client.post(
                "/v1/encode/test-model",
                json={"text": "hello"},
            )

        assert response.status_code == 502
        assert "connection error" in response.json()["detail"]["message"].lower()
        registry.mark_unhealthy.assert_called_once_with("http://worker-0:8080")


# =============================================================================
# All endpoint tests (encode, score, extract)
# =============================================================================


class TestAllProxyEndpoints:
    """Test that encode, score, and extract all use the same proxy logic."""

    @pytest.mark.parametrize("endpoint", ["/v1/encode/model", "/v1/score/model", "/v1/extract/model"])
    def test_all_endpoints_return_202_with_gpu(self, client: TestClient, registry: MagicMock, endpoint: str) -> None:
        """All proxy endpoints return 202 when GPU specified but no workers."""
        registry.select_worker.return_value = None

        with patch("sie_router.proxy.record_pending_demand"):
            with patch("sie_router.proxy.CONFIGURED_GPU_TYPES", ["l4"]):
                response = client.post(
                    endpoint,
                    headers={"X-SIE-MACHINE-PROFILE": "l4"},
                    json={"text": "hello"},
                )

        assert response.status_code == 202

    @pytest.mark.parametrize("endpoint", ["/v1/encode/model", "/v1/score/model", "/v1/extract/model"])
    def test_all_endpoints_return_503_without_gpu(self, client: TestClient, registry: MagicMock, endpoint: str) -> None:
        """All proxy endpoints return 503 when no GPU and no workers."""
        registry.select_worker.return_value = None

        response = client.post(
            endpoint,
            json={"text": "hello"},
        )

        assert response.status_code == 503


# =============================================================================
# Pool API tests
# =============================================================================


class TestPoolApi:
    """Tests for pool management API endpoints."""

    def test_create_pool(self, client: TestClient, pool_manager: MagicMock, registry: MagicMock) -> None:
        """Create pool endpoint."""
        pool = Pool(
            spec=PoolSpec(name="test-pool", gpus={"l4": 2}),
            status=PoolStatus(state=PoolState.PENDING),
        )
        pool_manager.create_pool = AsyncMock(return_value=pool)
        pool_manager._use_kubernetes = False

        response = client.post(
            "/v1/pools",
            json={"name": "test-pool", "gpus": {"l4": 2}},
        )

        assert response.status_code == 201
        assert response.json()["name"] == "test-pool"

    def test_create_pool_missing_name(self, client: TestClient, pool_manager: MagicMock) -> None:
        """Create pool without name returns 400."""
        response = client.post("/v1/pools", json={"gpus": {"l4": 1}})
        assert response.status_code == 400

    def test_get_pool(self, client: TestClient, pool_manager: MagicMock) -> None:
        """Get pool endpoint."""
        pool = Pool(
            spec=PoolSpec(name="my-pool", gpus={"l4": 1}),
            status=PoolStatus(state=PoolState.ACTIVE),
        )
        pool_manager.get_pool.return_value = pool

        response = client.get("/v1/pools/my-pool")

        assert response.status_code == 200
        assert response.json()["name"] == "my-pool"

    def test_get_pool_not_found(self, client: TestClient, pool_manager: MagicMock) -> None:
        """Get non-existent pool returns 404."""
        pool_manager.get_pool.return_value = None

        response = client.get("/v1/pools/missing")

        assert response.status_code == 404

    def test_get_pending_pool_assigns_workers(
        self, client: TestClient, pool_manager: MagicMock, registry: MagicMock
    ) -> None:
        """GET on pending pool with available workers assigns them (scale-from-zero).

        This test verifies the fix for pool 'pending' timeout issue: when KEDA
        scales up workers, the next GET pool call should assign them.
        """
        # Pool starts in pending state
        pool = Pool(
            spec=PoolSpec(name="eval-l4-0", gpus={"l4": 1}),
            status=PoolStatus(state=PoolState.PENDING),
        )
        pool_manager.get_pool.return_value = pool
        pool_manager.get_assigned_worker_urls.return_value = set()
        pool_manager._use_kubernetes = False

        # Workers are available (KEDA scaled them up)
        worker = MockWorker(name="worker-0", url="http://worker-0:8080", gpu="l4")
        registry.healthy_workers = [worker]

        # assign_workers succeeds and transitions pool to active
        pool_manager.assign_workers.return_value = True

        response = client.get("/v1/pools/eval-l4-0")

        assert response.status_code == 200
        # Verify assign_workers was called
        pool_manager.assign_workers.assert_called_once()
        call_args = pool_manager.assign_workers.call_args[0]
        assert call_args[0] == pool  # pool argument
        assert len(call_args[1]) == 1  # available workers list
        assert call_args[1][0] == ("worker-0", "http://worker-0:8080", "l4", "default")

    def test_list_pools(self, client: TestClient, pool_manager: MagicMock) -> None:
        """List pools endpoint."""
        pool = Pool(
            spec=PoolSpec(name="pool-1", gpus={"l4": 1}),
            status=PoolStatus(state=PoolState.ACTIVE),
        )
        pool_manager.pools = {"pool-1": pool}

        response = client.get("/v1/pools")

        assert response.status_code == 200
        assert len(response.json()["pools"]) == 1

    def test_delete_pool(self, client: TestClient, pool_manager: MagicMock) -> None:
        """Delete pool endpoint."""
        pool_manager.delete_pool = AsyncMock(return_value=True)

        response = client.delete("/v1/pools/my-pool")

        assert response.status_code == 200
        pool_manager.delete_pool.assert_called_once_with("my-pool")

    def test_delete_pool_not_found(self, client: TestClient, pool_manager: MagicMock) -> None:
        """Delete non-existent pool returns 404."""
        pool_manager.delete_pool = AsyncMock(return_value=False)

        response = client.delete("/v1/pools/missing")

        assert response.status_code == 404

    def test_renew_pool_lease(self, client: TestClient, pool_manager: MagicMock) -> None:
        """Renew pool lease endpoint."""
        pool_manager.renew_lease = AsyncMock(return_value=True)

        response = client.post("/v1/pools/my-pool/renew")

        assert response.status_code == 200
        pool_manager.renew_lease.assert_called_once_with("my-pool")

    def test_pool_management_disabled(self, app: FastAPI, registry: MagicMock) -> None:
        """Pool endpoints return 503 when pool management is disabled."""
        app.state.registry = registry
        app.state.pool_manager = None  # Disabled
        client = TestClient(app)

        response = client.get("/v1/pools")
        assert response.status_code == 503
        assert "not enabled" in response.json()["detail"]["message"]


# =============================================================================
# Audit logging tests
# =============================================================================


class TestMaskToken:
    """Tests for _mask_token helper."""

    def test_masks_long_token(self) -> None:
        """Should show only last 4 characters."""
        assert _mask_token("sk-abcdefgh1234ab3f") == "****ab3f"

    def test_masks_short_token(self) -> None:
        """Short tokens (<=4 chars) should be fully masked."""
        assert _mask_token("abcd") == "****"
        assert _mask_token("ab") == "****"

    def test_masks_exactly_five(self) -> None:
        """Token with 5 characters shows last 4."""
        assert _mask_token("12345") == "****2345"


class TestAuditLogging:
    """Tests for audit log emission."""

    @pytest.mark.parametrize("endpoint", ["/v1/encode/org/model", "/v1/score/org/model", "/v1/extract/org/model"])
    def test_successful_proxy_emits_audit_log(self, client: TestClient, registry: MagicMock, endpoint: str) -> None:
        """Successful proxy request emits audit log with correct fields."""
        registry.select_worker.return_value = WorkerState(
            name="worker-0", url="http://worker-0:8080", machine_profile="l4"
        )

        with (
            patch("sie_router.proxy._forward_request", new_callable=AsyncMock) as mock_forward,
            patch.object(audit_logger, "info") as mock_audit,
        ):
            from fastapi import Response as FastAPIResponse

            mock_forward.return_value = FastAPIResponse(content=b'{"result": []}', status_code=200)

            client.post(endpoint, json={"text": "hello"})

        mock_audit.assert_called_once()
        _msg, kwargs = mock_audit.call_args[0][0], mock_audit.call_args[1]
        extra = kwargs["extra"]

        assert extra["event"] == "api_request"
        assert extra["status"] == 200
        assert extra["worker"] == "worker-0"
        assert extra["model"] == "org/model"
        assert "latency_ms" in extra
        assert "body_bytes" in extra
        expected_ep = endpoint.split("/")[2]
        assert extra["endpoint"] == expected_ep

    def test_auth_failure_does_not_emit_audit_log(self, client: TestClient) -> None:
        """401 auth failure should NOT emit an audit log (rejected before handler)."""
        with (
            patch("sie_router.proxy.AUTH_MODE", "static"),
            patch("sie_router.proxy.AUTH_TOKENS", {"valid-token"}),
            patch.object(audit_logger, "info") as mock_audit,
        ):
            response = client.post(
                "/v1/encode/test-model",
                headers={"Authorization": "Bearer bad-token"},
                json={"text": "hello"},
            )

        assert response.status_code == 401
        mock_audit.assert_not_called()

    def test_202_provisioning_emits_audit_log(self, client: TestClient, registry: MagicMock) -> None:
        """202 provisioning response emits audit log."""
        registry.select_worker.return_value = None

        with (
            patch("sie_router.proxy.record_pending_demand"),
            patch("sie_router.proxy.CONFIGURED_GPU_TYPES", ["l4"]),
            patch.object(audit_logger, "info") as mock_audit,
        ):
            response = client.post(
                "/v1/encode/test-model",
                headers={"X-SIE-MACHINE-PROFILE": "l4"},
                json={"text": "hello"},
            )

        assert response.status_code == 202
        mock_audit.assert_called_once()
        extra = mock_audit.call_args[1]["extra"]
        assert extra["status"] == 202
        assert extra["gpu"] == "l4"
        assert extra["endpoint"] == "encode"

    def test_404_model_not_found_emits_audit_log(self, client: TestClient, registry: MagicMock) -> None:
        """404 model not found emits audit log."""
        from sie_router.model_registry import ModelNotFoundError, ModelRegistry

        mock_model_reg = MagicMock(spec=ModelRegistry)
        mock_model_reg.resolve_bundle.side_effect = ModelNotFoundError("unknown-model")
        client.app.state.model_registry = mock_model_reg

        with patch.object(audit_logger, "info") as mock_audit:
            response = client.post("/v1/encode/unknown-model", json={"text": "hello"})

        assert response.status_code == 404
        mock_audit.assert_called_once()
        extra = mock_audit.call_args[1]["extra"]
        assert extra["status"] == 404
        assert extra["model"] == "unknown-model"

        # Clean up
        client.app.state.model_registry = None

    def test_503_no_workers_emits_audit_log(self, client: TestClient, registry: MagicMock) -> None:
        """503 no workers emits audit log."""
        registry.select_worker.return_value = None

        with patch.object(audit_logger, "info") as mock_audit:
            response = client.post("/v1/encode/test-model", json={"text": "hello"})

        assert response.status_code == 503
        mock_audit.assert_called_once()
        extra = mock_audit.call_args[1]["extra"]
        assert extra["status"] == 503

    def test_token_id_is_masked_in_audit_log(self, client: TestClient, registry: MagicMock) -> None:
        """Token ID should be masked in audit log."""
        registry.select_worker.return_value = None

        with (
            patch("sie_router.proxy.AUTH_MODE", "static"),
            patch("sie_router.proxy.AUTH_TOKENS", {"sk-secret-token-ab3f"}),
            patch.object(audit_logger, "info") as mock_audit,
        ):
            response = client.post(
                "/v1/encode/test-model",
                headers={"Authorization": "Bearer sk-secret-token-ab3f"},
                json={"text": "hello"},
            )

        assert response.status_code == 503
        mock_audit.assert_called_once()
        extra = mock_audit.call_args[1]["extra"]
        assert extra["token_id"] == "****ab3f"  # noqa: S105

    def test_pool_create_emits_audit_log(
        self, client: TestClient, pool_manager: MagicMock, registry: MagicMock
    ) -> None:
        """Pool creation emits audit log."""
        pool = Pool(
            spec=PoolSpec(name="test-pool", gpus={"l4": 2}),
            status=PoolStatus(state=PoolState.PENDING),
        )
        pool_manager.create_pool = AsyncMock(return_value=pool)
        pool_manager._use_kubernetes = False

        with patch.object(audit_logger, "info") as mock_audit:
            response = client.post(
                "/v1/pools",
                json={"name": "test-pool", "gpus": {"l4": 2}},
            )

        assert response.status_code == 201
        mock_audit.assert_called_once()
        extra = mock_audit.call_args[1]["extra"]
        assert extra["event"] == "pool_create"
        assert extra["pool"] == "test-pool"
        assert extra["status"] == 201

    def test_pool_delete_emits_audit_log(self, client: TestClient, pool_manager: MagicMock) -> None:
        """Pool deletion emits audit log."""
        pool_manager.delete_pool = AsyncMock(return_value=True)

        with patch.object(audit_logger, "info") as mock_audit:
            response = client.delete("/v1/pools/my-pool")

        assert response.status_code == 200
        mock_audit.assert_called_once()
        extra = mock_audit.call_args[1]["extra"]
        assert extra["event"] == "pool_delete"
        assert extra["pool"] == "my-pool"
        assert extra["status"] == 200

    def test_no_token_id_when_auth_disabled(self, client: TestClient, registry: MagicMock) -> None:
        """When auth is disabled, token_id should not be in audit log extra."""
        registry.select_worker.return_value = None

        with patch.object(audit_logger, "info") as mock_audit:
            client.post("/v1/encode/test-model", json={"text": "hello"})

        mock_audit.assert_called_once()
        extra = mock_audit.call_args[1]["extra"]
        assert "token_id" not in extra
