"""Tests for graceful shutdown handling."""

from __future__ import annotations

import asyncio

import pytest
from sie_server.core.shutdown import ShutdownState


class TestShutdownState:
    """Tests for ShutdownState class."""

    @pytest.fixture
    def shutdown_state(self) -> ShutdownState:
        """Create a fresh shutdown state for each test."""
        return ShutdownState()

    def test_initial_state(self, shutdown_state: ShutdownState) -> None:
        """Shutdown state starts with correct defaults."""
        assert shutdown_state.shutting_down is False
        assert shutdown_state.in_flight == 0

    def test_start_shutdown_sets_flag(self, shutdown_state: ShutdownState) -> None:
        """start_shutdown sets shutting_down flag."""
        shutdown_state.start_shutdown()
        assert shutdown_state.shutting_down is True

    def test_start_shutdown_idempotent(self, shutdown_state: ShutdownState) -> None:
        """Multiple calls to start_shutdown are idempotent."""
        shutdown_state.start_shutdown()
        shutdown_state.start_shutdown()
        assert shutdown_state.shutting_down is True

    @pytest.mark.asyncio
    async def test_request_tracking(self, shutdown_state: ShutdownState) -> None:
        """Request tracking increments and decrements correctly."""
        assert shutdown_state.in_flight == 0

        await shutdown_state.request_started()
        assert shutdown_state.in_flight == 1

        await shutdown_state.request_started()
        assert shutdown_state.in_flight == 2

        await shutdown_state.request_finished()
        assert shutdown_state.in_flight == 1

        await shutdown_state.request_finished()
        assert shutdown_state.in_flight == 0

    @pytest.mark.asyncio
    async def test_drain_completes_immediately_when_no_requests(self, shutdown_state: ShutdownState) -> None:
        """wait_for_drain returns immediately when no requests in flight."""
        result = await shutdown_state.wait_for_drain(drain_timeout=1.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_drain_waits_for_requests(self, shutdown_state: ShutdownState) -> None:
        """wait_for_drain waits for in-flight requests to complete."""
        # Start a request
        await shutdown_state.request_started()

        # Start drain in background
        async def drain() -> bool:
            return await shutdown_state.wait_for_drain(drain_timeout=5.0)

        drain_task = asyncio.create_task(drain())

        # Give drain task time to start
        await asyncio.sleep(0.02)
        assert not drain_task.done()

        # Initiate shutdown (needed for drain_event to be set)
        shutdown_state.start_shutdown()

        # Complete the request
        await shutdown_state.request_finished()

        # Drain should complete now
        result = await asyncio.wait_for(drain_task, timeout=1.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_drain_timeout(self, shutdown_state: ShutdownState) -> None:
        """wait_for_drain returns False on timeout."""
        # Start a request that won't complete
        await shutdown_state.request_started()
        shutdown_state.start_shutdown()

        # Drain should timeout
        result = await shutdown_state.wait_for_drain(drain_timeout=0.01)
        assert result is False

    @pytest.mark.asyncio
    async def test_middleware_rejects_during_shutdown(self, shutdown_state: ShutdownState) -> None:
        """Middleware returns 503 when shutting down."""
        from unittest.mock import AsyncMock, MagicMock

        # Mock request and call_next
        request = MagicMock()
        call_next = AsyncMock()

        # Start shutdown
        shutdown_state.start_shutdown()

        # Call middleware
        response = await shutdown_state.middleware(request, call_next)

        # Should return 503 without calling next
        assert response.status_code == 503
        call_next.assert_not_called()

    @pytest.mark.asyncio
    async def test_middleware_tracks_requests(self, shutdown_state: ShutdownState) -> None:
        """Middleware tracks in-flight requests."""
        from unittest.mock import AsyncMock, MagicMock

        from fastapi.responses import JSONResponse

        # Mock request and call_next
        request = MagicMock()
        mock_response = JSONResponse(content={"ok": True})
        call_next = AsyncMock(return_value=mock_response)

        assert shutdown_state.in_flight == 0

        # Call middleware
        response = await shutdown_state.middleware(request, call_next)

        # Should have called next and returned response
        assert response == mock_response
        call_next.assert_called_once_with(request)

        # Request should be complete
        assert shutdown_state.in_flight == 0

    @pytest.mark.asyncio
    async def test_middleware_tracks_requests_on_error(self, shutdown_state: ShutdownState) -> None:
        """Middleware decrements counter even on error."""
        from unittest.mock import AsyncMock, MagicMock

        # Mock request and call_next that raises
        request = MagicMock()
        call_next = AsyncMock(side_effect=RuntimeError("boom"))

        assert shutdown_state.in_flight == 0

        # Call middleware - should raise but still clean up
        with pytest.raises(RuntimeError, match="boom"):
            await shutdown_state.middleware(request, call_next)

        # Request should still be complete
        assert shutdown_state.in_flight == 0


class TestSignalHandler:
    """Tests for signal handler setup."""

    def test_setup_signal_handlers_does_not_crash(self) -> None:
        """setup_signal_handlers handles not being in main thread gracefully."""
        from sie_server.core.shutdown import setup_signal_handlers

        shutdown_state = ShutdownState()
        # Should not raise even in test thread
        setup_signal_handlers(shutdown_state)
