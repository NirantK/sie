"""Graceful shutdown handling for spot instance preemption.

Handles SIGTERM signals to gracefully drain in-flight requests before shutdown.
This is critical for spot/preemptible instances which receive 30s warning.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint

if TYPE_CHECKING:
    from fastapi import Request, Response

logger = logging.getLogger(__name__)


@dataclass
class ShutdownState:
    """Tracks shutdown state and in-flight requests.

    Attributes:
        shutting_down: True when SIGTERM received, stops accepting new requests.
        in_flight: Count of currently processing requests.
        drain_timeout_s: Maximum time to wait for requests to drain.
    """

    shutting_down: bool = False
    in_flight: int = 0
    drain_timeout_s: float = 25.0  # Leave 5s buffer before 30s preemption
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _drain_event: asyncio.Event = field(default_factory=asyncio.Event)
    _shutdown_time: float | None = None

    def start_shutdown(self) -> None:
        """Called when SIGTERM received. Stops accepting new requests."""
        if not self.shutting_down:
            self.shutting_down = True
            self._shutdown_time = time.time()
            logger.warning(
                "Shutdown initiated, draining %d in-flight requests (timeout: %.1fs)",
                self.in_flight,
                self.drain_timeout_s,
            )
            # If no requests in flight, signal immediately
            if self.in_flight == 0:
                self._drain_event.set()

    async def request_started(self) -> None:
        """Called when a request starts processing."""
        async with self._lock:
            self.in_flight += 1

    async def request_finished(self) -> None:
        """Called when a request finishes processing."""
        async with self._lock:
            self.in_flight -= 1
            if self.shutting_down and self.in_flight == 0:
                logger.info("All requests drained, ready for shutdown")
                self._drain_event.set()

    async def wait_for_drain(self, drain_timeout: float | None = None) -> bool:
        """Wait for all in-flight requests to complete.

        Args:
            drain_timeout: Maximum seconds to wait. Uses drain_timeout_s if None.

        Returns:
            True if all requests drained, False if timeout.
        """
        if drain_timeout is None:
            drain_timeout = self.drain_timeout_s

        if self.in_flight == 0:
            return True

        logger.info("Waiting for %d requests to drain (timeout: %.1fs)", self.in_flight, drain_timeout)

        try:
            await asyncio.wait_for(self._drain_event.wait(), timeout=drain_timeout)
        except TimeoutError:
            logger.warning(
                "Drain timeout after %.1fs, %d requests still in flight",
                drain_timeout,
                self.in_flight,
            )
            return False
        else:
            return True

    async def middleware(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """FastAPI middleware to track requests and reject during shutdown.

        Returns 503 Service Unavailable when shutting down.
        """
        # Reject new requests during shutdown
        if self.shutting_down:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Service shutting down",
                    "detail": "Server is draining requests before shutdown. Retry on another instance.",
                },
                headers={"Retry-After": "5"},
            )

        # Track request lifecycle
        await self.request_started()
        try:
            return await call_next(request)
        finally:
            await self.request_finished()


def setup_signal_handlers(shutdown_state: ShutdownState) -> None:
    """Install SIGTERM handler for graceful shutdown.

    Args:
        shutdown_state: The shutdown state to update on signal.

    Note:
        Signal handlers can only be set in the main thread. In test environments
        or when running under certain frameworks, this may silently skip setup.
    """

    def handle_sigterm(_signum: int, _frame: object) -> None:
        logger.info("Received SIGTERM, initiating graceful shutdown")
        shutdown_state.start_shutdown()

    # Only set up signal handlers on Unix (not Windows)
    if sys.platform != "win32":
        try:
            signal.signal(signal.SIGTERM, handle_sigterm)
            logger.debug("SIGTERM handler installed for graceful shutdown")
        except ValueError:
            # Signal handlers can only be set in main thread
            # This is expected in test environments
            logger.debug("Skipping SIGTERM handler (not in main thread)")
