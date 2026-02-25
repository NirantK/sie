import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

from sie_server.api.serialization import MsgPackResponse, _convert_for_json, deserialize_msgpack
from sie_server.core.timing import RequestTiming
from sie_server.core.worker import QueueFullError
from sie_server.observability.metrics import record_request
from sie_server.observability.tracing import get_current_trace_id
from sie_server.types.responses import ErrorCode

if TYPE_CHECKING:
    from opentelemetry.trace import Span

    from sie_server.core.registry import ModelRegistry

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Content types
MSGPACK_CONTENT_TYPE = "application/msgpack"
JSON_CONTENT_TYPE = "application/json"


def _mask_api_key(key: str) -> str:
    """Mask an API key, showing only the last 4 characters."""
    mask = "****"
    if len(key) <= len(mask):
        return mask
    return f"{mask}{key[-4:]}"


@dataclass(frozen=True)
class RequestContext:
    """Request-scoped context for structured logging."""

    request_id: str
    api_key: str | None
    queue_depth: int | None


def extract_request_context(
    http_request: Request,
    model: str,
    registry: "ModelRegistry",
) -> RequestContext:
    """Extract request context from HTTP request for structured logging.

    Args:
        http_request: FastAPI request object.
        model: Model name (for queue depth lookup).
        registry: ModelRegistry to get queue depth.

    Returns:
        RequestContext with request_id, masked api_key, and queue_depth.
    """
    # request_id: from X-Request-ID header or generate new
    request_id = http_request.headers.get("x-request-id") or str(uuid.uuid4())

    # api_key: from Authorization header, masked
    auth_header = http_request.headers.get("authorization")
    api_key: str | None = None
    if auth_header:
        token = (auth_header[7:] if auth_header.lower().startswith("bearer ") else auth_header).strip()
        if token:
            api_key = _mask_api_key(token)

    # queue_depth: from worker's pending_count
    queue_depth: int | None = None
    worker = registry.get_worker(model)
    if worker is not None:
        queue_depth = worker.pending_count

    return RequestContext(request_id=request_id, api_key=api_key, queue_depth=queue_depth)


class ContentNegotiator:
    """Handles HTTP content negotiation for msgpack/JSON."""

    @staticmethod
    def wants_msgpack(accept: str | None) -> bool:
        """Check if client prefers msgpack based on Accept header.

        Default to msgpack if no preference specified (per DESIGN.md).
        """
        if not accept:
            return True  # Default to msgpack

        # Parse Accept header and check preferences
        accept_lower = accept.lower()
        if MSGPACK_CONTENT_TYPE in accept_lower:
            return True
        if "application/x-msgpack" in accept_lower:
            return True
        return JSON_CONTENT_TYPE not in accept_lower

    @staticmethod
    def is_msgpack_request(content_type: str | None) -> bool:
        """Check if request body is msgpack based on Content-Type header."""
        if not content_type:
            return False
        content_type_lower = content_type.lower()
        return MSGPACK_CONTENT_TYPE in content_type_lower or "application/x-msgpack" in content_type_lower


class RequestParser:
    """Parses and validates HTTP request bodies."""

    @staticmethod
    async def parse(
        http_request: Request,
        validator: Callable[[dict[str, Any]], None],
    ) -> dict[str, Any]:
        """Parse request body as msgpack or JSON based on Content-Type.

        Per DESIGN.md Section 4.3: All requests and responses use msgpack with msgpack-numpy.
        JSON is supported as fallback for debugging.

        Uses manual validation instead of Pydantic for zero overhead.

        Args:
            http_request: FastAPI Request object.
            validator: Validation function that raises HTTPException on invalid data.

        Returns:
            Parsed and validated request data as dict.

        Raises:
            HTTPException: 400 if parsing or validation fails.
        """
        content_type = http_request.headers.get("content-type")

        try:
            if ContentNegotiator.is_msgpack_request(content_type):
                # Parse msgpack body
                body = await http_request.body()
                data = deserialize_msgpack(body)
            else:
                # Parse JSON body (fallback)
                data = await http_request.json()

            # Manual validation (no Pydantic overhead)
            validator(data)
            return data
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": ErrorCode.INVALID_INPUT.value,
                    "message": f"Failed to parse request body: {e}",
                },
            ) from e


class ModelStateChecker:
    """Validates model state before inference."""

    def __init__(self, registry: "ModelRegistry", model: str, span: "Span") -> None:
        """Initialize checker.

        Args:
            registry: ModelRegistry instance.
            model: Model name to check.
            span: OpenTelemetry span for error attributes.
        """
        self.registry = registry
        self.model = model
        self.span = span

    def check_exists(self) -> None:
        """Check if model exists in registry.

        Raises:
            HTTPException: 404 if model not found.
        """
        if not self.registry.has_model(self.model):
            self.span.set_attribute("error", "model_not_found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": ErrorCode.MODEL_NOT_FOUND.value,
                    "message": f"Model '{self.model}' not found",
                },
            )

    def check_not_unloading(self) -> None:
        """Check if model is being unloaded.

        Raises:
            HTTPException: 503 if model is unloading.
        """
        if self.registry.is_unloading(self.model):
            self.span.set_attribute("error", "model_unloading")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": ErrorCode.MODEL_NOT_LOADED.value,
                    "message": f"Model '{self.model}' is unloading",
                },
            )

    def check_not_loading(self) -> None:
        """Check if model is currently loading.

        Raises:
            HTTPException: 503 with Retry-After if model is loading.
        """
        if self.registry.is_loading(self.model):
            self.span.set_attribute("error", "model_loading")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": ErrorCode.MODEL_LOADING.value,
                    "message": f"Model '{self.model}' is loading, please retry",
                },
                headers={"Retry-After": "5"},
            )

    async def ensure_loaded(self, device: str) -> None:
        """Start loading model if not loaded, raise 503 to retry.

        Args:
            device: Device to load model on (cpu, cuda, mps).

        Raises:
            HTTPException: 503 with Retry-After if model needs loading.
        """
        if not self.registry.is_loaded(self.model):
            logger.info("Starting background load for model %s on device %s", self.model, device)
            await self.registry.start_load_async(self.model, device=device)
            self.span.set_attribute("error", "model_loading")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": ErrorCode.MODEL_LOADING.value,
                    "message": f"Model '{self.model}' is loading, please retry",
                },
                headers={"Retry-After": "5"},
            )

    async def validate_ready(self, device: str) -> None:
        """Run all state checks to ensure model is ready for inference.

        Checks in order: exists, not unloading, not loading, ensure loaded.

        Args:
            device: Device to load model on if needed.

        Raises:
            HTTPException: Various status codes based on model state.
        """
        self.check_exists()
        self.check_not_unloading()
        self.check_not_loading()
        await self.ensure_loaded(device)


class InferenceErrorHandler:
    """Handles common inference errors and converts to HTTP responses."""

    def __init__(
        self,
        model: str,
        endpoint: str,
        span: "Span",
        ctx: RequestContext | None = None,
    ) -> None:
        """Initialize handler.

        Args:
            model: Model name for metrics.
            endpoint: Endpoint name for metrics (encode, score, extract).
            span: OpenTelemetry span for error attributes.
            ctx: Optional request context for structured logging.
        """
        self.model = model
        self.endpoint = endpoint
        self.span = span
        self.ctx = ctx

    def _log_kwargs(self) -> dict[str, Any]:
        """Build kwargs for record_request from context."""
        if self.ctx is None:
            return {}
        return {
            "request_id": self.ctx.request_id,
            "api_key": self.ctx.api_key,
            "queue_depth": self.ctx.queue_depth,
        }

    def handle_queue_full(self, error: QueueFullError) -> HTTPException:
        """Handle queue full backpressure error.

        Args:
            error: QueueFullError from worker.

        Returns:
            HTTPException with 503 status.
        """
        self.span.set_attribute("error", "queue_full")
        record_request(model=self.model, endpoint=self.endpoint, status="error", **self._log_kwargs())
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": ErrorCode.MODEL_NOT_LOADED.value,
                "message": str(error),
            },
        )

    def handle_value_error(self, error: ValueError) -> HTTPException:
        """Handle invalid input errors.

        Args:
            error: ValueError from adapter/worker.

        Returns:
            HTTPException with 400 status.
        """
        self.span.set_attribute("error", "invalid_input")
        record_request(model=self.model, endpoint=self.endpoint, status="error", **self._log_kwargs())
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": ErrorCode.INVALID_INPUT.value,
                "message": str(error),
            },
        )

    def handle_inference_error(self, error: Exception, operation: str = "Inference") -> HTTPException:
        """Handle generic inference errors.

        Args:
            error: Exception from inference.
            operation: Operation name for error message (Inference, Extraction, Scoring).

        Returns:
            HTTPException with 500 status.
        """
        logger.exception("%s error for model %s", operation, self.model)
        self.span.set_attribute("error", "inference_error")
        record_request(model=self.model, endpoint=self.endpoint, status="error", **self._log_kwargs())
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": ErrorCode.INFERENCE_ERROR.value,
                "message": f"{operation} error: {error}",
            },
        )


class ResponseBuilder:
    """Builds HTTP responses with common headers."""

    @staticmethod
    def build_headers(timing: RequestTiming | None = None) -> dict[str, str]:
        """Build response headers with trace ID and timing.

        Args:
            timing: Optional RequestTiming to include timing headers.

        Returns:
            Headers dict with X-Trace-ID and timing headers.
        """
        headers: dict[str, str] = {}

        # Add trace ID
        trace_id = get_current_trace_id()
        if trace_id:
            headers["X-Trace-ID"] = trace_id

        # Add timing headers
        if timing is not None:
            timing.finish()
            timing_headers = timing.to_headers()
            headers.update(timing_headers)

        return headers

    @staticmethod
    def build_response(
        content: Any,
        accept: str | None,
        headers: dict[str, str],
        *,
        convert_for_json: bool = False,
    ) -> MsgPackResponse | JSONResponse:
        """Build response with content negotiation.

        Args:
            content: Response content (TypedDict or dict).
            accept: Accept header value.
            headers: Response headers.
            convert_for_json: If True, convert numpy arrays for JSON response.

        Returns:
            MsgPackResponse or JSONResponse based on Accept header.
        """
        if ContentNegotiator.wants_msgpack(accept):
            return MsgPackResponse(content=content, headers=headers)

        # JSON fallback
        if convert_for_json:
            content = _convert_for_json(dict(content))

        return JSONResponse(content=content, headers=headers)
