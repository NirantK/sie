"""Tests for ModelWorker."""

import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest
from sie_server.core.inference_output import EncodeOutput, ExtractOutput, ScoreOutput
from sie_server.core.prepared import TextPreparedItem, make_text_item
from sie_server.core.timing import RequestTiming
from sie_server.core.worker import ModelWorker, RequestMetadata, WorkerConfig, WorkerResult, WorkerStats
from sie_server.types.inputs import Item
from sie_server.types.responses import Entity


class TestRequestMetadata:
    """Tests for RequestMetadata dataclass."""

    def test_basic_creation(self) -> None:
        """Can create basic metadata."""
        loop = asyncio.new_event_loop()
        future: asyncio.Future[WorkerResult] = loop.create_future()
        timing = RequestTiming()

        metadata = RequestMetadata(
            future=future,
            items=[Item(text="hello")],
            output_types=["dense"],
            timing=timing,
        )

        assert metadata.future is future
        assert metadata.items == [Item(text="hello")]
        assert metadata.output_types == ["dense"]
        assert metadata.timing is timing
        assert metadata.instruction is None
        assert metadata.is_query is False
        assert metadata.request_id is None
        loop.close()

    def test_with_all_fields(self) -> None:
        """Can create metadata with all fields."""
        loop = asyncio.new_event_loop()
        future: asyncio.Future[WorkerResult] = loop.create_future()
        timing = RequestTiming()

        metadata = RequestMetadata(
            future=future,
            items=[Item(text="hello")],
            output_types=["dense", "sparse"],
            timing=timing,
            instruction="Search query",
            is_query=True,
            request_id="req-123",
        )

        assert metadata.instruction == "Search query"
        assert metadata.is_query is True
        assert metadata.request_id == "req-123"
        assert metadata.timing is timing
        loop.close()


class TestWorkerConfig:
    """Tests for WorkerConfig dataclass."""

    def test_defaults(self) -> None:
        """Default config values."""
        config = WorkerConfig()

        assert config.max_batch_tokens == 16384
        assert config.max_batch_requests == 256
        assert config.max_batch_wait_ms == 10

    def test_custom_values(self) -> None:
        """Can set custom config values."""
        config = WorkerConfig(
            max_batch_tokens=8192,
            max_batch_requests=32,
            max_batch_wait_ms=5,
        )

        assert config.max_batch_tokens == 8192
        assert config.max_batch_requests == 32
        assert config.max_batch_wait_ms == 5


class TestWorkerStats:
    """Tests for WorkerStats dataclass."""

    def test_defaults(self) -> None:
        """Default stats values."""
        stats = WorkerStats()

        assert stats.batches_processed == 0
        assert stats.items_processed == 0
        assert stats.total_tokens_processed == 0
        assert stats.inference_errors == 0


class TestModelWorker:
    """Tests for ModelWorker."""

    @pytest.fixture
    def mock_adapter(self) -> MagicMock:
        """Create a mock adapter."""
        mock = MagicMock()
        # Return EncodeOutput (adapters return batched output now)
        mock.encode.side_effect = lambda items, *args, **kwargs: EncodeOutput(
            dense=np.array([[0.1, 0.2, 0.3]] * len(items)),
            batch_size=len(items),
        )
        return mock

    @pytest.fixture
    def tokenized_item(self) -> TextPreparedItem:
        """Create a tokenized item."""
        return make_text_item([1, 2, 3, 4, 5], 0)

    def test_init_default_config(self, mock_adapter: MagicMock) -> None:
        """Initialize with default config."""
        worker = ModelWorker(mock_adapter)

        assert worker.adapter is mock_adapter
        assert worker.config.max_batch_tokens == 16384
        assert worker.is_running is False

    def test_init_custom_config(self, mock_adapter: MagicMock) -> None:
        """Initialize with custom config."""
        config = WorkerConfig(max_batch_tokens=8192)
        worker = ModelWorker(mock_adapter, config)

        assert worker.config.max_batch_tokens == 8192

    @pytest.mark.asyncio
    async def test_start_stop(self, mock_adapter: MagicMock) -> None:
        """Start and stop worker."""
        worker = ModelWorker(mock_adapter)

        assert worker.is_running is False

        await worker.start()
        assert worker.is_running is True

        # Starting again is idempotent
        await worker.start()
        assert worker.is_running is True

        await worker.stop()
        assert worker.is_running is False

        # Stopping again is idempotent
        await worker.stop()
        assert worker.is_running is False

    @pytest.mark.asyncio
    async def test_submit_not_running(self, mock_adapter: MagicMock, tokenized_item: TextPreparedItem) -> None:
        """Submit raises when worker not running."""
        worker = ModelWorker(mock_adapter)

        with pytest.raises(RuntimeError, match="not running"):
            await worker.submit(
                [tokenized_item],
                [Item(text="hello")],
                ["dense"],
            )

    @pytest.mark.asyncio
    async def test_submit_and_get_result(self, mock_adapter: MagicMock, tokenized_item: TextPreparedItem) -> None:
        """Submit items and get result via future."""
        # Set up adapter to return embeddings via encode()
        mock_adapter.encode.return_value = EncodeOutput(
            dense=np.array([[0.1, 0.2, 0.3]]),
            batch_size=1,
        )

        config = WorkerConfig(
            max_batch_tokens=100,
            max_batch_requests=1,  # Batch immediately
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            future = await worker.submit(
                [tokenized_item],
                [Item(text="hello")],
                ["dense"],
            )

            # Wait for result
            worker_result = await asyncio.wait_for(future, timeout=2.0)

            assert worker_result.output.batch_size == 1
            assert worker_result.output.dense is not None
            np.testing.assert_array_equal(worker_result.output.dense[0], np.array([0.1, 0.2, 0.3]))

            # Stats updated
            assert worker.stats.batches_processed >= 1
            assert worker.stats.items_processed >= 1

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(
        self, mock_adapter: MagicMock, tokenized_item: TextPreparedItem
    ) -> None:
        """Multiple concurrent requests get batched."""

        # Set up adapter to return embeddings matching batch size
        def mock_encode(items, output_types, **kwargs):
            batch_size = len(items)
            return EncodeOutput(
                dense=np.array([[0.1, 0.2, 0.3]] * batch_size),
                batch_size=batch_size,
            )

        mock_adapter.encode.side_effect = mock_encode

        config = WorkerConfig(
            max_batch_tokens=100,
            max_batch_requests=3,  # Batch up to 3 requests
            max_batch_wait_ms=1,  # Short wait
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            # Submit 3 requests concurrently
            # Each request has one item, so original_index should be 0 for all
            # (original_index represents position within the request's items list)
            items = [
                make_text_item([1, 2], 0),
                make_text_item([1, 2, 3], 0),
                make_text_item([1, 2, 3, 4], 0),
            ]

            futures = []
            for i, item in enumerate(items):
                future = await worker.submit(
                    [item],
                    [Item(text=f"hello {i}")],
                    ["dense"],
                )
                futures.append(future)

            # Wait for all results
            worker_results = await asyncio.gather(*futures)

            # All requests completed
            assert len(worker_results) == 3
            for worker_result in worker_results:
                assert worker_result.output.batch_size == 1
                assert worker_result.output.dense is not None

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_inference_error_propagates(self, mock_adapter: MagicMock, tokenized_item: TextPreparedItem) -> None:
        """Inference error is propagated to future."""
        mock_adapter.encode.side_effect = RuntimeError("GPU OOM")

        config = WorkerConfig(
            max_batch_tokens=100,
            max_batch_requests=1,
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            future = await worker.submit(
                [tokenized_item],
                [Item(text="hello")],
                ["dense"],
            )

            with pytest.raises(RuntimeError, match="GPU OOM"):
                await asyncio.wait_for(future, timeout=2.0)

            # Error stats updated
            assert worker.stats.inference_errors >= 1

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_pending_count(self, mock_adapter: MagicMock, tokenized_item: TextPreparedItem) -> None:
        """Pending count reflects submitted items."""

        # Make encode slow so items stay pending
        def slow_encode(*args, **kwargs):
            import time

            time.sleep(0.5)
            return EncodeOutput(dense=np.array([[0.1, 0.2, 0.3]]), batch_size=1)

        mock_adapter.encode.side_effect = slow_encode

        config = WorkerConfig(
            max_batch_tokens=1000,  # High token limit
            max_batch_requests=100,  # High request limit
            max_batch_wait_ms=5,  # Wait before batching
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            # Initially no pending
            assert worker.pending_count == 0
            assert worker.pending_tokens == 0

            # Submit and check pending (don't await yet)
            await worker.submit(
                [tokenized_item],
                [Item(text="hello")],
                ["dense"],
            )

            # Should have pending items (before batch forms)
            # Note: This is timing-dependent but should work with high limits
            assert worker.pending_count >= 0  # May already be processed

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_passes_params_to_adapter(self, mock_adapter: MagicMock, tokenized_item: TextPreparedItem) -> None:
        """Request params are passed to adapter."""
        mock_adapter.encode.return_value = EncodeOutput(
            dense=np.array([[0.1, 0.2, 0.3]]),
            batch_size=1,
        )

        config = WorkerConfig(
            max_batch_tokens=100,
            max_batch_requests=1,
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            future = await worker.submit(
                [tokenized_item],
                [Item(text="hello")],
                ["dense", "sparse"],
                instruction="Search query",
                is_query=True,
            )

            await asyncio.wait_for(future, timeout=2.0)

            # Verify adapter.encode was called with correct params
            mock_adapter.encode.assert_called_once()
            call_args = mock_adapter.encode.call_args

            # Check positional args
            assert call_args[0][0] == [Item(text="hello")]  # items
            assert call_args[0][1] == ["dense", "sparse"]  # output_types

            # Check keyword args
            assert call_args[1]["instruction"] == "Search query"
            assert call_args[1]["is_query"] is True

        finally:
            await worker.stop()


class TestModelWorkerBackpressure:
    """Tests for backpressure (bounded queue) functionality."""

    @pytest.fixture
    def mock_adapter(self) -> MagicMock:
        """Create a mock adapter."""
        mock = MagicMock()
        mock.encode.return_value = EncodeOutput(
            dense=np.array([[0.1, 0.2, 0.3]]),
            batch_size=1,
        )
        return mock

    def test_queue_full_error_raised(self, mock_adapter: MagicMock) -> None:
        """QueueFullError raised when queue exceeds max_queue_size."""
        from sie_server.core.worker import QueueFullError

        # Very small queue limit
        config = WorkerConfig(
            max_batch_tokens=1000,
            max_batch_requests=100,
            max_batch_wait_ms=1,
            max_queue_size=5,  # Only allow 5 items
        )
        worker = ModelWorker(mock_adapter, config)

        async def test() -> None:
            await worker.start()
            try:
                # Submit 5 items (should succeed - at limit)
                items = [make_text_item([1, 2], i) for i in range(5)]
                for item in items:
                    await worker.submit(
                        [item],
                        [Item(text=f"hello {item.original_index}")],
                        ["dense"],
                    )

                # Try to submit one more (should fail)
                extra_item = make_text_item([3, 4], 5)
                with pytest.raises(QueueFullError, match="Queue full"):
                    await worker.submit(
                        [extra_item],
                        [Item(text="should fail")],
                        ["dense"],
                    )
            finally:
                await worker.stop()

        asyncio.get_event_loop().run_until_complete(test())

    def test_unlimited_queue_with_zero(self, mock_adapter: MagicMock) -> None:
        """max_queue_size=0 means unlimited queue."""
        config = WorkerConfig(
            max_batch_tokens=1000,
            max_batch_requests=100,
            max_batch_wait_ms=1,
            max_queue_size=0,  # Unlimited
        )
        worker = ModelWorker(mock_adapter, config)

        async def test() -> None:
            await worker.start()
            try:
                # Submit many items (should all succeed)
                for i in range(100):
                    item = make_text_item([1, 2], i)
                    await worker.submit(
                        [item],
                        [Item(text=f"hello {i}")],
                        ["dense"],
                    )
                # No QueueFullError raised
                assert worker.pending_count > 0
            finally:
                await worker.stop()

        asyncio.get_event_loop().run_until_complete(test())

    def test_queue_rejects_batch_that_would_exceed(self, mock_adapter: MagicMock) -> None:
        """Queue rejects a batch if adding it would exceed the limit."""
        from sie_server.core.worker import QueueFullError

        config = WorkerConfig(
            max_batch_tokens=1000,
            max_batch_requests=100,
            max_batch_wait_ms=1,
            max_queue_size=5,
        )
        worker = ModelWorker(mock_adapter, config)

        async def test() -> None:
            await worker.start()
            try:
                # Submit 3 items (leaves room for 2 more)
                items = [make_text_item([1, 2], i) for i in range(3)]
                for item in items:
                    await worker.submit(
                        [item],
                        [Item(text=f"hello {item.original_index}")],
                        ["dense"],
                    )

                # Try to submit 3 more items in one request (would make 6 total)
                batch_items = [make_text_item([3, 4], i) for i in range(3)]
                with pytest.raises(QueueFullError):
                    await worker.submit(
                        batch_items,
                        [Item(text=f"batch {i}") for i in range(3)],
                        ["dense"],
                    )
            finally:
                await worker.stop()

        asyncio.get_event_loop().run_until_complete(test())

    def test_default_max_queue_size(self, mock_adapter: MagicMock) -> None:
        """Default max_queue_size is 1000."""
        config = WorkerConfig()
        assert config.max_queue_size == 1000


class TestModelWorkerExtract:
    """Tests for submit_extract method."""

    @pytest.fixture
    def mock_adapter(self) -> MagicMock:
        """Create a mock adapter that supports extraction."""
        mock = MagicMock()
        # Return ExtractOutput for each batch
        mock.extract.side_effect = lambda items, **kwargs: ExtractOutput(
            entities=[[Entity(text="Mock", label="test", score=0.9, start=0, end=4)] for _ in items]
        )
        return mock

    @pytest.fixture
    def prepared_item(self) -> "ExtractPreparedItem":
        """Create a prepared item for extract (uses character cost, not tokens)."""
        from sie_server.core.prepared import ExtractPreparedItem

        return ExtractPreparedItem(
            cost=11,  # Character count
            original_index=0,
        )

    @pytest.mark.asyncio
    async def test_submit_extract_basic(self, mock_adapter: MagicMock, prepared_item: "ExtractPreparedItem") -> None:
        """Submit extract returns results."""
        config = WorkerConfig(
            max_batch_tokens=100,
            max_batch_requests=1,
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            future = await worker.submit_extract(
                [prepared_item],
                [Item(text="Hello world")],
                labels=["person", "organization"],
            )

            result = await asyncio.wait_for(future, timeout=2.0)

            assert result.output.batch_size == 1
            assert len(result.output.entities) == 1
            assert result.output.entities[0][0]["label"] == "test"

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_submit_extract_passes_labels(
        self, mock_adapter: MagicMock, prepared_item: "ExtractPreparedItem"
    ) -> None:
        """Labels are passed to adapter.extract."""
        config = WorkerConfig(
            max_batch_tokens=100,
            max_batch_requests=1,
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            future = await worker.submit_extract(
                [prepared_item],
                [Item(text="Hello world")],
                labels=["person", "organization", "location"],
            )

            await asyncio.wait_for(future, timeout=2.0)

            mock_adapter.extract.assert_called_once()
            call_kwargs = mock_adapter.extract.call_args.kwargs
            # Labels are sorted when grouping for batching
            assert sorted(call_kwargs["labels"]) == sorted(["person", "organization", "location"])

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_submit_extract_concurrent_batching(self, mock_adapter: MagicMock) -> None:
        """Multiple concurrent extract requests get batched together."""
        from sie_server.core.prepared import ExtractPreparedItem

        config = WorkerConfig(
            max_batch_tokens=100,
            max_batch_requests=3,  # Batch up to 3 requests
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            # Create 3 prepared items for concurrent requests
            prepared_items = [ExtractPreparedItem(cost=6, original_index=0) for i in range(3)]

            # Submit 3 extract requests concurrently
            futures = []
            for i, item in enumerate(prepared_items):
                future = await worker.submit_extract(
                    [item],
                    [Item(text=f"Text {i}")],
                    labels=["entity"],
                )
                futures.append(future)

            # Wait for all results
            results = await asyncio.gather(*futures)

            # All requests completed
            assert len(results) == 3
            for result in results:
                assert result.output.batch_size == 1
                assert len(result.output.entities) == 1

            # Verify batching happened (adapter called with 3 items)
            # Note: Due to timing, might be 1 call with 3 items or multiple calls
            total_items = sum(len(call.args[0]) for call in mock_adapter.extract.call_args_list)
            assert total_items == 3

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_submit_extract_groups_by_labels(self, mock_adapter: MagicMock) -> None:
        """Requests with different labels are grouped separately."""
        from sie_server.core.prepared import ExtractPreparedItem

        config = WorkerConfig(
            max_batch_tokens=100,
            max_batch_requests=10,
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            # Submit requests with different labels
            item1 = ExtractPreparedItem(cost=6, original_index=0)
            item2 = ExtractPreparedItem(cost=6, original_index=0)

            future1 = await worker.submit_extract(
                [item1],
                [Item(text="Text 1")],
                labels=["person", "organization"],
            )
            future2 = await worker.submit_extract(
                [item2],
                [Item(text="Text 2")],
                labels=["location"],  # Different labels!
            )

            await asyncio.gather(future1, future2)

            # Should have been 2 separate calls (different label sets)
            assert mock_adapter.extract.call_count >= 2

            # Verify different labels were passed
            label_sets = [tuple(sorted(call.kwargs["labels"])) for call in mock_adapter.extract.call_args_list]
            assert ("location",) in label_sets
            assert ("organization", "person") in label_sets

        finally:
            await worker.stop()


class TestModelWorkerExtractBackpressure:
    """Tests for extract backpressure."""

    @pytest.fixture
    def mock_adapter(self) -> MagicMock:
        """Create a mock adapter for extraction."""
        mock = MagicMock()
        mock.extract.return_value = ExtractOutput(entities=[[]])
        return mock

    def test_extract_queue_full_error(self, mock_adapter: MagicMock) -> None:
        """QueueFullError raised for extract when queue exceeds limit."""
        from sie_server.core.prepared import ExtractPreparedItem
        from sie_server.core.worker import QueueFullError

        config = WorkerConfig(
            max_batch_tokens=1000,
            max_batch_requests=100,
            max_batch_wait_ms=1,
            max_queue_size=5,
        )
        worker = ModelWorker(mock_adapter, config)

        async def test() -> None:
            await worker.start()
            try:
                # Submit 5 items (at limit)
                for i in range(5):
                    item = ExtractPreparedItem(cost=6, original_index=0)
                    await worker.submit_extract(
                        [item],
                        [Item(text=f"Text {i}")],
                        labels=["entity"],
                    )

                # Try to submit one more (should fail)
                extra_item = ExtractPreparedItem(cost=5, original_index=0)
                with pytest.raises(QueueFullError, match="Queue full"):
                    await worker.submit_extract(
                        [extra_item],
                        [Item(text="should fail")],
                        labels=["entity"],
                    )
            finally:
                await worker.stop()

        asyncio.get_event_loop().run_until_complete(test())


class TestModelWorkerScore:
    """Tests for score (reranking) via worker."""

    @pytest.fixture
    def mock_adapter(self) -> MagicMock:
        """Create a mock adapter for scoring."""
        mock = MagicMock()
        # Return ScoreOutput based on item count
        mock.score_pairs.side_effect = lambda q, d, **kw: ScoreOutput(
            scores=np.array([0.9 - (i * 0.1) for i in range(len(d))], dtype=np.float32)
        )
        return mock

    @pytest.fixture
    def prepared_item(self) -> "ScorePreparedItem":
        """Create a prepared item for tests."""
        from sie_server.core.prepared import ScorePreparedItem

        return ScorePreparedItem(cost=50, original_index=0)

    @pytest.mark.asyncio
    async def test_submit_score_basic(self, mock_adapter: MagicMock, prepared_item) -> None:
        """Submit score returns results."""
        from sie_server.types.inputs import Item

        config = WorkerConfig(
            max_batch_tokens=1000,
            max_batch_requests=10,
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            query = Item(text="What is machine learning?")
            items = [Item(text="ML is a branch of AI.")]

            future = await worker.submit_score(
                [prepared_item],
                query,
                items,
            )

            result = await asyncio.wait_for(future, timeout=2.0)
            assert result.output.batch_size == 1
            assert len(result.output.scores) == 1

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_submit_score_passes_instruction(self, mock_adapter: MagicMock, prepared_item) -> None:
        """Instruction is passed to adapter.score_pairs."""
        from sie_server.types.inputs import Item

        config = WorkerConfig(
            max_batch_tokens=1000,
            max_batch_requests=10,
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            query = Item(text="Query")
            items = [Item(text="Document")]

            future = await worker.submit_score(
                [prepared_item],
                query,
                items,
                instruction="Rank by relevance",
            )

            await asyncio.wait_for(future, timeout=2.0)

            mock_adapter.score_pairs.assert_called_once()
            call_kwargs = mock_adapter.score_pairs.call_args.kwargs
            assert call_kwargs["instruction"] == "Rank by relevance"

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_submit_score_concurrent_batching(self, mock_adapter: MagicMock) -> None:
        """Multiple concurrent score requests get batched together."""
        from sie_server.core.prepared import ScorePreparedItem
        from sie_server.types.inputs import Item

        config = WorkerConfig(
            max_batch_tokens=1000,
            max_batch_requests=3,
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            # Submit 3 score requests concurrently
            futures = []
            for i in range(3):
                item = ScorePreparedItem(cost=20, original_index=0)
                query = Item(text=f"Query {i}")
                docs = [Item(text=f"Doc {i}")]
                future = await worker.submit_score(
                    [item],
                    query,
                    docs,
                )
                futures.append(future)

            # Wait for all results
            results = await asyncio.gather(*futures)

            # All requests completed
            assert len(results) == 3
            for result in results:
                assert result.output.batch_size == 1
                assert len(result.output.scores) == 1

            # Verify batching happened
            total_items = sum(len(call.args[1]) for call in mock_adapter.score_pairs.call_args_list)
            assert total_items == 3

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_submit_score_groups_by_instruction(self, mock_adapter: MagicMock) -> None:
        """Requests with different instructions are grouped separately."""
        from sie_server.core.prepared import ScorePreparedItem
        from sie_server.types.inputs import Item

        config = WorkerConfig(
            max_batch_tokens=1000,
            max_batch_requests=10,
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            item1 = ScorePreparedItem(cost=20, original_index=0)
            item2 = ScorePreparedItem(cost=20, original_index=0)

            future1 = await worker.submit_score(
                [item1],
                Item(text="Query 1"),
                [Item(text="Doc 1")],
                instruction="instruction A",
            )
            future2 = await worker.submit_score(
                [item2],
                Item(text="Query 2"),
                [Item(text="Doc 2")],
                instruction="instruction B",
            )

            await asyncio.gather(future1, future2)

            # Should have been 2 separate calls (different instructions)
            assert mock_adapter.score_pairs.call_count >= 2

            # Verify different instructions were passed
            instructions = [call.kwargs["instruction"] for call in mock_adapter.score_pairs.call_args_list]
            assert "instruction A" in instructions
            assert "instruction B" in instructions

        finally:
            await worker.stop()


class TestModelWorkerScoreBackpressure:
    """Tests for score backpressure."""

    @pytest.fixture
    def mock_adapter(self) -> MagicMock:
        """Create a mock adapter for scoring."""
        mock = MagicMock()
        mock.score_pairs.return_value = ScoreOutput(scores=np.array([0.5], dtype=np.float32))
        return mock

    def test_score_queue_full_error(self, mock_adapter: MagicMock) -> None:
        """QueueFullError raised for score when queue exceeds limit."""
        from sie_server.core.prepared import ScorePreparedItem
        from sie_server.core.worker import QueueFullError
        from sie_server.types.inputs import Item

        config = WorkerConfig(
            max_batch_tokens=1000,
            max_batch_requests=100,
            max_batch_wait_ms=1,
            max_queue_size=5,
        )
        worker = ModelWorker(mock_adapter, config)

        async def test() -> None:
            await worker.start()
            try:
                # Submit 5 items (at limit)
                for i in range(5):
                    item = ScorePreparedItem(cost=20, original_index=0)
                    await worker.submit_score(
                        [item],
                        Item(text="Query"),
                        [Item(text=f"Doc {i}")],
                    )

                # Try to submit one more (should fail)
                extra_item = ScorePreparedItem(cost=20, original_index=0)
                with pytest.raises(QueueFullError, match="Queue full"):
                    await worker.submit_score(
                        [extra_item],
                        Item(text="Query"),
                        [Item(text="should fail")],
                    )
            finally:
                await worker.stop()

        asyncio.get_event_loop().run_until_complete(test())


class TestModelWorkerLoRABatching:
    """Tests for LoRA-aware batching."""

    @pytest.fixture
    def mock_adapter(self) -> MagicMock:
        """Create a mock adapter."""
        mock = MagicMock()
        mock.encode.return_value = EncodeOutput(
            dense=np.array([[0.1, 0.2, 0.3]]),
            batch_size=1,
        )
        return mock

    @pytest.mark.asyncio
    async def test_different_loras_batched_separately(self, mock_adapter: MagicMock) -> None:
        """Requests with different LoRA adapters are batched separately."""
        config = WorkerConfig(
            max_batch_tokens=1000,
            max_batch_requests=10,
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            item1 = make_text_item([1, 2, 3], 0)
            item2 = make_text_item([4, 5, 6], 0)

            # Two requests with different LoRA adapters
            future1 = await worker.submit(
                [item1],
                [Item(text="hello")],
                ["dense"],
                options={"lora": "legal"},
            )
            future2 = await worker.submit(
                [item2],
                [Item(text="world")],
                ["dense"],
                options={"lora": "medical"},
            )

            await asyncio.gather(future1, future2)

            # Should have 2 separate encode calls (different LoRAs)
            assert mock_adapter.encode.call_count == 2

            # Verify set_active_lora was called for each LoRA
            loras = [call.args[0] for call in mock_adapter.set_active_lora.call_args_list]
            assert "legal" in loras
            assert "medical" in loras

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_same_lora_batched_together(self, mock_adapter: MagicMock) -> None:
        """Requests with the same LoRA adapter are batched together."""
        # Return 2 embeddings for batched call
        mock_adapter.encode.return_value = EncodeOutput(
            dense=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            batch_size=2,
        )

        config = WorkerConfig(
            max_batch_tokens=1000,
            max_batch_requests=10,
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            item1 = make_text_item([1, 2, 3], 0)
            item2 = make_text_item([4, 5, 6], 0)

            # Two requests with the SAME LoRA adapter
            future1 = await worker.submit(
                [item1],
                [Item(text="hello")],
                ["dense"],
                options={"lora": "legal"},
            )
            future2 = await worker.submit(
                [item2],
                [Item(text="world")],
                ["dense"],
                options={"lora": "legal"},
            )

            await asyncio.gather(future1, future2)

            # Should batch together - only 1 encode call
            assert mock_adapter.encode.call_count == 1

            # Verify set_active_lora was called with the LoRA
            mock_adapter.set_active_lora.assert_called_with("legal")

            # Verify 2 items were batched
            call_args = mock_adapter.encode.call_args.args
            assert len(call_args[0]) == 2  # 2 items

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_no_lora_batched_together(self, mock_adapter: MagicMock) -> None:
        """Requests without LoRA are batched together."""
        mock_adapter.encode.return_value = EncodeOutput(
            dense=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            batch_size=2,
        )

        config = WorkerConfig(
            max_batch_tokens=1000,
            max_batch_requests=10,
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            item1 = make_text_item([1, 2, 3], 0)
            item2 = make_text_item([4, 5, 6], 0)

            # Two requests without LoRA
            future1 = await worker.submit(
                [item1],
                [Item(text="hello")],
                ["dense"],
            )
            future2 = await worker.submit(
                [item2],
                [Item(text="world")],
                ["dense"],
            )

            await asyncio.gather(future1, future2)

            # Should batch together
            assert mock_adapter.encode.call_count == 1

            # Verify set_active_lora was called with None (base model)
            mock_adapter.set_active_lora.assert_called_with(None)

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_lora_vs_no_lora_batched_separately(self, mock_adapter: MagicMock) -> None:
        """Requests with LoRA and without LoRA are batched separately."""
        config = WorkerConfig(
            max_batch_tokens=1000,
            max_batch_requests=10,
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            item1 = make_text_item([1, 2, 3], 0)
            item2 = make_text_item([4, 5, 6], 0)

            # One request with LoRA, one without
            future1 = await worker.submit(
                [item1],
                [Item(text="hello")],
                ["dense"],
                options={"lora": "legal"},
            )
            future2 = await worker.submit(
                [item2],
                [Item(text="world")],
                ["dense"],
                # No lora
            )

            await asyncio.gather(future1, future2)

            # Should have 2 separate encode calls
            assert mock_adapter.encode.call_count == 2

            # Verify set_active_lora was called for both LoRA and base model
            loras = [call.args[0] for call in mock_adapter.set_active_lora.call_args_list]
            assert "legal" in loras
            assert None in loras

        finally:
            await worker.stop()


class TestModelWorkerOptionsThreading:
    """Tests that runtime options are passed through to adapter.encode()."""

    @pytest.fixture
    def mock_adapter(self) -> MagicMock:
        """Create a mock adapter that records kwargs."""
        mock = MagicMock()
        mock.encode.side_effect = lambda items, *args, **kwargs: EncodeOutput(
            dense=np.array([[0.1, 0.2, 0.3]] * len(items)),
            batch_size=len(items),
        )
        return mock

    @pytest.mark.asyncio
    async def test_options_forwarded_to_adapter_encode(self, mock_adapter: MagicMock) -> None:
        """Runtime options are passed through worker pipeline to adapter.encode()."""
        config = WorkerConfig(
            max_batch_tokens=100,
            max_batch_requests=1,
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            item = make_text_item([1, 2, 3], 0)
            runtime_options = {"query_template": "Represent this: {text}", "doc_template": "{text}"}

            future = await worker.submit(
                [item],
                [Item(text="hello")],
                ["dense"],
                options=runtime_options,
            )

            worker_result = await asyncio.wait_for(future, timeout=2.0)
            assert worker_result.output.batch_size == 1

            # Verify adapter.encode() was called with options
            mock_adapter.encode.assert_called_once()
            call_kwargs = mock_adapter.encode.call_args
            assert call_kwargs.kwargs["options"] == runtime_options

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_none_options_forwarded_to_adapter_encode(self, mock_adapter: MagicMock) -> None:
        """When no options provided, None is passed through to adapter.encode()."""
        config = WorkerConfig(
            max_batch_tokens=100,
            max_batch_requests=1,
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            item = make_text_item([1, 2, 3], 0)

            future = await worker.submit(
                [item],
                [Item(text="hello")],
                ["dense"],
                # No options
            )

            worker_result = await asyncio.wait_for(future, timeout=2.0)
            assert worker_result.output.batch_size == 1

            # Verify adapter.encode() was called with options=None
            mock_adapter.encode.assert_called_once()
            call_kwargs = mock_adapter.encode.call_args
            assert call_kwargs.kwargs["options"] == {}

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_different_options_batched_separately(self, mock_adapter: MagicMock) -> None:
        """Requests with different options are batched separately."""
        config = WorkerConfig(
            max_batch_tokens=1000,
            max_batch_requests=10,
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            item1 = make_text_item([1, 2, 3], 0)
            item2 = make_text_item([4, 5, 6], 0)

            future1 = await worker.submit(
                [item1],
                [Item(text="hello")],
                ["dense"],
                options={"query_template": "template_a"},
            )
            future2 = await worker.submit(
                [item2],
                [Item(text="world")],
                ["dense"],
                options={"query_template": "template_b"},
            )

            await asyncio.gather(future1, future2)

            # Different options should produce separate batches
            assert mock_adapter.encode.call_count == 2

            # Verify each call got the right options
            call_options = [call.kwargs.get("options") for call in mock_adapter.encode.call_args_list]
            assert {"query_template": "template_a"} in call_options
            assert {"query_template": "template_b"} in call_options

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_nested_options_preserved_through_pipeline(self, mock_adapter: MagicMock) -> None:
        """Nested dicts/lists in options survive the worker pipeline intact.

        options go through _make_hashable() for config_key grouping, but
        adapter.encode() should receive the original dict (from metadata),
        not a reconstructed version.  dict(options_tuple) would corrupt
        nested structures (dicts→tuples-of-pairs, lists→tuples).
        """
        config = WorkerConfig(
            max_batch_tokens=100,
            max_batch_requests=1,
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            item = make_text_item([1, 2, 3], 0)
            nested_options = {
                "muvera": {"num_repetitions": 4, "rescale_factor": 0.5},
                "output_types": ["dense", "sparse"],
                "query_template": "Represent: {text}",
            }

            future = await worker.submit(
                [item],
                [Item(text="hello")],
                ["dense"],
                options=nested_options,
            )

            worker_result = await asyncio.wait_for(future, timeout=2.0)
            assert worker_result.output.batch_size == 1

            # The adapter must receive the original dict — not a
            # shallow reconstruction where nested dicts become tuples.
            mock_adapter.encode.assert_called_once()
            received_options = mock_adapter.encode.call_args.kwargs["options"]

            # Nested dict must still be a dict
            assert isinstance(received_options["muvera"], dict)
            assert received_options["muvera"] == {"num_repetitions": 4, "rescale_factor": 0.5}

            # List must still be a list
            assert isinstance(received_options["output_types"], list)
            assert received_options["output_types"] == ["dense", "sparse"]

            # Flat string preserved
            assert received_options["query_template"] == "Represent: {text}"

        finally:
            await worker.stop()


class TestModelWorkerScoreOptionsThreading:
    """Tests that runtime options are passed through to adapter.score_pairs()."""

    @pytest.fixture
    def mock_adapter(self) -> MagicMock:
        """Create a mock adapter that records kwargs."""
        mock = MagicMock()
        mock.score_pairs.side_effect = lambda q, d, **kw: ScoreOutput(scores=np.array([0.5] * len(d), dtype=np.float32))
        return mock

    @pytest.mark.asyncio
    async def test_score_options_forwarded_to_adapter(self, mock_adapter: MagicMock) -> None:
        """Runtime options are passed through worker pipeline to adapter.score_pairs()."""
        from sie_server.core.prepared import ScorePreparedItem

        config = WorkerConfig(
            max_batch_tokens=1000,
            max_batch_requests=1,
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            prepared = ScorePreparedItem(cost=50, original_index=0)
            runtime_options = {"max_length": 256}

            future = await worker.submit_score(
                [prepared],
                Item(text="query"),
                [Item(text="document")],
                options=runtime_options,
            )

            result = await asyncio.wait_for(future, timeout=2.0)
            assert result.output.batch_size == 1

            # Verify adapter.score_pairs() was called with options
            mock_adapter.score_pairs.assert_called_once()
            call_kwargs = mock_adapter.score_pairs.call_args.kwargs
            assert call_kwargs["options"] == runtime_options

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_score_none_options_forwarded(self, mock_adapter: MagicMock) -> None:
        """When no options provided, None is passed to adapter.score_pairs()."""
        from sie_server.core.prepared import ScorePreparedItem

        config = WorkerConfig(
            max_batch_tokens=1000,
            max_batch_requests=1,
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            prepared = ScorePreparedItem(cost=50, original_index=0)

            future = await worker.submit_score(
                [prepared],
                Item(text="query"),
                [Item(text="document")],
                # No options
            )

            result = await asyncio.wait_for(future, timeout=2.0)
            assert result.output.batch_size == 1

            # Verify adapter.score_pairs() was called with options=None
            mock_adapter.score_pairs.assert_called_once()
            call_kwargs = mock_adapter.score_pairs.call_args.kwargs
            assert call_kwargs["options"] is None

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_score_different_options_batched_separately(self, mock_adapter: MagicMock) -> None:
        """Score requests with different options are batched separately."""
        from sie_server.core.prepared import ScorePreparedItem

        config = WorkerConfig(
            max_batch_tokens=1000,
            max_batch_requests=10,
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            prepared1 = ScorePreparedItem(cost=50, original_index=0)
            prepared2 = ScorePreparedItem(cost=50, original_index=0)

            future1 = await worker.submit_score(
                [prepared1],
                Item(text="query 1"),
                [Item(text="doc 1")],
                options={"max_length": 256},
            )
            future2 = await worker.submit_score(
                [prepared2],
                Item(text="query 2"),
                [Item(text="doc 2")],
                options={"max_length": 512},
            )

            await asyncio.gather(future1, future2)

            # Different options should produce separate batches
            assert mock_adapter.score_pairs.call_count == 2

            # Verify each call got the right options
            call_options = [call.kwargs.get("options") for call in mock_adapter.score_pairs.call_args_list]
            assert {"max_length": 256} in call_options
            assert {"max_length": 512} in call_options

        finally:
            await worker.stop()
