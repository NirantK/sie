"""Tests for SDK scoring module and client-side MaxSim."""

import numpy as np
import pytest
from sie_sdk.scoring import maxsim, maxsim_batch

# Create a random generator for tests
_RNG = np.random.default_rng(42)


class TestMaxSim:
    """Tests for maxsim() function."""

    def test_maxsim_basic(self) -> None:
        """MaxSim returns correct number of scores."""
        query = _RNG.standard_normal((5, 128)).astype(np.float32)  # 5 tokens
        docs = [
            _RNG.standard_normal((10, 128)).astype(np.float32),  # doc 1: 10 tokens
            _RNG.standard_normal((8, 128)).astype(np.float32),  # doc 2: 8 tokens
        ]

        scores = maxsim(query, docs)

        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)

    def test_maxsim_single_document(self) -> None:
        """MaxSim works with a single document as 2D array."""
        query = _RNG.standard_normal((3, 64)).astype(np.float32)
        doc = _RNG.standard_normal((5, 64)).astype(np.float32)

        scores = maxsim(query, doc)

        assert len(scores) == 1

    def test_maxsim_identical_vectors(self) -> None:
        """MaxSim of identical normalized vectors equals num_query_tokens."""
        # Create normalized vectors
        query = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        doc = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        scores = maxsim(query, [doc])

        # Each query token has max sim of 1.0 with matching doc token, so total is 2.0
        assert abs(scores[0] - 2.0) < 1e-5

    def test_maxsim_orthogonal_vectors(self) -> None:
        """MaxSim of orthogonal vectors is lower."""
        query = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        # Doc 1: identical to query
        doc1 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        # Doc 2: rotated 45 degrees
        doc2 = np.array([[0.707, 0.707], [-0.707, 0.707]], dtype=np.float32)

        scores = maxsim(query, [doc1, doc2])

        # Doc 1 should have higher score (perfect match)
        assert scores[0] > scores[1]

    def test_maxsim_ranking_order(self) -> None:
        """MaxSim correctly ranks documents by similarity."""
        # Single-token query pointing in x direction
        query = np.array([[1.0, 0.0]], dtype=np.float32)

        # Docs with decreasing similarity to query
        doc_high = np.array([[1.0, 0.0]], dtype=np.float32)  # perfect match
        doc_mid = np.array([[0.707, 0.707]], dtype=np.float32)  # 45 degree
        doc_low = np.array([[0.0, 1.0]], dtype=np.float32)  # orthogonal

        scores = maxsim(query, [doc_high, doc_mid, doc_low])

        assert scores[0] > scores[1] > scores[2]

    def test_maxsim_with_variable_doc_lengths(self) -> None:
        """MaxSim handles documents with different token counts."""
        query = _RNG.standard_normal((4, 32)).astype(np.float32)
        docs = [
            _RNG.standard_normal((1, 32)).astype(np.float32),  # 1 token
            _RNG.standard_normal((10, 32)).astype(np.float32),  # 10 tokens
            _RNG.standard_normal((100, 32)).astype(np.float32),  # 100 tokens
        ]

        scores = maxsim(query, docs)

        assert len(scores) == 3
        # All scores should be finite
        assert all(np.isfinite(s) for s in scores)


class TestMaxSimBatch:
    """Tests for maxsim_batch() function."""

    def test_batch_shape(self) -> None:
        """Batch maxsim returns correct shape."""
        queries = [
            _RNG.standard_normal((3, 64)).astype(np.float32),
            _RNG.standard_normal((5, 64)).astype(np.float32),
        ]
        docs = [
            _RNG.standard_normal((4, 64)).astype(np.float32),
            _RNG.standard_normal((6, 64)).astype(np.float32),
            _RNG.standard_normal((8, 64)).astype(np.float32),
        ]

        scores = maxsim_batch(queries, docs)

        assert scores.shape == (2, 3)  # 2 queries x 3 docs

    def test_batch_matches_individual(self) -> None:
        """Batch results match individual maxsim calls."""
        queries = [
            _RNG.standard_normal((3, 32)).astype(np.float32),
            _RNG.standard_normal((4, 32)).astype(np.float32),
        ]
        docs = [
            _RNG.standard_normal((5, 32)).astype(np.float32),
            _RNG.standard_normal((6, 32)).astype(np.float32),
        ]

        batch_scores = maxsim_batch(queries, docs)

        # Compare with individual calls
        for i, query in enumerate(queries):
            individual_scores = maxsim(query, docs)
            for j, score in enumerate(individual_scores):
                assert abs(batch_scores[i, j] - score) < 1e-5


class TestClientSideMaxSim:
    """Tests for SIEClient.score() with pre-encoded multivectors."""

    def test_client_side_maxsim_basic(self) -> None:
        """Client-side MaxSim returns ScoreResult."""
        from sie_sdk.client import SIEClient

        # Create client (won't make any server calls for client-side scoring)
        client = SIEClient("http://localhost:8080")

        # Create pre-encoded multivectors
        query_mv = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        doc1_mv = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        doc2_mv = np.array([[0.5, 0.5], [0.5, -0.5]], dtype=np.float32)

        result = client.score(
            "test-colbert",
            query={"multivector": query_mv},
            items=[
                {"id": "doc-1", "multivector": doc1_mv},
                {"id": "doc-2", "multivector": doc2_mv},
            ],
        )

        assert result["model"] == "test-colbert"
        assert len(result["scores"]) == 2
        # Doc 1 should be ranked higher (identical to query)
        assert result["scores"][0]["item_id"] == "doc-1"
        assert result["scores"][0]["rank"] == 0
        assert result["scores"][1]["item_id"] == "doc-2"
        assert result["scores"][1]["rank"] == 1

    def test_client_side_maxsim_with_query_id(self) -> None:
        """Query ID is preserved in client-side scoring."""
        from sie_sdk.client import SIEClient

        client = SIEClient("http://localhost:8080")

        query_mv = np.array([[1.0, 0.0]], dtype=np.float32)
        doc_mv = np.array([[1.0, 0.0]], dtype=np.float32)

        result = client.score(
            "test-colbert",
            query={"id": "query-123", "multivector": query_mv},
            items=[{"multivector": doc_mv}],
        )

        assert result["query_id"] == "query-123"

    def test_client_side_maxsim_generates_item_ids(self) -> None:
        """Item IDs are generated if not provided."""
        from sie_sdk.client import SIEClient

        client = SIEClient("http://localhost:8080")

        query_mv = np.array([[1.0, 0.0]], dtype=np.float32)
        doc1_mv = np.array([[1.0, 0.0]], dtype=np.float32)
        doc2_mv = np.array([[0.0, 1.0]], dtype=np.float32)

        result = client.score(
            "test-colbert",
            query={"multivector": query_mv},
            items=[{"multivector": doc1_mv}, {"multivector": doc2_mv}],
        )

        item_ids = {s["item_id"] for s in result["scores"]}
        assert "item-0" in item_ids
        assert "item-1" in item_ids

    def test_mixed_mode_raises_error(self) -> None:
        """Mixing text and multivector items raises ValueError."""
        from sie_sdk.client import SIEClient

        client = SIEClient("http://localhost:8080")

        query_mv = np.array([[1.0, 0.0]], dtype=np.float32)

        with pytest.raises(ValueError, match="Cannot mix text and multivector"):
            client.score(
                "test-colbert",
                query={"multivector": query_mv},
                items=[{"text": "some text"}],  # text item with multivector query
            )

    def test_scores_sorted_descending(self) -> None:
        """Scores are sorted by relevance (descending)."""
        from sie_sdk.client import SIEClient

        client = SIEClient("http://localhost:8080")

        query_mv = np.array([[1.0, 0.0]], dtype=np.float32)
        # Create docs with known similarity order
        docs = [
            {"id": "low", "multivector": np.array([[0.0, 1.0]], dtype=np.float32)},
            {"id": "high", "multivector": np.array([[1.0, 0.0]], dtype=np.float32)},
            {"id": "mid", "multivector": np.array([[0.707, 0.707]], dtype=np.float32)},
        ]

        result = client.score("test-colbert", query={"multivector": query_mv}, items=docs)

        # Should be sorted: high > mid > low
        assert result["scores"][0]["item_id"] == "high"
        assert result["scores"][1]["item_id"] == "mid"
        assert result["scores"][2]["item_id"] == "low"
