import asyncio
import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock
from chunkrank.ranker import Ranker
from chunkrank.embeddings import EmbeddingConfig


def _mock_backend(all_vecs: np.ndarray) -> MagicMock:
    """Returns a mock EmbeddingBackend with embed() returning all_vecs."""
    mock = MagicMock()
    mock.embed.return_value = all_vecs
    return mock


def _make_ranker_with_mock(vecs: np.ndarray) -> Ranker:
    """Creates an embedding Ranker with a pre-injected mock backend."""
    ranker = Ranker(method="embedding")
    ranker._backend = _mock_backend(vecs)
    return ranker


def test_embedding_ranker_orders_correctly():
    """Most similar answer to question should rank first."""
    # q=[1,0,0], a=[1,0,0] (sim=1.0), b=[0,1,0] (sim=0.0)
    q_vec = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    a_vecs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    ranker = _make_ranker_with_mock(np.vstack([q_vec, a_vecs]))

    ranked = ranker.rank("question", ["answer_a", "answer_b"])

    assert ranked[0][0] == "answer_a"
    assert ranked[0][1] > ranked[1][1]


def test_embedding_ranker_scores_in_range():
    """Cosine similarities must be in [-1, 1] for normalised vectors."""
    q_vec = np.array([[1.0, 0.0]], dtype=np.float32)
    a_vecs = np.array([[0.6, 0.8], [0.8, 0.6]], dtype=np.float32)
    ranker = _make_ranker_with_mock(np.vstack([q_vec, a_vecs]))

    ranked = ranker.rank("q", ["a1", "a2"])
    for _, score in ranked:
        assert -1.0 <= score <= 1.0


def test_embedding_ranker_empty_answers():
    ranker = Ranker(method="embedding")
    assert ranker.rank("question", []) == []
    assert ranker.rank("question", ["", "  "]) == []


def test_embedding_ranker_filters_empty_strings():
    """Empty/whitespace answers must be ignored."""
    q_vec = np.array([[1.0, 0.0]], dtype=np.float32)
    a_vecs = np.array([[1.0, 0.0]], dtype=np.float32)
    ranker = _make_ranker_with_mock(np.vstack([q_vec, a_vecs]))

    ranked = ranker.rank("question", ["", "real answer", ""])

    assert len(ranked) == 1
    assert ranked[0][0] == "real answer"


def test_bm25_unchanged():
    ranker = Ranker(method="bm25")
    ranked = ranker.rank("machine learning", ["ML is great", "pasta is tasty"])
    assert len(ranked) == 2


def test_tfidf_unchanged():
    ranker = Ranker(method="tfidf")
    ranked = ranker.rank("machine learning", ["ML models learn from data", "the sky is blue"])
    assert len(ranked) == 2


def test_unknown_method_raises():
    ranker = Ranker(method="unknown")
    with pytest.raises(ValueError):
        ranker.rank("q", ["a"])


@pytest.mark.asyncio
async def test_rank_async_bm25():
    ranker = Ranker(method="bm25")
    ranked = await ranker.rank_async("machine learning", ["ML is great", "pasta"])
    assert ranked[0][0] == "ML is great"


@pytest.mark.asyncio
async def test_rank_async_embedding():
    q_vec = np.array([[1.0, 0.0]], dtype=np.float32)
    a_vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    all_vecs = np.vstack([q_vec, a_vecs])

    ranker = Ranker(method="embedding")
    mock_backend = MagicMock()
    mock_backend.embed_async = AsyncMock(return_value=all_vecs)
    ranker._backend = mock_backend

    ranked = await ranker.rank_async("question", ["answer_a", "answer_b"])
    assert ranked[0][0] == "answer_a"
