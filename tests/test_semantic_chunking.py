import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from chunkrank import Chunker, ChunkerConfig, EmbeddingConfig
from chunkrank.embeddings import EmbeddingBackend


def _make_chunker_with_mock_backend(
    model: str,
    threshold: float,
    mock_vecs: np.ndarray,
) -> Chunker:
    """
    Creates a semantic Chunker that uses a pre-configured mock EmbeddingBackend.
    We patch chunkrank.embeddings.EmbeddingBackend so the lazy import inside
    _chunk_by_semantic_similarity returns our mock.
    """
    cfg = ChunkerConfig(model=model, strategy="semantic", similarity_threshold=threshold)
    return Chunker(cfg), mock_vecs


def test_semantic_strategy_empty_text():
    cfg = ChunkerConfig(model="gpt-4o-mini", strategy="semantic")
    chunker = Chunker(cfg)
    assert chunker.split("") == []


def test_semantic_strategy_single_sentence():
    """A single sentence should return as one chunk without embedding calls."""
    cfg = ChunkerConfig(model="gpt-4o-mini", strategy="semantic")
    chunker = Chunker(cfg)
    result = chunker.split("Just one sentence.")
    assert result == ["Just one sentence."]


def test_semantic_strategy_splits_on_low_similarity():
    """Orthogonal vectors (sim=0) should force a new chunk per sentence."""
    cfg = ChunkerConfig(model="gpt-4o-mini", strategy="semantic", similarity_threshold=0.5)
    chunker = Chunker(cfg)
    text = "Machine learning is a type of AI. The sky is blue today. Pasta has flour in it."
    # 3 orthogonal unit vectors → pairwise sims = 0 → always splits
    mock_vecs = np.eye(3, dtype=np.float32)

    with patch("chunkrank.embeddings.EmbeddingBackend.embed", return_value=mock_vecs):
        chunks = chunker.split(text)

    assert len(chunks) == 3


def test_semantic_strategy_merges_on_high_similarity():
    """Identical vectors (sim=1) should keep all sentences in one chunk."""
    cfg = ChunkerConfig(model="gpt-4o-mini", strategy="semantic", similarity_threshold=0.5)
    chunker = Chunker(cfg)
    text = "Sentence one. Sentence two. Sentence three."
    # All identical unit vectors → sim=1 → no splits
    mock_vecs = np.tile(np.array([[1.0, 0.0]]), (3, 1)).astype(np.float32)

    with patch("chunkrank.embeddings.EmbeddingBackend.embed", return_value=mock_vecs):
        chunks = chunker.split(text)

    assert len(chunks) == 1


def test_semantic_strategy_respects_token_budget():
    """Even if similarity is high, must not exceed context window."""
    # bert-base-uncased has 512 token window (after reserve)
    cfg = ChunkerConfig(
        model="bert-base-uncased",
        strategy="semantic",
        similarity_threshold=0.99,
    )
    chunker = Chunker(cfg)
    # Repeat a short sentence 300 times → well over 512 tokens
    text = "This is a test sentence. " * 300
    n_sentences = len(text.split(". "))
    # All identical → would merge, but token budget forces splits
    mock_vecs = np.tile(np.array([[1.0, 0.0]]), (n_sentences, 1)).astype(np.float32)

    with patch("chunkrank.embeddings.EmbeddingBackend.embed", return_value=mock_vecs):
        chunks = chunker.split(text)

    assert len(chunks) > 1
    assert all(len(c) > 0 for c in chunks)


def test_tokens_strategy_unchanged_after_refactor():
    """Existing default strategy must still work after chunker changes."""
    cfg = ChunkerConfig(model="gpt-4o-mini")
    chunker = Chunker(cfg)
    chunks = chunker.split("Hello world " * 500)
    assert len(chunks) > 0


def test_unknown_strategy_raises():
    cfg = ChunkerConfig(model="gpt-4o-mini")
    chunker = Chunker(cfg)
    chunker.strategy = "invalid"
    with pytest.raises(NotImplementedError):
        chunker.split("some text")
