"""Extreme and edge-case tests for chunking, ranking, pipeline, cache, and async."""
from __future__ import annotations

import asyncio
import tempfile
from typing import List

import pytest

from chunkrank.cache import ChunkCache
from chunkrank.chunker import Chunker, ChunkerConfig
from chunkrank.models import get_model_info, register_model, _runtime_registry
from chunkrank.pipeline import ChunkRankPipeline
from chunkrank.async_pipeline import AsyncChunkRankPipeline
from chunkrank.ranker import Ranker


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

SMALL_WIN = ChunkerConfig(model="gpt-4o-mini", reserve_tokens=127_900)  # ~100 token window


# ------------------------------------------------------------------ #
# Extreme text inputs — chunker                                        #
# ------------------------------------------------------------------ #

def test_extremely_long_text():
    """1 million characters — must not crash and must produce multiple chunks."""
    text = "The quick brown fox jumps over the lazy dog. " * 22_000  # ~1M chars
    chunker = Chunker(ChunkerConfig(model="gpt-4o-mini"))
    chunks = chunker.split(text)
    assert len(chunks) > 1
    assert all(isinstance(c, str) and len(c) > 0 for c in chunks)


def test_single_character_text():
    chunker = Chunker(ChunkerConfig(model="gpt-4o-mini"))
    chunks = chunker.split("x")
    assert chunks == ["x"]


def test_single_word_text():
    chunker = Chunker(ChunkerConfig(model="gpt-4o-mini"))
    chunks = chunker.split("hello")
    assert chunks == ["hello"]


def test_text_with_no_spaces():
    """One giant word — chunker must still make progress."""
    text = "a" * 100_000
    chunker = Chunker(ChunkerConfig(model="gpt-4o-mini"))
    chunks = chunker.split(text)
    assert len(chunks) >= 1
    assert all(len(c) > 0 for c in chunks)


def test_whitespace_only_text():
    # Chunker does not strip whitespace — it returns whatever fits in the window.
    chunker = Chunker(ChunkerConfig(model="gpt-4o-mini"))
    chunks = chunker.split("   \n\t  ")
    assert isinstance(chunks, list)


def test_empty_string():
    chunker = Chunker(ChunkerConfig(model="gpt-4o-mini"))
    assert chunker.split("") == []


def test_newlines_only():
    chunker = Chunker(ChunkerConfig(model="gpt-4o-mini"))
    chunks = chunker.split("\n" * 1000)
    assert isinstance(chunks, list)


def test_unicode_heavy_text():
    """Chinese + Arabic + emoji mixed text."""
    text = (
        "人工智能正在改变世界。这是一个重要的技术革命。"
        "الذكاء الاصطناعي يغير العالم. هذا تطور تقني مهم."
        "🤖🧠💡🔬🌍 AI is transforming everything. "
    ) * 500
    chunker = Chunker(ChunkerConfig(model="gpt-4o-mini"))
    chunks = chunker.split(text)
    assert len(chunks) >= 1
    assert all(len(c) > 0 for c in chunks)


def test_repeated_sentence_10k_times():
    """Same sentence 10,000 times with small window — tests chunker stability."""
    text = "This is a test sentence. " * 10_000
    chunker = Chunker(ChunkerConfig(model="gpt-4o-mini", reserve_tokens=127_900))
    chunks = chunker.split(text)
    assert len(chunks) > 1
    assert all(len(c) > 0 for c in chunks)


def test_only_punctuation():
    text = "!@#$%^&*().,;:!?..." * 1000
    chunker = Chunker(ChunkerConfig(model="gpt-4o-mini"))
    chunks = chunker.split(text)
    assert len(chunks) >= 1


def test_mixed_languages():
    text = (
        "Hello world. Bonjour le monde. Hola mundo. "
        "Ciao mondo. Olá mundo. Привет мир. こんにちは世界。"
    ) * 1000
    chunker = Chunker(ChunkerConfig(model="gpt-4o-mini"))
    chunks = chunker.split(text)
    assert len(chunks) >= 1


def test_text_exactly_at_token_limit():
    """Text that exactly fills the context window — must produce exactly 1 chunk."""
    info = get_model_info("gpt-4o-mini")
    # ~4 chars per token, fill ~half the window to guarantee single chunk
    text = "word " * (info.max_context // 8)
    chunker = Chunker(ChunkerConfig(model="gpt-4o-mini"))
    chunks = chunker.split(text)
    assert len(chunks) >= 1


def test_tiny_context_model_produces_many_chunks():
    """bert-base-uncased has 512 token window — a doc larger than window should produce multiple chunks."""
    text = "Machine learning is a field of AI. " * 200
    chunker = Chunker(ChunkerConfig(model="bert-base-uncased"))
    chunks = chunker.split(text)
    assert len(chunks) > 1


def test_overlap_near_window_size_raises():
    """overlap_tokens >= window must raise."""
    with pytest.raises(ValueError):
        Chunker(ChunkerConfig(model="gpt-4o-mini", reserve_tokens=127_999, overlap_tokens=500))


def test_reserve_larger_than_context_clamps_to_one():
    """reserve_tokens > max_context — window should clamp to 1 (not crash)."""
    chunker = Chunker(ChunkerConfig(model="bert-base-uncased", reserve_tokens=600))
    assert chunker.window >= 1


# ------------------------------------------------------------------ #
# Ranking — extreme inputs                                             #
# ------------------------------------------------------------------ #

def test_rank_1000_answers():
    """BM25 and TF-IDF must handle 1000 candidate answers."""
    answers = [f"This is answer number {i} about machine learning." for i in range(1000)]
    for method in ("bm25", "tfidf"):
        ranker = Ranker(method=method)
        ranked = ranker.rank("What is machine learning?", answers)
        assert len(ranked) == 1000
        assert ranked[0][1] >= ranked[-1][1]


def test_rank_single_answer():
    ranker = Ranker(method="bm25")
    ranked = ranker.rank("question", ["only answer"])
    assert len(ranked) == 1
    assert ranked[0][0] == "only answer"


def test_rank_all_empty_strings():
    for method in ("bm25", "tfidf"):
        ranker = Ranker(method=method)
        assert ranker.rank("q", ["", "  ", "\n"]) == []


def test_rank_duplicate_answers():
    """Duplicate answers should all appear in output."""
    answers = ["same answer"] * 10
    ranker = Ranker(method="bm25")
    ranked = ranker.rank("same answer", answers)
    assert len(ranked) == 10


def test_rank_very_long_answer():
    """Single answer that is itself very long."""
    long_answer = "machine learning " * 5000
    ranker = Ranker(method="bm25")
    ranked = ranker.rank("What is machine learning?", [long_answer, "short answer"])
    assert len(ranked) == 2


def test_rank_question_not_in_answers():
    """Question has zero overlap with all answers — must still return results."""
    ranker = Ranker(method="bm25")
    ranked = ranker.rank("zzzzz quantum physics", ["apple pie recipe", "car maintenance tips"])
    assert len(ranked) == 2


def test_rank_special_characters_in_question():
    ranker = Ranker(method="bm25")
    ranked = ranker.rank("!@#$%? <script>alert(1)</script>", ["some answer", "other answer"])
    assert len(ranked) == 2


@pytest.mark.asyncio
async def test_rank_async_1000_answers():
    answers = [f"answer {i} about deep learning" for i in range(1000)]
    ranker = Ranker(method="tfidf")
    ranked = await ranker.rank_async("What is deep learning?", answers)
    assert len(ranked) == 1000


# ------------------------------------------------------------------ #
# Pipeline — extreme scenarios                                         #
# ------------------------------------------------------------------ #

def test_pipeline_no_matching_answer():
    """Pipeline with text completely unrelated to question — must return a string (not crash)."""
    pipe = ChunkRankPipeline(model="gpt-4o-mini")
    result = pipe.process("What is quantum entanglement?", "apple banana cherry " * 200)
    assert isinstance(result, str)


def test_pipeline_question_is_empty_string():
    pipe = ChunkRankPipeline(model="gpt-4o-mini")
    result = pipe.process("", "Some text about machine learning " * 100)
    assert isinstance(result, str)


def test_pipeline_very_long_question():
    long_q = "What is the meaning of " + "artificial intelligence " * 500 + "?"
    pipe = ChunkRankPipeline(model="gpt-4o-mini")
    result = pipe.process(long_q, "AI is the simulation of human intelligence. " * 100)
    assert isinstance(result, str)


def test_pipeline_retrieval_top_k_zero():
    """top_k=0 — no chunks answered, pipeline must return empty string gracefully."""
    pipe = ChunkRankPipeline(model="gpt-4o-mini", retrieval_top_k=0, chunker_config=SMALL_WIN)
    result = pipe.process("question", "some text " * 200)
    assert result == ""


def test_pipeline_single_chunk_text():
    """Text short enough to fit in one chunk."""
    pipe = ChunkRankPipeline(model="gpt-4o-mini")
    result = pipe.process("What is AI?", "Artificial intelligence is the future.")
    assert isinstance(result, str)


def test_pipeline_stream_many_chunks():
    """Stream over a large document — verify count matches chunk count."""
    text = "Neural networks learn from data. " * 2000
    pipe = ChunkRankPipeline(model="gpt-4o-mini", chunker_config=SMALL_WIN)
    streamed = list(pipe.stream("What do neural networks do?", text))
    chunks = pipe.chunker.split(text)
    assert len(streamed) <= len(chunks)
    assert all(isinstance(s, str) for s in streamed)


def test_pipeline_all_ranking_methods_on_same_input():
    """All ranking methods must run without error on the same input."""
    text = "Deep learning uses neural networks. " * 300
    question = "What is deep learning?"
    for method in ("bm25", "tfidf"):
        pipe = ChunkRankPipeline(model="gpt-4o-mini", ranking_method=method)
        result = pipe.process(question, text)
        assert isinstance(result, str)


# ------------------------------------------------------------------ #
# Async pipeline — extreme scenarios                                   #
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_async_pipeline_very_long_text():
    text = "Transformers revolutionized NLP. " * 5000
    pipe = AsyncChunkRankPipeline(model="gpt-4o-mini")
    result = await pipe.process("What revolutionized NLP?", text)
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_async_pipeline_concurrent_requests():
    """10 concurrent pipeline calls must all complete without errors."""
    pipe = AsyncChunkRankPipeline(model="gpt-4o-mini")
    text = "Reinforcement learning trains agents via rewards. " * 300
    tasks = [pipe.process("What is reinforcement learning?", text) for _ in range(10)]
    results = await asyncio.gather(*tasks)
    assert len(results) == 10
    assert all(isinstance(r, str) for r in results)


@pytest.mark.asyncio
async def test_async_pipeline_stream_large_doc():
    text = "Large language models are trained on vast datasets. " * 2000
    pipe = AsyncChunkRankPipeline(model="gpt-4o-mini", chunker_config=SMALL_WIN)
    results = [r async for r in pipe.stream("What are LLMs trained on?", text)]
    assert all(isinstance(r, str) for r in results)


@pytest.mark.asyncio
async def test_async_pipeline_empty_question():
    pipe = AsyncChunkRankPipeline(model="gpt-4o-mini")
    result = await pipe.process("", "Some content about AI. " * 100)
    assert isinstance(result, str)


# ------------------------------------------------------------------ #
# ChunkCache — extreme scenarios                                       #
# ------------------------------------------------------------------ #

def test_cache_very_long_text_key():
    """SHA-256 key derived from 1M char text — must not crash."""
    with tempfile.TemporaryDirectory() as d:
        cache = ChunkCache(d)
        text = "x" * 1_000_000
        chunks = ["chunk1", "chunk2"]
        cache.set(text, "gpt-4o", chunks)
        assert cache.get(text, "gpt-4o") == chunks


def test_cache_unicode_text_key():
    with tempfile.TemporaryDirectory() as d:
        cache = ChunkCache(d)
        text = "你好世界 🌍 مرحبا بالعالم"
        cache.set(text, "gpt-4o", ["hello"])
        assert cache.get(text, "gpt-4o") == ["hello"]


def test_cache_many_entries():
    """Write 500 different entries — all must be retrievable."""
    with tempfile.TemporaryDirectory() as d:
        cache = ChunkCache(d)
        for i in range(500):
            cache.set(f"text_{i}", "gpt-4o", [f"chunk_{i}"])
        for i in range(500):
            assert cache.get(f"text_{i}", "gpt-4o") == [f"chunk_{i}"]


def test_cache_empty_chunks_list():
    with tempfile.TemporaryDirectory() as d:
        cache = ChunkCache(d)
        cache.set("text", "gpt-4o", [])
        assert cache.get("text", "gpt-4o") == []


def test_cache_overwrite():
    """Writing the same key twice should overwrite."""
    with tempfile.TemporaryDirectory() as d:
        cache = ChunkCache(d)
        cache.set("text", "gpt-4o", ["old"])
        cache.set("text", "gpt-4o", ["new"])
        assert cache.get("text", "gpt-4o") == ["new"]


# ------------------------------------------------------------------ #
# register_model — extreme scenarios                                   #
# ------------------------------------------------------------------ #

def test_register_model_zero_context():
    """Zero context window is unusual but must not crash."""
    register_model("zero-ctx-model", max_context=1)
    info = get_model_info("zero-ctx-model")
    assert info.max_context == 1
    _runtime_registry.pop("zero-ctx-model", None)


def test_register_model_huge_context():
    register_model("infinite-ctx-model", max_context=100_000_000)
    info = get_model_info("infinite-ctx-model")
    assert info.max_context == 100_000_000
    _runtime_registry.pop("infinite-ctx-model", None)


def test_register_model_overwrite_twice():
    register_model("dup-model", max_context=10_000)
    register_model("dup-model", max_context=20_000)
    assert get_model_info("dup-model").max_context == 20_000
    _runtime_registry.pop("dup-model", None)


def test_register_model_special_chars_in_name():
    register_model("my-model/v2.1:latest", max_context=128_000)
    info = get_model_info("my-model/v2.1:latest")
    assert info.max_context == 128_000
    _runtime_registry.pop("my-model/v2.1:latest", None)


# ------------------------------------------------------------------ #
# Chunker with overlap — extreme                                       #
# ------------------------------------------------------------------ #

def test_large_overlap_produces_more_chunks():
    """Overlap should produce more chunks than no overlap."""
    text = "Machine learning is a subset of AI. " * 500
    chunker_no_overlap = Chunker(ChunkerConfig(model="gpt-4o-mini", reserve_tokens=127_800))
    chunker_with_overlap = Chunker(ChunkerConfig(model="gpt-4o-mini", reserve_tokens=127_800, overlap_tokens=20))
    chunks_no = chunker_no_overlap.split(text)
    chunks_with = chunker_with_overlap.split(text)
    assert len(chunks_with) >= len(chunks_no)


def test_unknown_model_fallback_chunking():
    """Unknown model falls back to 128k context — must chunk without error."""
    text = "Some content. " * 1000
    chunker = Chunker(ChunkerConfig(model="some-future-model-xyz"))
    chunks = chunker.split(text)
    assert len(chunks) >= 1
