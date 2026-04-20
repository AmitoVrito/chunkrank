"""Tests for ChunkCache, register_model, pipeline.stream, cross-encoder reranking."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from chunkrank.cache import ChunkCache
from chunkrank.models import register_model, get_model_info, _runtime_registry
from chunkrank.ranker import Ranker
from chunkrank.pipeline import ChunkRankPipeline
from chunkrank.async_pipeline import AsyncChunkRankPipeline


# ------------------------------------------------------------------ #
# ChunkCache                                                           #
# ------------------------------------------------------------------ #

def test_cache_miss_returns_none():
    with tempfile.TemporaryDirectory() as d:
        cache = ChunkCache(d)
        assert cache.get("some text", "gpt-4o") is None


def test_cache_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        cache = ChunkCache(d)
        chunks = ["chunk one", "chunk two", "chunk three"]
        cache.set("some text", "gpt-4o", chunks)
        assert cache.get("some text", "gpt-4o") == chunks


def test_cache_different_models_different_keys():
    with tempfile.TemporaryDirectory() as d:
        cache = ChunkCache(d)
        cache.set("hello", "gpt-4o", ["a"])
        cache.set("hello", "claude-sonnet-4-6", ["b"])
        assert cache.get("hello", "gpt-4o") == ["a"]
        assert cache.get("hello", "claude-sonnet-4-6") == ["b"]


def test_cache_clear():
    with tempfile.TemporaryDirectory() as d:
        cache = ChunkCache(d)
        cache.set("text", "gpt-4o", ["x"])
        cache.clear()
        assert cache.get("text", "gpt-4o") is None


def test_cache_creates_dir_if_missing():
    with tempfile.TemporaryDirectory() as d:
        new_dir = os.path.join(d, "nested", "cache")
        cache = ChunkCache(new_dir)
        assert os.path.isdir(new_dir)


# ------------------------------------------------------------------ #
# register_model                                                       #
# ------------------------------------------------------------------ #

def test_register_model_overrides_fallback():
    register_model("test-model-xyz", max_context=999_000)
    info = get_model_info("test-model-xyz")
    assert info.max_context == 999_000
    # cleanup
    _runtime_registry.pop("test-model-xyz", None)


def test_register_model_custom_tokenizer():
    register_model(
        "custom-model",
        max_context=50_000,
        tokenizer="tiktoken",
        tokenizer_id="cl100k_base",
        default_reserve=128,
    )
    info = get_model_info("custom-model")
    assert info.tokenizer_id == "cl100k_base"
    assert info.default_reserve == 128
    _runtime_registry.pop("custom-model", None)


def test_register_model_takes_priority_over_registry():
    # Override an existing registry model
    register_model("gpt-4o", max_context=256_000)
    info = get_model_info("gpt-4o")
    assert info.max_context == 256_000
    # cleanup — restore to registry value
    _runtime_registry.pop("gpt-4o", None)


# ------------------------------------------------------------------ #
# pipeline.stream                                                      #
# ------------------------------------------------------------------ #

def test_pipeline_stream_yields_strings():
    doc = (
        "Chunking is the process of splitting text into smaller pieces. "
        "Each piece fits within a model's context window. "
        "Ranking helps find the most relevant piece. "
    ) * 10
    pipeline = ChunkRankPipeline(model="gpt-4o-mini")
    results = list(pipeline.stream("What is chunking?", doc))
    assert len(results) > 0
    assert all(isinstance(r, str) for r in results)


def test_pipeline_stream_empty_text():
    pipeline = ChunkRankPipeline(model="gpt-4o-mini")
    results = list(pipeline.stream("question", ""))
    assert results == []


# ------------------------------------------------------------------ #
# async_pipeline.stream                                                #
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_async_pipeline_stream_yields_strings():
    doc = (
        "Async chunking enables non-blocking text processing. "
        "It uses asyncio to run tasks concurrently. "
        "This improves throughput in I/O-heavy pipelines. "
    ) * 10
    pipeline = AsyncChunkRankPipeline(model="gpt-4o-mini")
    results = [r async for r in pipeline.stream("What is async chunking?", doc)]
    assert len(results) > 0
    assert all(isinstance(r, str) for r in results)


@pytest.mark.asyncio
async def test_async_pipeline_stream_empty_text():
    pipeline = AsyncChunkRankPipeline(model="gpt-4o-mini")
    results = [r async for r in pipeline.stream("question", "")]
    assert results == []


# ------------------------------------------------------------------ #
# cross-encoder reranking (mocked)                                     #
# ------------------------------------------------------------------ #

def _make_cross_encoder_ranker(scores: list) -> Ranker:
    ranker = Ranker(method="cross-encoder")
    mock_ce = MagicMock()
    mock_ce.predict.return_value = np.array(scores, dtype=np.float32)
    ranker._cross_encoder = mock_ce
    return ranker


def test_cross_encoder_orders_correctly():
    ranker = _make_cross_encoder_ranker([0.1, 0.9, 0.5])
    answers = ["bad answer", "best answer", "ok answer"]
    ranked = ranker.rank("question", answers)
    assert ranked[0][0] == "best answer"
    assert ranked[-1][0] == "bad answer"


def test_cross_encoder_scores_returned():
    ranker = _make_cross_encoder_ranker([0.3, 0.7])
    ranked = ranker.rank("q", ["a1", "a2"])
    assert len(ranked) == 2
    assert ranked[0][1] > ranked[1][1]


def test_cross_encoder_empty_answers():
    ranker = Ranker(method="cross-encoder")
    assert ranker.rank("question", []) == []
    assert ranker.rank("question", ["", "  "]) == []


@pytest.mark.asyncio
async def test_cross_encoder_async():
    ranker = _make_cross_encoder_ranker([0.2, 0.8])
    ranked = await ranker.rank_async("question", ["a1", "a2"])
    assert ranked[0][0] == "a2"


# ------------------------------------------------------------------ #
# Model registry — new models present                                  #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("model", [
    "gpt-4o", "o1", "o3", "o3-mini", "o4-mini",
    "claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5-20251001",
    "gemini-1.5-pro", "gemini-2.0-flash", "gemini-2.5-pro",
    "Llama-3.3-70B", "Llama-4-Scout",
    "mistral-large-latest", "command-r-plus",
    "deepseek-v3", "deepseek-r1",
    "qwen2.5-72b-instruct",
])
def test_known_models_in_registry(model):
    info = get_model_info(model)
    assert info.max_context > 0
    assert info.name != ""
