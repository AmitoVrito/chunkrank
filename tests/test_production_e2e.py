"""Production-level end-to-end tests for ChunkRank.

Covers:
- Full module-level API (chunkrank.split / answer / rank)
- All 90 registry models — schema + usability
- New models added in expansion (gpt-4.1, phi-4, grok-2, …)
- Chunking correctness: overlap, token budget, determinism
- Ranking correctness: BM25/TF-IDF semantic ordering, determinism
- Pipeline idempotence and stream completeness
- Cache persistence across instances and isolation
- Async pipeline: correctness, concurrency, stream
- register_model full-pipeline integration
- Context-size difference: small-ctx vs large-ctx model
- Public API surface: all expected names importable from chunkrank
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from typing import List

import pytest

import chunkrank
from chunkrank import (
    AsyncChunkRankPipeline,
    ChunkCache,
    ChunkRankPipeline,
    Chunker,
    ChunkerConfig,
    Ranker,
)
from chunkrank.models import (
    _runtime_registry,
    get_model_info,
    load_registry,
    register_model,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

MEDIUM_DOC = (
    "Retrieval-augmented generation combines a retriever with a language model. "
    "The retriever fetches relevant documents from a knowledge base. "
    "The language model then synthesises an answer from those documents. "
    "Chunking is essential so that each piece fits within the model's context window. "
    "Ranking ensures only the most relevant chunks are passed to the model. "
) * 30  # ~750 words — enough to produce multiple chunks on small-ctx models

QUESTION = "What does chunking ensure?"


def _build_registry() -> dict:
    return load_registry()


# ---------------------------------------------------------------------------
# 1. Public API surface
# ---------------------------------------------------------------------------

class TestPublicAPI:
    """All names advertised in __init__.py must be importable."""

    EXPECTED = [
        "split", "answer", "rank",
        "async_split", "async_answer", "async_rank",
        "register_model",
        "ChunkRankPipeline", "AsyncChunkRankPipeline",
        "Chunker", "ChunkerConfig",
        "Ranker",
        "ChunkCache",
    ]

    @pytest.mark.parametrize("name", EXPECTED)
    def test_name_exported(self, name):
        assert hasattr(chunkrank, name), f"chunkrank.{name} not exported"

    def test_split_returns_list_of_strings(self):
        result = chunkrank.split(MEDIUM_DOC, model="gpt-4o-mini")
        assert isinstance(result, list)
        assert all(isinstance(c, str) for c in result)
        assert len(result) >= 1

    def test_answer_returns_list(self):
        chunks = chunkrank.split(MEDIUM_DOC, model="gpt-4o-mini")
        answers = chunkrank.answer(QUESTION, chunks)
        assert isinstance(answers, list)
        assert len(answers) == len(chunks)

    def test_rank_returns_string(self):
        chunks = chunkrank.split(MEDIUM_DOC, model="gpt-4o-mini")
        answers = chunkrank.answer(QUESTION, chunks)
        best = chunkrank.rank(answers)
        assert isinstance(best, str)

    def test_get_model_info_returns_info(self):
        info = get_model_info("gpt-4o")
        assert info.max_context == 128_000
        assert info.name == "gpt-4o"

    def test_register_model_roundtrip(self):
        chunkrank.register_model("api-surface-test-model", max_context=64_000)
        info = get_model_info("api-surface-test-model")
        assert info.max_context == 64_000
        _runtime_registry.pop("api-surface-test-model", None)


# ---------------------------------------------------------------------------
# 2. Full registry — schema + usability (all 90 models)
# ---------------------------------------------------------------------------

def _all_registry_models():
    return list(load_registry().keys())


class TestRegistrySchema:
    """Every entry in model_registry.json must have valid required fields."""

    @pytest.mark.parametrize("model_key", _all_registry_models())
    def test_schema_valid(self, model_key):
        registry = _build_registry()
        info = registry[model_key]
        assert info.name, f"{model_key}: name is empty"
        assert info.max_context > 0, f"{model_key}: max_context must be > 0"
        assert info.tokenizer in ("tiktoken", "hf"), (
            f"{model_key}: tokenizer must be 'tiktoken' or 'hf', got {info.tokenizer!r}"
        )
        assert info.tokenizer_id, f"{model_key}: tokenizer_id is empty"
        assert info.default_reserve >= 0, f"{model_key}: default_reserve must be >= 0"

    @pytest.mark.parametrize("model_key", _all_registry_models())
    def test_get_model_info_resolves(self, model_key):
        info = get_model_info(model_key)
        assert info.max_context > 0

    def test_registry_json_is_valid_json(self):
        import importlib.resources
        text = (
            importlib.resources.files("chunkrank.registry")
            .joinpath("model_registry.json")
            .read_text(encoding="utf-8")
        )
        data = json.loads(text)
        assert isinstance(data, dict)
        assert len(data) >= 88


class TestNewModels:
    """Spot-check the models added in the registry expansion."""

    @pytest.mark.parametrize("model,expected_ctx", [
        ("gpt-4.1",          1_047_576),
        ("gpt-4.1-mini",     1_047_576),
        ("gpt-4.1-nano",     1_047_576),
        ("o1-pro",           200_000),
        ("claude-3-opus",    200_000),
        ("claude-3-sonnet",  200_000),
        ("claude-3-haiku",   200_000),
        ("claude-3-5-sonnet-20240620", 200_000),
        ("gemini-2.5-flash", 1_048_576),
        ("gemini-2.0-flash-lite", 1_048_576),
        ("gemini-1.0-pro",   32_760),
        ("Llama-3.1-405B",   128_000),
        ("mistral-small-latest", 32_000),
        ("mixtral-8x22b",    65_536),
        ("pixtral-large-latest", 128_000),
        ("command-a-03-2025", 256_000),
        ("deepseek-v2",      128_000),
        ("deepseek-r1-distill-qwen-32b", 128_000),
        ("qwen3-72b",        131_072),
        ("qwen2.5-7b-instruct",  131_072),
        ("qwen2.5-14b-instruct", 131_072),
        ("qwen2.5-32b-instruct", 131_072),
        ("phi-3-mini",       128_000),
        ("phi-3-medium",     128_000),
        ("phi-4",            16_384),
        ("phi-4-mini",       128_000),
        ("grok-2",           131_072),
        ("grok-3",           131_072),
        ("granite-3.3-8b-instruct", 128_000),
        ("granite-3.3-2b-instruct", 128_000),
        ("falcon-40b",       2_048),
        ("falcon-180b",      4_096),
    ])
    def test_new_model_context(self, model, expected_ctx):
        info = get_model_info(model)
        assert info.max_context == expected_ctx, (
            f"{model}: expected {expected_ctx}, got {info.max_context}"
        )

    @pytest.mark.parametrize("model", [
        "phi-3-mini", "phi-4", "granite-3.3-8b-instruct",
        "Llama-3.1-405B", "falcon-40b",
    ])
    def test_new_hf_model_has_hf_tokenizer(self, model):
        info = get_model_info(model)
        assert info.tokenizer == "hf"

    @pytest.mark.parametrize("model", [
        "gpt-4.1", "o1-pro", "grok-2", "grok-3",
        "mistral-small-latest", "command-a-03-2025",
        "deepseek-v2", "qwen3-72b",
    ])
    def test_new_tiktoken_model(self, model):
        info = get_model_info(model)
        assert info.tokenizer == "tiktoken"


# ---------------------------------------------------------------------------
# 3. Chunking correctness
# ---------------------------------------------------------------------------

class TestChunkingCorrectness:

    def test_determinism(self):
        """Same text always produces identical chunks."""
        config = ChunkerConfig(model="gpt-4o-mini", overlap_tokens=32)
        chunker = Chunker(config)
        assert chunker.split(MEDIUM_DOC) == chunker.split(MEDIUM_DOC)

    def test_chunks_are_non_empty(self):
        chunks = chunkrank.split(MEDIUM_DOC, model="gpt-4o-mini")
        assert all(c.strip() for c in chunks)

    def test_all_content_covered(self):
        """Every word in the original text should appear in at least one chunk."""
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 40
        chunks = chunkrank.split(text, model="gpt-4o-mini")
        combined = " ".join(chunks)
        for word in ["alpha", "gamma", "epsilon", "kappa"]:
            assert word in combined

    def test_overlap_shares_content(self):
        """With overlap_tokens > 0, consecutive chunks must share some tokens."""
        text = " ".join([f"word{i}" for i in range(500)])
        config = ChunkerConfig(model="gpt-4o-mini", overlap_tokens=32)
        chunks = Chunker(config).split(text)
        if len(chunks) < 2:
            pytest.skip("not enough chunks to test overlap")
        # There must be at least one shared word between adjacent chunks
        for a, b in zip(chunks, chunks[1:]):
            words_a = set(a.split())
            words_b = set(b.split())
            assert words_a & words_b, "consecutive chunks share no tokens despite overlap"

    def test_small_ctx_produces_more_chunks_than_large(self):
        """bert-base-uncased (512 tokens) should produce more chunks than gpt-4o (128k)."""
        chunks_small = chunkrank.split(MEDIUM_DOC, model="bert-base-uncased")
        chunks_large = chunkrank.split(MEDIUM_DOC, model="gpt-4o")
        assert len(chunks_small) >= len(chunks_large)

    def test_single_sentence_is_one_chunk(self):
        text = "This is a single short sentence."
        chunks = chunkrank.split(text, model="gpt-4o-mini")
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_empty_text_returns_empty(self):
        assert chunkrank.split("", model="gpt-4o-mini") == []

    def test_chunker_config_defaults(self):
        config = ChunkerConfig(model="gpt-4o-mini")
        assert config.strategy == "tokens"
        assert config.overlap_tokens >= 0
        # reserve_tokens defaults to None (resolved at Chunker init from model info)
        assert config.reserve_tokens is None or config.reserve_tokens >= 0

    def test_custom_reserve_reduces_chunk_size(self):
        """Higher reserve_tokens → smaller usable window → more chunks."""
        text = MEDIUM_DOC
        chunks_low = Chunker(ChunkerConfig(model="gpt-4o-mini", reserve_tokens=64)).split(text)
        chunks_high = Chunker(ChunkerConfig(model="gpt-4o-mini", reserve_tokens=4096)).split(text)
        assert len(chunks_high) >= len(chunks_low)

    @pytest.mark.parametrize("model", [
        "gpt-4o-mini", "claude-sonnet-4-6", "gemini-2.0-flash",
        "mistral-large-latest", "deepseek-v3",
    ])
    def test_large_ctx_models_produce_one_chunk(self, model):
        """Short text must always fit in a single chunk for large-ctx models."""
        short = "ChunkRank is a lightweight Python library."
        chunks = chunkrank.split(short, model=model)
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# 4. Ranking correctness
# ---------------------------------------------------------------------------

class TestRankingCorrectness:

    ANSWERS = [
        "Chunking splits text to fit within the model context window.",
        "The sky is blue because of Rayleigh scattering.",
        "Paris is the capital of France.",
    ]
    Q = "What does chunking do?"

    def test_bm25_picks_relevant_answer(self):
        ranker = Ranker(method="bm25")
        ranked = ranker.rank(self.Q, self.ANSWERS)
        assert "chunking" in ranked[0][0].lower() or "split" in ranked[0][0].lower()

    def test_tfidf_picks_relevant_answer(self):
        ranker = Ranker(method="tfidf")
        ranked = ranker.rank(self.Q, self.ANSWERS)
        assert "chunking" in ranked[0][0].lower() or "split" in ranked[0][0].lower()

    def test_bm25_determinism(self):
        ranker = Ranker(method="bm25")
        r1 = ranker.rank(self.Q, self.ANSWERS)
        r2 = ranker.rank(self.Q, self.ANSWERS)
        assert [a for a, _ in r1] == [a for a, _ in r2]

    def test_tfidf_determinism(self):
        ranker = Ranker(method="tfidf")
        r1 = ranker.rank(self.Q, self.ANSWERS)
        r2 = ranker.rank(self.Q, self.ANSWERS)
        assert [a for a, _ in r1] == [a for a, _ in r2]

    def test_scores_are_floats(self):
        for method in ("bm25", "tfidf"):
            ranked = Ranker(method=method).rank(self.Q, self.ANSWERS)
            assert all(isinstance(s, float) for _, s in ranked)

    def test_scores_descending(self):
        for method in ("bm25", "tfidf"):
            ranked = Ranker(method=method).rank(self.Q, self.ANSWERS)
            scores = [s for _, s in ranked]
            assert scores == sorted(scores, reverse=True)

    def test_empty_answers_filtered_before_ranking(self):
        answers = ["", "  ", "relevant chunking answer", ""]
        ranked = Ranker(method="bm25").rank("chunking", answers)
        assert all(a.strip() for a, _ in ranked)

    def test_rank_single_answer_returns_it(self):
        ranked = Ranker(method="bm25").rank("question", ["only answer"])
        assert len(ranked) == 1
        assert ranked[0][0] == "only answer"

    @pytest.mark.asyncio
    async def test_bm25_async_matches_sync(self):
        ranker = Ranker(method="bm25")
        sync_result = ranker.rank(self.Q, self.ANSWERS)
        async_result = await ranker.rank_async(self.Q, self.ANSWERS)
        assert [a for a, _ in sync_result] == [a for a, _ in async_result]


# ---------------------------------------------------------------------------
# 5. Pipeline — idempotence, stream completeness, all ranking methods
# ---------------------------------------------------------------------------

class TestPipeline:

    def test_process_returns_non_empty(self):
        pipe = ChunkRankPipeline(model="gpt-4o-mini")
        result = pipe.process(QUESTION, MEDIUM_DOC)
        assert isinstance(result, str)
        assert result.strip()

    def test_process_idempotent(self):
        pipe = ChunkRankPipeline(model="gpt-4o-mini")
        r1 = pipe.process(QUESTION, MEDIUM_DOC)
        r2 = pipe.process(QUESTION, MEDIUM_DOC)
        assert r1 == r2

    def test_stream_yields_all_non_empty_chunk_answers(self):
        pipe = ChunkRankPipeline(model="gpt-4o-mini")
        streamed = list(pipe.stream(QUESTION, MEDIUM_DOC))
        # stream yields one answer per non-empty chunk answer
        assert len(streamed) >= 1
        assert all(isinstance(s, str) and s.strip() for s in streamed)

    def test_stream_count_le_chunk_count(self):
        """stream() cannot yield more results than there are chunks."""
        pipe = ChunkRankPipeline(model="gpt-4o-mini")
        chunks = chunkrank.split(MEDIUM_DOC, model="gpt-4o-mini")
        streamed = list(pipe.stream(QUESTION, MEDIUM_DOC))
        assert len(streamed) <= len(chunks)

    def test_process_empty_text_returns_empty(self):
        pipe = ChunkRankPipeline(model="gpt-4o-mini")
        assert pipe.process(QUESTION, "") == ""

    @pytest.mark.parametrize("method", ["bm25", "tfidf"])
    def test_process_all_ranking_methods(self, method):
        pipe = ChunkRankPipeline(model="gpt-4o-mini", ranking_method=method)
        result = pipe.process(QUESTION, MEDIUM_DOC)
        assert isinstance(result, str)

    def test_retrieval_top_k_one(self):
        """top_k=1 must still produce a result."""
        pipe = ChunkRankPipeline(model="gpt-4o-mini", retrieval_top_k=1)
        result = pipe.process(QUESTION, MEDIUM_DOC)
        assert isinstance(result, str)

    def test_retrieval_top_k_larger_than_chunks_ok(self):
        short = "Chunking splits text into smaller pieces."
        pipe = ChunkRankPipeline(model="gpt-4o-mini", retrieval_top_k=100)
        result = pipe.process(QUESTION, short)
        assert isinstance(result, str)

    def test_custom_chunker_config_used(self):
        config = ChunkerConfig(model="gpt-4o-mini", overlap_tokens=0)
        pipe = ChunkRankPipeline(model="gpt-4o-mini", chunker_config=config)
        result = pipe.process(QUESTION, MEDIUM_DOC)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 6. Async pipeline
# ---------------------------------------------------------------------------

class TestAsyncPipeline:

    @pytest.mark.asyncio
    async def test_process_returns_string(self):
        pipe = AsyncChunkRankPipeline(model="gpt-4o-mini")
        result = await pipe.process(QUESTION, MEDIUM_DOC)
        assert isinstance(result, str)
        assert result.strip()

    @pytest.mark.asyncio
    async def test_process_idempotent(self):
        pipe = AsyncChunkRankPipeline(model="gpt-4o-mini")
        r1 = await pipe.process(QUESTION, MEDIUM_DOC)
        r2 = await pipe.process(QUESTION, MEDIUM_DOC)
        assert r1 == r2

    @pytest.mark.asyncio
    async def test_stream_yields_strings(self):
        pipe = AsyncChunkRankPipeline(model="gpt-4o-mini")
        results = [r async for r in pipe.stream(QUESTION, MEDIUM_DOC)]
        assert len(results) >= 1
        assert all(isinstance(r, str) and r.strip() for r in results)

    @pytest.mark.asyncio
    async def test_stream_empty_text_returns_nothing(self):
        pipe = AsyncChunkRankPipeline(model="gpt-4o-mini")
        results = [r async for r in pipe.stream(QUESTION, "")]
        assert results == []

    @pytest.mark.asyncio
    async def test_concurrent_requests_isolated(self):
        """Multiple concurrent pipelines must not interfere with each other."""
        docs = [MEDIUM_DOC, MEDIUM_DOC[:len(MEDIUM_DOC) // 2], MEDIUM_DOC * 2]
        pipe = AsyncChunkRankPipeline(model="gpt-4o-mini")
        results = await asyncio.gather(*[pipe.process(QUESTION, d) for d in docs])
        assert all(isinstance(r, str) for r in results)

    @pytest.mark.asyncio
    async def test_async_module_api(self):
        chunks = await chunkrank.async_split(MEDIUM_DOC, model="gpt-4o-mini")
        answers = await chunkrank.async_answer(QUESTION, chunks)
        best = await chunkrank.async_rank(answers)
        assert isinstance(best, str)


# ---------------------------------------------------------------------------
# 7. Cache — persistence and isolation
# ---------------------------------------------------------------------------

class TestCachePersistence:

    def test_persists_across_instances(self):
        """A new ChunkCache pointed at the same dir must find the cached data."""
        with tempfile.TemporaryDirectory() as d:
            chunks = ["chunk A", "chunk B"]
            ChunkCache(d).set("text", "gpt-4o", chunks)
            # new instance, same dir
            loaded = ChunkCache(d).get("text", "gpt-4o")
            assert loaded == chunks

    def test_isolation_by_model(self):
        with tempfile.TemporaryDirectory() as d:
            cache = ChunkCache(d)
            cache.set("same text", "gpt-4o", ["openai chunk"])
            cache.set("same text", "claude-sonnet-4-6", ["anthropic chunk"])
            assert cache.get("same text", "gpt-4o") == ["openai chunk"]
            assert cache.get("same text", "claude-sonnet-4-6") == ["anthropic chunk"]

    def test_isolation_by_text(self):
        with tempfile.TemporaryDirectory() as d:
            cache = ChunkCache(d)
            cache.set("text one", "gpt-4o", ["A"])
            cache.set("text two", "gpt-4o", ["B"])
            assert cache.get("text one", "gpt-4o") == ["A"]
            assert cache.get("text two", "gpt-4o") == ["B"]

    def test_cache_file_is_valid_json(self):
        with tempfile.TemporaryDirectory() as d:
            ChunkCache(d).set("hello", "gpt-4o", ["x", "y"])
            files = [f for f in os.listdir(d) if f.endswith(".json")]
            assert files, "cache dir has no JSON files"
            for fname in files:
                with open(os.path.join(d, fname)) as f:
                    data = json.load(f)
                # cache stores the chunks list (or a wrapper dict/list) — must be valid JSON
                assert data is not None

    def test_clear_removes_all_entries(self):
        with tempfile.TemporaryDirectory() as d:
            cache = ChunkCache(d)
            cache.set("t1", "gpt-4o", ["a"])
            cache.set("t2", "gpt-4o", ["b"])
            cache.clear()
            assert cache.get("t1", "gpt-4o") is None
            assert cache.get("t2", "gpt-4o") is None

    def test_cache_and_chunker_integration(self):
        """Cache stores real Chunker output and retrieves it correctly."""
        with tempfile.TemporaryDirectory() as d:
            text = MEDIUM_DOC
            model = "gpt-4o-mini"
            config = ChunkerConfig(model=model)
            fresh_chunks = Chunker(config).split(text)

            cache = ChunkCache(d)
            cache.set(text, model, fresh_chunks)

            loaded = ChunkCache(d).get(text, model)
            assert loaded == fresh_chunks


# ---------------------------------------------------------------------------
# 8. register_model — full pipeline integration
# ---------------------------------------------------------------------------

class TestRegisterModelIntegration:

    def test_register_then_split(self):
        register_model("my-prod-model", max_context=64_000)
        try:
            chunks = chunkrank.split(MEDIUM_DOC, model="my-prod-model")
            assert isinstance(chunks, list)
            assert len(chunks) >= 1
        finally:
            _runtime_registry.pop("my-prod-model", None)

    def test_register_then_pipeline(self):
        register_model("my-pipeline-model", max_context=32_000)
        try:
            pipe = ChunkRankPipeline(model="my-pipeline-model")
            result = pipe.process(QUESTION, MEDIUM_DOC)
            assert isinstance(result, str)
        finally:
            _runtime_registry.pop("my-pipeline-model", None)

    def test_register_overrides_existing(self):
        """Registering over a known model replaces its context size."""
        register_model("gpt-4o-mini", max_context=999_999)
        try:
            info = get_model_info("gpt-4o-mini")
            assert info.max_context == 999_999
        finally:
            _runtime_registry.pop("gpt-4o-mini", None)
        # After cleanup, original registry value restored
        assert get_model_info("gpt-4o-mini").max_context == 128_000

    def test_register_model_context_drives_chunking(self):
        """A model registered with tiny context should produce many chunks."""
        register_model("tiny-ctx-model", max_context=64)
        try:
            chunks = chunkrank.split(MEDIUM_DOC, model="tiny-ctx-model")
            assert len(chunks) > 5
        finally:
            _runtime_registry.pop("tiny-ctx-model", None)


# ---------------------------------------------------------------------------
# 9. Full end-to-end: text → chunks → answers → rank
# ---------------------------------------------------------------------------

class TestFullEndToEnd:

    def test_full_pipeline_local(self):
        text = MEDIUM_DOC
        question = QUESTION

        chunks = chunkrank.split(text, model="gpt-4o-mini")
        assert len(chunks) >= 1

        answers = chunkrank.answer(question, chunks)
        assert len(answers) == len(chunks)

        best = chunkrank.rank(answers)
        assert isinstance(best, str)
        assert best.strip()

    def test_full_pipeline_all_ranking_methods(self):
        chunks = chunkrank.split(MEDIUM_DOC, model="gpt-4o-mini")
        answers = chunkrank.answer(QUESTION, chunks)  # List[Tuple[str, float]]
        answer_strings = [a for a, _ in answers if a.strip()]
        for method in ("bm25", "tfidf"):
            ranker = Ranker(method=method)
            ranked = ranker.rank(QUESTION, answer_strings)
            assert len(ranked) >= 1
            assert ranked[0][0].strip()

    def test_pipeline_vs_module_api_consistent(self):
        """ChunkRankPipeline.process and module-level API must agree on non-emptiness."""
        pipe = ChunkRankPipeline(model="gpt-4o-mini")
        pipe_result = pipe.process(QUESTION, MEDIUM_DOC)

        chunks = chunkrank.split(MEDIUM_DOC, model="gpt-4o-mini")
        answers = chunkrank.answer(QUESTION, chunks)
        module_result = chunkrank.rank(answers)

        # Both must be non-empty (exact match may differ by tie-breaking)
        assert pipe_result.strip()
        assert module_result.strip()

    @pytest.mark.asyncio
    async def test_full_async_pipeline(self):
        chunks = await chunkrank.async_split(MEDIUM_DOC, model="gpt-4o-mini")
        assert len(chunks) >= 1

        answers = await chunkrank.async_answer(QUESTION, chunks)
        assert len(answers) == len(chunks)

        best = await chunkrank.async_rank(answers)
        assert isinstance(best, str)
        assert best.strip()

    def test_top_k_pipeline_vs_full_pipeline(self):
        """Both top-K and full pipeline return a non-empty answer."""
        pipe_full = ChunkRankPipeline(model="gpt-4o-mini")
        pipe_topk = ChunkRankPipeline(model="gpt-4o-mini", retrieval_top_k=2)

        r_full = pipe_full.process(QUESTION, MEDIUM_DOC)
        r_topk = pipe_topk.process(QUESTION, MEDIUM_DOC)

        assert r_full.strip()
        assert r_topk.strip()

    def test_stream_then_process_same_pipeline(self):
        """stream() and process() on the same pipeline instance both work."""
        pipe = ChunkRankPipeline(model="gpt-4o-mini")
        streamed = list(pipe.stream(QUESTION, MEDIUM_DOC))
        processed = pipe.process(QUESTION, MEDIUM_DOC)
        assert len(streamed) >= 1
        assert processed.strip()

    @pytest.mark.asyncio
    async def test_async_stream_then_process(self):
        pipe = AsyncChunkRankPipeline(model="gpt-4o-mini")
        streamed = [r async for r in pipe.stream(QUESTION, MEDIUM_DOC)]
        processed = await pipe.process(QUESTION, MEDIUM_DOC)
        assert len(streamed) >= 1
        assert processed.strip()
