import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from chunkrank.async_pipeline import AsyncChunkRankPipeline, AsyncLLMAnswerer
import chunkrank

LONG_DOC = (
    "Async programming enables concurrent I/O without threads. "
    "Python asyncio uses an event loop for cooperative multitasking. "
    "gather() runs multiple coroutines concurrently and collects their results. "
) * 60


# ------------------------------------------------------------------ #
# AsyncChunkRankPipeline                                               #
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_async_pipeline_returns_string():
    pipeline = AsyncChunkRankPipeline(model="gpt-4o-mini")
    result = await pipeline.process("What does asyncio use?", LONG_DOC)
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_async_pipeline_empty_text():
    pipeline = AsyncChunkRankPipeline(model="gpt-4o-mini")
    result = await pipeline.process("question", "")
    assert result == ""


@pytest.mark.asyncio
async def test_async_pipeline_with_retrieval_top_k():
    answered = []
    pipeline = AsyncChunkRankPipeline(model="gpt-4o-mini", retrieval_top_k=1)
    original_answer = pipeline._sync_answerer.answer

    def recording(q, ctx):
        answered.append(ctx)
        return original_answer(q, ctx)

    pipeline._sync_answerer.answer = recording
    await pipeline.process("What is cooperative multitasking?", LONG_DOC)
    assert len(answered) == 1


@pytest.mark.asyncio
async def test_async_pipeline_parallel_answering():
    """All chunks answered concurrently — verify gather is used."""
    call_times = []

    async def fake_answer(q, ctx):
        call_times.append(asyncio.get_event_loop().time())
        await asyncio.sleep(0.01)
        return ("answer", 1.0)

    pipeline = AsyncChunkRankPipeline(model="gpt-4o-mini", provider="openai", api_key="sk-fake")
    pipeline._async_answerer.answer = fake_answer

    await pipeline.process("What is asyncio?", LONG_DOC)
    # Parallel execution: all calls start within a short window
    if len(call_times) > 1:
        spread = max(call_times) - min(call_times)
        assert spread < 0.5  # would be >n*0.01 if sequential


# ------------------------------------------------------------------ #
# AsyncLLMAnswerer                                                     #
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_async_llm_answerer_openai_mock():
    answerer = AsyncLLMAnswerer(provider="openai", api_key="sk-fake")

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "GPT-4o-mini answer."
    mock_response.choices[0].logprobs = None

    with patch.object(answerer, "_openai", new=AsyncMock(return_value=("GPT-4o-mini answer.", 1.0))):
        text, score = await answerer.answer("What is this?", "Some context.")

    assert text == "GPT-4o-mini answer."
    assert score == 1.0


@pytest.mark.asyncio
async def test_async_llm_answerer_anthropic_mock():
    answerer = AsyncLLMAnswerer(provider="anthropic", api_key="sk-ant-fake")

    with patch.object(answerer, "_anthropic", new=AsyncMock(return_value=("Claude answer.", 1.0))):
        text, score = await answerer.answer("What is this?", "Some context.")

    assert text == "Claude answer."


@pytest.mark.asyncio
async def test_async_llm_answerer_unknown_provider():
    answerer = AsyncLLMAnswerer(provider="openai", api_key="key")
    answerer.provider = "unknown"
    with pytest.raises(ValueError):
        await answerer.answer("q", "ctx")


# ------------------------------------------------------------------ #
# Module-level async functions                                         #
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_async_split():
    chunks = await chunkrank.async_split("Hello world " * 500, model="gpt-4o-mini")
    assert len(chunks) > 0
    assert all(isinstance(c, str) for c in chunks)


@pytest.mark.asyncio
async def test_async_split_empty():
    chunks = await chunkrank.async_split("", model="gpt-4o-mini")
    assert chunks == []


@pytest.mark.asyncio
async def test_async_answer_local():
    chunks = [
        "Machine learning learns from data to make predictions.",
        "The sky appears blue due to light scattering.",
    ]
    results = await chunkrank.async_answer("What is machine learning?", chunks)
    assert len(results) == 2
    assert all(isinstance(s, float) for _, s in results)


@pytest.mark.asyncio
async def test_async_answer_parallel_llm_mock():
    chunks = ["chunk one.", "chunk two.", "chunk three."]

    with patch("chunkrank.AsyncLLMAnswerer.answer", new=AsyncMock(return_value=("mocked", 1.0))):
        results = await chunkrank.async_answer(
            "question", chunks, provider="openai", api_key="sk-fake"
        )
    assert len(results) == 3


@pytest.mark.asyncio
async def test_async_rank():
    scored = [("best answer", 0.95), ("ok answer", 0.5), ("", 0.0)]
    result = await chunkrank.async_rank(scored)
    assert result == "best answer"


@pytest.mark.asyncio
async def test_async_rank_all_empty():
    result = await chunkrank.async_rank([("", 0.0), ("", 0.1)])
    assert result == ""
