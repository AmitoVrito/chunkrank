import pytest
from chunkrank.pipeline import ChunkRankPipeline
from chunkrank.chunker import ChunkerConfig

# Force small chunks by reserving almost all of the context window
SMALL_CHUNK_CFG = ChunkerConfig(model="gpt-4o-mini", reserve_tokens=127_900)

LONG_DOC = (
    "Machine learning is transforming many industries. "
    "Supervised learning requires labelled training data to work. "
    "Neural networks can approximate any continuous function given enough data. "
    "Deep learning uses multiple layers to learn hierarchical representations. "
) * 60


def test_retrieval_top_k_limits_answered_chunks():
    """With retrieval_top_k=1, only 1 chunk should be answered."""
    answered = []
    pipeline = ChunkRankPipeline(model="gpt-4o-mini", retrieval_top_k=1, chunker_config=SMALL_CHUNK_CFG)
    original = pipeline.answerer.answer

    def recording(q, ctx):
        answered.append(ctx)
        return original(q, ctx)

    pipeline.answerer.answer = recording
    pipeline.process("What is supervised learning?", LONG_DOC)
    assert len(answered) == 1


def test_retrieval_top_k_2():
    """With retrieval_top_k=2, exactly 2 chunks should be answered."""
    answered = []
    pipeline = ChunkRankPipeline(model="gpt-4o-mini", retrieval_top_k=2, chunker_config=SMALL_CHUNK_CFG)
    original = pipeline.answerer.answer

    def recording(q, ctx):
        answered.append(ctx)
        return original(q, ctx)

    pipeline.answerer.answer = recording
    pipeline.process("What is supervised learning?", LONG_DOC)
    assert len(answered) == 2


def test_no_retrieval_top_k_answers_all_chunks():
    """Default behaviour (retrieval_top_k=None): all chunks are answered."""
    answered = []
    pipeline = ChunkRankPipeline(model="gpt-4o-mini", chunker_config=SMALL_CHUNK_CFG)
    original = pipeline.answerer.answer

    def recording(q, ctx):
        answered.append(ctx)
        return original(q, ctx)

    pipeline.answerer.answer = recording
    pipeline.process("What is ML?", LONG_DOC)

    all_chunks = pipeline.chunker.split(LONG_DOC)
    assert len(answered) == len(all_chunks)


def test_retrieval_pipeline_returns_string():
    pipeline = ChunkRankPipeline(model="gpt-4o-mini", retrieval_top_k=2, chunker_config=SMALL_CHUNK_CFG)
    result = pipeline.process("What is machine learning?", LONG_DOC)
    assert isinstance(result, str)
    assert len(result) > 0


def test_retrieval_pipeline_empty_text():
    pipeline = ChunkRankPipeline(model="gpt-4o-mini", retrieval_top_k=2)
    assert pipeline.process("question", "") == ""


def test_retrieval_top_k_larger_than_chunks():
    """If top_k > number of chunks, all chunks are answered (no error)."""
    answered = []
    pipeline = ChunkRankPipeline(model="gpt-4o-mini", retrieval_top_k=1000, chunker_config=SMALL_CHUNK_CFG)
    original = pipeline.answerer.answer

    def recording(q, ctx):
        answered.append(ctx)
        return original(q, ctx)

    pipeline.answerer.answer = recording
    pipeline.process("What is ML?", LONG_DOC)
    all_chunks = pipeline.chunker.split(LONG_DOC)
    assert len(answered) == len(all_chunks)
