from __future__ import annotations

import asyncio
from typing import List, Optional, Tuple

from .chunker import Chunker, ChunkerConfig, chunk_text
from .pipeline import ChunkRankPipeline
from .async_pipeline import AsyncChunkRankPipeline, AsyncLLMAnswerer
from .ranker import Ranker
from .answerers import LocalExtractiveAnswerer, LLMAnswerer
from .embeddings import EmbeddingConfig, EmbeddingBackend


# ------------------------------------------------------------------ #
# Sync module-level API                                                #
# ------------------------------------------------------------------ #

def split(
    text: str,
    model: str,
    overlap_tokens: int = 0,
    reserve_tokens: Optional[int] = None,
    strategy: str = "tokens",
    similarity_threshold: float = 0.5,
    embedding_config: Optional[EmbeddingConfig] = None,
) -> List[str]:
    return chunk_text(
        text,
        model=model,
        overlap_tokens=overlap_tokens,
        reserve_tokens=reserve_tokens,
        strategy=strategy,
        similarity_threshold=similarity_threshold,
        embedding_config=embedding_config,
    )


def answer(
    question: str,
    chunks: List[str],
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> List[Tuple[str, float]]:
    if provider and api_key:
        answerer = LLMAnswerer(provider=provider, api_key=api_key, model=model or "")
    else:
        answerer = LocalExtractiveAnswerer()
    return [answerer.answer(question, chunk) for chunk in chunks]


def rank(scored_answers: List[Tuple[str, float]]) -> str:
    valid = [(a, s) for a, s in scored_answers if a]
    if not valid:
        return ""
    return sorted(valid, key=lambda x: x[1], reverse=True)[0][0]


# ------------------------------------------------------------------ #
# Async module-level API                                               #
# ------------------------------------------------------------------ #

async def async_split(
    text: str,
    model: str,
    overlap_tokens: int = 0,
    reserve_tokens: Optional[int] = None,
    strategy: str = "tokens",
    similarity_threshold: float = 0.5,
    embedding_config: Optional[EmbeddingConfig] = None,
) -> List[str]:
    """Non-blocking split; chunking runs in a thread pool."""
    cfg = ChunkerConfig(
        model=model,
        overlap_tokens=overlap_tokens,
        reserve_tokens=reserve_tokens,
        strategy=strategy,
        similarity_threshold=similarity_threshold,
        embedding_config=embedding_config,
    )
    chunker = Chunker(cfg)
    return await asyncio.to_thread(chunker.split, text)


async def async_answer(
    question: str,
    chunks: List[str],
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> List[Tuple[str, float]]:
    """Parallel answering: LLM calls run concurrently via asyncio.gather."""
    if provider and api_key:
        async_answerer = AsyncLLMAnswerer(provider=provider, api_key=api_key, model=model or "")
        tasks = [async_answerer.answer(question, chunk) for chunk in chunks]
        return list(await asyncio.gather(*tasks))
    local = LocalExtractiveAnswerer()
    tasks = [asyncio.to_thread(local.answer, question, chunk) for chunk in chunks]
    return list(await asyncio.gather(*tasks))


async def async_rank(scored_answers: List[Tuple[str, float]]) -> str:
    """Async wrapper around rank() for API symmetry."""
    return rank(scored_answers)


# ------------------------------------------------------------------ #
# Exports                                                              #
# ------------------------------------------------------------------ #

__all__ = [
    # Classes
    "Chunker",
    "ChunkerConfig",
    "ChunkRankPipeline",
    "AsyncChunkRankPipeline",
    "AsyncLLMAnswerer",
    "Ranker",
    "LocalExtractiveAnswerer",
    "LLMAnswerer",
    "EmbeddingConfig",
    "EmbeddingBackend",
    # Sync functions
    "chunk_text",
    "split",
    "answer",
    "rank",
    # Async functions
    "async_split",
    "async_answer",
    "async_rank",
]
