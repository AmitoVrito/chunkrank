from typing import List, Optional, Tuple

from .chunker import Chunker, ChunkerConfig, chunk_text
from .pipeline import ChunkRankPipeline
from .ranker import Ranker
from .answerers import LocalExtractiveAnswerer, LLMAnswerer


def split(text: str, model: str, overlap_tokens: int = 0, reserve_tokens: Optional[int] = None) -> List[str]:
    return chunk_text(text, model=model, overlap_tokens=overlap_tokens, reserve_tokens=reserve_tokens)


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


__all__ = [
    "Chunker",
    "ChunkerConfig",
    "chunk_text",
    "ChunkRankPipeline",
    "Ranker",
    "LocalExtractiveAnswerer",
    "LLMAnswerer",
    "split",
    "answer",
    "rank",
]
