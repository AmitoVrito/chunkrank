from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

from .chunker import Chunker, ChunkerConfig
from .ranker import Ranker
from .answerers import LocalExtractiveAnswerer, LLMAnswerer

if TYPE_CHECKING:
    from .embeddings import EmbeddingConfig


class ChunkRankPipeline:
    def __init__(
        self,
        model: str,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        answer_model: Optional[str] = None,
        ranking_method: str = "bm25",
        retrieval_top_k: Optional[int] = None,
        embedding_config: Optional["EmbeddingConfig"] = None,
        chunker_config: Optional[ChunkerConfig] = None,
    ):
        self.chunker = Chunker(chunker_config or ChunkerConfig(model=model))
        self.ranker = Ranker(method=ranking_method, embedding_config=embedding_config)
        self.retrieval_top_k = retrieval_top_k

        if provider and api_key:
            self.answerer = LLMAnswerer(
                provider=provider,
                api_key=api_key,
                model=answer_model or "",
            )
        else:
            self.answerer = LocalExtractiveAnswerer()

    def process(self, question: str, text: str) -> str:
        chunks = self.chunker.split(text)
        if not chunks:
            return ""

        if self.retrieval_top_k is not None:
            # Retrieve-then-answer: rank chunks first, answer top-K only
            ranked_chunks = self.ranker.rank_texts(question, chunks)
            top_chunks = [c for c, _ in ranked_chunks[: self.retrieval_top_k]]
            scored: List[Tuple[str, float]] = [
                self.answerer.answer(question, chunk) for chunk in top_chunks
            ]
        else:
            # Answer all chunks then rank (original behaviour)
            scored = [self.answerer.answer(question, chunk) for chunk in chunks]

        scored = [(a, s) for a, s in scored if a]
        if not scored:
            return ""

        ranked = self.ranker.rank(question, [a for a, _ in scored])
        return ranked[0][0] if ranked else ""
