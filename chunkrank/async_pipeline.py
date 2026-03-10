from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, List, Optional, Tuple

from .chunker import Chunker, ChunkerConfig
from .ranker import Ranker
from .answerers import LocalExtractiveAnswerer

if TYPE_CHECKING:
    from .embeddings import EmbeddingConfig


class AsyncLLMAnswerer:
    """
    Async LLM answerer using AsyncOpenAI / anthropic.AsyncAnthropic.
    Enables parallel chunk answering via asyncio.gather().
    """

    def __init__(self, provider: str, api_key: str, model: str = ""):
        self.provider = provider
        self.api_key = api_key
        if provider == "openai" and not model:
            model = "gpt-4o-mini"
        elif provider == "anthropic" and not model:
            model = "claude-haiku-4-5-20251001"
        self.model = model

    async def answer(self, question: str, context: str) -> Tuple[str, float]:
        if self.provider == "openai":
            return await self._openai(question, context)
        elif self.provider == "anthropic":
            return await self._anthropic(question, context)
        raise ValueError(f"Unknown provider: {self.provider!r}. Use 'openai' or 'anthropic'.")

    async def _openai(self, question: str, context: str) -> Tuple[str, float]:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")
        client = AsyncOpenAI(api_key=self.api_key)
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Answer the question using only the provided context. Be concise.",
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
            max_tokens=512,
        )
        text = response.choices[0].message.content.strip()
        lp = response.choices[0].logprobs
        score = lp.content[0].logprob if lp else 1.0
        return (text, float(score) if score else 1.0)

    async def _anthropic(self, question: str, context: str) -> Tuple[str, float]:
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic not installed. Run: pip install anthropic")
        client = anthropic.AsyncAnthropic(api_key=self.api_key)
        message = await client.messages.create(
            model=self.model,
            max_tokens=512,
            system="Answer the question using only the provided context. Be concise.",
            messages=[
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                }
            ],
        )
        return (message.content[0].text.strip(), 1.0)


class AsyncChunkRankPipeline:
    """
    Async version of ChunkRankPipeline.

    - Chunking: CPU-bound → runs in thread pool via asyncio.to_thread
    - Answering: I/O-bound → runs in parallel via asyncio.gather
    - Ranking: async-native for embeddings, thread pool for bm25/tfidf
    """

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
        self._chunker = Chunker(chunker_config or ChunkerConfig(model=model))
        self._ranker = Ranker(method=ranking_method, embedding_config=embedding_config)
        self._retrieval_top_k = retrieval_top_k

        if provider and api_key:
            self._async_answerer: Optional[AsyncLLMAnswerer] = AsyncLLMAnswerer(
                provider=provider,
                api_key=api_key,
                model=answer_model or "",
            )
            self._sync_answerer: Optional[LocalExtractiveAnswerer] = None
        else:
            self._async_answerer = None
            self._sync_answerer = LocalExtractiveAnswerer()

    async def process(self, question: str, text: str) -> str:
        # Chunking is CPU-bound — off-load to thread pool
        chunks: List[str] = await asyncio.to_thread(self._chunker.split, text)
        if not chunks:
            return ""

        # Optional retrieval: rank chunks, keep top-K before answering
        if self._retrieval_top_k is not None:
            ranked_chunks = await self._ranker.rank_async(question, chunks)
            chunks = [c for c, _ in ranked_chunks[: self._retrieval_top_k]]

        # Answer all (remaining) chunks in parallel
        scored: List[Tuple[str, float]] = await self._answer_chunks(question, chunks)
        scored = [(a, s) for a, s in scored if a]
        if not scored:
            return ""

        ranked = await self._ranker.rank_async(question, [a for a, _ in scored])
        return ranked[0][0] if ranked else ""

    async def _answer_chunks(
        self, question: str, chunks: List[str]
    ) -> List[Tuple[str, float]]:
        if self._async_answerer is not None:
            tasks = [self._async_answerer.answer(question, chunk) for chunk in chunks]
            return list(await asyncio.gather(*tasks))
        # LocalExtractiveAnswerer is sync + fast — still parallelise for consistency
        tasks = [
            asyncio.to_thread(self._sync_answerer.answer, question, chunk)
            for chunk in chunks
        ]
        return list(await asyncio.gather(*tasks))
