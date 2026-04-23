from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

if TYPE_CHECKING:
    from .embeddings import EmbeddingConfig


class Ranker:

    def __init__(
        self,
        method: str = "bm25",
        embedding_config: Optional["EmbeddingConfig"] = None,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.method = method
        self._embedding_config = embedding_config
        self._backend = None  # lazy-init EmbeddingBackend when method="embedding"
        self._cross_encoder = None  # lazy-init CrossEncoder when method="cross-encoder"
        self._cross_encoder_model = cross_encoder_model
        self._tfidf_vectorizer = TfidfVectorizer(stop_words="english")

    def rank(self, question: str, answers: List[str]) -> List[Tuple[str, float]]:
        clean = [a for a in answers if isinstance(a, str) and a.strip()]
        if not clean:
            return []

        if self.method == "tfidf":
            return self._rank_tfidf(question, clean)
        elif self.method == "bm25":
            return self._rank_bm25(question, clean)
        elif self.method == "embedding":
            return self._rank_embedding(question, clean)
        elif self.method == "cross-encoder":
            return self._rank_cross_encoder(question, clean)
        else:
            raise ValueError(f"Unknown ranking method: {self.method}")

    async def rank_async(
        self, question: str, answers: List[str]
    ) -> List[Tuple[str, float]]:
        """Async rank. Truly async for method='embedding' with API backends;
        dispatches bm25/tfidf to a thread pool for non-blocking behaviour."""
        clean = [a for a in answers if isinstance(a, str) and a.strip()]
        if not clean:
            return []

        if self.method == "embedding":
            from .embeddings import EmbeddingBackend, EmbeddingConfig
            if self._backend is None:
                self._backend = EmbeddingBackend(self._embedding_config or EmbeddingConfig())
            all_texts = [question] + clean
            vecs = await self._backend.embed_async(all_texts)
            q_vec = vecs[0:1]
            a_vecs = vecs[1:]
            scores = (q_vec @ a_vecs.T)[0].tolist()
            return sorted(zip(clean, scores), key=lambda x: x[1], reverse=True)

        # bm25 / tfidf / cross-encoder → thread pool
        return await asyncio.to_thread(self.rank, question, answers)

    def rank_texts(self, query: str, texts: List[str]) -> List[Tuple[str, float]]:
        return self.rank(query, texts)

    def _rank_tfidf(self, question: str, answers: List[str]) -> List[Tuple[str, float]]:
        corpus = [question] + answers
        vectors = self._tfidf_vectorizer.fit_transform(corpus)
        q_vec = vectors[0]
        a_vecs = vectors[1:]
        scores = cosine_similarity(q_vec, a_vecs)[0]
        return sorted(zip(answers, scores), key=lambda x: x[1], reverse=True)

    def _rank_bm25(self, question: str, answers: List[str]) -> List[Tuple[str, float]]:
        tokenized_answers = [a.split() for a in answers if a and a.strip()]
        if not tokenized_answers:
            return []
        bm25 = BM25Okapi(tokenized_answers)
        q_tokens = question.split()
        scores = bm25.get_scores(q_tokens)
        return sorted(zip(answers, scores), key=lambda x: x[1], reverse=True)

    def _rank_cross_encoder(
        self, question: str, answers: List[str]
    ) -> List[Tuple[str, float]]:
        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers  or  pip install chunkrank[semantic]"
                )
            self._cross_encoder = CrossEncoder(self._cross_encoder_model)
        pairs = [(question, a) for a in answers]
        scores = self._cross_encoder.predict(pairs)
        return sorted(zip(answers, scores.tolist()), key=lambda x: x[1], reverse=True)

    def _rank_embedding(
        self, question: str, answers: List[str]
    ) -> List[Tuple[str, float]]:
        from .embeddings import EmbeddingBackend, EmbeddingConfig
        if self._backend is None:
            self._backend = EmbeddingBackend(self._embedding_config or EmbeddingConfig())
        all_texts = [question] + answers
        vecs = self._backend.embed(all_texts)
        q_vec = vecs[0:1]   # (1, dim)
        a_vecs = vecs[1:]   # (n, dim)
        scores = (q_vec @ a_vecs.T)[0].tolist()
        return sorted(zip(answers, scores), key=lambda x: x[1], reverse=True)


def rank_answers(question: str, answers: List[str], method: str = "bm25") -> str:
    ranker = Ranker(method=method)
    ranked = ranker.rank(question, answers)
    return ranked[0][0]
