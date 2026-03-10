from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np

EmbeddingProvider = Literal["local", "openai", "cohere"]


@dataclass
class EmbeddingConfig:
    provider: EmbeddingProvider = "local"
    model: str = ""
    api_key: Optional[str] = None
    batch_size: int = 64

    def __post_init__(self) -> None:
        if self.provider == "local" and not self.model:
            self.model = "all-MiniLM-L6-v2"
        elif self.provider == "openai" and not self.model:
            self.model = "text-embedding-3-small"
        elif self.provider == "cohere" and not self.model:
            self.model = "embed-english-v3.0"


class EmbeddingBackend:
    """
    Unified embedding interface for local (sentence-transformers),
    OpenAI, and Cohere providers.

    All methods return L2-normalised float32 numpy arrays of shape (n, dim).
    Cosine similarity between any two rows is therefore just their dot product.
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._local_model = None  # lazy-loaded SentenceTransformer

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        if self.config.provider == "local":
            return self._embed_local(texts)
        elif self.config.provider == "openai":
            return self._embed_openai(texts)
        elif self.config.provider == "cohere":
            return self._embed_cohere(texts)
        raise ValueError(f"Unknown embedding provider: {self.config.provider!r}")

    async def embed_async(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        if self.config.provider == "local":
            return await asyncio.to_thread(self._embed_local, texts)
        elif self.config.provider == "openai":
            return await self._embed_openai_async(texts)
        elif self.config.provider == "cohere":
            return await asyncio.to_thread(self._embed_cohere, texts)
        raise ValueError(f"Unknown embedding provider: {self.config.provider!r}")

    # ------------------------------------------------------------------ #
    # Local (sentence-transformers)                                        #
    # ------------------------------------------------------------------ #

    def _embed_local(self, texts: List[str]) -> np.ndarray:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers  or  pip install chunkrank[semantic]"
            )
        if self._local_model is None:
            self._local_model = SentenceTransformer(self.config.model)
        vecs = self._local_model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(vecs, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # OpenAI                                                               #
    # ------------------------------------------------------------------ #

    def _embed_openai(self, texts: List[str]) -> np.ndarray:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")
        client = OpenAI(api_key=self.config.api_key)
        response = client.embeddings.create(model=self.config.model, input=texts)
        vecs = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        return self._normalise(np.array(vecs, dtype=np.float32))

    async def _embed_openai_async(self, texts: List[str]) -> np.ndarray:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")
        client = AsyncOpenAI(api_key=self.config.api_key)
        response = await client.embeddings.create(model=self.config.model, input=texts)
        vecs = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        return self._normalise(np.array(vecs, dtype=np.float32))

    # ------------------------------------------------------------------ #
    # Cohere                                                               #
    # ------------------------------------------------------------------ #

    def _embed_cohere(self, texts: List[str]) -> np.ndarray:
        try:
            import cohere
        except ImportError:
            raise ImportError("cohere not installed. Run: pip install cohere")
        co = cohere.Client(self.config.api_key)
        response = co.embed(
            texts=texts,
            model=self.config.model,
            input_type="search_document",
        )
        return self._normalise(np.array(response.embeddings, dtype=np.float32))

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalise(arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-10)
