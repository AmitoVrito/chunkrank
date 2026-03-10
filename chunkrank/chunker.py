from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, TYPE_CHECKING

from .models import get_model_info
from .tokenizers import build_tokenizer

if TYPE_CHECKING:
    from .embeddings import EmbeddingConfig

Strategy = Literal["tokens", "semantic"]


@dataclass
class ChunkerConfig:
    model: str
    strategy: Strategy = "tokens"
    overlap_tokens: int = 0
    reserve_tokens: Optional[int] = None
    # Semantic strategy options (ignored when strategy="tokens")
    similarity_threshold: float = 0.5
    embedding_config: Optional["EmbeddingConfig"] = None


class Chunker:
    def __init__(self, config: ChunkerConfig):
        info = get_model_info(config.model)

        reserve = config.reserve_tokens if config.reserve_tokens is not None else info.default_reserve
        self.window = max(1, info.max_context - max(0, reserve))

        self.overlap = max(0, config.overlap_tokens)
        if self.overlap >= self.window:
            raise ValueError("overlap_tokens must be < usable window size")

        self.strategy = config.strategy
        self.tok = build_tokenizer(info.tokenizer, info.tokenizer_id)
        self._similarity_threshold = config.similarity_threshold
        self._embedding_config = config.embedding_config

    def split(self, text: str) -> List[str]:
        if not isinstance(text, str) or not text:
            return []

        if self.strategy == "tokens":
            return list(self._chunk_by_token_budget(text))
        elif self.strategy == "semantic":
            return list(self._chunk_by_semantic_similarity(text))
        else:
            raise NotImplementedError(f"Unknown strategy: {self.strategy!r}")

    def _chunk_by_token_budget(self, text: str):
        """
        Robust approach: grow a slice until token budget reached, then emit slice.
        Avoids needing tokenizer.decode() (so no None chunks).
        """
        start = 0
        n = len(text)

        # Fast path: already fits
        if self.tok.count(text) <= self.window:
            yield text
            return

        # Character-based upper bound for initial probe (roughly 4 chars/token)
        approx_chars = max(64, self.window * 4)

        while start < n:
            end = min(n, start + approx_chars)
            chunk = text[start:end]

            # If too big, shrink
            while end > start and self.tok.count(chunk) > self.window:
                end = start + max(1, (end - start) * 9 // 10)
                chunk = text[start:end]

            # If somehow cannot shrink (pathological), force a minimal progress
            if end <= start:
                end = min(n, start + 200)
                chunk = text[start:end]

            yield chunk

            if end >= n:
                break

            # overlap handling (approx char backoff)
            if self.overlap > 0:
                backoff_chars = self.overlap * 4
                start = max(0, end - backoff_chars)
            else:
                start = end

    def _chunk_by_semantic_similarity(self, text: str):
        """
        Split on embedding similarity drops between adjacent sentences.
        Also enforces the token budget so chunks never exceed the context window.
        """
        from .embeddings import EmbeddingBackend, EmbeddingConfig
        from .utils import split_sentences

        sentences = split_sentences(text)
        if not sentences:
            return
        if len(sentences) == 1:
            yield text
            return

        cfg = self._embedding_config or EmbeddingConfig()
        backend = EmbeddingBackend(cfg)
        vecs = backend.embed(sentences)  # (n, dim), L2-normalised

        current_group: List[str] = [sentences[0]]
        current_tokens: int = self.tok.count(sentences[0])

        for i in range(1, len(sentences)):
            sim = float(vecs[i - 1] @ vecs[i])  # cosine sim (dot of normalised vecs)
            sentence_tokens = self.tok.count(sentences[i])
            would_exceed = (current_tokens + sentence_tokens) > self.window

            if sim < self._similarity_threshold or would_exceed:
                yield " ".join(current_group)
                current_group = [sentences[i]]
                current_tokens = sentence_tokens
            else:
                current_group.append(sentences[i])
                current_tokens += sentence_tokens

        if current_group:
            yield " ".join(current_group)


def chunk_text(
    text: str,
    model: str,
    overlap_tokens: int = 0,
    reserve_tokens: Optional[int] = None,
    strategy: Strategy = "tokens",
    similarity_threshold: float = 0.5,
    embedding_config: Optional["EmbeddingConfig"] = None,
) -> List[str]:
    cfg = ChunkerConfig(
        model=model,
        overlap_tokens=overlap_tokens,
        reserve_tokens=reserve_tokens,
        strategy=strategy,
        similarity_threshold=similarity_threshold,
        embedding_config=embedding_config,
    )
    return Chunker(cfg).split(text)
