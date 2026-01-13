from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

from .models import get_model_info
from .tokenizers import build_tokenizer

Strategy = Literal["tokens"]  # keep it simple for now


@dataclass
class ChunkerConfig:
    model: str
    strategy: Strategy = "tokens"
    overlap_tokens: int = 0
    reserve_tokens: Optional[int] = None


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

    def split(self, text: str) -> List[str]:
        if not isinstance(text, str) or not text:
            return []

        if self.strategy != "tokens":
            raise NotImplementedError("Only 'tokens' strategy is implemented in this version.")

        return list(self._chunk_by_token_budget(text))

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


def chunk_text(
    text: str,
    model: str,
    overlap_tokens: int = 0,
    reserve_tokens: Optional[int] = None,
) -> List[str]:
    cfg = ChunkerConfig(model=model, overlap_tokens=overlap_tokens, reserve_tokens=reserve_tokens)
    return Chunker(cfg).split(text)
