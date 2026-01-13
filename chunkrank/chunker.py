from dataclasses import dataclass
from typing import List, Literal
from .models import get_model_info
from .tokenizers import build_tokenizer

Strategy = Literal["tokens", "sentences", "paragraphs"]


@dataclass
class ChunkerConfig:
    model: str
    strategy: Strategy = "tokens"
    overlap_tokens: int = 0
    reserve_tokens: int = None


class Chunker:
    def __init__(self, config: ChunkerConfig):
        info = get_model_info(config.model)
        self.window = info.max_context - (config.reserve_tokens or info.default_reserve)
        self.overlap = config.overlap_tokens
        self.strategy = config.strategy
        self.tok = build_tokenizer(info.tokenizer, info.tokenizer_id)

    def split(self):
        ...

    def decode(self):
        def _decode(self, token_ids: List[int]) -> str:
            if hasattr(self.tok, "decode"):
                return self.tok.decode(token_ids)
            return f"[Chunk {len(token_ids)} tokens]"
