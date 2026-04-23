from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

# Module-level cache: (backend, tokenizer_id) → TokenizerAdapter
_tokenizer_cache: Dict[Tuple[Optional[str], Optional[str]], "TokenizerAdapter"] = {}


def _try_importing_tiktoken():
    try:
        import tiktoken
        return tiktoken
    except ImportError:
        return None

def _try_importing_transformers():
    try:
        import transformers
        return transformers
    except ImportError:
        return None


class TokenizerAdapter:
    def __init__(self, encode_fn: Callable[[str], List[int]]):
        self._encode = encode_fn

    def encode(self, text: str) -> List[int]:
        return self._encode(text)

    def count(self, text: str) -> int:
        return len(self._encode(text))


def build_tokenizer(backend: Optional[str], tokenizer_id: Optional[str]) -> TokenizerAdapter:
    cache_key = (backend, tokenizer_id)
    if cache_key in _tokenizer_cache:
        return _tokenizer_cache[cache_key]

    adapter: TokenizerAdapter
    if backend == "tiktoken":
        tiktoken = _try_importing_tiktoken()
        if tiktoken:
            enc = tiktoken.get_encoding(tokenizer_id or "o200k_base")
            adapter = TokenizerAdapter(lambda s: enc.encode(s, disallowed_special=()))
            _tokenizer_cache[cache_key] = adapter
            return adapter
    elif backend == "hf":
        transformers = _try_importing_transformers()
        if transformers:
            tok = transformers.AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
            adapter = TokenizerAdapter(lambda s: tok.encode(s, add_special_tokens=False))
            _tokenizer_cache[cache_key] = adapter
            return adapter

    adapter = TokenizerAdapter(lambda s: list(range(len(s) // 4)))
    _tokenizer_cache[cache_key] = adapter
    return adapter
