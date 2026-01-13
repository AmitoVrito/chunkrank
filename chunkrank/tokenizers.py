
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
    if backend == "tiktoken":
        tiktoken = _try_importing_tiktoken()
        if tiktoken:
            enc = tiktoken.get_encoding(tokenizer_id or "o200k_base")
            return TokenizerAdapter(lambda s: enc.encode(s, disallowed_special=()))
    elif backend == "hf":
        transformers = _try_importing_transformers()
        if transformers:
            tok = transformers.AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
            return TokenizerAdapter(lambda s: tok.encode(s, add_special_tokens=False))

    return TokenizerAdapter(lambda s: list(range(len(s) // 4)))
