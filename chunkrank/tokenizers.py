
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
    ...


def build_tokenizer():
    ...