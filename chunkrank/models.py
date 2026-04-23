import importlib.resources
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional


@dataclass
class ModelInfo:
    name: str
    max_context: int
    tokenizer: Optional[str]
    tokenizer_id: Optional[str]
    default_reserve: int = 256


# Runtime registry — checked before the static JSON file.
# Populated via register_model(); survives the process lifetime.
_runtime_registry: Dict[str, ModelInfo] = {}


def register_model(
    name: str,
    max_context: int,
    tokenizer: Optional[str] = "tiktoken",
    tokenizer_id: Optional[str] = "o200k_base",
    default_reserve: int = 256,
) -> None:
    """Register a model at runtime without editing model_registry.json.

    Useful for new model releases between library versions::

        import chunkrank
        chunkrank.register_model("gpt-5", max_context=200_000)
    """
    _runtime_registry[name] = ModelInfo(
        name=name,
        max_context=max_context,
        tokenizer=tokenizer,
        tokenizer_id=tokenizer_id,
        default_reserve=default_reserve,
    )


@lru_cache(maxsize=1)
def load_registry() -> Dict[str, ModelInfo]:
    "Loads the model registry from the json file (cached after first call)."
    text = importlib.resources.files("chunkrank.registry").joinpath("model_registry.json").read_text(encoding="utf-8")
    data = json.loads(text)
    return {k: ModelInfo(**v) for k, v in data.items()}


def get_model_info(model: str) -> ModelInfo:
    if model in _runtime_registry:
        return _runtime_registry[model]
    registry = load_registry()
    if model in registry:
        return registry[model]
    return ModelInfo(model, 128_000, "tiktoken", "o200k_base", 512)
