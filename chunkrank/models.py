import importlib.resources
import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelInfo:
    name: str
    max_context: int
    tokenizer: Optional[str]
    tokenizer_id: Optional[str]
    default_reserve: int = 256


def load_registry() -> Dict[str, ModelInfo]:
    "Loads the model registry from the json file"
    with importlib.resources.open_text("chunkrank.registry", "model_registry.json") as file:
        data = json.load(file)
        return {k: ModelInfo(**v) for k, v in data.items()}


def get_model_info(model: str) -> ModelInfo:
    registry = load_registry()
    if model in registry:
        return registry[model]
    return ModelInfo(model,
                     128_000,
                     "tiktoken",
                     "o200k_base",
                     512)
