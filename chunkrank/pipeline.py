from typing import List, Optional, Tuple
from .chunker import Chunker, ChunkerConfig
from .ranker import Ranker
from .answerers import LocalExtractiveAnswerer, LLMAnswerer


class ChunkRankPipeline:
    def __init__(
        self,
        model: str,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        answer_model: Optional[str] = None,
    ):
        self.chunker = Chunker(ChunkerConfig(model=model))
        self.ranker = Ranker()

        if provider and api_key:
            self.answerer = LLMAnswerer(
                provider=provider,
                api_key=api_key,
                model=answer_model or "",
            )
        else:
            self.answerer = LocalExtractiveAnswerer()

    def process(self, question: str, text: str) -> str:
        chunks = self.chunker.split(text)
        scored: List[Tuple[str, float]] = [
            self.answerer.answer(question, chunk) for chunk in chunks
        ]
        scored = [(a, s) for a, s in scored if a]
        if not scored:
            return ""
        ranked = self.ranker.rank(question, [a for a, _ in scored])
        return ranked[0][0] if ranked else ""
