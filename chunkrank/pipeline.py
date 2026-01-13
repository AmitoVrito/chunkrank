from typing import List
from .chunker import Chunker, ChunkerConfig
from .ranker import Ranker


class ChunkRankPipeline:
    def __init__(self, model: str):
        self.chunker = Chunker(ChunkerConfig(model=model))
        self.ranker = Ranker()

    def process(self, question: str, text: str) -> str:
        chunks = self.chunker.split(text)
        answers = [f"Answer from chunk {i}" for i, _ in enumerate(chunks, 1)]
        best = self.ranker.rank(question, answers)[0][0]
        return best
