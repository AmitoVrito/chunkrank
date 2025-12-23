from .chunker import Chunker, ChunkerConfig
from .ranker import Ranker

class ChunkRankPipeline:
    def __init__(self, model: str):
        self.chunker = Chunker(ChunkerConfig(model=model))
        self.ranker = Ranker()
