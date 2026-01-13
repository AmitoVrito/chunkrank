from chunkrank.chunker import Chunker, ChunkerConfig
from chunkrank.ranker import Ranker
from chunkrank.answerers import LocalExtractiveAnswerer

def test_rank_chunks_then_answer():
    doc = (
        "Model-aware chunking selects chunk sizes based on a model's context window. "
        "After chunking, ranking helps choose the best relevant chunk. "
    ) * 50

    question = "Why does ranking help?"
    chunker = Chunker(ChunkerConfig(model="gpt-4o-mini", overlap_tokens=32))
    chunks = chunker.split(doc)

    ranker = Ranker(method="bm25")
    top_chunk, _ = ranker.rank_texts(question, chunks)[0]

    answerer = LocalExtractiveAnswerer(min_overlap=1)
    answer = answerer.answer(question, top_chunk)

    assert "ranking" in answer.lower() or answer != ""
