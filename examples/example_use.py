from chunkrank.chunker import Chunker, ChunkerConfig
from chunkrank.ranker import Ranker
from chunkrank.answerers import LocalExtractiveAnswerer

text = ("BM25 ranks documents by term relevance. "
        "Chunking splits long documents into pieces. "
        "Ranking helps choose the best chunk for a question. ") * 100

question = "What does BM25 do?"

chunker = Chunker(ChunkerConfig(model="gpt-4o-mini", overlap_tokens=32))
chunks = chunker.split(text)

ranker = Ranker(method="bm25")
top_chunk, score = ranker.rank_texts(question, chunks)[0]

answerer = LocalExtractiveAnswerer(min_overlap=1)
answer = answerer.answer(question, top_chunk)

print("Top score:", score)
print("Answer:", answer or "[no answer found]")
