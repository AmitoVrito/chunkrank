"""
Basic usage: module-level API (no LLM required).
"""
import chunkrank

text = ("BM25 ranks documents by term relevance. "
        "Chunking splits long documents into pieces. "
        "Ranking helps choose the best chunk for a question. ") * 100

question = "What does BM25 do?"

chunks = chunkrank.split(text, model="gpt-4o-mini")
answers = chunkrank.answer(question, chunks)
best = chunkrank.rank(answers)

print("Chunks:", len(chunks))
print("Answer:", best or "[no answer found]")
