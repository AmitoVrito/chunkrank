"""
Rank chunks first by relevance, then answer using the top chunk.
Uses local extractive answerer by default. Pass an API key to use an LLM.

Local (no API key needed):
    python example_rank_chunks_then_answer.py

With OpenAI:
    OPENAI_API_KEY=sk-... python example_rank_chunks_then_answer.py
"""
import os
import chunkrank
from chunkrank.ranker import Ranker

DOC = """
ChunkRank is a library for model-aware chunking and answer ranking.
Each LLM has a different context window and tokenizer. Some models support 128k tokens.
Once documents are chunked, answering per chunk creates multiple candidate answers.
A ranking step is needed to select the best answer.
BM25 is a classic ranking method used in information retrieval.
TF-IDF cosine similarity is another lightweight alternative.
""" * 20

question = "Why do we need ranking after chunking?"
model = "gpt-4o-mini"

# 1) Chunk
chunks = chunkrank.split(text=DOC, model=model, overlap_tokens=64)
print(f"Chunks: {len(chunks)}")

# 2) Rank chunks by relevance, pick the top one
ranker = Ranker(method="bm25")
top_chunk, top_score = ranker.rank_texts(question, chunks)[0]
print(f"Top chunk score: {top_score:.4f}")

# 3) Answer using the best chunk
if os.getenv("OPENAI_API_KEY"):
    answers = chunkrank.answer(question, [top_chunk], provider="openai", api_key=os.environ["OPENAI_API_KEY"])
elif os.getenv("ANTHROPIC_API_KEY"):
    answers = chunkrank.answer(question, [top_chunk], provider="anthropic", api_key=os.environ["ANTHROPIC_API_KEY"])
else:
    print("No API key found — using local extractive answerer.")
    answers = chunkrank.answer(question, [top_chunk])

best = chunkrank.rank(answers)
print("Answer:", best or "[no answer found]")
