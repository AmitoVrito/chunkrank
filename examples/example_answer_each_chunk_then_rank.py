"""
Answer each chunk then rank — using local extractor (default) or an LLM.

Local (no API key needed):
    python example_answer_each_chunk_then_rank.py

With OpenAI:
    OPENAI_API_KEY=sk-... python example_answer_each_chunk_then_rank.py

With Anthropic:
    ANTHROPIC_API_KEY=sk-ant-... python example_answer_each_chunk_then_rank.py
"""
import os
import chunkrank

DOC = """
ChunkRank aims to combine model-aware text chunking with answer ranking.
Model-aware chunking means selecting chunk sizes based on the model's context window and tokenizer.
After chunking, you might get multiple answers from different chunks.
To pick the best one, you rank the answers against the question.
BM25 often works well for ranking short answer candidates.
""" * 30

question = "What is model-aware chunking?"
model = "gpt-4o-mini"

chunks = chunkrank.split(model=model, text=DOC, overlap_tokens=64)
print(f"Chunks: {len(chunks)}")

# Detect provider from environment
if os.getenv("OPENAI_API_KEY"):
    answers = chunkrank.answer(question, chunks, provider="openai", api_key=os.environ["OPENAI_API_KEY"])
elif os.getenv("ANTHROPIC_API_KEY"):
    answers = chunkrank.answer(question, chunks, provider="anthropic", api_key=os.environ["ANTHROPIC_API_KEY"])
else:
    print("No API key found — using local extractive answerer.")
    answers = chunkrank.answer(question, chunks)

non_empty = [(a, s) for a, s in answers if a]
print(f"Candidate answers: {len(non_empty)}")

best = chunkrank.rank(answers)
print("Best answer:", best or "[no answer found]")
