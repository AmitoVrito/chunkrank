# ChunkRank: Model-Aware Chunking + Answer Ranking

```
Used internally for long-document QA and evaluation pipelines handling 1,000+ PDFs.
```

ChunkRank is a lightweight Python library that automatically chunks text based on an LLM's
tokenizer and context window, then consolidates and ranks answers across chunks.

🔗 PyPI: https://pypi.org/project/chunkrank/

---

## Why ChunkRank?

When working with LLMs, long documents must be split into chunks, but:
- Every model has **different tokenizers and context limits**
- Chunk sizes are usually **hard-coded and error-prone**
- Answer quality drops when responses come from **multiple chunks**
- Existing RAG frameworks are **heavy** when you only need chunking + ranking

**ChunkRank solves this gap.**

---

## Installation

```bash
pip install chunkrank
```

With semantic chunking + cross-encoder reranking:
```bash
pip install chunkrank[semantic]
```

With all optional backends:
```bash
pip install chunkrank[all]
```

For development:
```bash
poetry install --with dev
```

---

## Quick Example

```python
import chunkrank

text = open("document.txt").read()
question = "What is the main topic of this document?"

chunks = chunkrank.split(text, model="gpt-4o-mini")
answers = chunkrank.answer(question, chunks)
best = chunkrank.rank(answers)

print(best)
```

---

## Core API

```python
import chunkrank

# 1. Split text into model-aware chunks
chunks = chunkrank.split(text, model="gpt-4o-mini")

# 2. Answer the question across all chunks
#    Default: local extractive (no API key required)
answers = chunkrank.answer(question, chunks)

#    With OpenAI:
answers = chunkrank.answer(question, chunks, provider="openai", api_key="sk-...")

#    With Anthropic:
answers = chunkrank.answer(question, chunks, provider="anthropic", api_key="sk-ant-...")

# 3. Rank and return the best answer
best_answer = chunkrank.rank(answers)
```

---

## Pipeline API

```python
from chunkrank import ChunkRankPipeline

# Local (no LLM required)
pipe = ChunkRankPipeline(model="gpt-4o-mini")

# With OpenAI
pipe = ChunkRankPipeline(model="gpt-4o-mini", provider="openai", api_key="sk-...")

# With Anthropic
pipe = ChunkRankPipeline(model="gpt-4o-mini", provider="anthropic", api_key="sk-ant-...")

# Process — returns best answer
answer = pipe.process(question="What is the main topic?", text=text)

# Stream — yields answers progressively as each chunk is processed
for partial in pipe.stream(question="What is the main topic?", text=text):
    print(partial)
```

---

## Async API

```python
from chunkrank import AsyncChunkRankPipeline

pipe = AsyncChunkRankPipeline(model="gpt-4o-mini", provider="openai", api_key="sk-...")

# Parallel chunk answering via asyncio.gather
answer = await pipe.process(question, text)

# Async streaming
async for partial in pipe.stream(question, text):
    print(partial)
```

Module-level async functions:

```python
import chunkrank

chunks = await chunkrank.async_split(text, model="gpt-4o-mini")
answers = await chunkrank.async_answer(question, chunks)   # parallel LLM calls
best = await chunkrank.async_rank(answers)
```

---

## Ranking Methods

| Method | Description | Extra dep |
|---|---|---|
| `bm25` (default) | BM25 lexical ranking | none |
| `tfidf` | TF-IDF cosine similarity | none |
| `embedding` | Dense vector similarity | `[semantic]` or `openai-embed` |
| `cross-encoder` | Semantic cross-encoder (most accurate) | `[semantic]` |

```python
from chunkrank import Ranker

ranker = Ranker(method="cross-encoder")
ranked = ranker.rank(question, answers)
```

---

## Chunking Strategies

```python
# Token-budget sliding window (default)
chunks = chunkrank.split(text, model="gpt-4o-mini", strategy="tokens", overlap_tokens=64)

# Semantic — splits on embedding similarity drops between sentences
chunks = chunkrank.split(text, model="gpt-4o-mini", strategy="semantic", similarity_threshold=0.5)
```

---

## Retrieve-then-Answer (top-K)

Rank chunks first, answer only the top-K — reduces LLM calls on large documents:

```python
pipe = ChunkRankPipeline(model="gpt-4o-mini", retrieval_top_k=3)
answer = pipe.process(question, text)
```

---

## Disk Cache

Avoid re-chunking the same document on repeated runs:

```python
from chunkrank import ChunkCache, Chunker, ChunkerConfig

cache = ChunkCache(".chunkrank_cache")
chunks = cache.get(text, model="gpt-4o-mini")
if chunks is None:
    chunks = Chunker(ChunkerConfig(model="gpt-4o-mini")).split(text)
    cache.set(text, model="gpt-4o-mini", chunks=chunks)
```

---

## Runtime Model Registration

Register new models without editing the registry JSON:

```python
import chunkrank

chunkrank.register_model("my-custom-model", max_context=200_000)
```

---

## Supported Models

54 models in the built-in registry, including:

| Provider | Models |
|---|---|
| OpenAI | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo, o1, o3, o3-mini, o4-mini |
| Anthropic | claude-3-opus/sonnet/haiku, claude-3-5-sonnet/haiku, claude-sonnet-4-6, claude-opus-4-6 |
| Google | gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash, gemini-2.5-pro |
| Meta | Llama-3.1/3.2/3.3, Llama-4-Scout (10M ctx), Llama-4-Maverick |
| Mistral | mistral-7b, mixtral-8x7b, mistral-large, codestral |
| Cohere | command-r, command-r-plus, command-r7b |
| DeepSeek | deepseek-v3, deepseek-r1 |
| Qwen | qwen2.5-72b-instruct, qwen2.5-coder-32b-instruct |

Unknown models fall back to 128k context with tiktoken (`o200k_base`).

---

## How It Fits

| Tool | What it does |
|---|---|
| LangChain / LlamaIndex | Full RAG pipelines |
| Haystack | End-to-end retrieval frameworks |
| **ChunkRank** | Focused, model-aware chunking + answer ranking |

**ChunkRank complements RAG frameworks — it doesn't replace them.**

---

## Requirements

- Python 3.10+
- numpy, scikit-learn, rank-bm25

---

## Community

- [Contributors](CONTRIBUTORS.md)
- [Changelog](CHANGELOG.md)
