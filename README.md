# ChunkRank

<p align="center">
  <b>Model-aware text chunking and answer re-ranking for LLM pipelines</b>
</p>

<p align="center">
  <a href="https://pypi.org/project/chunkrank/"><img src="https://img.shields.io/pypi/v/chunkrank?color=blue&label=PyPI" alt="PyPI version"></a>
  <a href="https://pypi.org/project/chunkrank/"><img src="https://img.shields.io/pypi/dm/chunkrank?color=green&label=downloads" alt="PyPI downloads"></a>
  <a href="https://pypi.org/project/chunkrank/"><img src="https://img.shields.io/pypi/pyversions/chunkrank?label=python" alt="Python versions"></a>
  <a href="https://github.com/AmitoVrito/chunkrank/blob/main/LICENCE"><img src="https://img.shields.io/badge/license-Apache%202.0-orange" alt="License"></a>
  <a href="https://github.com/AmitoVrito/chunkrank/blob/main/CHANGELOG.md"><img src="https://img.shields.io/badge/changelog-v1.2.0-blue" alt="Changelog"></a>
  <a href="https://github.com/AmitoVrito/chunkrank/actions"><img src="https://img.shields.io/badge/tests-116%20passing-brightgreen" alt="Tests"></a>
</p>

---

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

## More Examples

### Read from a file

```python
import chunkrank

with open("report.txt") as f:
    text = f.read()

chunks = chunkrank.split(text, model="gpt-4o-mini")
answers = chunkrank.answer("What are the key findings?", chunks)
print(chunkrank.rank(answers))
```

### Batch processing multiple documents

```python
import chunkrank

files = ["doc1.txt", "doc2.txt", "doc3.txt"]
question = "What is the main conclusion?"

for path in files:
    text = open(path).read()
    chunks = chunkrank.split(text, model="gpt-4o-mini")
    answers = chunkrank.answer(question, chunks)
    print(f"{path}: {chunkrank.rank(answers)}")
```

### Custom chunker config

```python
from chunkrank import Chunker, ChunkerConfig

config = ChunkerConfig(
    model="claude-sonnet-4-6",
    strategy="tokens",
    overlap_tokens=128,
    reserve_tokens=1024,
)
chunker = Chunker(config)
chunks = chunker.split(text)
print(f"{len(chunks)} chunks created")
```

### Semantic chunking with similarity threshold

```python
# pip install chunkrank[semantic]
chunks = chunkrank.split(
    text,
    model="gpt-4o-mini",
    strategy="semantic",
    similarity_threshold=0.6,  # higher = fewer, larger chunks
)
```

### Compare all ranking methods

```python
from chunkrank import Ranker

question = "What is the capital of France?"
answers = ["Paris is the capital.", "France is in Europe.", "The city of Paris."]

for method in ["bm25", "tfidf", "embedding", "cross-encoder"]:
    ranker = Ranker(method=method)
    best = ranker.rank(question, answers)
    print(f"{method}: {best}")
```

### Async batch with asyncio.gather

```python
import asyncio
from chunkrank import AsyncChunkRankPipeline

async def process_all(docs, question):
    pipe = AsyncChunkRankPipeline(
        model="gpt-4o-mini", provider="openai", api_key="sk-..."
    )
    tasks = [pipe.process(question, doc) for doc in docs]
    return await asyncio.gather(*tasks)

results = asyncio.run(process_all(docs, "What is the summary?"))
```

### FastAPI streaming endpoint

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from chunkrank import ChunkRankPipeline

app = FastAPI()
pipe = ChunkRankPipeline(model="gpt-4o-mini", provider="openai", api_key="sk-...")

@app.post("/ask")
def ask(question: str, text: str):
    return StreamingResponse(
        pipe.stream(question=question, text=text),
        media_type="text/plain",
    )
```

### Inspect a model's context info

```python
import chunkrank

info = chunkrank.get_model_info("gpt-4o")
print(info)
# {'name': 'gpt-4o', 'max_context': 128000, 'tokenizer': 'tiktoken', ...}
```

### Top-K retrieval + cross-encoder reranking

```python
from chunkrank import ChunkRankPipeline

# Rank chunks first, then answer only the top 3 — fewer LLM calls on large docs
pipe = ChunkRankPipeline(
    model="claude-sonnet-4-6",
    provider="anthropic",
    api_key="sk-ant-...",
    retrieval_top_k=3,
)
answer = pipe.process("What is the conclusion?", text)
```

### Register and use a custom model

```python
import chunkrank

chunkrank.register_model(
    "my-llm-v2",
    max_context=512_000,
    tokenizer="tiktoken",
    tokenizer_id="o200k_base",
    default_reserve=1024,
)

chunks = chunkrank.split(text, model="my-llm-v2")
print(f"Split into {len(chunks)} chunks")
```

### Disk cache with custom chunker

```python
from chunkrank import ChunkCache, Chunker, ChunkerConfig

cache = ChunkCache(".chunkrank_cache")
chunks = cache.get(text, model="gpt-4o")

if chunks is None:
    config = ChunkerConfig(model="gpt-4o", overlap_tokens=64)
    chunks = Chunker(config).split(text)
    cache.set(text, model="gpt-4o", chunks=chunks)

answers = chunkrank.answer(question, chunks)
print(chunkrank.rank(answers))
```

---

## Supported Models

90 models in the built-in registry, including:

| Provider | Models |
|---|---|
| OpenAI | gpt-4o, gpt-4o-mini, gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, gpt-4-turbo, gpt-4, gpt-3.5-turbo, o1, o1-mini, o1-pro, o3, o3-mini, o4-mini |
| Anthropic | claude-3-opus, claude-3-sonnet, claude-3-haiku, claude-3-5-sonnet, claude-3-5-haiku, claude-haiku-4-5, claude-sonnet-4-6, claude-opus-4-6 |
| Google | gemini-1.0-pro, gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash, gemini-2.0-flash-lite, gemini-2.5-pro, gemini-2.5-flash |
| Meta | Llama-3.1-8B/70B/405B, Llama-3.2-1B/3B/11B/90B, Llama-3.3-70B, Llama-4-Scout (10M ctx), Llama-4-Maverick |
| Mistral | mistral-7b, mistral-small, mistral-nemo, mistral-large, mixtral-8x7b, mixtral-8x22b, codestral, pixtral-large |
| Microsoft | phi-3-mini, phi-3-medium, phi-4, phi-4-mini |
| xAI | grok-2, grok-3 |
| Cohere | command-r, command-r-plus, command-r7b, command-a |
| DeepSeek | deepseek-v2, deepseek-v3, deepseek-r1, deepseek-r1-distill-qwen-32b |
| Qwen | qwen3-72b, qwen2.5-7b/14b/32b/72b-instruct, qwen2.5-coder-32b-instruct |
| IBM | granite-3.3-2b-instruct, granite-3.3-8b-instruct |
| EleutherAI | gpt-neo-2.7B, gpt-j-6B |
| Falcon | falcon-40b, falcon-180b |
| HuggingFace | BERT, DistilBERT, DeBERTa-v3, BigBird, Longformer, T5, FLAN-T5 |

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

## License

Apache 2.0 — see [LICENCE](LICENCE).

---

## Community

- [Contributors](https://github.com/AmitoVrito/chunkrank/blob/main/CONTRIBUTORS.md)
- [Changelog](https://github.com/AmitoVrito/chunkrank/blob/main/CHANGELOG.md)
- Issues: https://github.com/AmitoVrito/chunkrank/issues
