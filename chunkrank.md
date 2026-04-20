# ChunkRank — Project Overview

> Model-Aware Text Chunking + Answer Ranking for LLM Pipelines

Available at: https://pypi.org/project/chunkrank/

---

## What It Does

ChunkRank solves three problems in LLM pipelines:

1. **Model-Aware Chunking** — given a model name, automatically resolves the tokenizer and context window, then splits text into correctly-sized chunks with optional overlap.

2. **Answer Consolidation & Ranking** — runs a question across all chunks, collects candidate answers, then re-ranks them to return the best one.

3. **Unified Workflow** — everything available as a single lightweight package, usable standalone or inside a RAG pipeline.

---

## Why It Exists

| Problem | Common Approach | ChunkRank |
|---|---|---|
| Different models have different token limits | Hard-code chunk size | Auto-resolves from model registry |
| Chunking + ranking are separate concerns | Use two different libs | One package, one API |
| Full RAG frameworks are heavy | LangChain / LlamaIndex | ~3 deps, drop-in utility |

---

## Architecture

```
text + model
    │
    ▼
Chunker (token-budget or semantic)
    │
    ▼
Answerer (local extractive / OpenAI / Anthropic)  ──→  parallel via asyncio
    │
    ▼
Ranker (bm25 / tfidf / embedding / cross-encoder)
    │
    ▼
best answer
```

---

## Key Features (v1.1.0)

- 54 models in the built-in registry (OpenAI, Anthropic, Gemini, Llama, Mistral, Cohere, DeepSeek, Qwen)
- Chunking strategies: `tokens` (sliding window) and `semantic` (embedding similarity)
- Ranking methods: `bm25`, `tfidf`, `embedding`, `cross-encoder`
- Sync + async pipelines with streaming support
- `ChunkCache` — disk-backed cache to avoid re-chunking
- `register_model()` — runtime model registration
- Python 3.10+, Apache 2.0

---

## Quick Start

```python
import chunkrank

chunks = chunkrank.split(text, model="gpt-4o-mini")
answers = chunkrank.answer(question, chunks)
best = chunkrank.rank(answers)
```

---

## Compared to Alternatives

- **LangChain / LlamaIndex** — full RAG frameworks; ChunkRank complements, not replaces
- **Chonkie** — chunking only, no ranking
- **Ragatouille** — ranking only (ColBERT), no chunking
- **tiktoken** — tokenization only, OpenAI-specific

ChunkRank is the only lightweight library that does both chunking and ranking in one package.
