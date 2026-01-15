# ChunkRank: Model-Aware Chunking + Answer Ranking
```
Used internally for long-document QA and evaluation pipelines handling 1,000+ PDFs.
```
```
ChunkRank is a lightweight Python library that automatically chunks 
text based on an LLM’s tokenizer and context window, then consolidates
and ranks answers across chunks. In short ChunkRank is a model-aware text 
chunking and answer re-ranking library for LLM pipelines.
```

🔗 PyPI : https://pypi.org/project/chunkrank/

---

## Why ChunkRank?

When working with LLMs, long documents must be split into chunks, but:
- Every model has **different tokenizers and context limits**
- Chunk sizes are usually **hard-coded and error-prone**
- Answer quality drops when responses come from **multiple chunks**
- Existing RAG frameworks are **heavy** when you only need chunking + ranking

**ChunkRank solves this gap.**

---

## What It Does

✅**Model-aware chunking**  
- Pass a model name (`gpt-4o-mini`, `claude-3.5-sonnet`, `Llama-3.1-8B` etc.)   
- ChunkRank automatically:
  - Selects the correct tokenizer
  - Applies the correct context window
  - Reserves token space for prompts and responses

No manual token math. No trial-and-error.
  
✅**Answer consolidation & ranking**  
- Query runs across multiple chunks
- Multiple candidate answers are produced
- ChunkRank **re-ranks** them to return the best answer
Works standalone — no full RAG stack required.

---

## Installation

```bash
pip install chunkrank
```
or for development:
```bash
poetry install

```
## Quick Example

``` python
from chunkrank import ChunkRankPipeline

text = open("document.txt").read()

pipe = ChunkRankPipeline(model="gpt-4o-mini")

answer = pipe.process(
    question="What is the main topic of this document?",
    text=text
)

print(answer)
```
---


## Core API

``` python
chunks = chunkrank.split(text, model="gpt-4o-mini")

answers = chunkrank.answer(question, chunks)

best_answer = chunkrank.rank(answers)
```
---


## Supported Capabilities

- Automatic model → tokenizer → context resolution
- Token, sentence, and paragraph chunking strategies
- Cross-encoder based answer re-ranking
- Works with OpenAI, Anthropic, HF, Llama-based models
- Drop-in utility for QA, summarization, extraction

---

## How It Fits

| Tool | What it does |
|------|-------------|
| LangChain / LlamaIndex | Full RAG pipelines |
| Haystack | End-to-end retrieval frameworks |
| **ChunkRank** | Focused, model-aware chunking + answer ranking |

**ChunkRank complements RAG frameworks — it doesn’t replace them.**

---
## Roadmap
1. Build the **model registry** (model → context window + tokenizer).  
2. Implement **chunking strategies** (tokens, sentences, paragraphs).  
3. Integrate a **re-ranking engine** (start with Hugging Face cross-encoder).  
4. Package and release to PyPI with a simple API.  
---

## Community

- [Contributors](CONTRIBUTORS.md)
- [Maintainers](MAINTAINERS.md)

---
