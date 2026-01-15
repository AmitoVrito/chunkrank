# ChunkRank

## Existing Libraries & Gaps

### Chunking
- **LangChain Text Splitters** → Token-based, works with `tiktoken`, but requires manual chunk size config.  
- **LlamaIndex `TokenTextSplitter`** → Similar functionality, manual sizing.  
- **Haystack `PreProcessor`** → Can split by tokens, overlap supported, but not model-aware by default.  
- **semantic-text-splitter / semchunk** → Standalone, supports tiktoken/HF tokenizers, still needs user-specified chunk length.

**Gap:** None of these libraries automatically map a model → tokenizer → context window → chunk size.

---

### Ranking
- **pygaggle** (Waterloo CAST) → neural re-ranker.  
- **Tevatron** → dense retrieval + re-ranking toolkit.  
- **Pyserini** (with pygaggle) → BM25 + neural re-rankers.  
- **Haystack, LlamaIndex** → include ranking in RAG pipelines.  

**Gap:** Ranking exists, but **not combined with chunking** in a single, simple package.

---

## What We Want to Build
A standalone Python library that:

1. **Model-Aware Chunking**
   - User specifies a model name (e.g., `gpt-4o-mini`, `claude-3.5-sonnet`, `Llama-3.1-8B`).
   - Library looks up the model’s max context window and tokenizer.
   - Automatically chunks text into model-compatible pieces with optional overlap and reserve space.

2. **Answer Consolidation & Ranking**
   - Given multiple answers from chunks, apply a re-ranking step to select the best one.
   - Should integrate with existing ranking models (cross-encoder, bi-encoder, BM25 + re-ranker).
   - Should work standalone, without needing a full RAG pipeline.

3. **Unified Workflow**
   - `chunks = chunkrank.split(text, model="gpt-4o-mini")`
   - `answers = chunkrank.answer(question, chunks)`
   - `best = chunkrank.rank(answers)`

---

## Vision
- Lightweight, model-agnostic utility library.  
- Bridges the gap between **text preparation** (chunking) and **answer quality** (ranking).  
- Complements existing RAG frameworks but can also work independently.  
- Easy to drop into pipelines: preprocessing for QA, summarization, or information extraction.

---

> Model-Aware Text Chunking + Answer Ranking for LLM Pipelines

Available at : https://pypi.org/project/chunkrank/
## Installation

```bash
pip install chunkrank
```
or for development:
```bash
poetry install

```
## Usage

``` python
from chunkrank import ChunkRankPipeline

text = "..."  # Some long document
pipe = ChunkRankPipeline(model="gpt-4o-mini")
answer = pipe.process("What is the topic?", text)
```

## Features

- Automatic model-aware chunking (context-size, tokenizer)
- Sentence/paragraph strategies
- Answer re-ranking via cross-encoder
- Works standalone or with RAG pipelines