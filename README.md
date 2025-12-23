# ChunkRank: Model-Aware Chunking + Answer Ranking

## Problem
When using LLMs, text often exceeds the model’s context window.  
To handle this, text must be **chunked** into pieces that fit the model’s maximum token length.  

Two challenges arise:
1. **Model-aware chunking**  
   Each model (OpenAI, Anthropic, Llama, Gemini, t5, Bert, BigBert, LangBert etc.) has a different context length and tokenizer.  
   Current libraries require users to manually configure chunk sizes; no unified library automatically adapts to the chosen model.

2. **Answer consolidation & ranking**  
   Once text is chunked, a query may return multiple answers from different chunks.  
   A **ranking step** is needed to decide the best, most relevant answer.  
   Existing solutions (e.g., RAG frameworks) combine retrieval + generation, but there’s no standalone library that couples **chunking** and **answer re-ranking**.

---

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

## Next Steps
1. Build the **model registry** (model → context window + tokenizer).  
2. Implement **chunking strategies** (tokens, sentences, paragraphs).  
3. Integrate a **re-ranking engine** (start with Hugging Face cross-encoder).  
4. Package and release to PyPI with a simple API.  

---

## Community

- [Contributors](CONTRIBUTORS.md)
- [Maintainers](MAINTAINERS.md)
- [Contributing Guidelines](CONTRIBUTING.md)

