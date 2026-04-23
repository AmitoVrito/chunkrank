# Changelog

All notable changes to this project will be documented here.

---

## [1.2.0] — 2026-04-23

### Added
- Expanded model registry from 58 to 90 models: gpt-4.1 family, o1-pro, claude-3 aliases, gemini-2.5-flash, gemini-2.0-flash-lite, Llama-3.1-405B, Mistral small/pixtral/mixtral-8x22b, command-a, DeepSeek v2/r1-distill, Qwen3-72b, Qwen2.5 size variants, Microsoft Phi-3/4, xAI Grok-2/3, IBM Granite, Falcon
- 9 new README examples: file reading, batch processing, custom chunker config, semantic chunking, ranking method comparison, async gather, FastAPI streaming, `get_model_info`, cache integration
- Production-level test suite (`test_production_e2e.py`): 299 tests covering all 90 registry models, chunking correctness, ranking determinism, pipeline idempotence, async concurrency, cache persistence

### Changed
- Author metadata updated to `Amit.N <research.amit.n@gmail.com>`
- `CONTRIBUTORS.md`: Gaurav Nautiyal → Gaurav
- Fixed PyPI 404 links for Contributors and Changelog (relative → absolute GitHub URLs in README)

### Performance
- `load_registry()` now cached with `@lru_cache` — JSON parsed once per process instead of on every `get_model_info()` call
- `build_tokenizer()` caches tokenizer instances by `(backend, tokenizer_id)` — tiktoken/HF tokenizers instantiated once and reused
- `TfidfVectorizer` instantiated once per `Ranker` instance instead of per ranking call
- LLM API clients (`OpenAI`, `Anthropic`) cached as instance variables — connection pools reused across chunks
- Chunk token-budget loop replaced with binary search — O(log₂ N) tokenizer calls instead of O(log₀.₉ N)
- Regex patterns in `utils` pre-compiled at module load

---

## [1.1.3] — 2026-04-20

### Fixed
- Fix TOML structure in pyproject.toml so PyPI correctly reads Changelog URL

## [1.1.2] — 2026-04-20

### Fixed
- Add Changelog URL to PyPI project metadata

---

## [1.1.1] — 2026-04-20

### Changed
- Add Ayush as maintainer in MAINTAINERS.md
- Update CONTRIBUTING.md with dev extras and Apache 2.0 note
- Update chunkrank.md to reflect current v1.1.x state

---

## [1.1.0] — 2026-04-20

### Added
- `ChunkCache` — disk-backed JSON cache for chunked text (stdlib only, no new deps)
- `Ranker(method="cross-encoder")` — semantic re-ranking via `sentence-transformers` CrossEncoder
- `register_model()` — register new models at runtime without editing `model_registry.json`
- `ChunkRankPipeline.stream()` — yields answers progressively per chunk (`Iterator[str]`)
- `AsyncChunkRankPipeline.stream()` — async progressive streaming (`AsyncIterator[str]`)
- `AsyncChunkRankPipeline` and `AsyncLLMAnswerer` (parallel chunk answering via `asyncio.gather`)
- `retrieval_top_k` parameter on both pipelines (rank-first, answer top-K only)
- Expanded model registry: OpenAI o-series, Claude 3/3.5/4.x, Gemini 1.5/2.x, Llama 3.1–4, Mistral, Cohere, DeepSeek, Qwen

### Changed
- License changed from MIT to Apache 2.0
- `requires-python` bumped from `>=3.14` to `>=3.10`
- `mypy`, `ruff`, `deptry` moved from runtime deps to `[dev]` optional deps
- `regex` removed (was unused runtime dep)
- `sentence-transformers` version constraint relaxed (`>=2.7` instead of `>=2.7,<3.0`)
- Replaced deprecated `importlib.resources.open_text()` with `files().joinpath().read_text()` (compatible through Python 3.13+)
- Fixed missing `typing` imports in `tokenizers.py`

### Ranking methods
`bm25` (default) | `tfidf` | `embedding` | `cross-encoder`

---

## [0.2.4] — (previous PyPI release)

- Token-budget chunking with sliding window overlap
- BM25 and TF-IDF re-ranking
- Embedding-based ranking (local via sentence-transformers, OpenAI, Cohere)
- Semantic chunking strategy
- `ChunkRankPipeline` sync pipeline
- Initial model registry (GPT-4o-mini, Claude 3.5 Sonnet, Llama 3.1, Mistral, BERT family)
