# ChunkRank

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