from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Optional


class ChunkCache:
    """Disk-backed cache for chunked text.

    Chunks are stored as JSON files keyed by SHA-256 of ``text + model``.
    No external dependencies — uses stdlib only.

    Usage::

        cache = ChunkCache()
        chunks = cache.get(text, model)
        if chunks is None:
            chunks = chunker.split(text)
            cache.set(text, model, chunks)
    """

    def __init__(self, cache_dir: str = ".chunkrank_cache") -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, text: str, model: str) -> Path:
        digest = hashlib.sha256(f"{model}:{text}".encode()).hexdigest()
        return self._dir / f"{digest}.json"

    def get(self, text: str, model: str) -> Optional[List[str]]:
        """Return cached chunks, or ``None`` if not cached."""
        path = self._path(text, model)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def set(self, text: str, model: str, chunks: List[str]) -> None:
        """Persist chunks to disk."""
        self._path(text, model).write_text(
            json.dumps(chunks, ensure_ascii=False), encoding="utf-8"
        )

    def clear(self) -> None:
        """Delete all cached entries."""
        for f in self._dir.glob("*.json"):
            f.unlink()
