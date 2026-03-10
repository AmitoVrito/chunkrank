from __future__ import annotations
import re
from typing import List


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def norm_words(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    return set(words)
