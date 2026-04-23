from __future__ import annotations
import re
from typing import List

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def split_sentences(text: str) -> List[str]:
    parts = _SENTENCE_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def norm_words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))
