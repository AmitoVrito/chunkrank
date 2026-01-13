from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import re


@dataclass
class LocalExtractiveAnswerer:
    min_overlap: int = 2

    def answer(self, question: str, context: str) -> str:
        sentences = _split_sentences(context)
        if not sentences:
            return ""

        q_words = _norm_words(question)
        best: Tuple[str, int] = ("", 0)

        for s in sentences:
            s_words = _norm_words(s)
            overlap = len(q_words.intersection(s_words))
            if overlap > best[1]:
                best = (s, overlap)

        if best[1] < self.min_overlap:
            return ""
        return best[0].strip()


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _norm_words(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    return set(words)
