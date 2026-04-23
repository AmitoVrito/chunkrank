from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from .utils import split_sentences, norm_words


@dataclass
class LocalExtractiveAnswerer:
    min_overlap: int = 1

    def answer(self, question: str, context: str) -> Tuple[str, float]:
        sentences = split_sentences(context)
        if not sentences:
            return ("", 0.0)

        q_words = norm_words(question)
        best: Tuple[str, float] = ("", 0.0)

        for s in sentences:
            s_words = norm_words(s)
            overlap = len(q_words.intersection(s_words))
            if overlap > best[1]:
                best = (s.strip(), float(overlap))

        if best[1] < self.min_overlap:
            return ("", 0.0)
        return best


@dataclass
class LLMAnswerer:
    provider: str          # "openai" or "anthropic"
    api_key: str
    model: str = ""
    _openai_client: object = field(default=None, init=False, repr=False, compare=False)
    _anthropic_client: object = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.provider == "openai" and not self.model:
            self.model = "gpt-4o-mini"
        elif self.provider == "anthropic" and not self.model:
            self.model = "claude-haiku-4-5-20251001"

    def answer(self, question: str, context: str) -> Tuple[str, float]:
        if self.provider == "openai":
            return self._openai(question, context)
        elif self.provider == "anthropic":
            return self._anthropic(question, context)
        else:
            raise ValueError(f"Unknown provider: {self.provider}. Use 'openai' or 'anthropic'.")

    def _openai(self, question: str, context: str) -> Tuple[str, float]:
        if self._openai_client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
            self._openai_client = OpenAI(api_key=self.api_key)

        client = self._openai_client
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Answer the question using only the provided context. Be concise."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}"
                }
            ],
            max_tokens=512,
        )
        text = response.choices[0].message.content.strip()
        score = response.choices[0].logprobs.content[0].logprob if response.choices[0].logprobs else 1.0
        return (text, float(score) if score else 1.0)

    def _anthropic(self, question: str, context: str) -> Tuple[str, float]:
        if self._anthropic_client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
            self._anthropic_client = anthropic.Anthropic(api_key=self.api_key)

        client = self._anthropic_client
        message = client.messages.create(
            model=self.model,
            max_tokens=512,
            system="Answer the question using only the provided context. Be concise.",
            messages=[
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}"
                }
            ]
        )
        text = message.content[0].text.strip()
        return (text, 1.0)


