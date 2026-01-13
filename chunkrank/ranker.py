from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi


class Ranker:

    def __init__(self, method: str = "bm25"):
        self.method = method

    def rank(self, question: str, answers: List[str]) -> List[Tuple[str, float]]:
        clean = [a for a in answers if isinstance(a, str) and a.strip()]
        if not clean:
            return []

        if self.method == "tfidf":
            return self._rank_tfidf(question, clean)
        elif self.method == "bm25":
            return self._rank_bm25(question, clean)
        else:
            raise ValueError(f"Unknown ranking method: {self.method}")

    def _rank_tfidf(self, question: str, answers: List[str]) -> List[Tuple[str, float]]:
        vectorizer = TfidfVectorizer(stop_words="english")
        corpus = [question] + answers
        vectors = vectorizer.fit_transform(corpus)

        q_vec = vectors[0]
        a_vecs = vectors[1:]

        scores = cosine_similarity(q_vec, a_vecs)[0]
        return sorted(zip(answers, scores), key=lambda x: x[1], reverse=True)

    def _rank_bm25(self, question: str, answers: List[str]) -> List[Tuple[str, float]]:
        tokenized_answers = [a.split() for a in answers if a and a.strip()]
        if not tokenized_answers:
            return []
        bm25 = BM25Okapi(tokenized_answers)

        q_tokens = question.split()
        scores = bm25.get_scores(q_tokens)
        return sorted(zip(answers, scores), key=lambda x: x[1], reverse=True)

    def rank_texts(self, query: str, texts: List[str]) -> List[Tuple[str, float]]:
        """
        Rank raw texts (chunks) against a query.
        """
        return self.rank(query, texts)


def rank_answers(question: str, answers: List[str], method: str = "bm25") -> str:
    ranker = Ranker(method=method)
    ranked = ranker.rank(question, answers)
    return ranked[0][0]
