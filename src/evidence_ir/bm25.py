"""Okapi BM25 over an inverted index (Robertson & Zaragoza, 2009)."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .text import tokenize


def top_k(doc_ids: list[str], scores: np.ndarray, k: int) -> list[tuple[str, float]]:
    """Return the ``k`` best (doc_id, score) pairs, highest first, ties broken by doc order."""
    k = min(k, len(doc_ids))
    if k <= 0:
        return []
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.lexsort((idx, -scores[idx]))]
    return [(doc_ids[i], float(scores[i])) for i in idx]


class BM25Retriever:
    """BM25 with the Lucene-style non-negative IDF ``log(1 + (N - df + 0.5) / (df + 0.5))``."""

    name = "bm25"

    def __init__(self, k1: float = 0.9, b: float = 0.4) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self._postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._idf: dict[str, float] = {}
        self._norm: np.ndarray = np.zeros(0)

    def index(self, corpus: dict[str, str]) -> BM25Retriever:
        self.doc_ids = list(corpus)
        lengths = np.zeros(len(self.doc_ids))
        postings: dict[str, list[tuple[int, int]]] = {}
        for i, did in enumerate(self.doc_ids):
            counts = Counter(tokenize(corpus[did]))
            lengths[i] = sum(counts.values())
            for term, tf in counts.items():
                postings.setdefault(term, []).append((i, tf))
        n = len(self.doc_ids)
        avgdl = float(lengths.mean()) if n else 0.0
        self._postings = {
            t: (np.array([i for i, _ in p]), np.array([f for _, f in p], dtype=float))
            for t, p in postings.items()
        }
        self._idf = {
            t: math.log(1.0 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in postings.items()
        }
        self._norm = self.k1 * (1.0 - self.b + self.b * lengths / max(avgdl, 1e-9))
        return self

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.doc_ids))
        for term in set(tokenize(query)):
            if term not in self._postings:
                continue
            idx, tf = self._postings[term]
            out[idx] += self._idf[term] * tf * (self.k1 + 1.0) / (tf + self._norm[idx])
        return out

    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        scores = self.scores(query)
        hits = top_k(self.doc_ids, scores, k)
        return [(d, s) for d, s in hits if s > 0]
