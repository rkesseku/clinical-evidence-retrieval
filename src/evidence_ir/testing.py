"""Deterministic, network-free stand-ins for the neural models (used by tests and demos)."""

from __future__ import annotations

import hashlib

import numpy as np

from .text import tokenize


class HashingEncoder:
    """Bag-of-words hashed into a fixed-size, L2-normalised vector. Fast and offline."""

    model_name = "hashing-encoder"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vec(t) for t in texts])

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        return self.encode_queries(texts)


class OverlapScorer:
    """Scores (query, doc) by the fraction of query terms present in the doc."""

    model_name = "overlap-scorer"

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        q = set(tokenize(query))
        if not q:
            return np.zeros(len(docs))
        return np.array([len(q.intersection(tokenize(d))) / len(q) for d in docs], dtype=float)


TOY_CORPUS = {
    "d1": "Statin use and breast cancer survival. Statins lower cholesterol.",
    "d2": "Dietary fibre intake and colorectal cancer risk in a cohort study.",
    "d3": "Vitamin D supplementation and asthma exacerbations in children.",
    "d4": "Cholesterol feeds breast cancer cells in laboratory experiments.",
    "d5": "Green tea consumption and cardiovascular mortality.",
    "d6": "Eczema and asthma improved on a plant-based diet.",
}
TOY_QUERIES = {
    "q1": "breast cancer cholesterol statins",
    "q2": "diet asthma eczema",
}
TOY_QRELS = {
    "q1": {"d1": 2, "d4": 1},
    "q2": {"d6": 2, "d3": 1},
}
