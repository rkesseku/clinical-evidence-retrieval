"""Cross-encoder re-ranking of a candidate list."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class PairScorer(Protocol):
    def score(self, query: str, docs: list[str]) -> np.ndarray: ...


class CrossEncoderScorer:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
        max_length: int = 512,
        device: str | None = None,
    ) -> None:
        from sentence_transformers import CrossEncoder  # optional dependency

        self.model_name = model_name
        self.model = CrossEncoder(model_name, max_length=max_length, device=device)
        self.batch_size = batch_size

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        if not docs:
            return np.zeros(0)
        pairs = [(query, d) for d in docs]
        out = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return np.asarray(out, dtype=np.float64)


class Reranker:
    """Re-scores the top ``depth`` candidates; anything deeper keeps its order underneath."""

    name = "rerank"

    def __init__(self, scorer: PairScorer, corpus: dict[str, str], depth: int = 100) -> None:
        self.scorer = scorer
        self.corpus = corpus
        self.depth = depth

    def rerank(self, query: str, candidates: list[tuple[str, float]]) -> list[tuple[str, float]]:
        head, tail = candidates[: self.depth], candidates[self.depth :]
        if not head:
            return list(tail)
        scores = np.asarray(self.scorer.score(query, [self.corpus[d] for d, _ in head]))
        order = np.lexsort((np.arange(len(head)), -scores))
        reranked = [(head[i][0], float(scores[i])) for i in order]
        floor = float(scores.min()) - 1.0
        return reranked + [(d, floor - j) for j, (d, _) in enumerate(tail)]
