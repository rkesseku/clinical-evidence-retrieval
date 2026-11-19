"""Compose BM25, dense, RRF and cross-encoder stages into the four compared systems."""

from __future__ import annotations

import time

import numpy as np

from .bm25 import BM25Retriever
from .dense import DenseRetriever, Encoder
from .fusion import reciprocal_rank_fusion
from .rerank import PairScorer, Reranker

SYSTEMS = ("bm25", "dense", "hybrid", "hybrid+rerank")
# query-time stages each system actually depends on (dense-only does not need BM25)
STAGES = {
    "bm25": ["bm25"],
    "dense": ["dense"],
    "hybrid": ["bm25", "dense", "fusion"],
    "hybrid+rerank": ["bm25", "dense", "fusion", "rerank"],
}


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


class SearchPipeline:
    """One index per retriever, shared by every system.

    * ``bm25``: lexical only
    * ``dense``: bi-encoder only
    * ``hybrid``: reciprocal rank fusion of the BM25 and dense top-``depth`` lists
    * ``hybrid+rerank``: the hybrid list with its top ``rerank_depth`` re-scored by a cross-encoder
    """

    def __init__(
        self,
        corpus: dict[str, str],
        encoder: Encoder | None = None,
        scorer: PairScorer | None = None,
        depth: int = 100,
        rrf_k: int = 60,
        rerank_depth: int = 100,
    ) -> None:
        self.corpus = corpus
        self.depth = depth
        self.rrf_k = rrf_k
        self.index_seconds: dict[str, float] = {}
        t0 = time.perf_counter()
        self.bm25 = BM25Retriever().index(corpus)
        self.index_seconds["bm25"] = time.perf_counter() - t0
        self.dense: DenseRetriever | None = None
        if encoder is not None:
            t0 = time.perf_counter()
            self.dense = DenseRetriever(encoder).index(corpus)
            self.index_seconds["dense"] = time.perf_counter() - t0
        self.reranker = Reranker(scorer, corpus, rerank_depth) if scorer is not None else None

    @property
    def available(self) -> list[str]:
        out = ["bm25"]
        if self.dense is not None:
            out += ["dense", "hybrid"]
            if self.reranker is not None:
                out.append("hybrid+rerank")
        return out

    def run_query(self, query: str, systems: list[str] | None = None) -> tuple[dict, dict]:
        """Ranked lists for the requested systems plus per-stage latency (ms).

        Stages are shared: ``hybrid+rerank`` reuses the BM25, dense and fused lists.
        """
        systems = systems or self.available
        bad = [s for s in systems if s not in self.available]
        if bad:
            raise ValueError(f"unavailable system(s) {bad}; available: {self.available}")
        need_dense = any(s != "bm25" for s in systems)
        need_fusion = any(s.startswith("hybrid") for s in systems)
        runs: dict[str, list[tuple[str, float]]] = {}
        ms: dict[str, float] = {}

        t0 = time.perf_counter()
        lexical = self.bm25.search(query, self.depth)
        ms["bm25"] = _ms(t0)
        runs["bm25"] = lexical
        if need_dense:
            assert self.dense is not None
            t0 = time.perf_counter()
            runs["dense"] = self.dense.search(query, self.depth)
            ms["dense"] = _ms(t0)
        if need_fusion:
            t0 = time.perf_counter()
            runs["hybrid"] = reciprocal_rank_fusion(
                [lexical, runs["dense"]], k=self.rrf_k, top_n=self.depth
            )
            ms["fusion"] = _ms(t0)
        if "hybrid+rerank" in systems:
            assert self.reranker is not None
            t0 = time.perf_counter()
            runs["hybrid+rerank"] = self.reranker.rerank(query, runs["hybrid"])
            ms["rerank"] = _ms(t0)
        return {s: runs[s] for s in systems}, ms

    def search(self, query: str, system: str = "hybrid", k: int = 10) -> dict:
        runs, ms = self.run_query(query, [system])
        stages = STAGES[system]
        timings = {f"{s}_ms": round(ms[s], 3) for s in stages if s in ms}
        timings["total_ms"] = round(sum(ms[s] for s in stages if s in ms), 3)
        return {"system": system, "hits": runs[system][:k], "timings": timings}

    def run_all(self, queries: dict[str, str]) -> tuple[dict, dict[str, np.ndarray]]:
        """Every available system over every query, one query at a time (realistic latency)."""
        runs: dict[str, dict] = {s: {} for s in self.available}
        stage_ms: dict[str, list[float]] = {}
        for qid, text in queries.items():
            per_system, ms = self.run_query(text)
            for s, ranked in per_system.items():
                runs[s][qid] = ranked
            for stage, value in ms.items():
                stage_ms.setdefault(stage, []).append(value)
        return runs, {k: np.asarray(v) for k, v in stage_ms.items()}
