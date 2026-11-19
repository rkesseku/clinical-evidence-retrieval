"""Per-query IR metrics (trec_eval / BEIR conventions) and a paired bootstrap comparison."""

from __future__ import annotations

import math

import numpy as np

Run = dict[str, list[tuple[str, float]]]
Qrels = dict[str, dict[str, int]]


def ndcg_at_k(ranked: list[str], rels: dict[str, int], k: int = 10) -> float:
    """nDCG@k with linear gain (the graded label) and log2 discount, as in trec_eval / BEIR."""
    dcg = sum(rels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted((r for r in rels.values() if r > 0), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: list[str], rels: dict[str, int], k: int = 100) -> float:
    relevant = {d for d, r in rels.items() if r > 0}
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def mrr_at_k(ranked: list[str], rels: dict[str, int], k: int = 10) -> float:
    for i, d in enumerate(ranked[:k]):
        if rels.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


METRICS = {
    "ndcg@10": lambda ranked, rels: ndcg_at_k(ranked, rels, 10),
    "recall@100": lambda ranked, rels: recall_at_k(ranked, rels, 100),
    "mrr@10": lambda ranked, rels: mrr_at_k(ranked, rels, 10),
}


def per_query(run: Run, qrels: Qrels) -> dict[str, np.ndarray]:
    """Per-query metric arrays, ordered by sorted query id; a missing query scores 0."""
    qids = sorted(qrels)
    return {
        name: np.array([fn([d for d, _ in run.get(q, [])], qrels[q]) for q in qids])
        for name, fn in METRICS.items()
    }


def summarize(run: Run, qrels: Qrels) -> dict[str, float]:
    return {name: float(v.mean()) for name, v in per_query(run, qrels).items()}


def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> dict[str, float]:
    """Paired bootstrap over queries for mean(b - a): estimate, 95% percentile CI, one-sided p.

    ``p_one_sided`` is the share of resamples in which the mean difference is <= 0, i.e. the
    evidence against "b improves on a".
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("paired arrays must be non-empty and the same shape")
    diff = b - a
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, diff.size, size=(n_boot, diff.size))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "diff": float(diff.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_one_sided": float((means <= 0).mean()),
    }
