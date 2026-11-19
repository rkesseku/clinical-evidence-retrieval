"""Reciprocal rank fusion (Cormack, Clarke & Buettcher, 2009)."""

from __future__ import annotations


def reciprocal_rank_fusion(
    runs: list[list[tuple[str, float]]],
    k: float = 60,
    top_n: int | None = None,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked lists: score(d) = sum over runs of 1 / (k + rank(d)), ranks starting at 1.

    Only ranks are used, so BM25 and cosine scores never need to be put on a common scale.
    Ties are broken by the order in which documents first appear across the runs. Optional
    ``weights`` scale each run's contribution (weighted RRF); ``None`` means all 1.
    """
    weights = weights or [1.0] * len(runs)
    if len(weights) != len(runs):
        raise ValueError("one weight per run")
    fused: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    for w, run in zip(weights, runs, strict=True):
        for rank, (doc_id, _) in enumerate(run, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + w / (k + rank)
            first_seen.setdefault(doc_id, len(first_seen))
    ranked = sorted(fused.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))
    return ranked[:top_n] if top_n is not None else ranked
