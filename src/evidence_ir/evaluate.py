"""Run every system on a BEIR split and write metrics, per-query scores, a report and figures."""

from __future__ import annotations

import csv
import json
import platform
import time
from pathlib import Path

import numpy as np

from .data import BeirDataset
from .metrics import METRICS, paired_bootstrap, per_query, summarize
from .pipeline import STAGES, SearchPipeline

COMPARISONS = [
    ("bm25", "dense"),
    ("bm25", "hybrid"),
    ("bm25", "hybrid+rerank"),
    ("hybrid", "hybrid+rerank"),
    ("dense", "hybrid"),
]


def latency_summary(stage_ms: dict[str, np.ndarray], systems: list[str]) -> dict:
    """Median / p95 milliseconds per stage, and per system (sum of the stages it needs)."""
    stages = {
        s: {"median_ms": float(np.median(v)), "p95_ms": float(np.percentile(v, 95))}
        for s, v in stage_ms.items()
    }
    per_system = {}
    for system in systems:
        total = sum(stage_ms[s] for s in STAGES[system])
        per_system[system] = {
            "median_ms": float(np.median(total)),
            "p95_ms": float(np.percentile(total, 95)),
        }
    return {"stages": stages, "systems": per_system}


def evaluate(
    dataset: BeirDataset,
    pipeline: SearchPipeline,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[dict, dict]:
    """Evaluate all available systems; returns (metrics, per_query_scores)."""
    t0 = time.perf_counter()
    runs, stage_ms = pipeline.run_all(dataset.queries)
    wall = time.perf_counter() - t0
    systems = pipeline.available
    scores = {s: per_query(runs[s], dataset.qrels) for s in systems}
    comparisons = []
    for a, b in COMPARISONS:
        if a in scores and b in scores:
            for metric in METRICS:
                res = paired_bootstrap(scores[a][metric], scores[b][metric], n_boot, seed)
                comparisons.append({"baseline": a, "system": b, "metric": metric, **res})
    encoder = pipeline.dense.encoder if pipeline.dense is not None else None
    scorer = pipeline.reranker.scorer if pipeline.reranker is not None else None
    metrics = {
        "n_queries": len(dataset.queries),
        "n_docs": len(dataset.corpus),
        "systems": {s: summarize(runs[s], dataset.qrels) for s in systems},
        "comparisons": comparisons,
        "latency": latency_summary(stage_ms, systems),
        "index_seconds": {k: round(v, 3) for k, v in pipeline.index_seconds.items()},
        "eval_wall_seconds": round(wall, 2),
        "config": {
            "bm25": {"k1": pipeline.bm25.k1, "b": pipeline.bm25.b},
            "depth": pipeline.depth,
            "rrf_k": pipeline.rrf_k,
            "rerank_depth": pipeline.reranker.depth if pipeline.reranker else None,
            "encoder": getattr(encoder, "model_name", None),
            "reranker": getattr(scorer, "model_name", None),
            "bootstrap": {"n_boot": n_boot, "seed": seed},
        },
        "hardware": {"platform": platform.platform(), "processor": platform.processor()},
    }
    return metrics, scores


def write_outputs(metrics: dict, scores: dict, qids: list[str], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with open(out / "per_query.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        cols = [(s, m) for s in scores for m in METRICS]
        w.writerow(["query_id", *[f"{s}:{m}" for s, m in cols]])
        for i, q in enumerate(qids):
            w.writerow([q, *[f"{scores[s][m][i]:.4f}" for s, m in cols]])
    (out / "report.md").write_text(render_report(metrics), encoding="utf-8")
    try:
        plot_results(metrics, out)
    except ImportError:  # matplotlib missing: metrics are still written
        pass


def render_report(m: dict) -> str:
    lines = [
        "# Results",
        "",
        f"BEIR NFCorpus test split: {m['n_queries']} queries, {m['n_docs']} documents.",
        "",
        "| System | nDCG@10 | Recall@100 | MRR@10 | Median latency (ms) | p95 latency (ms) |",
        "|---|---|---|---|---|---|",
    ]
    for s, v in m["systems"].items():
        lat = m["latency"]["systems"][s]
        lines.append(
            f"| {s} | {v['ndcg@10']:.4f} | {v['recall@100']:.4f} | {v['mrr@10']:.4f} | "
            f"{lat['median_ms']:.1f} | {lat['p95_ms']:.1f} |"
        )
    lines += [
        "",
        "Paired bootstrap over queries (difference = system minus baseline):",
        "",
        "| Baseline | System | Metric | Difference | 95% CI | one-sided p |",
        "|---|---|---|---|---|---|",
    ]
    for c in m["comparisons"]:
        lines.append(
            f"| {c['baseline']} | {c['system']} | {c['metric']} | {c['diff']:+.4f} | "
            f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] | {c['p_one_sided']:.4f} |"
        )
    lines += ["", f"Config: `{json.dumps(m['config'])}`", ""]
    return "\n".join(lines)


def plot_results(m: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    systems = list(m["systems"])
    ndcg = [m["systems"][s]["ndcg@10"] for s in systems]
    base = m["systems"]["bm25"]["ndcg@10"]
    err_lo, err_hi = [], []
    for s in systems:
        c = next(
            (
                c
                for c in m["comparisons"]
                if c["baseline"] == "bm25" and c["system"] == s and c["metric"] == "ndcg@10"
            ),
            None,
        )
        if c is None:
            err_lo.append(0.0)
            err_hi.append(0.0)
        else:  # CI of the system's mean implied by the paired difference CI
            err_lo.append(max(0.0, base + c["diff"] - (base + c["ci_low"])))
            err_hi.append(max(0.0, base + c["ci_high"] - (base + c["diff"])))
    colors = ["#9aa5b1", "#5b8def", "#1f4e9e", "#0b2a5b"][: len(systems)]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(systems, ndcg, color=colors, yerr=[err_lo, err_hi], capsize=4)
    for i, v in enumerate(ndcg):
        ax.text(i, v + err_hi[i] + 0.008, f"{v:.3f}", ha="center", fontsize=9)
    ax.axhline(base, color="#9aa5b1", ls="--", lw=1)
    ax.set_ylabel("nDCG@10 (test)")
    ax.set_ylim(0, max(ndcg) * 1.25)
    ax.set_title("NFCorpus test: nDCG@10 by system (bars: 95% paired-bootstrap CI vs BM25)",
                 fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "ndcg_by_system.png", dpi=200)
    plt.close(fig)

    lat = m["latency"]["systems"]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    med = [lat[s]["median_ms"] for s in systems]
    p95 = [lat[s]["p95_ms"] for s in systems]
    x = np.arange(len(systems))
    ax.bar(x - 0.2, med, 0.4, label="median", color="#1f4e9e")
    ax.bar(x + 0.2, p95, 0.4, label="p95", color="#9aa5b1")
    ax.set_xticks(x, systems)
    ax.set_yscale("log")
    ax.set_ylabel("ms per query (CPU, log scale)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "latency_by_system.png", dpi=200)
    plt.close(fig)
