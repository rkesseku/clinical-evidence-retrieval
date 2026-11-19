"""Build the brief's figures from docs/results/metrics.json (run from the repo root)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs" / "results"
FIG = Path(__file__).resolve().parent / "figures"
BLUE, GREY, DARK = "#1f4e9e", "#eef3fb", "#2d2d2d"


def box(ax, x, y, w, h, text, fill=GREY, edge=BLUE, size=7.6, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fill, ec=edge, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=DARK,
            weight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1.1))


def architecture() -> None:
    m = json.loads((RESULTS / "metrics.json").read_text())
    cfg = m["config"]
    fig, ax = plt.subplots(figsize=(7.6, 3.3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.4)
    ax.axis("off")
    box(ax, 0.05, 1.7, 1.45, 0.9, "Query\n(plain\nEnglish)", bold=True)
    box(ax, 2.0, 2.65, 2.6, 1.05, f"BM25 (inverted index)\nk1={cfg['bm25']['k1']}, "
        f"b={cfg['bm25']['b']}\ntop {cfg['depth']}")
    box(ax, 2.0, 0.55, 2.6, 1.05, f"Dense bi-encoder\n{cfg['encoder'].split('/')[-1]}\n"
        f"exact cosine, top {cfg['depth']}")
    box(ax, 5.05, 1.6, 1.8, 1.05, f"Reciprocal rank\nfusion (k={cfg['rrf_k']})")
    box(ax, 7.25, 1.6, 2.7, 1.05, f"Cross-encoder re-rank\n{cfg['reranker'].split('/')[-1]}\n"
        f"top {cfg['rerank_depth']} candidates")
    arrow(ax, 1.55, 2.3, 2.0, 3.15)
    arrow(ax, 1.55, 2.0, 2.0, 1.1)
    arrow(ax, 4.65, 3.15, 5.05, 2.45)
    arrow(ax, 4.65, 1.1, 5.05, 1.85)
    arrow(ax, 6.9, 2.12, 7.25, 2.12)
    ax.text(3.3, 3.85, "bm25", ha="center", fontsize=8, color=BLUE, style="italic")
    ax.text(3.3, 0.2, "dense", ha="center", fontsize=8, color=BLUE, style="italic")
    ax.text(5.9, 1.3, "hybrid", ha="center", fontsize=8, color=BLUE, style="italic")
    ax.text(8.6, 1.3, "hybrid+rerank", ha="center", fontsize=8, color=BLUE, style="italic")
    ax.text(8.6, 3.35, "One scorer for every system:\nnDCG@10, Recall@100, MRR@10\n"
            "+ paired bootstrap over queries\nServing: FastAPI POST /search", ha="center",
            va="center", fontsize=7.2, color=DARK)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(FIG / "architecture.png", dpi=220)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    architecture()
    for name in ("ndcg_by_system.png", "latency_by_system.png"):
        shutil.copy(RESULTS / name, FIG / name)


if __name__ == "__main__":
    main()
