"""Optuna search for BM25 and hybrid-fusion settings, chosen on the dev split, tested once.

Two studies, both maximising mean nDCG@10 on the NFCorpus **dev** queries:

* ``bm25``: BM25 ``k1`` and ``b``.
* ``hybrid``: BM25 ``k1`` and ``b`` (for the lexical list inside the fusion), the RRF constant
  ``k`` and the weight on the dense list (BM25 weight fixed at 1), i.e. weighted RRF.

The library defaults are enqueued as the first trial of each study, so the selected setting is
never worse than the defaults *on dev*. Dense embeddings are computed once (and cached on disk)
because the dense retriever has nothing to tune here. The test queries are scored once at the
end, for the default and the tuned setting of each system, with a paired bootstrap over queries.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .bm25 import BM25Retriever, top_k
from .data import BeirDataset
from .dense import Encoder
from .fusion import reciprocal_rank_fusion
from .metrics import METRICS, paired_bootstrap, per_query, summarize

DEFAULTS = {"k1": 0.9, "b": 0.4, "rrf_k": 60.0, "w_dense": 1.0}
SPACE = {"k1": (0.2, 3.0), "b": (0.0, 1.0), "rrf_k": (1.0, 200.0), "w_dense": (0.2, 5.0)}


def dense_runs(
    encoder: Encoder,
    corpus: dict[str, str],
    query_sets: dict[str, dict[str, str]],
    depth: int = 100,
    cache: str | Path | None = None,
) -> dict[str, dict[str, list[tuple[str, float]]]]:
    """Top-``depth`` dense results for each named query set; document vectors cached on disk."""
    doc_ids = list(corpus)
    path = Path(cache) if cache else None
    if path is not None and path.exists():
        z = np.load(path, allow_pickle=False)
        if list(z["doc_ids"]) != doc_ids:
            raise ValueError(f"embedding cache {path} does not match the corpus")
        docs = z["vectors"]
    else:
        docs = np.asarray(encoder.encode_docs([corpus[d] for d in doc_ids]), dtype=np.float32)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, doc_ids=np.array(doc_ids), vectors=docs)
    out: dict[str, dict[str, list[tuple[str, float]]]] = {}
    for name, queries in query_sets.items():
        qids = sorted(queries)
        qv = np.asarray(encoder.encode_queries([queries[q] for q in qids]), dtype=np.float32)
        sims = qv @ docs.T
        out[name] = {q: top_k(doc_ids, sims[i], depth) for i, q in enumerate(qids)}
    return out


def bm25_runs(
    bm25: BM25Retriever, queries: dict[str, str], k1: float, b: float, depth: int = 100
) -> dict[str, list[tuple[str, float]]]:
    bm25.set_params(k1, b)
    return {q: bm25.search(queries[q], depth) for q in sorted(queries)}


def hybrid_runs(
    lexical: dict[str, list], dense: dict[str, list], rrf_k: float, w_dense: float, depth: int
) -> dict[str, list[tuple[str, float]]]:
    return {
        q: reciprocal_rank_fusion(
            [lexical[q], dense[q]], k=rrf_k, top_n=depth, weights=[1.0, w_dense]
        )
        for q in lexical
    }


def _ndcg(run, qrels) -> float:
    return summarize(run, qrels)["ndcg@10"]


def run_tuning(
    dev: BeirDataset,
    test: BeirDataset,
    encoder: Encoder,
    out_dir: str | Path,
    n_trials: int = 200,
    depth: int = 100,
    seed: int = 0,
    n_boot: int = 2000,
    cache: str | Path | None = None,
) -> dict:
    """Run both studies on dev, then score defaults and tuned settings once on test."""
    import optuna

    if set(dev.queries) & set(test.queries):
        raise ValueError("dev and test queries overlap")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t0 = time.perf_counter()
    dense = dense_runs(
        encoder, dev.corpus, {"dev": dev.queries, "test": test.queries}, depth, cache
    )
    t_dense = time.perf_counter() - t0
    bm25 = BM25Retriever(DEFAULTS["k1"], DEFAULTS["b"]).index(dev.corpus)

    def bm25_objective(trial) -> float:
        k1 = trial.suggest_float("k1", *SPACE["k1"])
        b = trial.suggest_float("b", *SPACE["b"])
        return _ndcg(bm25_runs(bm25, dev.queries, k1, b, depth), dev.qrels)

    def hybrid_objective(trial) -> float:
        k1 = trial.suggest_float("k1", *SPACE["k1"])
        b = trial.suggest_float("b", *SPACE["b"])
        rrf_k = trial.suggest_float("rrf_k", *SPACE["rrf_k"], log=True)
        w = trial.suggest_float("w_dense", *SPACE["w_dense"], log=True)
        lex = bm25_runs(bm25, dev.queries, k1, b, depth)
        return _ndcg(hybrid_runs(lex, dense["dev"], rrf_k, w, depth), dev.qrels)

    studies = {}
    t1 = time.perf_counter()
    for name, objective, keys in (
        ("bm25", bm25_objective, ("k1", "b")),
        ("hybrid", hybrid_objective, ("k1", "b", "rrf_k", "w_dense")),
    ):
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
        )
        study.enqueue_trial({k: DEFAULTS[k] for k in keys})  # trial 0 = library defaults
        study.optimize(objective, n_trials=n_trials)
        studies[name] = study
    t_search = time.perf_counter() - t1

    def test_runs(params: dict) -> dict:
        lex = bm25_runs(bm25, test.queries, params["k1"], params["b"], depth)
        return {
            "bm25": lex,
            "hybrid": hybrid_runs(lex, dense["test"], params["rrf_k"], params["w_dense"], depth),
        }

    tuned_bm25 = {**DEFAULTS, **studies["bm25"].best_params}
    tuned_hybrid = {**DEFAULTS, **studies["hybrid"].best_params}
    default_test = test_runs(DEFAULTS)  # the test split is scored here, once per setting
    tuned_test = {
        "bm25": test_runs(tuned_bm25)["bm25"],
        "hybrid": test_runs(tuned_hybrid)["hybrid"],
    }

    systems: dict[str, dict] = {}
    per_query_scores: dict[str, dict[str, list[float]]] = {}
    for system in ("bm25", "hybrid"):
        study = studies[system]
        pq_default = per_query(default_test[system], test.qrels)
        pq_tuned = per_query(tuned_test[system], test.qrels)
        systems[system] = {
            "params_default": {k: DEFAULTS[k] for k in study.best_params},
            "params_tuned": {k: float(v) for k, v in study.best_params.items()},
            "dev_ndcg@10_default": float(study.trials[0].value),
            "dev_ndcg@10_tuned": float(study.best_value),
            "best_trial": int(study.best_trial.number),
            "test_default": {m: float(v.mean()) for m, v in pq_default.items()},
            "test_tuned": {m: float(v.mean()) for m, v in pq_tuned.items()},
            "test_tuned_minus_default": {
                m: paired_bootstrap(pq_default[m], pq_tuned[m], n_boot=n_boot, seed=seed)
                for m in METRICS
            },
        }
        per_query_scores[system] = {
            "default": pq_default["ndcg@10"].tolist(),
            "tuned": pq_tuned["ndcg@10"].tolist(),
        }
    dense_test = {m: float(v.mean()) for m, v in per_query(dense["test"], test.qrels).items()}
    results = {
        "protocol": (
            "Optuna TPE on dev nDCG@10; library defaults enqueued as trial 0; test scored once "
            "for the default and the tuned setting of each system"
        ),
        "n_trials": n_trials,
        "seed": seed,
        "depth": depth,
        "n_dev_queries": len(dev.queries),
        "n_test_queries": len(test.queries),
        "search_space": {k: list(v) for k, v in SPACE.items()},
        "systems": systems,
        "dense_test": dense_test,
        "seconds": {"dense_encoding_and_search": t_dense, "optuna_search": t_search},
        "per_query_ndcg@10": per_query_scores,
    }
    write_tuning_outputs(results, out_dir)
    return results


def render_tuning_report(r: dict) -> str:
    lines = [
        "# Optuna tuning of BM25 and hybrid fusion",
        "",
        f"{r['n_trials']} TPE trials per study on {r['n_dev_queries']} dev queries (seed "
        f"{r['seed']}); defaults were trial 0. Test: {r['n_test_queries']} queries, scored once "
        "per setting.",
        "",
        "| System | Setting | Parameters | Dev nDCG@10 | Test nDCG@10 | Test Recall@100 "
        "| Test MRR@10 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, s in r["systems"].items():
        for label, params, dev_v, test_v in (
            ("default", s["params_default"], s["dev_ndcg@10_default"], s["test_default"]),
            ("tuned", s["params_tuned"], s["dev_ndcg@10_tuned"], s["test_tuned"]),
        ):
            p = ", ".join(f"{k}={v:.3g}" for k, v in params.items())
            lines.append(
                f"| {name} | {label} | {p} | {dev_v:.4f} | {test_v['ndcg@10']:.4f} | "
                f"{test_v['recall@100']:.4f} | {test_v['mrr@10']:.4f} |"
            )
    lines += ["", "Tuned minus default on test (paired bootstrap, 95% CI):", ""]
    for name, s in r["systems"].items():
        for m, d in s["test_tuned_minus_default"].items():
            lines.append(
                f"- {name} {m}: {d['diff']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}], "
                f"one-sided p = {d['p_one_sided']:.3f}"
            )
    return "\n".join(lines) + "\n"


def write_tuning_outputs(results: dict, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"json": out / "tuning.json", "report": out / "tuning.md"}
    paths["json"].write_text(json.dumps(results, indent=2), encoding="utf-8")
    paths["report"].write_text(render_tuning_report(results), encoding="utf-8")
    return paths
