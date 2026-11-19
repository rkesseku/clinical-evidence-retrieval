"""Command line: ``evidence-ir run-all`` (evaluate), ``tune`` (Optuna on dev) and ``serve``."""

from __future__ import annotations

import argparse
import json
import sys

from .data import load_beir
from .pipeline import SearchPipeline

DEFAULT_ENCODER = "BAAI/bge-small-en-v1.5"
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _build(args: argparse.Namespace, corpus: dict[str, str]) -> SearchPipeline:
    encoder = scorer = None
    if not args.bm25_only:
        from .dense import SentenceTransformerEncoder

        encoder = SentenceTransformerEncoder(args.encoder)
        if not args.no_rerank:
            from .rerank import CrossEncoderScorer

            scorer = CrossEncoderScorer(args.reranker)
    return SearchPipeline(
        corpus, encoder=encoder, scorer=scorer, depth=args.depth, rerank_depth=args.rerank_depth
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="evidence-ir")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run-all", "serve"):
        s = sub.add_parser(name)
        s.add_argument("--data", default="data/raw/nfcorpus", help="BEIR dataset directory")
        s.add_argument("--encoder", default=DEFAULT_ENCODER)
        s.add_argument("--reranker", default=DEFAULT_RERANKER)
        s.add_argument("--depth", type=int, default=100)
        s.add_argument("--rerank-depth", type=int, default=100)
        s.add_argument("--bm25-only", action="store_true", help="skip neural models")
        s.add_argument("--no-rerank", action="store_true")
    sub.choices["run-all"].add_argument("--split", default="test")
    sub.choices["run-all"].add_argument("--out", default="docs/results")
    sub.choices["run-all"].add_argument("--n-boot", type=int, default=2000)
    sub.choices["serve"].add_argument("--host", default="127.0.0.1")
    sub.choices["serve"].add_argument("--port", type=int, default=8000)
    t = sub.add_parser("tune", help="Optuna search for BM25 / fusion settings on the dev split")
    t.add_argument("--data", default="data/raw/nfcorpus")
    t.add_argument("--encoder", default=DEFAULT_ENCODER)
    t.add_argument("--trials", type=int, default=200)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--n-boot", type=int, default=2000)
    t.add_argument("--threads", type=int, default=None, help="torch threads for encoding")
    t.add_argument("--cache", default="data/cache/nfcorpus_bge-small_docs.npz")
    t.add_argument("--out", default="docs/results/tuning")
    args = p.parse_args(argv)

    if args.cmd == "tune":
        from .dense import SentenceTransformerEncoder
        from .tuning import run_tuning

        if args.threads:
            import torch

            torch.set_num_threads(args.threads)
        dev, test = load_beir(args.data, "dev"), load_beir(args.data, "test")
        r = run_tuning(
            dev,
            test,
            SentenceTransformerEncoder(args.encoder),
            args.out,
            n_trials=args.trials,
            seed=args.seed,
            n_boot=args.n_boot,
            cache=args.cache,
        )
        for name, s in r["systems"].items():
            d = s["test_tuned_minus_default"]["ndcg@10"]
            ndcg = s["test_tuned"]["ndcg@10"]
            print(f"{name}: tuned {s['params_tuned']} test nDCG@10 {ndcg:.4f} "
                  f"({d['diff']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}])")
        return 0

    if args.cmd == "run-all":
        from .evaluate import evaluate, write_outputs

        ds = load_beir(args.data, args.split)
        pipe = _build(args, ds.corpus)
        metrics, scores = evaluate(ds, pipe, n_boot=args.n_boot)
        write_outputs(metrics, scores, sorted(ds.qrels), args.out)
        json.dump(metrics["systems"], sys.stdout, indent=2)
        print()
        return 0

    import uvicorn

    from .data import load_corpus
    from .serve import create_app

    corpus = load_corpus(f"{args.data}/corpus.jsonl")
    uvicorn.run(create_app(_build(args, corpus)), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
