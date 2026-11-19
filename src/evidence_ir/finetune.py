"""Fine-tune the dense encoder with dev-split model selection and a single test evaluation.

Procedure (``run_finetune_experiment``):

1. Build (query, relevant document) pairs from the **train** qrels only.
2. For each learning rate in a small grid, train from the base model for up to ``max_epochs``
   epochs. After every epoch the current model is scored on the **dev** split (dense nDCG@10,
   computed with ``evidence_ir.metrics``). The untuned base model is also a dev candidate
   ("epoch 0"), so if fine-tuning never helps on dev, the base model is kept.
3. Only the single best dev checkpoint is saved; nothing about the test split is seen during
   training or selection.
4. The selected model and the base model are each evaluated **once** on the **test** split, with
   the same pipeline as the CPU results (BM25, dense, hybrid RRF), and compared with the paired
   bootstrap from ``evidence_ir.metrics``.

The training itself sits behind ``FineTuneBackend`` so the selection logic can be tested on CPU
with a fake encoder; ``SentenceTransformerBackend`` is the real (GPU) implementation.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .data import BeirDataset, load_beir, load_corpus, load_qrels, load_queries
from .dense import DenseRetriever, Encoder, SentenceTransformerEncoder
from .metrics import METRICS, Qrels, paired_bootstrap, per_query, summarize
from .pipeline import SearchPipeline

TEST_SYSTEMS = ("bm25", "dense", "hybrid")
COMPARED_SYSTEMS = ("dense", "hybrid")
SELECTION_METRIC = "ndcg@10"


class TrainableEncoder(Encoder, Protocol):
    def save(self, path: str | Path) -> None: ...


EpochCallback = Callable[[int, TrainableEncoder], None]


class FineTuneBackend(Protocol):
    """What the experiment needs from a training library."""

    def base_encoder(self) -> Encoder: ...

    def train(
        self, pairs: list[tuple[str, str]], lr: float, epochs: int, on_epoch_end: EpochCallback
    ) -> None:
        """Train a fresh copy of the base model; call ``on_epoch_end(epoch, encoder)`` (1-based)."""
        ...

    def load_encoder(self, path: str | Path) -> Encoder: ...


@dataclass
class FineTuneData:
    corpus: dict[str, str]
    queries: dict[str, str]
    train_qrels: Qrels
    dev: BeirDataset
    test: BeirDataset


def load_finetune_data(root: str | Path) -> FineTuneData:
    """Corpus, all queries, train qrels and the dev / test splits of a BEIR directory."""
    root = Path(root)
    return FineTuneData(
        corpus=load_corpus(root / "corpus.jsonl"),
        queries=load_queries(root / "queries.jsonl"),
        train_qrels=load_qrels(root / "qrels" / "train.tsv"),
        dev=load_beir(root, "dev"),
        test=load_beir(root, "test"),
    )


def check_disjoint(train_qrels: Qrels, dev: BeirDataset, test: BeirDataset) -> None:
    """Train, dev and test query sets must not overlap (no leakage into training or selection)."""
    tr, dv, te = set(train_qrels), set(dev.queries), set(test.queries)
    overlaps = {"train&dev": tr & dv, "train&test": tr & te, "dev&test": dv & te}
    bad = {k: len(v) for k, v in overlaps.items() if v}
    if bad:
        raise ValueError(f"query splits overlap: {bad}")


def build_training_pairs(
    train_qrels: Qrels, queries: dict[str, str], corpus: dict[str, str]
) -> list[tuple[str, str]]:
    """(query text, document text) for every positive train judgement (raw text, no prefix)."""
    return [
        (queries[q], corpus[d])
        for q, rels in train_qrels.items()
        if q in queries
        for d, s in rels.items()
        if s > 0 and d in corpus
    ]


def dense_scores(encoder: Encoder, dataset: BeirDataset, depth: int = 100) -> dict[str, float]:
    """Dense-only summary metrics on a split (used for dev selection)."""
    retriever = DenseRetriever(encoder).index(dataset.corpus)
    run = {q: retriever.search(t, depth) for q, t in dataset.queries.items()}
    return summarize(run, dataset.qrels)


def estimate_minutes(
    n_pairs: int,
    batch_size: int,
    max_epochs: int,
    n_lrs: int,
    steps_per_second: float = 6.0,
    eval_seconds: float = 45.0,
) -> dict[str, float]:
    """Rough wall-clock estimate. ``steps_per_second`` and ``eval_seconds`` are assumptions
    (bge-small, seq len 256, fp16 on a T4), not measurements; the run logs the real times."""
    steps = int(np.ceil(n_pairs / batch_size)) * max_epochs * n_lrs
    n_dev_evals = max_epochs * n_lrs + 1
    train_min = steps / steps_per_second / 60
    eval_min = (n_dev_evals + 2) * eval_seconds / 60  # + base and selected on test
    return {
        "train_steps": steps,
        "train_minutes": round(train_min, 1),
        "eval_minutes": round(eval_min, 1),
        "total_minutes": round(train_min + eval_min, 1),
    }


def _evaluate_test(
    encoder: Encoder, test: BeirDataset, depth: int, rrf_k: int
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, np.ndarray]]]:
    pipe = SearchPipeline(test.corpus, encoder=encoder, depth=depth, rrf_k=rrf_k)
    runs, _ = pipe.run_all(test.queries)
    systems = {s: summarize(runs[s], test.qrels) for s in TEST_SYSTEMS}
    scores = {s: per_query(runs[s], test.qrels) for s in TEST_SYSTEMS}
    return systems, scores


def run_finetune_experiment(
    data: FineTuneData,
    backend: FineTuneBackend,
    out_dir: str | Path,
    lrs: Sequence[float] = (1e-5, 2e-5, 5e-5),
    max_epochs: int = 3,
    depth: int = 100,
    rrf_k: int = 60,
    n_boot: int = 2000,
    seed: int = 0,
    log: Callable[[str], None] = print,
    extra_info: dict | None = None,
) -> dict:
    """Train with dev-based checkpoint selection, then evaluate base and selected once on test."""
    check_disjoint(data.train_qrels, data.dev, data.test)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    best_dir = out / "best_model"
    pairs = build_training_pairs(data.train_qrels, data.queries, data.corpus)
    eval_log: list[dict] = []  # every split evaluation, in order (audited by the tests)
    t_start = time.perf_counter()

    def dev_eval(encoder: Encoder, label: str) -> dict[str, float]:
        eval_log.append({"split": "dev", "model": label})
        return dense_scores(encoder, data.dev, depth)

    base = backend.base_encoder()
    base_dev = dev_eval(base, "base")
    log(f"base model: dev {SELECTION_METRIC} = {base_dev[SELECTION_METRIC]:.4f}")
    curve = [{"lr": None, "epoch": 0, "dev": base_dev, "train_seconds": 0.0}]
    best = {"lr": None, "epoch": 0, "score": base_dev[SELECTION_METRIC]}

    for lr in lrs:
        t_lr = time.perf_counter()

        def on_epoch_end(epoch: int, encoder: TrainableEncoder, lr: float = lr,
                         t_lr: float = t_lr) -> None:
            elapsed = time.perf_counter() - t_lr
            dev = dev_eval(encoder, f"lr={lr:g},epoch={epoch}")
            score = dev[SELECTION_METRIC]
            curve.append({"lr": lr, "epoch": epoch, "dev": dev, "train_seconds": round(elapsed, 1)})
            improved = score > best["score"]
            if improved:  # keep only the best dev checkpoint on disk
                if best_dir.exists():
                    shutil.rmtree(best_dir)
                encoder.save(best_dir)
                best.update(lr=lr, epoch=epoch, score=score)
            log(f"lr={lr:g} epoch {epoch}: dev {SELECTION_METRIC} = {score:.4f}"
                f"{'  (new best)' if improved else ''}  [{elapsed / 60:.1f} min]")

        backend.train(pairs, lr, max_epochs, on_epoch_end)

    selected_is_base = best["epoch"] == 0
    if selected_is_base:
        log("no fine-tuned checkpoint beat the base model on dev; the base model is selected")
        selected: Encoder = base
    else:
        log(f"selected lr={best['lr']:g}, epoch {best['epoch']} by dev {SELECTION_METRIC}")
        selected = backend.load_encoder(best_dir)

    # the test split is touched exactly twice: once for the base model, once for the selection
    eval_log.append({"split": "test", "model": "base"})
    base_test, base_scores = _evaluate_test(base, data.test, depth, rrf_k)
    eval_log.append({"split": "test", "model": "selected"})
    ft_test, ft_scores = _evaluate_test(selected, data.test, depth, rrf_k)

    comparisons = [
        {
            "system": s,
            "metric": m,
            **paired_bootstrap(base_scores[s][m], ft_scores[s][m], n_boot, seed),
        }
        for s in COMPARED_SYSTEMS
        for m in METRICS
    ]
    result = {
        "procedure": (
            "train on train qrels; after each epoch score dense nDCG@10 on dev; keep the best "
            "dev checkpoint (base model included as epoch 0); evaluate base and selected once "
            "on test (bm25, dense, hybrid RRF); paired bootstrap selected minus base"
        ),
        "n_train_queries": len(data.train_qrels),
        "n_train_pairs": len(pairs),
        "n_dev_queries": len(data.dev.queries),
        "n_test_queries": len(data.test.queries),
        "n_docs": len(data.corpus),
        "grid": {"lrs": list(lrs), "max_epochs": max_epochs},
        "selection": {
            "metric": f"dense {SELECTION_METRIC} on dev",
            "lr": best["lr"],
            "epoch": best["epoch"],
            "dev_score": best["score"],
            "selected_is_base": selected_is_base,
        },
        "dev_curve": curve,
        "test": {"base": base_test, "finetuned": ft_test},
        "finetuned_minus_base": comparisons,
        "config": {"depth": depth, "rrf_k": rrf_k, "bootstrap": {"n_boot": n_boot, "seed": seed}},
        "eval_log": eval_log,
        "wall_minutes": round((time.perf_counter() - t_start) / 60, 2),
        **(extra_info or {}),
    }
    write_finetune_outputs(result, base_scores, ft_scores, sorted(data.test.qrels), out)
    return result


def write_finetune_outputs(
    result: dict, base_scores: dict, ft_scores: dict, qids: list[str], out: Path
) -> None:
    (out / "finetune_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    cols = [(label, s, m) for label in ("base", "finetuned") for s in TEST_SYSTEMS for m in METRICS]
    src = {"base": base_scores, "finetuned": ft_scores}
    lines = ["query_id," + ",".join(f"{lab}:{s}:{m}" for lab, s, m in cols)]
    for i, q in enumerate(qids):
        lines.append(q + "," + ",".join(f"{src[lab][s][m][i]:.4f}" for lab, s, m in cols))
    (out / "finetune_per_query_test.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "finetune_report.md").write_text(render_finetune_report(result), encoding="utf-8")


def render_finetune_report(r: dict) -> str:
    sel = r["selection"]
    chosen = "base model (no fine-tuning)" if sel["selected_is_base"] else (
        f"lr {sel['lr']:g}, epoch {sel['epoch']}"
    )
    lines = [
        "# Encoder fine-tuning results",
        "",
        f"Train: {r['n_train_queries']} queries ({r['n_train_pairs']} positive pairs). "
        f"Dev: {r['n_dev_queries']} queries (model selection). "
        f"Test: {r['n_test_queries']} queries (evaluated once). Corpus: {r['n_docs']} documents.",
        "",
        "## Dev curve (dense nDCG@10; selection uses dev only)",
        "",
        "| lr | Epoch | Dev nDCG@10 | Dev Recall@100 | Selected |",
        "|---|---|---|---|---|",
    ]
    for c in r["dev_curve"]:
        is_sel = c["lr"] == sel["lr"] and c["epoch"] == sel["epoch"]
        lr = "base" if c["lr"] is None else f"{c['lr']:g}"
        lines.append(
            f"| {lr} | {c['epoch']} | {c['dev']['ndcg@10']:.4f} | {c['dev']['recall@100']:.4f} | "
            f"{'yes' if is_sel else ''} |"
        )
    lines += [
        "",
        f"Selected: {chosen}.",
        "",
        "## Test (single evaluation of base and selected encoder)",
        "",
        "| System | Encoder | nDCG@10 | Recall@100 | MRR@10 |",
        "|---|---|---|---|---|",
    ]
    for s in TEST_SYSTEMS:
        for label in ("base", "finetuned"):
            if s == "bm25" and label == "finetuned":
                continue  # BM25 does not use the encoder
            v = r["test"][label][s]
            enc = "-" if s == "bm25" else label
            lines.append(
                f"| {s} | {enc} | {v['ndcg@10']:.4f} | {v['recall@100']:.4f} | {v['mrr@10']:.4f} |"
            )
    lines += [
        "",
        "Paired bootstrap over test queries (difference = fine-tuned minus base):",
        "",
        "| System | Metric | Difference | 95% CI | one-sided p |",
        "|---|---|---|---|---|",
    ]
    for c in r["finetuned_minus_base"]:
        lines.append(
            f"| {c['system']} | {c['metric']} | {c['diff']:+.4f} | "
            f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] | {c['p_one_sided']:.4f} |"
        )
    lines += ["", f"Wall time: {r['wall_minutes']:.1f} min.", ""]
    return "\n".join(lines)


class _LiveEncoder:
    """Encodes with the model that is being trained; uses the evaluation sequence length."""

    def __init__(self, model, query_prefix: str, max_seq_length: int, batch_size: int) -> None:
        self.model = model
        self.query_prefix = query_prefix
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def _encode(self, texts: list[str]) -> np.ndarray:
        train_len = self.model.max_seq_length
        self.model.max_seq_length = self.max_seq_length
        try:
            vecs = self.model.encode(
                texts, batch_size=self.batch_size, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False,
            )
        finally:
            self.model.max_seq_length = train_len
        return np.asarray(vecs, dtype=np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.query_prefix + t for t in texts])

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def save(self, path: str | Path) -> None:
        self.model.save(str(path))


class SentenceTransformerBackend:
    """MultipleNegativesRankingLoss (in-batch negatives, no duplicate queries per batch).

    Training uses ``train_max_seq_length`` (256) for speed; every evaluation, base model included,
    uses ``eval_max_seq_length`` (512, the setting of the committed CPU results).
    """

    def __init__(
        self,
        base_model: str = "BAAI/bge-small-en-v1.5",
        device: str | None = None,
        batch_size: int = 64,
        warmup_ratio: float = 0.1,
        train_max_seq_length: int = 256,
        eval_max_seq_length: int = 512,
        fp16: bool = True,
        seed: int = 13,
        work_dir: str | Path = "ft_work",
    ) -> None:
        self.base_model = base_model
        self.device = device
        self.batch_size = batch_size
        self.warmup_ratio = warmup_ratio
        self.train_max_seq_length = train_max_seq_length
        self.eval_max_seq_length = eval_max_seq_length
        self.fp16 = fp16
        self.seed = seed
        self.work_dir = str(work_dir)
        self.query_prefix = (
            SentenceTransformerEncoder.BGE_QUERY_PREFIX if "bge" in base_model.lower() else ""
        )

    def hyperparameters(self) -> dict:
        return {
            "base_model": self.base_model, "batch_size": self.batch_size,
            "warmup_ratio": self.warmup_ratio, "scheduler": "linear decay over all epochs",
            "train_max_seq_length": self.train_max_seq_length,
            "eval_max_seq_length": self.eval_max_seq_length, "fp16": self.fp16,
            "seed": self.seed, "loss": "MultipleNegativesRankingLoss, no-duplicate batches",
        }

    def base_encoder(self) -> Encoder:
        return SentenceTransformerEncoder(
            self.base_model, query_prefix=self.query_prefix,
            max_seq_length=self.eval_max_seq_length, device=self.device,
        )

    def load_encoder(self, path: str | Path) -> Encoder:
        return SentenceTransformerEncoder(
            str(path), query_prefix=self.query_prefix,
            max_seq_length=self.eval_max_seq_length, device=self.device,
        )

    def train(
        self, pairs: list[tuple[str, str]], lr: float, epochs: int, on_epoch_end: EpochCallback
    ) -> None:
        from datasets import Dataset
        from sentence_transformers import (
            SentenceTransformer,
            SentenceTransformerTrainer,
            SentenceTransformerTrainingArguments,
            losses,
        )
        from sentence_transformers.training_args import BatchSamplers
        from transformers import TrainerCallback, set_seed

        set_seed(self.seed)
        model = SentenceTransformer(self.base_model, device=self.device)
        model.max_seq_length = self.train_max_seq_length
        live = _LiveEncoder(model, self.query_prefix, self.eval_max_seq_length, 128)
        ds = Dataset.from_dict({
            "anchor": [self.query_prefix + q for q, _ in pairs],
            "positive": [d for _, d in pairs],
        })

        class _EpochEnd(TrainerCallback):
            def on_epoch_end(self, args, state, control, **kwargs):
                on_epoch_end(int(round(state.epoch)), live)

        args = SentenceTransformerTrainingArguments(
            output_dir=self.work_dir, num_train_epochs=epochs,
            per_device_train_batch_size=self.batch_size, learning_rate=lr,
            warmup_ratio=self.warmup_ratio, fp16=self.fp16,
            batch_sampler=BatchSamplers.NO_DUPLICATES, logging_steps=200,
            save_strategy="no", seed=self.seed, report_to="none",
        )
        trainer = SentenceTransformerTrainer(
            model=model, args=args, train_dataset=ds,
            loss=losses.MultipleNegativesRankingLoss(model), callbacks=[_EpochEnd()],
        )
        trainer.train()
        del trainer, model
