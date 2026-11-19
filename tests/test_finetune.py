"""CPU checks of the fine-tuning procedure with a fake trainable encoder (no models, no data)."""

import json
from collections import Counter

import numpy as np
import pytest

from evidence_ir.data import BeirDataset
from evidence_ir.finetune import (
    FineTuneData,
    build_training_pairs,
    check_disjoint,
    estimate_minutes,
    run_finetune_experiment,
)

N_DOCS = 12
CORPUS = {f"d{i}": f"document number {i}" for i in range(N_DOCS)}
# each query is relevant to exactly one document
TRAIN_Q = {f"tr{i}": f"train question {i}" for i in range(4)}
DEV_Q = {f"dv{i}": f"dev question {i}" for i in range(4)}
TEST_Q = {f"te{i}": f"test question {i}" for i in range(4)}
REL = {**{f"tr{i}": f"d{i}" for i in range(4)},
       **{f"dv{i}": f"d{4 + i}" for i in range(4)},
       **{f"te{i}": f"d{8 + i}" for i in range(4)}}
QUERIES = {**TRAIN_Q, **DEV_Q, **TEST_Q}
TEXT_TO_QID = {t: q for q, t in QUERIES.items()}


def split(qs: dict[str, str]) -> BeirDataset:
    return BeirDataset(corpus=CORPUS, queries=dict(qs), qrels={q: {REL[q]: 1} for q in qs})


class FakeEncoder:
    """One-hot documents; a query maps to its relevant doc if its id is in ``good``, else to a
    wrong doc. Records how many times it encoded each split's queries."""

    def __init__(self, tag: str, good: set[str], counter: Counter) -> None:
        self.tag, self.good, self.counter = tag, set(good), counter

    def encode_docs(self, texts):
        out = np.zeros((len(texts), N_DOCS), dtype=np.float32)
        for row, t in enumerate(texts):
            out[row, int(t.split()[-1])] = 1.0
        return out

    def encode_queries(self, texts):
        out = np.zeros((len(texts), N_DOCS), dtype=np.float32)
        for row, t in enumerate(texts):
            qid = TEXT_TO_QID[t]
            self.counter[(self.tag, qid)] += 1
            rel = int(REL[qid][1:])
            out[row, rel if qid in self.good else (rel + 1) % N_DOCS] = 1.0
        return out

    def save(self, path):
        path.mkdir(parents=True)
        (path / "fake.json").write_text(json.dumps({"tag": self.tag, "good": sorted(self.good)}))


class FakeBackend:
    """Scripted training: ``schedule[(lr, epoch)]`` is the set of queries the model gets right."""

    def __init__(self, schedule: dict, base_good: set[str] = frozenset()) -> None:
        self.schedule = schedule
        self.base_good = set(base_good)
        self.counter: Counter = Counter()
        self.trained_on: list[list[tuple[str, str]]] = []
        self.loaded: list[str] = []

    def base_encoder(self):
        return FakeEncoder("base", self.base_good, self.counter)

    def train(self, pairs, lr, epochs, on_epoch_end):
        self.trained_on.append(pairs)
        for epoch in range(1, epochs + 1):
            on_epoch_end(epoch, FakeEncoder(f"{lr}:{epoch}", self.schedule[(lr, epoch)],
                                            self.counter))

    def load_encoder(self, path):
        spec = json.loads((path / "fake.json").read_text())
        self.loaded.append(spec["tag"])
        return FakeEncoder("selected", set(spec["good"]), self.counter)


def data() -> FineTuneData:
    return FineTuneData(
        corpus=CORPUS,
        queries=QUERIES,
        train_qrels={q: {REL[q]: 1} for q in TRAIN_Q},
        dev=split(DEV_Q),
        test=split(TEST_Q),
    )


ALL_DEV, ALL_TEST = set(DEV_Q), set(TEST_Q)
HALF_DEV, HALF_TEST = {"dv0", "dv1"}, {"te0", "te1"}


def test_selection_uses_dev_not_test_and_test_is_evaluated_once(tmp_path):
    schedule = {
        (1e-5, 1): ALL_TEST,  # best on test, worst on dev: must NOT be chosen
        (1e-5, 2): {"dv0"},
        (2e-5, 1): HALF_DEV | {"te0"},
        (2e-5, 2): ALL_DEV | {"te1"},  # best on dev: must be chosen
    }
    backend = FakeBackend(schedule)
    r = run_finetune_experiment(data(), backend, tmp_path, lrs=(1e-5, 2e-5), max_epochs=2,
                                n_boot=200, log=lambda _: None)
    assert (r["selection"]["lr"], r["selection"]["epoch"]) == (2e-5, 2)
    assert r["selection"]["dev_score"] == pytest.approx(1.0)
    assert backend.loaded == ["2e-05:2"]
    # dev was scored for the base model and after every epoch; test only for base and selected
    assert [e["split"] for e in r["eval_log"]].count("dev") == 1 + 2 * 2
    assert [e for e in r["eval_log"] if e["split"] == "test"] == [
        {"split": "test", "model": "base"},
        {"split": "test", "model": "selected"},
    ]
    # independent audit through the encoder: no candidate ever saw a test query; base and the
    # selected model encoded each test query exactly once (one pass serves dense and hybrid)
    for (tag, kind), n in backend.counter.items():
        if kind == "te":
            assert tag in {"base", "selected"} and n == 1
        if kind == "dv":
            assert tag != "selected"
    assert sum(n for (_, q), n in backend.counter.items() if q.startswith("te")) == 2 * len(TEST_Q)
    # training data is the train split only
    assert all(q.startswith("train question") for q, _ in backend.trained_on[0])
    # the selected model scores what it should on test
    # te1 is ranked first; te0 finds its doc at rank 10 behind the tie-broken zeros; te2/te3 miss
    expected = (1.0 + 1.0 / np.log2(11)) / len(TEST_Q)
    assert r["test"]["finetuned"]["dense"]["ndcg@10"] == pytest.approx(expected)
    assert {c["system"] for c in r["finetuned_minus_base"]} == {"dense", "hybrid"}
    for name in ("finetune_metrics.json", "finetune_report.md", "finetune_per_query_test.csv"):
        assert (tmp_path / name).exists()
    assert json.loads((tmp_path / "finetune_metrics.json").read_text())["selection"]["epoch"] == 2


def test_base_model_kept_when_fine_tuning_never_helps_on_dev(tmp_path):
    backend = FakeBackend({(1e-5, 1): ALL_TEST, (1e-5, 2): {"dv0"}}, base_good=HALF_DEV)
    r = run_finetune_experiment(data(), backend, tmp_path, lrs=(1e-5,), max_epochs=2,
                                n_boot=200, log=lambda _: None)
    assert r["selection"]["selected_is_base"] and r["selection"]["epoch"] == 0
    assert backend.loaded == []
    assert not (tmp_path / "best_model").exists()
    diffs = [c["diff"] for c in r["finetuned_minus_base"]]
    assert diffs == pytest.approx([0.0] * len(diffs))


def test_split_overlap_is_rejected():
    d = data()
    with pytest.raises(ValueError, match="dev&test"):
        check_disjoint(d.train_qrels, d.dev, split({**DEV_Q, **TEST_Q}))


def test_training_pairs_and_estimate():
    pairs = build_training_pairs({"tr0": {"d0": 1, "d1": 0, "missing": 2}}, QUERIES, CORPUS)
    assert pairs == [("train question 0", "document number 0")]
    est = estimate_minutes(n_pairs=6400, batch_size=64, max_epochs=3, n_lrs=2,
                           steps_per_second=10.0, eval_seconds=60.0)
    assert est["train_steps"] == 600 and est["train_minutes"] == pytest.approx(1.0)
    assert est["eval_minutes"] == pytest.approx(9.0)
