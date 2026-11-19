"""CPU tests for the Optuna tuning study (offline hashing encoder, toy corpus)."""

import json

import numpy as np

from evidence_ir.bm25 import BM25Retriever
from evidence_ir.data import BeirDataset
from evidence_ir.fusion import reciprocal_rank_fusion
from evidence_ir.testing import TOY_CORPUS, HashingEncoder
from evidence_ir.tuning import DEFAULTS, SPACE, run_tuning


def test_set_params_matches_fresh_index():
    a = BM25Retriever(1.5, 0.75).index(TOY_CORPUS)
    b = BM25Retriever().index(TOY_CORPUS).set_params(1.5, 0.75)
    assert np.allclose(a.scores("breast cancer statin"), b.scores("breast cancer statin"))


def test_weighted_rrf_reduces_to_rrf_and_respects_weights():
    r1, r2 = [("a", 1), ("b", 1)], [("b", 1), ("c", 1)]
    assert reciprocal_rank_fusion([r1, r2]) == reciprocal_rank_fusion([r1, r2], weights=[1, 1])
    top = reciprocal_rank_fusion([r1, r2], k=1, weights=[1.0, 10.0])[0][0]
    assert top == "b"


class CountingEncoder(HashingEncoder):
    def __init__(self):
        super().__init__()
        self.queries: list[str] = []
        self.doc_calls = 0

    def encode_queries(self, texts):
        self.queries += list(texts)
        return super().encode_queries(texts)

    def encode_docs(self, texts):
        self.doc_calls += 1
        return super().encode_queries(texts)


def test_tuning_selects_on_dev_and_scores_test_once(tmp_path):
    dev = BeirDataset(
        TOY_CORPUS,
        {"q1": "breast cancer cholesterol statins", "q2": "diet asthma eczema"},
        {"q1": {"d1": 2, "d4": 1}, "q2": {"d6": 2, "d3": 1}},
    )
    test = BeirDataset(
        TOY_CORPUS,
        {"t1": "vitamin asthma children", "t2": "green tea heart mortality"},
        {"t1": {"d3": 2}, "t2": {"d5": 2}},
    )
    enc = CountingEncoder()
    cache = tmp_path / "cache" / "docs.npz"
    r = run_tuning(dev, test, enc, tmp_path / "out", n_trials=12, depth=6, n_boot=50, cache=cache)
    # dense vectors: documents encoded once and cached; each query encoded exactly once
    assert enc.doc_calls == 1 and cache.exists()
    assert sorted(enc.queries) == sorted(list(dev.queries.values()) + list(test.queries.values()))
    for s in r["systems"].values():
        assert s["dev_ndcg@10_tuned"] >= s["dev_ndcg@10_default"]  # defaults are trial 0
        for k, v in s["params_tuned"].items():
            lo, hi = SPACE[k]
            assert lo <= v <= hi or v == DEFAULTS[k]
        assert set(s["test_tuned_minus_default"]) == {"ndcg@10", "recall@100", "mrr@10"}
    saved = json.loads((tmp_path / "out" / "tuning.json").read_text())
    assert saved["n_dev_queries"] == 2 and saved["n_test_queries"] == 2
    # second run reuses the cached document vectors
    enc2 = CountingEncoder()
    run_tuning(dev, test, enc2, tmp_path / "out2", n_trials=3, depth=6, n_boot=20, cache=cache)
    assert enc2.doc_calls == 0
