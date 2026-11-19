import json

import numpy as np
import pytest

from evidence_ir.bm25 import BM25Retriever, top_k
from evidence_ir.data import load_beir
from evidence_ir.dense import DenseRetriever
from evidence_ir.fusion import reciprocal_rank_fusion
from evidence_ir.metrics import mrr_at_k, ndcg_at_k, paired_bootstrap, recall_at_k, summarize
from evidence_ir.rerank import Reranker
from evidence_ir.testing import TOY_CORPUS, HashingEncoder, OverlapScorer
from evidence_ir.text import stem, tokenize


def test_tokenize_drops_stopwords_and_stems():
    assert tokenize("The statins are lowering cholesterol levels") == [
        "statin",
        "lower",
        "cholesterol",
        "level",
    ]
    assert stem("studies") == "study"
    assert stem("glass") == "glass"
    assert stem("virus") == "virus"


def test_bm25_ranks_matching_docs_first():
    bm = BM25Retriever().index(TOY_CORPUS)
    hits = bm.search("breast cancer cholesterol", k=3)
    assert {d for d, _ in hits[:2]} == {"d1", "d4"}
    assert all(s > 0 for _, s in hits)
    assert bm.search("zzzz unknown", k=5) == []


def test_bm25_idf_rewards_rare_terms():
    corpus = {"a": "cancer cancer statin", "b": "cancer diet", "c": "cancer tea"}
    bm = BM25Retriever().index(corpus)
    s = bm.scores("statin")
    assert s[0] > 0 and s[1] == 0 and s[2] == 0
    # a term in every document still gets a small positive weight (Lucene IDF)
    assert (bm.scores("cancer") > 0).all()


def test_top_k_orders_and_breaks_ties_by_position():
    ids = ["a", "b", "c", "d"]
    out = top_k(ids, np.array([0.5, 0.9, 0.5, 0.1]), 3)
    assert [d for d, _ in out] == ["b", "a", "c"]
    assert top_k(ids, np.zeros(4), 0) == []


def test_rrf_matches_hand_computation():
    a = [("x", 9.0), ("y", 5.0)]
    b = [("y", 0.9), ("z", 0.8)]
    fused = dict(reciprocal_rank_fusion([a, b], k=60))
    assert fused["y"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["x"] == pytest.approx(1 / 61)
    assert list(dict(reciprocal_rank_fusion([a, b], k=60)))[0] == "y"
    assert len(reciprocal_rank_fusion([a, b], top_n=2)) == 2


def test_metrics_known_values():
    rels = {"a": 2, "b": 1, "c": 0}
    assert ndcg_at_k(["a", "b"], rels) == pytest.approx(1.0)
    swapped = (1 + 2 / np.log2(3)) / (2 + 1 / np.log2(3))
    assert ndcg_at_k(["b", "a"], rels) == pytest.approx(swapped)
    assert recall_at_k(["a", "x"], rels, 100) == 0.5
    assert mrr_at_k(["c", "x", "b"], rels) == pytest.approx(1 / 3)
    assert mrr_at_k(["c"], rels) == 0.0
    assert ndcg_at_k(["a"], {"a": 0}) == 0.0


def test_summarize_counts_missing_queries_as_zero():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    out = summarize({"q1": [("a", 1.0)]}, qrels)
    assert out["mrr@10"] == pytest.approx(0.5)


def test_paired_bootstrap_detects_clear_improvement():
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 0.5, 200)
    res = paired_bootstrap(a, a + 0.1, n_boot=500)
    assert res["diff"] == pytest.approx(0.1)
    assert res["ci_low"] == pytest.approx(0.1) and res["p_one_sided"] == 0.0
    noisy = paired_bootstrap(a, a + rng.normal(0, 0.2, 200), n_boot=500)
    assert noisy["ci_low"] < noisy["diff"] < noisy["ci_high"]
    with pytest.raises(ValueError):
        paired_bootstrap(np.zeros(3), np.zeros(4))


def test_dense_retriever_with_fake_encoder():
    dense = DenseRetriever(HashingEncoder(dim=512)).index(TOY_CORPUS)
    hits = dense.search("asthma eczema diet", k=2)
    assert hits[0][0] == "d6"
    assert dense.doc_matrix.shape == (len(TOY_CORPUS), 512)


def test_reranker_reorders_head_and_keeps_tail_below():
    cands = [("d5", 3.0), ("d2", 2.0), ("d1", 1.0), ("d3", 0.5)]
    rr = Reranker(OverlapScorer(), TOY_CORPUS, depth=3)
    out = rr.rerank("statins cholesterol", cands)
    assert out[0][0] == "d1"
    assert out[-1][0] == "d3"
    assert out[-1][1] < min(s for _, s in out[:3])


def test_load_beir_filters_unjudged_and_missing(tmp_path):
    root = tmp_path / "ds"
    (root / "qrels").mkdir(parents=True)
    docs = [{"_id": "d1", "title": "T", "text": "body"}, {"_id": "d2", "title": "", "text": "x"}]
    (root / "corpus.jsonl").write_text("\n".join(json.dumps(d) for d in docs))
    qs = [{"_id": "q1", "text": "one"}, {"_id": "q2", "text": "two"}, {"_id": "q3", "text": "3"}]
    (root / "queries.jsonl").write_text("\n".join(json.dumps(q) for q in qs))
    (root / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t2\nq2\tmissing\t1\nq3\td2\t0\n"
    )
    ds = load_beir(root, "test")
    assert ds.corpus == {"d1": "T. body", "d2": "x"}
    assert ds.queries == {"q1": "one"}
    assert ds.qrels == {"q1": {"d1": 2}}
