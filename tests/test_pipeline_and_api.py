import json

import pytest
from fastapi.testclient import TestClient

from evidence_ir.data import BeirDataset
from evidence_ir.evaluate import evaluate, write_outputs
from evidence_ir.pipeline import SearchPipeline
from evidence_ir.serve import create_app
from evidence_ir.testing import (
    TOY_CORPUS,
    TOY_QRELS,
    TOY_QUERIES,
    HashingEncoder,
    OverlapScorer,
)


@pytest.fixture()
def pipe():
    return SearchPipeline(
        TOY_CORPUS, encoder=HashingEncoder(), scorer=OverlapScorer(), depth=5, rerank_depth=5
    )


def test_bm25_only_pipeline_has_one_system():
    p = SearchPipeline(TOY_CORPUS)
    assert p.available == ["bm25"]
    with pytest.raises(ValueError):
        p.search("asthma", "dense")


def test_all_systems_and_timings(pipe):
    assert pipe.available == ["bm25", "dense", "hybrid", "hybrid+rerank"]
    res = pipe.search("asthma eczema diet", "hybrid+rerank", k=3)
    assert res["hits"][0][0] == "d6"
    assert set(res["timings"]) == {"bm25_ms", "dense_ms", "fusion_ms", "rerank_ms", "total_ms"}
    assert set(pipe.search("asthma", "dense")["timings"]) == {"dense_ms", "total_ms"}


def test_evaluate_and_write_outputs(pipe, tmp_path):
    ds = BeirDataset(corpus=TOY_CORPUS, queries=TOY_QUERIES, qrels=TOY_QRELS)
    metrics, scores = evaluate(ds, pipe, n_boot=200)
    assert metrics["n_queries"] == 2
    assert set(metrics["systems"]) == {"bm25", "dense", "hybrid", "hybrid+rerank"}
    for v in metrics["systems"].values():
        assert 0.0 <= v["ndcg@10"] <= 1.0
    assert metrics["systems"]["bm25"]["recall@100"] == 1.0
    assert {c["system"] for c in metrics["comparisons"]} >= {"dense", "hybrid+rerank"}
    write_outputs(metrics, scores, sorted(ds.qrels), tmp_path)
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["config"]["encoder"] == "hashing-encoder"
    assert (tmp_path / "per_query.csv").read_text().count("\n") == 3
    assert "nDCG@10" in (tmp_path / "report.md").read_text()
    assert (tmp_path / "ndcg_by_system.png").exists()


def test_api_search_and_errors(pipe):
    client = TestClient(create_app(pipe))
    assert client.get("/health").json()["systems"][-1] == "hybrid+rerank"
    r = client.post("/search", json={"query": "breast cancer cholesterol", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert len(body["hits"]) == 2 and body["hits"][0]["doc_id"] in {"d1", "d4"}
    assert "rerank_ms" in body["timings"]
    assert client.post("/search", json={"query": "x", "system": "nope"}).status_code == 400
    assert client.post("/search", json={"query": ""}).status_code == 422
