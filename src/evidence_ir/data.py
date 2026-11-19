"""Load a BEIR-format dataset (corpus.jsonl, queries.jsonl, qrels/<split>.tsv) from plain files."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

BEIR_NFCORPUS_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"


@dataclass
class BeirDataset:
    """Corpus, queries and relevance judgements for one split."""

    corpus: dict[str, str]  # doc_id -> "title. text"
    queries: dict[str, str]  # query_id -> text (only queries judged in this split)
    qrels: dict[str, dict[str, int]]  # query_id -> {doc_id: graded relevance}


def _doc_text(record: dict) -> str:
    title = (record.get("title") or "").strip()
    text = (record.get("text") or "").strip()
    return f"{title}. {text}" if title else text


def load_corpus(path: str | Path) -> dict[str, str]:
    corpus: dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                corpus[str(rec["_id"])] = _doc_text(rec)
    return corpus


def load_queries(path: str | Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                queries[str(rec["_id"])] = str(rec["text"]).strip()
    return queries


def load_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    """Read a BEIR qrels TSV (``query-id  corpus-id  score``), header optional."""
    qrels: dict[str, dict[str, int]] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.reader(fh, delimiter="\t"):
            if len(row) < 3 or row[0] == "query-id":
                continue
            qrels.setdefault(row[0], {})[row[1]] = int(row[2])
    return qrels


def load_beir(root: str | Path, split: str = "test") -> BeirDataset:
    """Load a BEIR dataset directory, keeping only queries judged relevant in ``split``."""
    root = Path(root)
    corpus = load_corpus(root / "corpus.jsonl")
    all_queries = load_queries(root / "queries.jsonl")
    raw = load_qrels(root / "qrels" / f"{split}.tsv")
    qrels = {
        q: {d: s for d, s in rels.items() if d in corpus}
        for q, rels in raw.items()
        if q in all_queries
    }
    qrels = {q: r for q, r in qrels.items() if any(s > 0 for s in r.values())}
    queries = {q: all_queries[q] for q in qrels}
    return BeirDataset(corpus=corpus, queries=queries, qrels=qrels)
