"""Dense bi-encoder retrieval with exact (brute-force) inner-product search."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .bm25 import top_k


class Encoder(Protocol):
    def encode_queries(self, texts: list[str]) -> np.ndarray: ...

    def encode_docs(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEncoder:
    """Wraps a sentence-transformers model; vectors are L2-normalised so dot = cosine.

    BGE models expect an instruction prefix on queries only.
    """

    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        query_prefix: str | None = None,
        batch_size: int = 32,
        max_seq_length: int = 512,
        device: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer  # optional dependency

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = max_seq_length
        if query_prefix is None:
            query_prefix = self.BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""
        self.query_prefix = query_prefix
        self.batch_size = batch_size

    def _encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.query_prefix + t for t in texts])

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)


class DenseRetriever:
    name = "dense"

    def __init__(self, encoder: Encoder) -> None:
        self.encoder = encoder
        self.doc_ids: list[str] = []
        self.doc_matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)

    def index(self, corpus: dict[str, str]) -> DenseRetriever:
        self.doc_ids = list(corpus)
        self.doc_matrix = np.asarray(
            self.encoder.encode_docs([corpus[d] for d in self.doc_ids]), dtype=np.float32
        )
        return self

    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        qvec = np.asarray(self.encoder.encode_queries([query])[0], dtype=np.float32)
        return top_k(self.doc_ids, self.doc_matrix @ qvec, k)
