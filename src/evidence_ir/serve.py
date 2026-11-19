"""FastAPI service: POST /search returns ranked evidence with per-stage latency."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .pipeline import SearchPipeline


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    system: str = "hybrid+rerank"
    k: int = Field(10, ge=1, le=100)


class Hit(BaseModel):
    doc_id: str
    score: float
    snippet: str


class SearchResponse(BaseModel):
    system: str
    hits: list[Hit]
    timings: dict[str, float]


def create_app(pipeline: SearchPipeline, snippet_chars: int = 240) -> FastAPI:
    app = FastAPI(title="clinical-evidence-retrieval", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "systems": pipeline.available, "n_docs": len(pipeline.corpus)}

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest) -> SearchResponse:
        if req.system not in pipeline.available:
            raise HTTPException(
                status_code=400,
                detail=f"system {req.system!r} not available; choose from {pipeline.available}",
            )
        res = pipeline.search(req.query, req.system, req.k)
        hits = [
            Hit(doc_id=d, score=s, snippet=pipeline.corpus[d][:snippet_chars])
            for d, s in res["hits"]
        ]
        return SearchResponse(system=req.system, hits=hits, timings=res["timings"])

    return app
