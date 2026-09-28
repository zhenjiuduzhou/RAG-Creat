"""HTTP retrieval endpoints for generic workflows and Dify."""
from __future__ import annotations

import hmac
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .embedding import Embedder, model_fingerprint
from .search import SearchIndex


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievalSetting(BaseModel):
    top_k: int = Field(ge=1, le=20)
    score_threshold: float = Field(ge=0, le=1)


class DifyRequest(BaseModel):
    knowledge_id: str
    query: str = Field(min_length=1, max_length=2000)
    retrieval_setting: RetrievalSetting


def create_app(bundle: Path, token: str | None = None) -> FastAPI:
    bundle = Path(bundle)
    embedder = Embedder.load(bundle / "model", allow_download=False)
    index = SearchIndex(bundle / "data", embedder)
    if model_fingerprint(bundle / "model") != index.manifest["model_fingerprint"]:
        raise ValueError("模型文件与知识库索引不匹配")
    app = FastAPI(title="RAG Studio Retrieval", version="0.1.0", docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def request_limit(request, call_next):
        if request.method == "POST":
            from fastapi.responses import JSONResponse

            if int(request.headers.get("content-length", "0") or 0) > 65536:
                return JSONResponse({"detail": "Request too large"}, status_code=413)
            if len(await request.body()) > 65536:
                return JSONResponse({"detail": "Request too large"}, status_code=413)
        return await call_next(request)

    def authorize(header: str | None) -> None:
        if token and (not header or not header.startswith("Bearer ") or
                      not hmac.compare_digest(header[7:], token)):
            raise HTTPException(status_code=401, detail="Unauthorized")

    @app.get("/health")
    def health():
        return {"status": "ok", "knowledge_id": index.manifest["knowledge_id"],
                "chunks": index.manifest["chunk_count"]}

    @app.post("/search")
    def search(body: SearchRequest, authorization: str | None = Header(default=None)):
        authorize(authorization)
        return {"results": index.search(body.query, body.top_k)}

    @app.post("/retrieval")
    def retrieval(body: DifyRequest, authorization: str | None = Header(default=None)):
        authorize(authorization)
        if body.knowledge_id != index.manifest["knowledge_id"]:
            raise HTTPException(status_code=404, detail="Knowledge base not found")
        found = index.search(body.query, body.retrieval_setting.top_k)
        return {"records": [{"content": row["text"], "score": row["score"],
                             "title": row["source"],
                             "metadata": {"location": row["location"], "file_id": row["file_id"]}}
                            for row in found if row["score"] >= body.retrieval_setting.score_threshold]}

    return app
