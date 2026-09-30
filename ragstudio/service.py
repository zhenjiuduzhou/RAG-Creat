"""HTTP retrieval endpoints for generic workflows and Dify."""
from __future__ import annotations

import hmac
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .embedding import Embedder, model_fingerprint
from .search import SearchIndex
from .project import load_json
from .providers import load_provider


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
    metadata_condition: dict | None = None


def create_app(bundle: Path, token: str | None = None) -> FastAPI:
    bundle = Path(bundle)
    manifest = load_json(bundle / "data" / "manifest.json")
    embedder = (load_provider(manifest["embedding_profile"], bundle, allow_download=False, bundle_model=bundle / "model")
                if manifest.get("embedding_profile") else Embedder.load(bundle / "model", allow_download=False))
    index = SearchIndex(bundle / "data", embedder)
    if not manifest.get("embedding_profile") and model_fingerprint(bundle / "model") != index.manifest["model_fingerprint"]:
        raise ValueError("模型文件与知识库索引不匹配")
    app = FastAPI(title="RAG Studio Retrieval", version="0.1.0", docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def request_limit(request, call_next):
        if request.method == "POST":
            from fastapi.responses import JSONResponse

            header = request.headers.get("authorization")
            if token and (not header or not header.startswith("Bearer ") or
                          not hmac.compare_digest(header[7:], token)):
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)
            if int(request.headers.get("content-length", "0") or 0) > 65536:
                return JSONResponse({"detail": "Request too large"}, status_code=413)
            body = bytearray()
            async for chunk in request.stream():
                if len(body) + len(chunk) > 65536:
                    return JSONResponse({"detail": "Request too large"}, status_code=413)
                body.extend(chunk)
            request._body = bytes(body)
        return await call_next(request)

    def authorize(header: str | None) -> None:
        if token and (not header or not header.startswith("Bearer ") or
                      not hmac.compare_digest(header[7:], token)):
            raise HTTPException(status_code=401, detail="Unauthorized")

    @app.get("/health")
    def health():
        return {"status": "ok", "knowledge_id": index.manifest["knowledge_id"],
                "chunks": index.manifest["chunk_count"]}

    def retrieve(query, top_k):
        try:
            return index.search(query, top_k)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from None

    @app.post("/search")
    def search(body: SearchRequest, authorization: str | None = Header(default=None)):
        authorize(authorization)
        return {"results": retrieve(body.query, body.top_k)}

    @app.post("/retrieval")
    def retrieval(body: DifyRequest, authorization: str | None = Header(default=None)):
        authorize(authorization)
        if body.metadata_condition and body.metadata_condition.get("conditions") != []:
            raise HTTPException(status_code=422, detail="暂不支持 Dify 元数据过滤条件")
        if body.knowledge_id != index.manifest["knowledge_id"]:
            raise HTTPException(status_code=404, detail="Knowledge base not found")
        found = retrieve(body.query, body.retrieval_setting.top_k)
        return {"records": [{"content": row["text"], "score": row["score"],
                             "title": row["source"],
                             "metadata": {"location": row["location"], "file_id": row["file_id"]}}
                            for row in found if row["score"] >= body.retrieval_setting.score_threshold]}

    return app
