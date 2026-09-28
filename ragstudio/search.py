"""Load and query a portable knowledge-base index."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .project import load_json, sha256_file


def verify_index(folder: Path) -> dict:
    folder = Path(folder)
    manifest = load_json(folder / "manifest.json")
    if manifest.get("format_version") != 1:
        raise ValueError("不支持的知识库格式版本")
    for filename in ("chunks.jsonl", "vectors.npy"):
        if sha256_file(folder / filename) != manifest["files"][filename]:
            raise ValueError(f"索引校验失败：{filename}")
    vectors = np.load(folder / "vectors.npy", mmap_mode="r", allow_pickle=False)
    if vectors.ndim != 2 or vectors.shape != (manifest["chunk_count"], manifest["dimension"]):
        raise ValueError("向量数量或维度不匹配")
    return manifest


class SearchIndex:
    def __init__(self, folder: Path, embedder):
        self.folder = Path(folder)
        self.manifest = verify_index(self.folder)
        if self.manifest["dimension"] != embedder.dimension:
            raise ValueError("检索模型与向量维度不一致")
        self.embedder = embedder
        self.vectors = np.load(self.folder / "vectors.npy", mmap_mode="r", allow_pickle=False)
        self.chunks = [json.loads(line) for line in (self.folder / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
        if len(self.chunks) != len(self.vectors):
            raise ValueError("文本块与向量数量不匹配")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query = query.strip()
        if not query or not self.chunks:
            return []
        top_k = max(1, min(int(top_k), 20))
        q = self.embedder.encode_query(query)
        cosine = np.asarray(self.vectors @ q, dtype=np.float32)
        # Exact identifiers and Chinese phrases receive a modest boost, while
        # every result retains a comparable 0..1 semantic relevance score.
        terms = set(re.findall(r"[A-Za-z0-9][A-Za-z0-9._-]*", query.casefold()))
        scores = np.clip((cosine + 1.0) / 2.0, 0.0, 1.0)
        for i, chunk in enumerate(self.chunks):
            text = chunk["text"].casefold()
            hits = sum(term in text for term in terms) / len(terms) if terms else 0.0
            phrase = query.casefold() in text
            scores[i] = min(1.0, float(scores[i]) * 0.88 + 0.12 * max(hits, float(phrase)))
        best = np.argsort(-scores, kind="stable")[:top_k]
        return [{"text": self.chunks[int(i)]["text"], "score": round(float(scores[int(i)]), 4),
                 "source": self.chunks[int(i)]["source"], "location": self.chunks[int(i)]["location"],
                 "file_id": self.chunks[int(i)]["file_id"]} for i in best]

