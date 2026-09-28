"""Build a versioned knowledge base without exposing partial results."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np

from .embedding import MODEL_ID, PASSAGE_PREFIX, QUERY_PREFIX, model_fingerprint
from .extract import ExtractionError, Section, extract_file
from .project import ProjectStore, atomic_json, load_json, sha256_file
from .search import verify_index

EXTRACT_VERSION = 2
CHUNK_VERSION = 1
TARGET_TOKENS = 384
OVERLAP_TOKENS = 48


class BuildCancelled(Exception):
    pass


def active_index(root: Path) -> Path:
    current = load_json(root / "current.json")
    folder = root / "versions" / current["version"]
    verify_index(folder)
    return folder


def chunk_sections(sections: list[Section], embedder, file_id: str, filename: str) -> list[dict]:
    chunks: list[dict] = []
    prefix_tokens = len(embedder.tokenize(PASSAGE_PREFIX))
    max_seq = int(getattr(embedder.model, "max_seq_length", 512) or 512)
    special = int(embedder.tokenizer.num_special_tokens_to_add(pair=False))
    target = min(TARGET_TOKENS, max_seq - prefix_tokens - special)
    if target <= OVERLAP_TOKENS:
        raise ValueError("模型最大输入长度不足以进行分块")
    for section in sections:
        tokens = embedder.tokenize(section.text)
        if not tokens:
            continue
        start = 0
        while start < len(tokens):
            end = min(start + target, len(tokens))
            text = embedder.decode(tokens[start:end])
            if text:
                if len(embedder.tokenize(PASSAGE_PREFIX + text)) + special > max_seq:
                    raise ValueError("文本块超过模型最大输入长度")
                chunks.append({"id": uuid.uuid5(uuid.NAMESPACE_URL, f"{file_id}:{section.location}:{start}").hex,
                               "file_id": file_id, "source": filename, "location": section.location,
                               "method": section.method, "text": text})
            if end == len(tokens):
                break
            start = end - OVERLAP_TOKENS
    return chunks


def _check_cancel(cancel: threading.Event | None) -> None:
    if cancel and cancel.is_set():
        raise BuildCancelled("用户取消构建")


def build_project(store: ProjectStore, project_id: str, embedder, *,
                  model_dir: Path, progress: Callable[[int, str], None] | None = None,
                  cancel: threading.Event | None = None,
                  tesseract_cmd: str | None = None) -> dict:
    root = store.root(project_id)
    project = store.get(project_id)
    files = project["files"]
    fingerprint = model_fingerprint(model_dir)
    cache = root / "cache"
    cache.mkdir(exist_ok=True)
    all_chunks: list[dict] = []
    matrices: list[np.ndarray] = []
    used_cache: set[Path] = set()
    errors: list[dict] = []
    progress = progress or (lambda _percent, _message: None)
    _check_cancel(cancel)
    for i, item in enumerate(files):
        _check_cancel(cancel)
        progress(int(i / len(files) * 85), f"处理 {item['name']}")
        extract_cache = cache / f"extract-{item['sha256']}-{EXTRACT_VERSION}.json"
        try:
            if extract_cache.is_file():
                sections = [Section(**row) for row in json.loads(extract_cache.read_text(encoding="utf-8"))]
            else:
                sections = extract_file(root / "sources" / item["path"], tesseract_cmd)
                atomic_json(extract_cache, [asdict(section) for section in sections])
            used_cache.add(extract_cache)
            chunks = chunk_sections(sections, embedder, item["id"], item["name"])
            if not chunks:
                raise ExtractionError("文件没有可用文本块")
            text_fingerprint = hashlib.sha256(json.dumps([chunk["text"] for chunk in chunks],
                                                       ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
            vector_cache = cache / f"vectors-{item['sha256']}-{EXTRACT_VERSION}-{CHUNK_VERSION}-{text_fingerprint}-{fingerprint[:16]}.npy"
            if vector_cache.is_file():
                vectors = np.load(vector_cache, allow_pickle=False)
                if vectors.shape != (len(chunks), embedder.dimension):
                    vector_cache.unlink()
                    vectors = embedder.encode_passages([chunk["text"] for chunk in chunks])
                    np.save(vector_cache, vectors)
            else:
                vectors = embedder.encode_passages([chunk["text"] for chunk in chunks])
                np.save(vector_cache, vectors)
            used_cache.add(vector_cache)
            _check_cancel(cancel)
            matrices.append(vectors)
            all_chunks.extend(chunks)
            item["status"] = f"已处理 · {len(chunks)} 块"
        except BuildCancelled:
            raise
        except Exception as exc:
            item["status"] = "失败"
            errors.append({"file": item["name"], "error": str(exc)})
    if not all_chunks and files:
        current = store.get(project_id)
        if [(row["id"], row["sha256"]) for row in current["files"]] == [(row["id"], row["sha256"]) for row in files]:
            atomic_json(root / "project.json", project)
        detail = "；".join(f"{row['file']}: {row['error']}" for row in errors)
        raise ValueError(f"没有可用文本块。{detail[:1000]}")
    _check_cancel(cancel)
    progress(88, "保存并校验索引")
    versions = root / "versions"
    versions.mkdir(exist_ok=True)
    version = uuid.uuid4().hex
    staging = versions / f".tmp-{version}"
    final = versions / version
    staging.mkdir()
    try:
        with (staging / "chunks.jsonl").open("w", encoding="utf-8") as stream:
            for chunk in all_chunks:
                stream.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        np.save(staging / "vectors.npy", np.concatenate(matrices).astype(np.float32)
                if matrices else np.empty((0, embedder.dimension), dtype=np.float32))
        model_meta_path = model_dir / "ragstudio-model.json"
        model_meta = load_json(model_meta_path) if model_meta_path.exists() else {}
        manifest = {"format_version": 1, "knowledge_id": project_id, "name": project["name"],
                    "model_id": MODEL_ID, "model_revision": model_meta.get("revision", "local"),
                    "model_fingerprint": fingerprint, "dimension": embedder.dimension,
                    "normalization": "l2", "query_prefix": QUERY_PREFIX,
                    "passage_prefix": PASSAGE_PREFIX, "chunk_version": CHUNK_VERSION,
                    "chunk_count": len(all_chunks), "errors": errors,
                    "sources": [{"id": item["id"], "name": item["name"], "sha256": item["sha256"]}
                                for item in files],
                    "files": {name: sha256_file(staging / name) for name in ("chunks.jsonl", "vectors.npy")}}
        atomic_json(staging / "manifest.json", manifest)
        verify_index(staging)
        _check_cancel(cancel)
        current = store.get(project_id)
        if [(row["id"], row["sha256"]) for row in current["files"]] != [(row["id"], row["sha256"]) for row in files]:
            raise BuildCancelled("资料清单已变更，请重新构建")
        os.replace(staging, final)
        atomic_json(root / "current.json", {"version": version})
        current = store.get(project_id)
        statuses = {row["id"]: row["status"] for row in files}
        for row in current["files"]:
            if row["id"] in statuses:
                row["status"] = statuses[row["id"]]
        current["dirty"] = [(row["id"], row["sha256"]) for row in current["files"]] != [(row["id"], row["sha256"]) for row in files]
        atomic_json(root / "project.json", current)
        # Retain the old version throughout construction, then discard stale
        # source-derived data only after the new index becomes active.
        resolved_versions = versions.resolve()
        for previous in versions.iterdir():
            if previous != final and previous.is_dir() and previous.resolve().is_relative_to(resolved_versions):
                try:
                    shutil.rmtree(previous)
                except OSError:
                    pass  # A running reader may still hold a Windows file handle.
        for cached in cache.iterdir():
            if cached.is_file() and cached not in used_cache:
                cached.unlink(missing_ok=True)
        progress(100, "构建完成")
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)
