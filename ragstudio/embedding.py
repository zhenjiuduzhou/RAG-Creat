"""Pinned local E5 embedding model and tokenizer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

MODEL_ID = "intfloat/multilingual-e5-small"
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


def model_fingerprint(folder: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(folder.rglob("*")):
        if file.is_file() and file.name != "ragstudio-model.json":
            digest.update(file.relative_to(folder).as_posix().encode())
            with file.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


class Embedder:
    def __init__(self, model):
        self.model = model
        self.tokenizer = model.tokenizer
        self.query_prefix = QUERY_PREFIX
        self.passage_prefix = PASSAGE_PREFIX
        self.dimension = int(model.get_embedding_dimension())
        if not 0 < self.dimension <= 65536:
            raise ValueError(f"模型向量维度异常：{self.dimension}")

    @classmethod
    def load(cls, folder: Path, allow_download: bool = True) -> "Embedder":
        from sentence_transformers import SentenceTransformer

        folder = Path(folder)
        if (folder / "model.safetensors").is_file() and (folder / "ragstudio-model.json").is_file():
            return cls(SentenceTransformer(str(folder), device="cpu", local_files_only=True))
        if not allow_download:
            raise FileNotFoundError("离线模型不存在，请在联网时先完成首次模型下载")
        folder.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import HfApi

        revision = HfApi().model_info(MODEL_ID).sha
        model = SentenceTransformer(MODEL_ID, revision=revision, device="cpu",
                                    cache_folder=str(folder.parent / "download-cache"))
        model.save(str(folder))
        (folder / "ragstudio-model.json").write_text(
            json.dumps({"id": MODEL_ID, "revision": revision}, indent=2), encoding="utf-8"
        )
        return cls(model)

    def encode_passages(self, passages: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode([self.passage_prefix + text for text in passages],
                              batch_size=16, normalize_embeddings=True, show_progress_bar=False,
                              convert_to_numpy=True), dtype=np.float32
        )

    def encode_query(self, query: str) -> np.ndarray:
        vector = self.model.encode(self.query_prefix + query, normalize_embeddings=True,
                                   show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(vector, dtype=np.float32)

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
