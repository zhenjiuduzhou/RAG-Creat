"""Validated local and OpenAI-compatible embedding providers."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .credentials import CredentialsStore
from .embedding import Embedder, MODEL_ID, model_fingerprint

_MODULES = {"sentence_transformers.models.Transformer", "sentence_transformers.models.Pooling",
            "sentence_transformers.models.Normalize",
            "sentence_transformers.base.modules.transformer.Transformer",
            "sentence_transformers.sentence_transformer.modules.pooling.Pooling",
            "sentence_transformers.base.modules.normalize.Normalize"}


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _public(profile: dict) -> dict:
    permitted = ("provider", "model_id", "endpoint", "prefixes", "max_length", "dimension", "revision")
    return {name: profile[name] for name in permitted if name in profile and profile[name] is not None}


def _prefixes(profile: dict) -> tuple[str, str]:
    value = profile.get("prefixes") or {}
    if not isinstance(value, dict):
        raise ValueError("prefixes must be an object")
    query = profile.get("query_prefix", value.get("query", "" if profile.get("provider") == "api" else "query: "))
    passage = profile.get("passage_prefix", value.get("passage", "" if profile.get("provider") == "api" else "passage: "))
    if not isinstance(query, str) or not isinstance(passage, str):
        raise ValueError("embedding prefixes must be strings")
    return query, passage


def _validate_local(folder: Path) -> None:
    folder = Path(folder)
    if not folder.is_dir() or folder.is_symlink():
        raise ValueError("local model directory does not exist or is a symlink")
    for path in folder.rglob("*"):
        if path.is_symlink():
            raise ValueError("local model contains a symlink")
    modules_file = folder / "modules.json"
    if not modules_file.is_file():
        raise ValueError("local model is missing modules.json")
    modules = json.loads(modules_file.read_text(encoding="utf-8"))
    if not isinstance(modules, list) or not modules:
        raise ValueError("invalid SentenceTransformer modules.json")
    seen = set()
    transformer_count = 0
    for module in modules:
        if not isinstance(module, dict) or module.get("type") not in _MODULES:
            raise ValueError("unknown or executable SentenceTransformer module")
        relative = module.get("path")
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("unsafe SentenceTransformer module path")
        target = folder / relative
        if not target.is_dir() or relative in seen:
            raise ValueError("incomplete SentenceTransformer module")
        seen.add(relative)
        kind = module["type"].rsplit(".", 1)[-1]
        if kind == "Transformer":
            transformer_count += 1
            if not (target / "config.json").is_file() or not any((target / name).is_file() for name in
                     ("model.safetensors", "model.safetensors.index.json")):
                raise ValueError("incomplete Transformer weights")
            if not any((target / name).is_file() for name in
                       ("tokenizer.json", "tokenizer.model", "vocab.txt", "vocab.json")):
                raise ValueError("incomplete Transformer tokenizer")
            index = target / "model.safetensors.index.json"
            if index.is_file():
                weights = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
                if not weights or any(not isinstance(name, str) or "/" in name or "\\" in name or
                                      not name.endswith(".safetensors") or not (target / name).is_file()
                                      for name in weights.values()):
                    raise ValueError("incomplete Transformer weight shards")
        elif kind == "Pooling" and not (target / "config.json").is_file():
            raise ValueError("incomplete Pooling module")
    if transformer_count != 1:
        raise ValueError("local model requires one Transformer module")


def install_local(source: Path, home: Path) -> Path:
    """Copy a validated snapshot into content-addressed managed storage."""
    source = Path(source)
    _validate_local(source)
    digest = model_fingerprint(source)
    parent = Path(home) / "models" / "local"
    destination = parent / digest
    if destination.is_dir():
        _validate_local(destination)
        if model_fingerprint(destination) != digest:
            raise ValueError("managed model snapshot was modified")
        return destination
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".install-", dir=parent))
    try:
        shutil.rmtree(temporary)
        shutil.copytree(source, temporary, symlinks=False)
        _validate_local(temporary)
        if model_fingerprint(temporary) != digest:
            raise ValueError("local model changed during installation")
        for file in temporary.rglob("*"):
            if file.is_file():
                file.chmod(0o644)
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


class _LocalEmbedder:
    def __init__(self, embedder: Embedder, profile: dict, folder: Path):
        self.model = embedder.model
        self.tokenizer = embedder.tokenizer
        self.dimension = embedder.dimension
        self.query_prefix, self.passage_prefix = _prefixes(profile)
        if profile.get("max_length") is not None:
            maximum = int(profile["max_length"])
            if maximum <= 0:
                raise ValueError("max_length must be positive")
            self.model.max_seq_length = min(maximum, int(self.model.max_seq_length))
        self.profile = _public(profile)
        meta = folder / "ragstudio-model.json"
        if meta.is_file():
            self.profile["revision"] = json.loads(meta.read_text(encoding="utf-8")).get("revision", "local")
        self.profile.update({"provider": profile.get("provider", "builtin"),
                             "model_id": profile.get("model_id") or (MODEL_ID if profile.get("provider", "builtin") == "builtin" else folder.name),
                             "prefixes": {"query": self.query_prefix, "passage": self.passage_prefix},
                             "max_length": int(self.model.max_seq_length), "dimension": self.dimension})
        self.fingerprint = _hash({"weights": model_fingerprint(folder), "profile": self.profile})

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True).strip()

    def _check_length(self, text: str) -> None:
        if len(self.tokenize(text)) + self.tokenizer.num_special_tokens_to_add(pair=False) > self.model.max_seq_length:
            raise ValueError("embedding input exceeds max_length")

    def encode_passages(self, passages: list[str]) -> np.ndarray:
        if not passages:
            return np.empty((0, self.dimension), dtype=np.float32)
        for text in passages:
            self._check_length(self.passage_prefix + text)
        return np.asarray(self.model.encode([self.passage_prefix + text for text in passages], batch_size=16,
                                            normalize_embeddings=True, show_progress_bar=False, prompt="",
                                            convert_to_numpy=True), dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        self._check_length(self.query_prefix + query)
        return np.asarray(self.model.encode(self.query_prefix + query, normalize_embeddings=True,
                                            show_progress_bar=False, convert_to_numpy=True), dtype=np.float32)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        raise ValueError("embedding endpoint redirects are disabled")


def _endpoint(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if parsed.username or parsed.password or not parsed.hostname or parsed.fragment:
        raise ValueError("invalid embedding endpoint")
    if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("HTTP embedding endpoint must be loopback")
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("embedding endpoint must use HTTPS")
    if any(any(word in name.casefold() for word in ("key", "token", "secret", "password", "credential"))
           for name, _ in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)):
        raise ValueError("embedding credentials cannot appear in the URL")
    return value


class RemoteEmbedder:
    def __init__(self, profile: dict, key: str):
        self.endpoint = _endpoint(str(profile.get("endpoint", "")))
        self.model_id = str(profile.get("model_id", ""))
        if not self.model_id:
            raise ValueError("请填写 API 模型 ID")
        self.key = key
        self.query_prefix, self.passage_prefix = _prefixes(profile)
        self.max_length = int(profile.get("max_length") or 512)
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        self.model = SimpleNamespace(max_seq_length=self.max_length)
        self.tokenizer = self
        self.cancel = None
        self._single_input_only = False
        self._opener = urllib.request.build_opener(_NoRedirect())
        declared = profile.get("dimension")
        if declared is None:
            self.dimension = len(self._request([self.query_prefix + "dimension probe"])[0])
        else:
            self.dimension = int(declared)
            if not 1 <= self.dimension <= 65536:
                raise ValueError("embedding dimension out of range")
        self.profile = _public(profile)
        self.profile.update({"provider": "api", "endpoint": self.endpoint, "model_id": self.model_id,
                             "prefixes": {"query": self.query_prefix, "passage": self.passage_prefix},
                             "max_length": self.max_length, "dimension": self.dimension})
        self.fingerprint = _hash(self.profile)

    def tokenize(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.tokenize(text)

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 0

    def _request(self, inputs: list[str]) -> np.ndarray:
        self._check_cancel()
        if any(len(item) > self.max_length for item in inputs):
            raise ValueError("embedding input exceeds max_length")
        if self._single_input_only and len(inputs) > 1:
            return np.concatenate([self._request([text]) for text in inputs])
        payload = json.dumps({"model": self.model_id, "input": inputs}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        request = urllib.request.Request(self.endpoint, payload, headers, method="POST")
        for attempt in range(3):
            self._check_cancel()
            try:
                with self._opener.open(request, timeout=60) as response:
                    raw = response.read(16 * 1024 * 1024 + 1)
                    if len(raw) > 16 * 1024 * 1024:
                        raise ValueError("模型 API 响应超过 16 MB")
                    try:
                        data = json.loads(raw)
                    except (ValueError, UnicodeError):
                        raise RuntimeError("模型 API 未返回合法 JSON") from None
                break
            except urllib.error.HTTPError as exc:
                code = exc.code
                exc.close()
                if code not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise RuntimeError(f"模型 API 请求失败（HTTP {code}），请检查地址、模型名、密钥或服务额度") from None
            except (urllib.error.URLError, TimeoutError):
                if attempt == 2:
                    raise RuntimeError("无法连接模型 API，请检查网络和服务地址") from None
            waiter = self.cancel or threading.Event()
            waiter.wait(1 + attempt)
            self._check_cancel()
        self._check_cancel()
        rows = data.get("data") if isinstance(data, dict) else None
        if not isinstance(rows, list) or len(rows) != len(inputs):
            raise RuntimeError("embedding API returned incomplete data")
        indices = [row.get("index") if isinstance(row, dict) else None for row in rows]
        if (any(type(index) is not int for index in indices)
                or sorted(indices) != list(range(len(inputs)))):
            if len(inputs) > 1:
                # Never infer ordering from an ambiguous batch response. Obtain
                # each vector separately and retain this mode for this provider.
                self._single_input_only = True
                return self._request(inputs)
            raise RuntimeError("模型 API 返回无效 index，无法确认文本与向量的对应关系")
        ordered = [None] * len(inputs)
        for row in rows:
            index = row["index"]
            values = row.get("embedding")
            if not isinstance(values, list) or not 1 <= len(values) <= 65536:
                raise RuntimeError("embedding API returned invalid dimension")
            vector = np.asarray(values, dtype=np.float32)
            if vector.ndim != 1 or not np.all(np.isfinite(vector)):
                raise RuntimeError("embedding API returned nonfinite vector")
            norm = float(np.linalg.norm(vector.astype(np.float64)))
            if not math.isfinite(norm) or norm == 0:
                raise RuntimeError("embedding API returned zero vector")
            ordered[index] = vector / norm
        lengths = {len(row) for row in ordered}
        if len(lengths) != 1:
            raise RuntimeError("embedding API returned inconsistent dimensions")
        if hasattr(self, "dimension") and lengths != {self.dimension}:
            raise RuntimeError("embedding API dimension changed")
        return np.stack(ordered)

    def encode_passages(self, passages: list[str]) -> np.ndarray:
        if not passages:
            return np.empty((0, self.dimension), dtype=np.float32)
        chunks = []
        for start in range(0, len(passages), 16):
            self._check_cancel()
            chunks.append(self._request([self.passage_prefix + text for text in passages[start:start + 16]]))
        return np.concatenate(chunks)

    def encode_query(self, query: str) -> np.ndarray:
        return self._request([self.query_prefix + query])[0]

    def _check_cancel(self) -> None:
        if self.cancel is not None and self.cancel.is_set():
            from .errors import BuildCancelled
            raise BuildCancelled("构建已取消")


def load_provider(profile: dict, home: Path, allow_download: bool = True,
                  bundle_model: Path | None = None, key: str | None = None):
    provider = profile.get("provider", "builtin")
    if provider == "api":
        endpoint = _endpoint(str(profile.get("endpoint", "")))
        model_id = str(profile.get("model_id", ""))
        if key is None:
            key = (os.getenv("RAG_STUDIO_EMBEDDING_KEY") if bundle_model is not None
                   else CredentialsStore(home).lookup(endpoint, model_id))
        return RemoteEmbedder(profile, key or "")
    if provider == "builtin":
        folder = Path(bundle_model) if bundle_model is not None else Path(home) / "models" / "multilingual-e5-small"
        embedder = Embedder.load(folder, allow_download=allow_download)
    elif provider == "local":
        if bundle_model is not None:
            folder = Path(bundle_model)
        else:
            raw = profile.get("local_path")
            if not raw:
                raise ValueError("local_path is required")
            folder = Path(raw)
        _validate_local(folder)
        from sentence_transformers import SentenceTransformer
        embedder = Embedder(SentenceTransformer(str(folder), device="cpu", local_files_only=True,
                                               trust_remote_code=False))
    else:
        raise ValueError("unknown embedding provider")
    if profile.get("dimension") is not None and int(profile["dimension"]) != embedder.dimension:
        raise ValueError("local embedding dimension does not match profile")
    return _LocalEmbedder(embedder, profile, folder)
