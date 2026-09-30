from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from ragstudio import credentials
from ragstudio.providers import RemoteEmbedder, install_local
from ragstudio.pipeline import BuildCancelled


class Handler(BaseHTTPRequestHandler):
    requests = []
    vector = [3, 4]

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.requests.append((body, self.headers.get("Authorization")))
        rows = [{"index": i, "embedding": self.vector} for i in range(len(body["input"]))]
        rows.reverse()
        payload = json.dumps({"data": rows}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_):
        pass


@pytest.fixture
def endpoint():
    Handler.requests = []
    Handler.vector = [3, 4]
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1/embeddings"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_remote_probe_batch_prefix_and_normalization(endpoint):
    embedder = RemoteEmbedder({"provider": "api", "endpoint": endpoint, "model_id": "test",
                               "prefixes": {"query": "Q:", "passage": "P:"}}, "secret")
    assert embedder.dimension == 2
    result = embedder.encode_passages([str(i) for i in range(17)])
    assert result.shape == (17, 2)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[0], [.6, .8])
    assert [len(request[0]["input"]) for request in Handler.requests] == [1, 16, 1]
    assert Handler.requests[1][0]["input"][0] == "P:0"
    assert all(auth == "Bearer secret" for _, auth in Handler.requests)
    assert embedder.tokenize("abc") == [97, 98, 99]
    assert embedder.decode([97, 98]) == "ab"


def test_remote_rejects_overlimit_and_bad_vectors(endpoint):
    embedder = RemoteEmbedder({"provider": "api", "endpoint": endpoint, "model_id": "test",
                               "dimension": 2, "max_length": 8}, "secret")
    with pytest.raises(ValueError, match="max_length"):
        embedder.encode_query("123456789")
    assert not Handler.requests
    Handler.vector = [0, 0]
    with pytest.raises(RuntimeError, match="zero"):
        embedder.encode_query("a")
    Handler.vector = [1, 2, 3]
    with pytest.raises(RuntimeError, match="dimension"):
        embedder.encode_query("a")


def test_remote_cancel_stops_before_next_request(endpoint):
    embedder = RemoteEmbedder({"provider": "api", "endpoint": endpoint, "model_id": "test", "dimension": 2}, "key")
    cancel = threading.Event()
    cancel.set()
    embedder.cancel = cancel
    with pytest.raises(BuildCancelled):
        embedder.encode_passages(["x"])
    assert not Handler.requests


def test_remote_endpoint_rejects_credentials_and_external_http():
    for endpoint in ("http://example.com/embeddings", "https://user:pass@example.com/embeddings"):
        with pytest.raises(ValueError):
            RemoteEmbedder({"provider": "api", "endpoint": endpoint, "model_id": "test", "dimension": 2}, "key")


def test_install_local_rejects_custom_module_and_symlink(tmp_path):
    source = tmp_path / "model"
    source.mkdir()
    (source / "modules.json").write_text(json.dumps([{"type": "evil.Custom", "path": ""}]))
    with pytest.raises(ValueError, match="unknown"):
        install_local(source, tmp_path / "home")


def test_credentials_never_persist_plaintext(tmp_path, monkeypatch):
    monkeypatch.setattr(credentials, "_crypt", lambda data, *, decrypt: data[::-1])
    store = credentials.CredentialStore(tmp_path)
    profile = {"endpoint": "https://example.com/embeddings", "model_id": "test"}
    store.save(profile, "secret-key")
    assert "secret-key" not in store.path.read_text()
    assert store.lookup(profile) == "secret-key"
    monkeypatch.setenv("RAG_STUDIO_EMBEDDING_KEY", "override")
    assert store.lookup(profile) == "override"


def test_api_build_service_and_model_identity(endpoint, tmp_path, monkeypatch):
    from ragstudio.providers import load_provider
    from ragstudio.project import ProjectStore, atomic_json
    from ragstudio.pipeline import build_project, active_index
    from ragstudio.search import SearchIndex
    from ragstudio.service import create_app
    from fastapi.testclient import TestClient
    from ragstudio.export import export_project
    import zipfile

    store = ProjectStore(tmp_path / "home")
    project = store.create_project("API 客户")
    source = tmp_path / "contract.txt"
    source.write_text("ABC-123 保修两年。" * 100, encoding="utf-8")
    store.add_files(project["id"], [str(source)])
    profile = {"provider": "api", "endpoint": endpoint, "model_id": "test", "revision": "v1",
               "query_prefix": "Q:", "passage_prefix": "P:", "max_length": 512}
    embedder = load_provider(profile, store.home, key="test-key")
    current = store.get(project["id"])
    current["embedding"] = embedder.profile
    atomic_json(store.root(project["id"]) / "project.json", current)
    manifest = build_project(store, project["id"], embedder, model_dir=tmp_path / "absent-model")
    assert manifest["dimension"] == 2
    assert manifest["chunk_units"] == "characters"
    assert manifest["query_prefix"] == "Q:"
    assert manifest["passage_prefix"] == "P:"
    index = SearchIndex(active_index(store.root(project["id"])), embedder)
    expected = index.search("ABC-123")
    monkeypatch.setenv("RAG_STUDIO_EMBEDDING_KEY", "test-key")
    archive = export_project(store, project["id"], tmp_path / "export", "developer", tmp_path / "absent-model", embedder=embedder)
    bundle = tmp_path / "moved-bundle"
    with zipfile.ZipFile(archive) as data:
        assert not any(name.startswith("model/") for name in data.namelist())
        assert "test-key" not in data.read("data/manifest.json").decode()
        assert "providers.py" in " ".join(data.namelist())
        data.extractall(bundle)
    client = TestClient(create_app(bundle))
    assert client.get("/health").json()["chunks"] == manifest["chunk_count"]
    assert client.post("/search", json={"query": "ABC-123"}).json()["results"] == expected
    assert client.post("/search", json={"query": "x" * 600}).status_code == 422
    assert client.post("/retrieval", json={"knowledge_id": project["id"], "query": "x" * 600,
                                           "retrieval_setting": {"top_k": 2, "score_threshold": 0}}).status_code == 422
    changed = dict(embedder.profile, revision="v2")
    other = load_provider(changed, store.home, key="test-key")
    with pytest.raises(ValueError, match="重新构建"):
        SearchIndex(active_index(store.root(project["id"])), other)


def test_native_dpapi_roundtrip(tmp_path):
    import os
    if os.name != "nt":
        pytest.skip("Windows DPAPI")
    store = credentials.CredentialStore(tmp_path)
    store.save("https://example.com/v1/embeddings", "test", "sensitive-value")
    assert "sensitive-value" not in store.path.read_text()
    assert store.lookup("https://example.com/v1/embeddings", "test") == "sensitive-value"


def test_current_sentence_transformers_snapshot_modules(tmp_path):
    model = tmp_path / "saved-model"
    (model / "1_Pooling").mkdir(parents=True)
    (model / "2_Normalize").mkdir()
    (model / "config.json").write_text("{}")
    (model / "tokenizer.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"weights")
    (model / "1_Pooling" / "config.json").write_text("{}")
    (model / "modules.json").write_text(json.dumps([
        {"path": "", "type": "sentence_transformers.base.modules.transformer.Transformer"},
        {"path": "1_Pooling", "type": "sentence_transformers.sentence_transformer.modules.pooling.Pooling"},
        {"path": "2_Normalize", "type": "sentence_transformers.base.modules.normalize.Normalize"}]))
    installed = install_local(model, tmp_path / "home")
    assert installed.is_dir() and installed != model
    assert install_local(model, tmp_path / "home") == installed


def test_local_api_without_key_and_default_prefixes(endpoint):
    from ragstudio.providers import load_provider
    embedder = load_provider({"provider": "api", "endpoint": endpoint, "model_id": "test"}, ".", key="")
    embedder.encode_query("query")
    assert Handler.requests[-1][0]["input"] == ["query"]
    assert Handler.requests[-1][1] is None
