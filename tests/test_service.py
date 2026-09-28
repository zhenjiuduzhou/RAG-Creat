from pathlib import Path
import shutil
from types import SimpleNamespace

from fastapi.testclient import TestClient

from ragstudio.pipeline import active_index, build_project
from ragstudio.project import ProjectStore
from ragstudio.service import create_app
from test_pipeline import FakeEmbedder


def test_search_and_dify_protocol(tmp_path: Path, monkeypatch):
    store = ProjectStore(tmp_path / "home")
    project = store.create_project("示例")
    source = tmp_path / "sample.txt"
    source.write_text("产品 ABC-123 保修两年", encoding="utf-8")
    store.add_files(project["id"], [str(source)])
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "dummy").write_text("model", encoding="utf-8")
    embedder = FakeEmbedder()
    manifest = build_project(store, project["id"], embedder, model_dir=model_dir)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "model").mkdir()
    shutil.copytree(active_index(store.root(project["id"])), bundle / "data")
    monkeypatch.setattr("ragstudio.service.Embedder", SimpleNamespace(load=lambda *args, **kwargs: embedder))
    monkeypatch.setattr("ragstudio.service.model_fingerprint", lambda _folder: manifest["model_fingerprint"])
    client = TestClient(create_app(bundle, token="secret"))
    assert client.get("/health").json()["status"] == "ok"
    assert client.post("/search", json={"query": "ABC"}).status_code == 401
    headers = {"Authorization": "Bearer secret"}
    response = client.post("/search", json={"query": "ABC", "top_k": 1}, headers=headers)
    assert response.status_code == 200
    assert "ABC-123" in response.json()["results"][0]["text"]
    body = {"knowledge_id": project["id"], "query": "ABC",
            "retrieval_setting": {"top_k": 1, "score_threshold": 0}}
    records = client.post("/retrieval", json=body, headers=headers).json()["records"]
    assert records[0]["title"] == "sample.txt"
    assert isinstance(records[0]["metadata"], dict)
    body["knowledge_id"] = "wrong"
    assert client.post("/retrieval", json=body, headers=headers).status_code == 404
    body["knowledge_id"] = project["id"]
    body["metadata_condition"] = {"conditions": [{"name": "file_id", "comparison_operator": "is", "value": "other"}]}
    assert client.post("/retrieval", json=body, headers=headers).status_code == 422
    assert client.post("/search", content="x" * 70000, headers=headers).status_code == 413
    assert client.post("/search", content=(b"x" * 10000 for _ in range(8)), headers=headers).status_code == 413
    assert client.post("/search", content="x" * 70000).status_code == 401
