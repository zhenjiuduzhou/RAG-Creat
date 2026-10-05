import json
import zipfile

import pytest

from ragstudio.export import export_project
from ragstudio.pipeline import build_project
from ragstudio.project import ProjectStore, atomic_json
from ragstudio.service import create_app
from test_pipeline import FakeEmbedder


def make_project(tmp_path, provider):
    store = ProjectStore(tmp_path / "home")
    project = store.create_project("runtime test")
    source = tmp_path / "source.txt"
    source.write_text("知识库测试内容", encoding="utf-8")
    store.add_files(project["id"], [str(source)])
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights.txt").write_text("current model only")
    embedder = FakeEmbedder()
    embedder.profile = {"provider": provider, "dimension": embedder.dimension, "model_id": "test"}
    embedder.fingerprint = "test-fingerprint"
    current = store.get(project["id"])
    current["embedding"] = dict(embedder.profile, local_path=str(model))
    atomic_json(store.root(project["id"]) / "project.json", current)
    build_project(store, project["id"], embedder, model_dir=model)
    return store, project["id"], model, embedder


def make_runtime(path, kind):
    path.mkdir()
    (path / "ragstudio-server.exe").write_bytes(kind.encode())
    (path / "runtime-profile.json").write_text(json.dumps({"kind": kind}))
    return path


@pytest.mark.parametrize("provider", ["api", "builtin", "local"])
def test_export_selects_only_matching_runtime_and_model(tmp_path, monkeypatch, provider):
    store, project_id, model, embedder = make_project(tmp_path, provider)
    local = make_runtime(tmp_path / "ragstudio-server", "local")
    make_runtime(tmp_path / "ragstudio-server-api", "api")
    monkeypatch.setattr("ragstudio.export.load_provider", lambda *args, **kwargs: embedder)
    archive = export_project(store, project_id, tmp_path / "export", "windows-full", model,
                             local, embedder=embedder)
    expected = "api" if provider == "api" else "local"
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.read("runtime/ragstudio-server.exe") == expected.encode()
        assert ("model/weights.txt" in bundle.namelist()) == (provider != "api")
    developer = export_project(store, project_id, tmp_path / "export", "developer", model, embedder=embedder)
    with zipfile.ZipFile(developer) as bundle:
        requirements = bundle.read("requirements.txt").decode()
        assert ("sentence-transformers" in requirements) == (provider != "api")
        assert not any(name.startswith("runtime/") for name in bundle.namelist())


def test_api_export_never_falls_back_to_heavy_runtime(tmp_path):
    store, project_id, model, embedder = make_project(tmp_path, "api")
    local = make_runtime(tmp_path / "ragstudio-server", "local")
    with pytest.raises(FileNotFoundError, match="api"):
        export_project(store, project_id, tmp_path / "export", "windows-full", model,
                       local, embedder=embedder)
    wrong = make_runtime(tmp_path / "ragstudio-server-api", "local")
    with pytest.raises(ValueError, match="类型不匹配"):
        export_project(store, project_id, tmp_path / "export", "windows-full", model,
                       local, embedder=embedder, api_server_runtime=wrong)


def test_service_rejects_runtime_mismatch_before_loading_model(tmp_path):
    (tmp_path / "data").mkdir()
    atomic_json(tmp_path / "data" / "manifest.json", {"embedding_profile": {"provider": "local"}})
    make_runtime(tmp_path / "runtime", "api")
    with pytest.raises(ValueError, match="模型类型不匹配"):
        create_app(tmp_path)
