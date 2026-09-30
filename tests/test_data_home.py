import json
import shutil
from pathlib import Path
from ragstudio import project
from ragstudio.project import ProjectStore, atomic_json, migrate_legacy_home


def test_default_home_is_sibling_of_frozen_app(monkeypatch, tmp_path):
    monkeypatch.delenv("RAG_STUDIO_HOME", raising=False)
    monkeypatch.setattr(project.sys, "frozen", True, raising=False)
    monkeypatch.setattr(project.sys, "executable", str(tmp_path / "RAGStudio" / "RAGStudio.exe"))
    assert project.default_home() == tmp_path / "RAGStudioDate"
    monkeypatch.setenv("RAG_STUDIO_HOME", str(tmp_path / "override"))
    assert project.default_home() == tmp_path / "override"


def test_verified_migration_rebases_model_and_preserves_secrets(tmp_path):
    legacy = tmp_path / "old" / "RAGStudio"
    store = ProjectStore(legacy)
    customer = store.create_project("旧客户")
    model = legacy / "models" / "local" / "fingerprint"
    model.mkdir(parents=True)
    (model / "weights").write_bytes(b"model")
    record = store.get(customer["id"])
    record["embedding"] = {"provider":"local", "local_path":str(model)}
    # Simulate an older release writing absolute paths.
    (store.root(customer["id"]) / "project.json").write_text(json.dumps(record))
    (legacy / "credentials.json").write_bytes(b"encrypted-key")
    destination = tmp_path / "dist" / "RAGStudioDate"
    assert migrate_legacy_home(destination, legacy)
    assert legacy.exists()
    assert (destination / "credentials.json").read_bytes() == b"encrypted-key"
    moved = ProjectStore(destination)
    assert moved.get(customer["id"])["embedding"]["local_path"] == str(destination / "models" / "local" / "fingerprint")
    assert not migrate_legacy_home(destination, legacy)
    elsewhere = tmp_path / "other-drive" / "RAGStudioDate"
    shutil.copytree(destination, elsewhere)
    assert ProjectStore(elsewhere).get(customer["id"])["embedding"]["local_path"] == str(elsewhere / "models" / "local" / "fingerprint")


def test_managed_model_paths_saved_relative(tmp_path):
    store = ProjectStore(tmp_path / "RAGStudioDate")
    customer = store.create_project("客户")
    model = store.home / "models" / "local" / "abc"
    model.mkdir(parents=True)
    customer["embedding"] = {"provider":"local", "local_path":str(model)}
    atomic_json(store.root(customer["id"]) / "project.json", customer)
    assert json.loads((store.root(customer["id"]) / "project.json").read_text())["embedding"]["local_path"] == "models/local/abc"
    assert store.get(customer["id"])["embedding"]["local_path"] == str(model)


def test_migration_failure_keeps_legacy_and_does_not_activate(tmp_path, monkeypatch):
    import pytest
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "weights").write_bytes(b"original")
    destination = tmp_path / "dist" / "RAGStudioDate"
    real_hash = project.sha256_file
    monkeypatch.setattr(project, "sha256_file", lambda file: real_hash(file) if file.parent == legacy else "bad")
    with pytest.raises(ValueError, match="校验失败"):
        migrate_legacy_home(destination, legacy)
    assert (legacy / "weights").read_bytes() == b"original"
    assert not destination.exists()
