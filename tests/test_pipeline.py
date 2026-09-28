from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from ragstudio.pipeline import BuildCancelled, active_index, build_project
from ragstudio.export import export_project
from ragstudio.project import ProjectStore
from ragstudio.search import SearchIndex


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, values, skip_special_tokens=True):
        return "".join(chr(value) for value in values)

    def num_special_tokens_to_add(self, pair=False):
        return 2


class FakeModel:
    max_seq_length = 512


class FakeEmbedder:
    dimension = 384
    tokenizer = Tokenizer()
    model = FakeModel()

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def decode(self, values):
        return self.tokenizer.decode(values)

    def encode_passages(self, passages):
        out = np.zeros((len(passages), 384), dtype=np.float32)
        for i, passage in enumerate(passages):
            out[i, ord(passage[-1]) % 384] = 1.0
        return out

    def encode_query(self, query):
        out = np.zeros(384, dtype=np.float32)
        out[ord(query[-1]) % 384] = 1.0
        return out


def test_build_and_remove_does_not_leave_old_content(tmp_path: Path):
    store = ProjectStore(tmp_path / "home")
    project = store.create_project("客户 A")
    model = tmp_path / "model"
    model.mkdir()
    (model / "dummy").write_text("model", encoding="utf-8")
    one = tmp_path / "one.txt"
    two = tmp_path / "two.txt"
    one.write_text("独特的甲方合同内容", encoding="utf-8")
    two.write_text("普通的乙方文档内容", encoding="utf-8")
    files = store.add_files(project["id"], [str(one), str(two)])
    embedder = FakeEmbedder()
    build_project(store, project["id"], embedder, model_dir=model)
    old = SearchIndex(active_index(store.root(project["id"])), embedder)
    assert any("甲方" in row["text"] for row in old.chunks)
    store.remove_file(project["id"], files[0]["id"])
    build_project(store, project["id"], embedder, model_dir=model)
    new = SearchIndex(active_index(store.root(project["id"])), embedder)
    assert all("甲方" not in row["text"] for row in new.chunks)
    assert new.search("文档", 1)


def test_cancel_preserves_current_index(tmp_path: Path):
    store = ProjectStore(tmp_path / "home")
    project = store.create_project("客户 B")
    model = tmp_path / "model"
    model.mkdir()
    (model / "dummy").write_text("model", encoding="utf-8")
    source = tmp_path / "doc.txt"
    source.write_text("合同条款", encoding="utf-8")
    store.add_files(project["id"], [str(source)])
    embedder = FakeEmbedder()
    build_project(store, project["id"], embedder, model_dir=model)
    before = active_index(store.root(project["id"]))
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(BuildCancelled):
        build_project(store, project["id"], embedder, model_dir=model, cancel=cancel)
    assert active_index(store.root(project["id"])) == before


def test_removing_last_file_commits_empty_index_and_clears_cached_text(tmp_path: Path):
    store = ProjectStore(tmp_path / "home")
    project = store.create_project("待清空")
    model = tmp_path / "model"
    model.mkdir()
    (model / "dummy").write_text("model", encoding="utf-8")
    source = tmp_path / "private.txt"
    source.write_text("需要清除的企业正文", encoding="utf-8")
    item = store.add_files(project["id"], [str(source)])[0]
    embedder = FakeEmbedder()
    build_project(store, project["id"], embedder, model_dir=model)
    store.remove_file(project["id"], item["id"])
    build_project(store, project["id"], embedder, model_dir=model)
    root = store.root(project["id"])
    index = SearchIndex(active_index(root), embedder)
    assert index.chunks == []
    assert index.search("企业正文") == []
    assert list((root / "cache").iterdir()) == []
    assert len(list((root / "versions").iterdir())) == 1


def test_mid_build_cancel_preserves_previous_version(tmp_path: Path):
    store = ProjectStore(tmp_path / "home")
    project = store.create_project("客户 C")
    model = tmp_path / "model"
    model.mkdir()
    (model / "dummy").write_text("model", encoding="utf-8")
    source = tmp_path / "doc.txt"
    source.write_text("第一份文件", encoding="utf-8")
    store.add_files(project["id"], [str(source)])
    embedder = FakeEmbedder()
    build_project(store, project["id"], embedder, model_dir=model)
    before = active_index(store.root(project["id"]))
    next_file = tmp_path / "next.txt"
    next_file.write_text("第二份文件", encoding="utf-8")
    store.add_files(project["id"], [str(next_file)])
    cancel = threading.Event()

    def progress(_percent, message):
        if "next.txt" in message:
            cancel.set()

    with pytest.raises(BuildCancelled):
        build_project(store, project["id"], embedder, model_dir=model,
                      cancel=cancel, progress=progress)
    assert active_index(store.root(project["id"])) == before
    with pytest.raises(ValueError, match="重新构建"):
        export_project(store, project["id"], tmp_path / "export", "developer", model)


def test_inventory_change_during_build_keeps_new_file_and_old_index(tmp_path: Path):
    store = ProjectStore(tmp_path / "home")
    project = store.create_project("并发修改")
    model = tmp_path / "model"
    model.mkdir()
    (model / "dummy").write_text("model", encoding="utf-8")
    source = tmp_path / "old.txt"
    source.write_text("旧资料", encoding="utf-8")
    store.add_files(project["id"], [str(source)])
    embedder = FakeEmbedder()
    build_project(store, project["id"], embedder, model_dir=model)
    previous = active_index(store.root(project["id"]))
    newer = tmp_path / "new.txt"
    newer.write_text("后来导入的资料", encoding="utf-8")
    another = tmp_path / "another.txt"
    another.write_text("正在构建的资料", encoding="utf-8")
    store.add_files(project["id"], [str(another)])
    changed = False

    def progress(_percent, message):
        nonlocal changed
        if not changed and "another.txt" in message:
            store.add_files(project["id"], [str(newer)])
            changed = True

    with pytest.raises(BuildCancelled, match="资料清单已变更"):
        build_project(store, project["id"], embedder, model_dir=model, progress=progress)
    assert active_index(store.root(project["id"])) == previous
    assert "new.txt" in [row["name"] for row in store.get(project["id"])["files"]]
    assert store.get(project["id"])["dirty"]
