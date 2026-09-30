"""Isolated customer projects and source-file inventory."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path


def default_home() -> Path:
    override = os.getenv("RAG_STUDIO_HOME")
    return Path(override) if override else Path(os.getenv("LOCALAPPDATA", Path.home())) / "RAGStudio"


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ProjectStore:
    def __init__(self, home: Path | None = None):
        self.home = Path(home or default_home())
        self.projects = self.home / "projects"
        self.projects.mkdir(parents=True, exist_ok=True)

    def list_projects(self) -> list[dict]:
        result = []
        for file in self.projects.glob("*/project.json"):
            try:
                data = load_json(file)
                result.append({"id": data["id"], "name": data["name"]})
            except (ValueError, KeyError):
                continue
        return sorted(result, key=lambda item: item["name"].casefold())

    def create_project(self, name: str) -> dict:
        name = name.strip()
        if not name or len(name) > 100:
            raise ValueError("项目名称不能为空且不能超过 100 个字符")
        project_id = uuid.uuid4().hex
        root = self.projects / project_id
        (root / "sources").mkdir(parents=True)
        data = {"id": project_id, "name": name, "created_at": datetime.now(timezone.utc).isoformat(),
                "files": [], "dirty": True}
        atomic_json(root / "project.json", data)
        return data

    def root(self, project_id: str) -> Path:
        if not project_id or any(c not in "0123456789abcdef" for c in project_id):
            raise ValueError("无效项目 ID")
        root = self.projects / project_id
        if not (root / "project.json").is_file():
            raise FileNotFoundError("项目不存在")
        return root

    def get(self, project_id: str) -> dict:
        return load_json(self.root(project_id) / "project.json")

    def delete_project(self, project_id: str) -> None:
        """Delete only this project's managed data, never shared model storage."""
        root = self.root(project_id)
        projects = self.projects.resolve()
        if root.is_symlink() or root.is_junction() or root.resolve().parent != projects:
            raise ValueError("拒绝删除项目目录之外的数据")
        for child in root.rglob("*"):
            if child.is_symlink() or child.is_junction():
                raise ValueError("项目目录包含外部链接，请先移除链接再删除项目")
        shutil.rmtree(root)

    def add_files(self, project_id: str, paths: list[str]) -> list[dict]:
        root = self.root(project_id)
        project = self.get(project_id)
        added = []
        known = {item["sha256"] for item in project["files"]}
        for raw in paths:
            source = Path(raw)
            if not source.is_file():
                continue
            if source.stat().st_size > 100 * 1024 * 1024:
                raise ValueError(f"文件超过 100 MB：{source.name}")
            digest = sha256_file(source)
            if digest in known:
                continue
            file_id = uuid.uuid4().hex
            suffix = source.suffix.lower()
            destination = root / "sources" / f"{file_id}{suffix}"
            shutil.copy2(source, destination)
            item = {"id": file_id, "name": source.name, "path": destination.name, "sha256": digest, "status": "待构建"}
            project["files"].append(item)
            known.add(digest)
            added.append(item)
        if added:
            project["dirty"] = True
        atomic_json(root / "project.json", project)
        return added

    def remove_file(self, project_id: str, file_id: str) -> None:
        root = self.root(project_id)
        project = self.get(project_id)
        kept = []
        for item in project["files"]:
            if item["id"] == file_id:
                (root / "sources" / item["path"]).unlink(missing_ok=True)
            else:
                kept.append(item)
        project["files"] = kept
        project["dirty"] = True
        atomic_json(root / "project.json", project)
