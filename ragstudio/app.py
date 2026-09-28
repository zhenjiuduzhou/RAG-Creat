"""Desktop controller: connect user actions to background work."""
from __future__ import annotations

import sys
import threading
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import QApplication, QMessageBox

from .embedding import Embedder
from .export import export_project
from .extract import extract_file
from .pipeline import BuildCancelled, active_index, build_project
from .project import ProjectStore
from .search import SearchIndex
from .ui import MainWindow


def tesseract_path() -> str | None:
    import shutil

    base = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
    for path in (base / "tesseract" / "tesseract.exe",
                 Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")):
        if path.is_file():
            return str(path)
    return shutil.which("tesseract")


class Worker(QObject):
    progress = Signal(int, str)
    succeeded = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.succeeded.emit(self.fn(self.progress.emit))
        except BuildCancelled:
            self.failed.emit("构建已取消，旧知识库保持可用")
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class StudioController(QObject):
    def __init__(self):
        super().__init__()
        self.store = ProjectStore()
        self.model_dir = self.store.home / "models" / "multilingual-e5-small"
        self.embedder = None
        self.window = MainWindow()
        self.cancel = threading.Event()
        self._thread = None
        self._worker = None
        self._connect()
        self.window.set_projects(self.store.list_projects())
        if not self.store.list_projects():
            self.window.set_status("请先新建项目")

    def _connect(self):
        w = self.window
        w.create_project.connect(self.create_project)
        w.project_selected.connect(self.select_project)
        w.import_files.connect(self.add_files)
        w.remove_file.connect(self.remove_file)
        w.preview_requested.connect(self.preview)
        w.build_requested.connect(self.build)
        w.cancel_requested.connect(self.cancel.set)
        w.search_requested.connect(self.search)
        w.export_requested.connect(self.export)

    def _project_id(self) -> str:
        project_id = self.window.current_project_id()
        if not project_id:
            raise ValueError("请先新建或选择项目")
        return project_id

    def _error(self, message: str):
        self.window.set_status("操作失败")
        QMessageBox.warning(self.window, "RAG Studio", message)

    def _start(self, fn, on_success, initial: str):
        if self._thread is not None:
            return
        self.cancel.clear()
        self.window.set_busy(True)
        self.window.set_status(initial)
        thread = QThread(self)
        worker = Worker(fn)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self.window.set_progress)
        worker.succeeded.connect(on_success)
        worker.failed.connect(self._error)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        def cleanup():
            self._thread = None
            self._worker = None
            self.window.set_busy(False)

        thread.finished.connect(cleanup)
        self._thread, self._worker = thread, worker
        thread.start()

    def _get_embedder(self):
        if self.embedder is None:
            self.embedder = Embedder.load(self.model_dir)
        return self.embedder

    def create_project(self, name: str):
        try:
            project = self.store.create_project(name)
            self.window.set_projects(self.store.list_projects())
            self.window.set_current_project(project["id"])
            self.window.set_status("项目已创建")
        except Exception as exc:
            self._error(str(exc))

    def select_project(self, project_id: str):
        try:
            project = self.store.get(project_id)
            self.window.set_files(project["files"])
            self.window.set_results([])
        except Exception as exc:
            self._error(str(exc))

    def add_files(self, paths: list[str]):
        if self._thread is not None:
            return
        try:
            project_id = self._project_id()
            added = self.store.add_files(project_id, paths)
            self.window.set_files(self.store.get(project_id)["files"])
            self.window.set_status(f"已导入 {len(added)} 个文件；请构建知识库")
        except Exception as exc:
            self._error(str(exc))

    def remove_file(self, file_id: str):
        if self._thread is not None:
            return
        try:
            project_id = self._project_id()
            self.store.remove_file(project_id, file_id)
            self.window.set_files(self.store.get(project_id)["files"])
            self.window.set_status("文件已移除；请重新构建知识库")
        except Exception as exc:
            self._error(str(exc))

    def preview(self, file_id: str):
        try:
            project_id = self._project_id()
            root = self.store.root(project_id)
            item = next(f for f in self.store.get(project_id)["files"] if f["id"] == file_id)
        except Exception as exc:
            self._error(str(exc))
            return

        def work(progress):
            progress(0, "提取文件预览")
            sections = extract_file(root / "sources" / item["path"], tesseract_path())
            preview = "【提取文本】\n\n" + "\n\n".join(
                f"[{s.location} · {s.method}]\n{s.text}" for s in sections)
            try:
                if self.store.get(project_id).get("dirty", False):
                    return preview[:50000] + "\n\n资料已变更，请重新构建后查看文本块。"
                import json

                index = active_index(root)
                chunks = [json.loads(line) for line in (index / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
                matching = [c for c in chunks if c["file_id"] == file_id]
                if matching:
                    preview += "\n\n【当前知识库文本块】\n\n" + "\n\n".join(
                        f"[{c['location']}]\n{c['text']}" for c in matching)
            except FileNotFoundError:
                pass
            return preview[:50000]

        self._start(work, self.window.show_preview, "正在预览")

    def build(self):
        try:
            project_id = self._project_id()
        except Exception as exc:
            self._error(str(exc))
            return

        def work(progress):
            progress(0, "加载本地模型；首次使用需要下载")
            embedder = self._get_embedder()
            return build_project(self.store, project_id, embedder, model_dir=self.model_dir,
                                 progress=progress, cancel=self.cancel, tesseract_cmd=tesseract_path())

        def done(manifest):
            self.window.set_files(self.store.get(project_id)["files"])
            self.window.set_status(f"构建完成 · {manifest['chunk_count']} 个文本块")
            if manifest["errors"]:
                self.window.set_status("构建完成，部分文件处理失败")
                QMessageBox.warning(self.window, "部分文件处理失败", "\n".join(
                    f"{row['file']}: {row['error']}" for row in manifest["errors"]))

        self._start(work, done, "正在构建")

    def search(self, query: str):
        try:
            project_id = self._project_id()
            if self.store.get(project_id).get("dirty", False):
                raise ValueError("资料有变更，请重新构建知识库后再检索")
            root = self.store.root(project_id)
        except Exception as exc:
            self._error(str(exc))
            return

        def work(_progress):
            return SearchIndex(active_index(root), self._get_embedder()).search(query)

        self._start(work, self.window.set_results, "正在检索")

    def export(self, destination: str, mode: str):
        try:
            project_id = self._project_id()
        except Exception as exc:
            self._error(str(exc))
            return
        base = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
        runtime = base / "server-runtime" if getattr(sys, "frozen", False) else base / "dist" / "ragstudio-server"

        def work(progress):
            progress(0, "校验并打包")
            return export_project(self.store, project_id, Path(destination), mode, self.model_dir, runtime)

        def done(path):
            self.window.set_status(f"已导出：{path}")
            QMessageBox.information(self.window, "导出完成", f"知识库包已保存到：\n{path}")

        self._start(work, done, "正在导出")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("RAG Studio")
    controller = StudioController()
    controller.window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
