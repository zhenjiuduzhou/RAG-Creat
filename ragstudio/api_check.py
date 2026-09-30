"""Exercise model settings, build and retrieval inside the normal Qt event loop."""
from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer, QCoreApplication, QEvent
from .app import StudioController
from .model_dialog import ModelSettingsDialog
from .pipeline import build_project, active_index
from .search import SearchIndex


def check_api(endpoint: str, model: str, output: Path) -> int:
    with tempfile.TemporaryDirectory(prefix="ragstudio-api-check-") as temp:
        os.environ["RAG_STUDIO_HOME"] = temp
        app = QApplication([])
        app.setQuitOnLastWindowClosed(False)
        controller = StudioController()
        controller.window.show()
        project = controller.store.create_project("API 接入验收")
        controller.window.set_projects(controller.store.list_projects())
        controller.window.set_current_project(project["id"])

        def apply_dialog():
            dialog = next(widget for widget in app.topLevelWidgets() if isinstance(widget, ModelSettingsDialog) and widget.isVisible())
            dialog.provider.setCurrentIndex(dialog.provider.findData("api"))
            dialog.endpoint.setText(endpoint)
            dialog.model_id.setText(model)
            dialog.api_key.setText("diagnostic-key")
            dialog.apply_button.click()

        def exercise():
            try:
                QTimer.singleShot(100, apply_dialog)
                controller.model_settings()
                saved = controller.store.get(project["id"])
                assert saved["embedding"]["provider"] == "api"
                embedder = controller._get_embedder(project["id"])
                source = Path(temp) / "contract.txt"
                source.write_text("产品 ABC-123 保修两年，支持换货。", encoding="utf-8")
                controller.store.add_files(project["id"], [str(source)])
                manifest = build_project(controller.store, project["id"], embedder, model_dir=Path(temp) / "no-model")
                results = SearchIndex(active_index(controller.store.root(project["id"])), embedder).search("ABC-123")
                controller.window.set_results(results)
                assert controller.window.isVisible() and results
                output.write_text(json.dumps({"ok": True, "dimension": embedder.dimension, "chunks": manifest["chunk_count"], "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
                QTimer.singleShot(100, app.quit)
            except Exception as exc:
                output.write_text(json.dumps({"ok":False,"error":str(exc)},ensure_ascii=False),encoding="utf-8")
                app.exit(1)

        QTimer.singleShot(0, exercise)
        code = app.exec()
        controller.window.close()
        controller.window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        return code
