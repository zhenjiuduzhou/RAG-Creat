"""Exercise the frozen desktop's real build, search and modal-dialog lifecycle."""
from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox, QPushButton

from .app import StudioController


def check_desktop(model_dir: Path, output: Path) -> int:
    with tempfile.TemporaryDirectory(prefix="ragstudio-desktop-check-") as home:
        os.environ["RAG_STUDIO_HOME"] = home
        app = QApplication([])
        controller = StudioController()
        controller.model_dir = model_dir
        controller.window.show()
        project = controller.store.create_project("桌面验收")
        source = Path(home) / "check.txt"
        source.write_text("产品 ABC-123 保修两年，售后支持电话 400-123-4567。", encoding="utf-8")
        file_id = controller.store.add_files(project["id"], [str(source)])[0]["id"]
        controller.window.set_projects(controller.store.list_projects())
        controller.window.set_current_project(project["id"])
        main_thread = threading.get_ident()
        state = {"stage": 0, "errors": [], "gui_thread": True, "dialogs": 0}

        def error(message):
            state["errors"].append(message)

        controller._error = error
        # The connected Slot remains the controller method; inspect callbacks
        # through their actual GUI operations rather than replacing that Slot.
        original_files = controller.window.set_files

        def files(rows):
            state["gui_thread"] &= threading.get_ident() == main_thread
            original_files(rows)

        controller.window.set_files = files

        def tick():
            for widget in app.topLevelWidgets():
                if isinstance(widget, QDialog) and widget.isVisible():
                    state["dialogs"] += 1
                    if isinstance(widget, QMessageBox):
                        widget.button(QMessageBox.StandardButton.Yes).click()
                    else:
                        widget.accept()
                    return
            if controller._thread is not None:
                return
            if state["errors"]:
                app.exit(1)
                return
            stage = state["stage"]
            state["stage"] += 1
            if stage < 2:
                controller.build()
            elif stage == 2:
                controller.search("ABC-123 保修多久")
            elif stage == 3:
                assert controller.window.results.count() > 0
                controller.preview(file_id)
            elif stage == 4:
                row = controller.window.projects.itemWidget(controller.window.projects.currentItem())
                QTimer.singleShot(0, row.findChild(QPushButton).click)
            else:
                state["deleted"] = not (controller.store.projects / project["id"]).exists()
                assert state["deleted"] and controller.window.projects.count() == 0
                state["visible"] = controller.window.isVisible()
                state["busy"] = controller.window._busy
                app.exit(0)

        timer = QTimer()
        timer.timeout.connect(tick)
        timer.start(50)
        QTimer.singleShot(120000, lambda: app.exit(2))
        exit_code = app.exec()
        output.write_text(json.dumps({**state, "exit_code": exit_code}, ensure_ascii=False), encoding="utf-8")
        controller.window.close()
        return exit_code
