from pathlib import Path
import time
import pytest
from PySide6.QtCore import Qt, QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton
from ragstudio.project import ProjectStore
from ragstudio.app import StudioController


def test_delete_project_removes_only_managed_customer_data(tmp_path):
    store = ProjectStore(tmp_path / "home")
    first = store.create_project("客户甲")
    second = store.create_project("客户乙")
    root = store.root(first["id"])
    original = tmp_path / "original.txt"
    original.write_text("合同")
    store.add_files(first["id"], [str(original)])
    for name in ("cache", "versions"):
        (root / name).mkdir()
        (root / name / "data").write_text("旧构建")
    model = store.home / "models" / "shared"
    model.mkdir(parents=True)
    (model / "weights").write_text("模型")
    exported = tmp_path / "delivery.zip"
    exported.write_bytes(b"delivered")
    store.delete_project(first["id"])
    assert not root.exists()
    assert store.get(second["id"])["name"] == "客户乙"
    assert original.exists() and exported.exists() and (model / "weights").exists()
    with pytest.raises(ValueError):
        store.delete_project("../models")


def test_delete_button_confirmation_cancel_busy_and_last_project(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("RAG_STUDIO_HOME", str(tmp_path))
    app = QApplication.instance() or QApplication([])
    controller = StudioController()
    project = controller.store.create_project("要删除的项目")
    other = controller.store.create_project("其他项目")
    controller.window.set_projects(controller.store.list_projects())
    controller.window.set_current_project(other["id"])
    row = next(controller.window.projects.item(i) for i in range(2)
               if controller.window.projects.item(i).data(Qt.ItemDataRole.UserRole) == project["id"])
    button = controller.window.projects.itemWidget(row).findChild(QPushButton)
    controller.window.set_busy(True)
    assert not button.isEnabled()
    controller.window.set_busy(False)
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.No)
    button.click()
    assert controller.store.get(project["id"])
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
    button.click()
    deadline = time.monotonic() + 5
    while controller._thread is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.001)
    assert controller._thread is None
    assert not (controller.store.projects / project["id"]).exists()
    assert controller.window.current_project_id() == other["id"]
    assert controller.window.project_title.text() == "其他项目"
    controller.delete_project(other["id"])
    deadline = time.monotonic() + 5
    while controller._thread is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.001)
    assert controller.window.projects.count() == 0
    assert not controller.window.current_project_id()
    assert controller.window.files_table.rowCount() == 0
    assert controller.window.project_title.text() == "选择一个项目"
    controller.window.close()
    controller.window.deleteLater()
    controller.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_delete_refuses_resolved_target_outside_projects(tmp_path, monkeypatch):
    store = ProjectStore(tmp_path / "home")
    project = store.create_project("客户")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "project.json").write_text("{}")
    monkeypatch.setattr(store, "root", lambda project_id: outside)
    with pytest.raises(ValueError, match="目录之外"):
        store.delete_project(project["id"])
    assert outside.is_dir()
