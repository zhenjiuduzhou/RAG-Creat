import threading
import time

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QDialog

from ragstudio.app import StudioController


def test_background_completion_and_dialogs_run_on_gui_thread(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("RAG_STUDIO_HOME", str(tmp_path))
    app = QApplication.instance() or QApplication([])
    controller = StudioController()
    controller.window.show()
    main_thread = threading.get_ident()
    seen = []

    def complete(value):
        seen.append((threading.get_ident(), value))
        dialog = QDialog(controller.window)
        QTimer.singleShot(0, dialog.accept)
        dialog.exec()
        assert controller.window.isVisible()

    def drain():
        deadline = time.monotonic() + 5
        while controller._thread is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.001)
        assert controller._thread is None
        assert not controller.window._busy

    for value in range(3):
        controller._start(lambda progress, value=value: value, complete, "测试")
        drain()
    assert seen == [(main_thread, value) for value in range(3)]
    errors = []
    monkeypatch.setattr(controller, "_error", lambda message: errors.append((threading.get_ident(), message)))

    def fail(_progress):
        raise ValueError("expected failure")

    controller._start(fail, complete, "测试失败")
    drain()
    assert errors == [(main_thread, "expected failure")]
    controller.window.close()
