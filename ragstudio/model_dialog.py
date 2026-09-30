"""Project embedding model settings form.

Validation and model loading belong to the controller. This dialog only gathers
settings and reports an apply request, leaving it open until the controller accepts.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


class ModelSettingsDialog(QDialog):
    apply_requested = Signal(object, str)

    def __init__(self, profile: dict | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pending = False
        self.setWindowTitle("模型设置")
        self.setMinimumWidth(540)
        self.setModal(True)
        profile = dict(profile or {})
        for field, prefix in (("query_prefix", "query"), ("passage_prefix", "passage")):
            if field not in profile and isinstance(profile.get("prefixes"), dict):
                profile[field] = profile["prefixes"].get(prefix, "")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 20)
        layout.setSpacing(14)
        intro = QLabel("为当前项目选择向量模型。更换模型后需要重新构建知识库。")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.provider = QComboBox()
        self.provider.addItem("内置 E5 模型", "builtin")
        self.provider.addItem("本地模型目录", "local")
        self.provider.addItem("远程 API", "api")
        self.provider.setAccessibleName("模型来源")
        layout.addWidget(self.provider)

        self.pages = QStackedWidget()
        builtin_page = QWidget()
        builtin_layout = QVBoxLayout(builtin_page)
        builtin_layout.setContentsMargins(0, 0, 0, 0)
        builtin_label = QLabel("intfloat/multilingual-e5-small · 首次使用时下载模型")
        builtin_label.setWordWrap(True)
        builtin_layout.addWidget(builtin_label)
        builtin_layout.addStretch()
        self.pages.addWidget(builtin_page)

        local_page = QWidget()
        local_layout = QVBoxLayout(local_page)
        local_layout.setContentsMargins(0, 0, 0, 0)
        local_form = QFormLayout()
        self.local_path = QLineEdit(str(profile.get("local_path") or ""))
        self.local_path.setPlaceholderText("选择模型所在目录")
        self.local_path.setAccessibleName("本地模型目录")
        path_row = QHBoxLayout()
        path_row.addWidget(self.local_path, 1)
        browse = QPushButton("浏览…")
        browse.clicked.connect(self._browse_local)
        path_row.addWidget(browse)
        local_form.addRow("模型目录", path_row)
        local_layout.addLayout(local_form)
        local_hint = QLabel("请选择完整的 Sentence Transformers 模型目录，包含模型、配置和分词器文件。")
        local_hint.setWordWrap(True)
        local_layout.addWidget(local_hint)
        local_layout.addStretch()
        self.pages.addWidget(local_page)

        api_page = QWidget()
        api_layout = QVBoxLayout(api_page)
        api_layout.setContentsMargins(0, 0, 0, 0)
        api_form = QFormLayout()
        self.endpoint = QLineEdit(str(profile.get("endpoint") or ""))
        self.endpoint.setPlaceholderText("https://example.com/v1/embeddings")
        self.endpoint.setAccessibleName("API 地址")
        api_form.addRow("API 地址", self.endpoint)
        self.model_id = QLineEdit(str(profile.get("model_id") or ""))
        self.model_id.setAccessibleName("API 模型 ID")
        api_form.addRow("模型 ID", self.model_id)
        self.revision = QLineEdit(str(profile.get("revision") or ""))
        self.revision.setPlaceholderText("可选：部署版本标识；更新服务模型后修改并重建")
        api_form.addRow("模型版本", self.revision)
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key.setPlaceholderText("输入密钥；留空可沿用已保存的密钥")
        self.api_key.setAccessibleName("API 密钥")
        api_form.addRow("API 密钥", self.api_key)
        self.max_length = QSpinBox()
        self.max_length.setRange(64, 8192)
        self.max_length.setValue(int(profile.get("max_length") or 512))
        self.max_length.setSuffix(" 字符")
        self.max_length.setAccessibleName("每段最大字符数")
        api_form.addRow("分段上限", self.max_length)
        api_layout.addLayout(api_form)
        api_hint = QLabel("远程 API 按字符数分段；这里的上限单位是字符，不是 token。")
        api_hint.setWordWrap(True)
        api_layout.addWidget(api_hint)
        api_layout.addStretch()
        self.pages.addWidget(api_page)

        layout.addWidget(self.pages)
        self.provider.currentIndexChanged.connect(self.pages.setCurrentIndex)
        provider_index = self.provider.findData(profile.get("provider", "builtin"))
        self.provider.setCurrentIndex(max(0, provider_index))
        self.pages.setCurrentIndex(self.provider.currentIndex())

        prefix_form = QFormLayout()
        self.query_prefix = QLineEdit(str(profile.get("query_prefix", "" if profile.get("provider") == "api" else "query: ")))
        self.query_prefix.setAccessibleName("查询前缀")
        prefix_form.addRow("查询前缀", self.query_prefix)
        self.passage_prefix = QLineEdit(str(profile.get("passage_prefix", "" if profile.get("provider") == "api" else "passage: ")))
        self.passage_prefix.setAccessibleName("文档前缀")
        prefix_form.addRow("文档前缀", self.passage_prefix)
        layout.addLayout(prefix_form)
        self.provider.currentIndexChanged.connect(self._provider_changed)

        self.error_label = QLabel()
        self.error_label.setObjectName("modelError")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        layout.addWidget(self.error_label)
        actions = QHBoxLayout()
        actions.addStretch()
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.reject)
        actions.addWidget(self.cancel_button)
        self.apply_button = QPushButton("测试并应用")
        self.apply_button.setDefault(True)
        self.apply_button.clicked.connect(self._request_apply)
        actions.addWidget(self.apply_button)
        layout.addLayout(actions)

    def _provider_changed(self):
        builtin = self.provider.currentData() == "builtin"
        self.query_prefix.setText("query: " if builtin else "")
        self.passage_prefix.setText("passage: " if builtin else "")

    def _browse_local(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择完整的 Sentence Transformers 模型目录", self.local_path.text())
        if path:
            self.local_path.setText(path)

    def configuration(self) -> tuple[dict, str]:
        provider = str(self.provider.currentData())
        local_path = self.local_path.text().strip() if provider == "local" else ""
        profile = {
            "provider": provider,
            "revision": self.revision.text().strip() if provider == "api" else "local",
            "local_path": local_path,
            "model_id": (self.model_id.text().strip() if provider == "api" else
                         Path(local_path).name if provider == "local" and local_path else
                         "intfloat/multilingual-e5-small"),
            "endpoint": self.endpoint.text().strip() if provider == "api" else "",
            "query_prefix": self.query_prefix.text(),
            "passage_prefix": self.passage_prefix.text(),
            "max_length": self.max_length.value() if provider == "api" else 512,
        }
        return profile, self.api_key.text() if provider == "api" else ""

    def _request_apply(self) -> None:
        self.show_error("")
        profile, key = self.configuration()
        self.apply_requested.emit(profile, key)

    def set_pending(self, pending: bool) -> None:
        self._pending = bool(pending)
        self.provider.setEnabled(not pending)
        self.pages.setEnabled(not pending)
        self.query_prefix.setEnabled(not pending)
        self.passage_prefix.setEnabled(not pending)
        self.apply_button.setEnabled(not pending)
        self.cancel_button.setEnabled(not pending)
        self.apply_button.setText("正在测试…" if pending else "测试并应用")

    def reject(self) -> None:
        if not self._pending:
            super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._pending:
            event.ignore()
        else:
            super().closeEvent(event)

    def show_error(self, message: str) -> None:
        self.error_label.setText(message)
        self.error_label.setVisible(bool(message))
