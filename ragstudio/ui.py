"""Main window for RAG Studio.

The window emits intent signals only. Long-running work belongs in the caller's
worker thread, which can feed progress and results back through the setters below.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal, QSettings
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QComboBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QTextEdit,
)


class MainWindow(QMainWindow):
    create_project = Signal(str)
    import_files = Signal(list)
    remove_file = Signal(str)
    build_requested = Signal()
    cancel_requested = Signal()
    search_requested = Signal(str)
    export_requested = Signal(str, str)
    project_selected = Signal(str)
    preview_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("RAG Studio")
        self.setMinimumSize(900, 620)
        self.resize(1240, 800)
        self.setAcceptDrops(True)
        self._files: list[dict] = []
        self._busy = False
        self._settings = QSettings("RAGStudio", "RAGStudio")
        self._theme = str(self._settings.value("ui/theme", "dark"))
        if self._theme not in {"dark", "light"}:
            self._theme = "dark"
        self._build_ui()
        self._apply_style()

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self.setCentralWidget(root)

        side = QFrame()
        side.setObjectName("sidebar")
        side.setMinimumWidth(220)
        side.setMaximumWidth(280)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(18, 22, 18, 18)
        side_layout.setSpacing(14)
        brand = QLabel("◈  RAG Studio")
        brand.setObjectName("brand")
        side_layout.addWidget(brand)
        side_layout.addWidget(self._eyebrow("工作区"))
        self.projects = QListWidget()
        self.projects.setAccessibleName("项目列表")
        self.projects.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.projects.currentItemChanged.connect(self._on_project_changed)
        side_layout.addWidget(self.projects, 1)
        new_project = QPushButton("＋  新建项目")
        self.new_project_button = new_project
        new_project.setObjectName("secondaryButton")
        new_project.clicked.connect(self._create_project_dialog)
        side_layout.addWidget(new_project)
        side_layout.addStretch(0)
        root_layout.addWidget(side)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(32, 22, 32, 24)
        content_layout.setSpacing(18)
        root_layout.addWidget(content, 1)

        header = QHBoxLayout()
        titles = QVBoxLayout()
        self.project_title = QLabel("选择一个项目")
        self.project_title.setObjectName("pageTitle")
        self.project_subtitle = QLabel("导入资料，构建知识库，并测试检索效果。")
        self.project_subtitle.setObjectName("muted")
        titles.addWidget(self.project_title)
        titles.addWidget(self.project_subtitle)
        header.addLayout(titles, 1)
        self.theme_button = QPushButton()
        self.theme_button.setObjectName("secondaryButton")
        self.theme_button.setAccessibleName("切换界面主题")
        self.theme_button.clicked.connect(self._toggle_theme)
        header.addWidget(self.theme_button, 0, Qt.AlignmentFlag.AlignTop)
        self.status_badge = QLabel("就绪")
        self.status_badge.setObjectName("statusBadge")
        header.addWidget(self.status_badge, 0, Qt.AlignmentFlag.AlignTop)
        content_layout.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        content_layout.addWidget(splitter, 1)
        top_panel = QWidget()
        top_layout = QVBoxLayout(top_panel)
        top_layout.setContentsMargins(0, 0, 0, 4)
        top_layout.setSpacing(14)
        self._section_heading(top_layout, "知识库资料", "支持 PDF、Word、Markdown、文本等文件")
        self.drop_zone = QFrame()
        self.drop_zone.setObjectName("dropZone")
        self.drop_zone.setMinimumHeight(104)
        dz = QHBoxLayout(self.drop_zone)
        dz.setContentsMargins(20, 14, 20, 14)
        drop_copy = QVBoxLayout()
        drop_title = QLabel("拖放文件到这里")
        drop_title.setObjectName("dropTitle")
        drop_hint = QLabel("或从电脑中选择多个文件导入")
        drop_hint.setObjectName("muted")
        drop_copy.addWidget(drop_title)
        drop_copy.addWidget(drop_hint)
        dz.addLayout(drop_copy, 1)
        self.import_button = QPushButton("选择文件")
        self.import_button.setObjectName("primaryButton")
        self.import_button.clicked.connect(self._choose_files)
        dz.addWidget(self.import_button)
        top_layout.addWidget(self.drop_zone)
        self.files_table = QTableWidget(0, 3)
        self.files_table.setHorizontalHeaderLabels(["文件", "状态", "操作"])
        self.files_table.setAlternatingRowColors(True)
        self.files_table.setShowGrid(False)
        self.files_table.verticalHeader().setVisible(False)
        self.files_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.files_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.files_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.files_table.horizontalHeader().setStretchLastSection(False)
        self.files_table.horizontalHeader().setSectionResizeMode(0, self.files_table.horizontalHeader().ResizeMode.Stretch)
        self.files_table.horizontalHeader().setSectionResizeMode(1, self.files_table.horizontalHeader().ResizeMode.ResizeToContents)
        self.files_table.horizontalHeader().setSectionResizeMode(2, self.files_table.horizontalHeader().ResizeMode.ResizeToContents)
        self.files_table.setAccessibleName("项目文件")
        top_layout.addWidget(self.files_table, 1)
        splitter.addWidget(top_panel)

        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 4, 0, 0)
        bottom_layout.setSpacing(14)
        actions = QHBoxLayout()
        actions.addWidget(QLabel("知识库构建"))
        actions.addStretch(1)
        self.build_button = QPushButton("构建知识库")
        self.build_button.setObjectName("primaryButton")
        self.build_button.clicked.connect(self.build_requested.emit)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setObjectName("secondaryButton")
        self.cancel_button.clicked.connect(self.cancel_requested.emit)
        self.cancel_button.hide()
        actions.addWidget(self.build_button)
        actions.addWidget(self.cancel_button)
        bottom_layout.addLayout(actions)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("%p%   %v")
        self.progress.setAccessibleName("构建进度")
        bottom_layout.addWidget(self.progress)

        self._section_heading(bottom_layout, "检索测试", "输入问题，查看知识库召回的相关内容")
        search_row = QHBoxLayout()
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("例如：产品的退货政策是什么？")
        self.query_input.setAccessibleName("检索问题")
        self.query_input.returnPressed.connect(self._search)
        search_row.addWidget(self.query_input, 1)
        self.search_button = QPushButton("搜索")
        self.search_button.clicked.connect(self._search)
        search_row.addWidget(self.search_button)
        bottom_layout.addLayout(search_row)
        self.results = QListWidget()
        self.results.setObjectName("results")
        self.results.setAccessibleName("检索结果")
        self.results.setMinimumHeight(100)
        bottom_layout.addWidget(self.results, 1)

        export_row = QHBoxLayout()
        export_label = QLabel("导出")
        export_label.setObjectName("muted")
        export_row.addWidget(export_label)
        self.export_mode = QComboBox()
        self.export_mode.addItem("Windows 完整版", "windows-full")
        self.export_mode.addItem("开发者包", "developer")
        self.export_mode.setAccessibleName("导出类型")
        export_row.addWidget(self.export_mode)
        export_row.addStretch(1)
        self.export_button = QPushButton("导出项目…")
        self.export_button.setObjectName("secondaryButton")
        self.export_button.clicked.connect(self._export)
        export_row.addWidget(self.export_button)
        bottom_layout.addLayout(export_row)
        splitter.addWidget(bottom)
        splitter.setSizes([390, 340])

        self.statusBar().showMessage("准备就绪")

    @staticmethod
    def _eyebrow(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("eyebrow")
        return label

    @staticmethod
    def _section_heading(layout: QVBoxLayout, title: str, subtitle: str) -> None:
        row = QHBoxLayout()
        copy = QVBoxLayout()
        heading = QLabel(title)
        heading.setObjectName("sectionTitle")
        sub = QLabel(subtitle)
        sub.setObjectName("muted")
        copy.addWidget(heading)
        copy.addWidget(sub)
        row.addLayout(copy)
        row.addStretch(1)
        layout.addLayout(row)

    def _apply_style(self) -> None:
        dark = """
            QWidget { font-family: 'Segoe UI', 'Microsoft YaHei UI', sans-serif; font-size: 13px; color: #e7ebf3; }
            QMainWindow, QWidget { background: #11151d; }
            #sidebar { background: #171c26; border-right: 1px solid #252c39; }
            #brand { font-size: 19px; font-weight: 700; color: #f4f6fb; padding-bottom: 16px; }
            #eyebrow { color: #8d98aa; font-size: 11px; font-weight: 700; letter-spacing: 1px; }
            #pageTitle { font-size: 24px; font-weight: 700; color: #f4f6fb; }
            #sectionTitle { font-size: 15px; font-weight: 650; color: #f2f4f8; }
            #muted { color: #8d98aa; font-size: 12px; }
            QListWidget { background: transparent; border: none; outline: none; }
            QListWidget#results { background: #171c26; border: 1px solid #293142; border-radius: 10px; }
            QListWidget::item { padding: 10px 11px; border-radius: 7px; margin: 2px 0; }
            QListWidget::item:selected { background: #273553; color: #fff; }
            QTableWidget { background: #171c26; alternate-background-color: #1a202b; border: 1px solid #293142; border-radius: 10px; padding: 4px; }
            QHeaderView::section { background: #1b2230; color: #99a5b8; border: none; padding: 9px; font-weight: 600; }
            QTableWidget::item { padding: 6px; border: none; }
            QFrame#dropZone { background: #171e2b; border: 1px dashed #485a79; border-radius: 12px; }
            #dropTitle { font-size: 15px; font-weight: 650; }
            QPushButton { background: #252d3b; color: #e8edf6; border: 1px solid #364153; border-radius: 8px; padding: 9px 15px; font-weight: 600; }
            QPushButton:hover { background: #303b4d; }
            QPushButton:disabled { color: #70798a; background: #202530; }
            QPushButton#primaryButton { color: white; background: #586fea; border-color: #586fea; }
            QPushButton#primaryButton:hover { background: #6b80f4; }
            QPushButton#secondaryButton { background: transparent; }
            QLineEdit, QComboBox { background: #171c26; border: 1px solid #313b4c; border-radius: 8px; padding: 10px 12px; selection-background-color: #586fea; }
            QLineEdit:focus, QComboBox:focus { border-color: #7185ff; }
            QComboBox::drop-down { border: none; width: 24px; }
            QProgressBar { background: #202633; color: #dbe3f2; border: none; border-radius: 6px; height: 12px; text-align: center; }
            QProgressBar::chunk { background: #6278f0; border-radius: 6px; }
            #statusBadge { background: #1f2e2b; color: #82d9b5; border-radius: 10px; padding: 6px 11px; font-size: 12px; }
            QStatusBar { color: #8d98aa; background: #11151d; }
            QSplitter::handle { height: 10px; background: transparent; }
        """
        light = """
            QWidget { font-family: 'Segoe UI', 'Microsoft YaHei UI', sans-serif; font-size: 13px; color: #263246; }
            QMainWindow, QWidget { background: #f5f7fb; }
            #sidebar { background: #edf1f7; border-right: 1px solid #dce3ee; }
            #brand { font-size: 19px; font-weight: 700; color: #172238; padding-bottom: 16px; }
            #eyebrow { color: #69778d; font-size: 11px; font-weight: 700; letter-spacing: 1px; }
            #pageTitle { font-size: 24px; font-weight: 700; color: #172238; }
            #sectionTitle { font-size: 15px; font-weight: 650; color: #1e2b40; }
            #muted { color: #69778d; font-size: 12px; }
            QListWidget { background: transparent; border: none; outline: none; }
            QListWidget#results { background: #fff; border: 1px solid #dce3ee; border-radius: 10px; }
            QListWidget::item { padding: 10px 11px; border-radius: 7px; margin: 2px 0; }
            QListWidget::item:selected { background: #dfe8ff; color: #1d2d51; }
            QTableWidget { background: #fff; alternate-background-color: #f7f9fc; border: 1px solid #dce3ee; border-radius: 10px; padding: 4px; }
            QHeaderView::section { background: #edf1f7; color: #53627a; border: none; padding: 9px; font-weight: 600; }
            QTableWidget::item { padding: 6px; border: none; }
            QFrame#dropZone { background: #f0f4fb; border: 1px dashed #aab9d2; border-radius: 12px; }
            #dropTitle { font-size: 15px; font-weight: 650; }
            QPushButton { background: #fff; color: #29374e; border: 1px solid #cbd5e3; border-radius: 8px; padding: 9px 15px; font-weight: 600; }
            QPushButton:hover { background: #edf2fa; }
            QPushButton:disabled { color: #929bad; background: #eef1f5; }
            QPushButton#primaryButton { color: white; background: #536be3; border-color: #536be3; }
            QPushButton#primaryButton:hover { background: #465dcf; }
            QPushButton#secondaryButton { background: transparent; }
            QLineEdit, QComboBox { background: #fff; border: 1px solid #cbd5e3; border-radius: 8px; padding: 10px 12px; selection-background-color: #536be3; }
            QLineEdit:focus, QComboBox:focus { border-color: #647bf0; }
            QComboBox::drop-down { border: none; width: 24px; }
            QProgressBar { background: #e5eaf2; color: #344259; border: none; border-radius: 6px; height: 12px; text-align: center; }
            QProgressBar::chunk { background: #6077eb; border-radius: 6px; }
            #statusBadge { background: #e3f4ec; color: #247451; border-radius: 10px; padding: 6px 11px; font-size: 12px; }
            QStatusBar { color: #69778d; background: #f5f7fb; }
            QSplitter::handle { height: 10px; background: transparent; }
        """
        self.setStyleSheet(dark if self._theme == "dark" else light)
        self.theme_button.setText("☀  浅色模式" if self._theme == "dark" else "☾  深色模式")

    def _toggle_theme(self) -> None:
        self._theme = "light" if self._theme == "dark" else "dark"
        self._settings.setValue("ui/theme", self._theme)
        self._apply_style()

    def _on_project_changed(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current:
            pid = str(current.data(Qt.ItemDataRole.UserRole) or "")
            self.project_title.setText(current.text())
            self.project_selected.emit(pid)

    def _create_project_dialog(self) -> None:
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "新建项目", "项目名称")
        if ok and name.strip():
            self.create_project.emit(name.strip())

    def _choose_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "选择要导入的文件", "", "所有文件 (*);;文档 (*.pdf *.docx *.md *.txt *.html *.csv)")
        if paths:
            self.import_files.emit(paths)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.import_files.emit(paths)
            event.acceptProposedAction()

    def _search(self) -> None:
        text = self.query_input.text().strip()
        if text:
            self.search_requested.emit(text)

    def _export(self) -> None:
        dest = QFileDialog.getExistingDirectory(self, "选择导出位置")
        if dest:
            self.export_requested.emit(dest, str(self.export_mode.currentData()))

    def set_projects(self, projects: list[dict]) -> None:
        selected = self.current_project_id()
        self.projects.blockSignals(True)
        self.projects.clear()
        for project in projects:
            item = QListWidgetItem(str(project.get("name", "未命名项目")))
            item.setData(Qt.ItemDataRole.UserRole, str(project.get("id", "")))
            self.projects.addItem(item)
            if item.data(Qt.ItemDataRole.UserRole) == selected:
                self.projects.setCurrentItem(item)
        self.projects.blockSignals(False)
        if self.projects.count() and self.projects.currentRow() < 0:
            self.projects.setCurrentRow(0)

    def current_project_id(self) -> str:
        item = self.projects.currentItem()
        return str(item.data(Qt.ItemDataRole.UserRole) or "") if item else ""

    def set_current_project(self, project_id: str) -> None:
        for i in range(self.projects.count()):
            item = self.projects.item(i)
            if str(item.data(Qt.ItemDataRole.UserRole) or "") == str(project_id):
                self.projects.setCurrentItem(item)
                return

    def set_files(self, files: list[dict]) -> None:
        self._files = list(files)
        self.files_table.setRowCount(len(files))
        for row, file in enumerate(files):
            fid = str(file.get("id", ""))
            name = str(file.get("name", "未命名文件"))
            name_item = QTableWidgetItem(name)
            name_item.setData(Qt.ItemDataRole.UserRole, fid)
            self.files_table.setItem(row, 0, name_item)
            self.files_table.setItem(row, 1, QTableWidgetItem(str(file.get("status", "已导入"))))
            actions = QWidget()
            action_layout = QHBoxLayout(actions)
            action_layout.setContentsMargins(3, 2, 3, 2)
            preview = QPushButton("预览")
            preview.setAccessibleName(f"预览 {name}")
            preview.clicked.connect(lambda _=False, file_id=fid: self.preview_requested.emit(file_id))
            remove = QPushButton("移除")
            remove.setAccessibleName(f"移除 {name}")
            remove.clicked.connect(lambda _=False, file_id=fid: self.remove_file.emit(file_id))
            action_layout.addWidget(preview)
            action_layout.addWidget(remove)
            self.files_table.setCellWidget(row, 2, actions)
            self.files_table.setRowHeight(row, 46)

    def show_preview(self, text: str) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("文件预览")
        dialog.setMinimumSize(560, 360)
        dialog.resize(850, 600)
        layout = QVBoxLayout(dialog)
        preview = QTextEdit(dialog)
        preview.setReadOnly(True)
        preview.setAccessibleName("文件预览内容")
        preview.setPlainText(text if text else "暂无可显示的预览内容。")
        layout.addWidget(preview, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.accept)
        buttons.button(QDialogButtonBox.StandardButton.Close).clicked.connect(dialog.accept)
        layout.addWidget(buttons)
        dialog.exec()

    def set_results(self, results: list[dict]) -> None:
        self.results.clear()
        if not results:
            self.results.addItem("没有找到相关内容。")
            return
        for result in results:
            score = result.get("score")
            score_text = f" · 相关度 {float(score):.2f}" if isinstance(score, (int, float)) else ""
            source = str(result.get("source", ""))
            location = str(result.get("location", ""))
            text = str(result.get("text", ""))
            metadata = []
            if source:
                metadata.append(f"来源：{source}")
            if location:
                metadata.append(f"位置：{location}")
            if score_text:
                metadata.append(score_text.strip(" ·"))
            item = QListWidgetItem(f"{text}\n" + "   ·   ".join(metadata) if metadata else text)
            item.setToolTip(text)
            self.results.addItem(item)

    def set_progress(self, value: int, message: str) -> None:
        self.progress.setValue(max(0, min(100, int(value))))
        if message:
            self.progress.setFormat(f"%p%   {message}")

    def set_status(self, message: str) -> None:
        self.status_badge.setText(message or "就绪")
        self.statusBar().showMessage(message or "准备就绪")

    def set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        self.build_button.setEnabled(not busy)
        self.cancel_button.setVisible(busy)
        self.import_button.setEnabled(not busy)
        self.search_button.setEnabled(not busy)
        self.export_button.setEnabled(not busy)
        self.projects.setEnabled(not busy)
        self.new_project_button.setEnabled(not busy)
        self.files_table.setEnabled(not busy)
