# 第三方组件说明

RAG Studio 使用动态链接的 Qt for Python / PySide6。当前 PySide6 包元数据列出 LGPL-3.0-only、GPL-2.0-only 和 GPL-3.0-only 等许可选项。本目录附 LGPL 3.0 与 GPL 3.0 正文；发布方应保留 Qt 动态库可替换性，并按实际分发方式核对相应义务。许可证来源：[GNU LGPL 3.0](https://www.gnu.org/licenses/lgpl-3.0.txt)、[GNU GPL 3.0](https://www.gnu.org/licenses/gpl-3.0.txt)、[Qt for Python 商业使用说明](https://doc.qt.io/qtforpython-6/commercial-use.html)。

PDF 文本解析采用 BSD-3-Clause 的 pypdf；扫描页渲染采用 Apache-2.0 / BSD-3-Clause 的 pypdfium2。pypdfium2 随附的 PDFium 二进制还涉及依赖许可证，应保留其发行包提供的许可证文件。[pypdf 许可说明](https://pypdf.readthedocs.io/en/stable/meta/faq.html)、[pypdfium2 许可说明](https://pypdfium2.readthedocs.io/en/stable/readme.html)

嵌入模型 `intfloat/multilingual-e5-small` 的模型卡标注 MIT 许可，模型卡随导出包保存于 `model/README.md`。[模型卡](https://huggingface.co/intfloat/multilingual-e5-small/raw/main/README.md)

`THIRD_PARTY.txt` 记录构建环境中的主要依赖及其包元数据声明。具体商业分发前仍应核对所有二进制组件的许可证与公告要求。
