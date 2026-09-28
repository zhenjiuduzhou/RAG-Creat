# RAG Studio

RAG Studio 是本机运行的 Windows 知识库制作工具。为每个客户建立项目，导入文档，检查提取结果，构建向量索引并测试检索，然后导出可交付的 ZIP。软件不调用云端大模型，也不会主动上传客户资料。

## 运行开发版

在 Windows PowerShell 中：

```powershell
python -m venv .venv  # 请使用 Python 3.12
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
.\.venv\Scripts\python.exe main.py
```

首次构建知识库时下载多语言 E5 模型，约 466 MB；下载完成后解析、向量化和检索均在本机进行。OCR 使用随 RapidOCR Python 包提供的中英文 ONNX 模型，能够离线处理扫描 PDF 和图片。[模型许可及说明](https://huggingface.co/intfloat/multilingual-e5-small/raw/main/README.md)

## Windows 成品

```powershell
./scripts/build_windows.ps1
```

打包成功后，`dist/RAGStudio/RAGStudio.exe` 是桌面程序。完整包导出会将 `dist/ragstudio-server` 中的服务运行程序和本地模型一起装入 ZIP，客户解压后运行 `start_windows.cmd`，无需安装 Python。

完整包可能超过 1 GB，因为包含嵌入模型和独立运行环境。`--host 0.0.0.0` 对外开放前，必须设置 `RAG_STUDIO_TOKEN`；还应在企业网关配置 HTTPS、防火墙和访问控制。云端 Dify 无法访问桌面的 `127.0.0.1`，需将导出包部署到它可访问的服务器。软件会在每个导出包内生成更具体的接入指南。

## 资料与限制

- 支持 PDF、DOCX、PPTX、XLSX、CSV、TXT、MD、HTML、JSON、PNG、JPG 和 TIFF。旧版 `.doc`、`.xls` 请先转换。
- 单文件最大 100 MB，OCR 图片最大 5000 万像素。密码保护、损坏或未提取到文字的文件会在构建结果中列出。
- 原始资料默认不放入导出包；文本块仍含有业务资料，须按客户数据要求保管。
- 初版使用精确向量检索，适用于中小型资料集。大型资料集构建速度和服务内存占用取决于机器资源。
- 导入或移除资料后，需点击“构建知识库”才能更新可检索版本。

## 验证

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
