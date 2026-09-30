# RAG Studio

RAG Studio 是本机运行的 Windows 知识库制作工具。为每个客户建立项目，导入文档，检查提取结果，构建向量索引并测试检索，然后导出可交付的 ZIP。默认使用本地嵌入模型，可在模型设置中切换到自定义本地模型或 OpenAI 兼容 Embedding API。API 模式会将资料片段与查询发送到配置的模型服务。

## 运行开发版

在 Windows PowerShell 中：

```powershell
python -m venv .venv  # 请使用 Python 3.12
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
.\.venv\Scripts\python.exe main.py
```

使用默认模型时，首次构建知识库或测试模型接入会下载多语言 E5 模型，约 466 MB；下载完成后解析、向量化和检索均在本机进行。OCR 使用随 RapidOCR Python 包提供的中英文 ONNX 模型，能够离线处理扫描 PDF 和图片。[模型许可及说明](https://huggingface.co/intfloat/multilingual-e5-small/raw/main/README.md)

## Windows 成品

```powershell
./scripts/build_windows.ps1
```

打包成功后，`dist/RAGStudio/RAGStudio.exe` 是桌面程序。完整包导出会将 `dist/ragstudio-server` 中的服务运行程序和本地模型一起装入 ZIP，客户解压后运行 `start_windows.cmd`，无需安装 Python。

完整包可能超过 1 GB，因为包含嵌入模型和独立运行环境。`--host 0.0.0.0` 对外开放前，必须设置 `RAG_STUDIO_TOKEN`；还应在企业网关配置 HTTPS、防火墙和访问控制。云端 Dify 无法访问桌面的 `127.0.0.1`，需将导出包部署到它可访问的服务器。软件会在每个导出包内生成更具体的接入指南。

## 模型接入

点击项目右上方“模型设置”。支持内置 E5、完整 Sentence Transformers 模型目录，以及云端或本机 OpenAI 兼容 embeddings API。密钥使用 Windows DPAPI 加密保存；不同客户项目可以分别选择模型。详见 [模型接入说明](docs/模型接入.md)。

本地模型可离线使用；API 模式需要模型服务可访问。API 模式导出包不包含模型权重或 API Key，客户需配置环境变量 `RAG_STUDIO_EMBEDDING_KEY`。更换模型后必须重建。

## 资料与限制

- 支持 PDF、DOCX、PPTX、XLSX、CSV、TXT、MD、HTML、JSON、PNG、JPG 和 TIFF。旧版 `.doc`、`.xls` 请先转换。
- 单文件最大 100 MB，OCR 图片最大 5000 万像素。密码保护、损坏或未提取到文字的文件会在构建结果中列出。
- 原始资料默认不放入导出包；文本块仍含有业务资料，须按客户数据要求保管。
- 初版使用精确向量检索，适用于中小型资料集。大型资料集构建速度和服务内存占用取决于机器资源。
- 导入或移除资料后，需点击“构建知识库”才能更新可检索版本。
- 删除最后一份资料后再次构建，会生成空知识库并清除本项目旧索引及缓存。已交付的旧 ZIP 需另行回收。
- Dify 的程序化 `metadata_condition` 非空过滤请求暂返回 422；如需按部门、密级等元数据隔离，请给每个授权范围建立独立项目和服务。
- 图文混排 PDF 以每页可提取文字长度决定是否 OCR；含有长电子页眉的扫描正文可能需要先另行 OCR 后导入。关闭窗口时会等待后台任务停止。

## 验证

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
