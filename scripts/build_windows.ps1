$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$pyinstaller = Join-Path $projectRoot '.venv\Scripts\pyinstaller.exe'
if (-not (Test-Path -LiteralPath $python) -or -not (Test-Path -LiteralPath $pyinstaller)) {
    throw '请先创建 .venv 并安装 pyproject.toml 的 dev 依赖。'
}
Push-Location $projectRoot
try {
    & $pyinstaller --noconfirm --onedir --name ragstudio-server `
        --collect-all sentence_transformers --collect-all transformers --collect-all tokenizers `
        --hidden-import uvicorn.logging --hidden-import uvicorn.loops.auto `
        --hidden-import uvicorn.protocols.http.auto --hidden-import uvicorn.lifespan.on server_main.py
    if ($LASTEXITCODE -ne 0) { throw '检索服务打包失败' }

    & $pyinstaller --noconfirm --onedir --windowed --name RAGStudio `
        --collect-all sentence_transformers --collect-all transformers --collect-all tokenizers `
        --collect-all rapidocr --collect-all pypdfium2 `
        --collect-all docx --collect-all pptx --collect-all openpyxl `
        --collect-all pypdf --collect-all bs4 --collect-all PIL `
        --hidden-import uvicorn.logging --hidden-import uvicorn.loops.auto `
        --hidden-import uvicorn.protocols.http.auto --hidden-import uvicorn.lifespan.on main.py
    if ($LASTEXITCODE -ne 0) { throw '桌面程序打包失败' }

    # PyInstaller can collect an unrelated Poppler ICU DLL from the build
    # machine's PATH. It shadows the Windows ICU API expected by Qt6Core.
    $appInternal = (Resolve-Path -LiteralPath (Join-Path $projectRoot 'dist\RAGStudio\_internal')).Path
    $foreignIcu = Join-Path $appInternal 'icuuc.dll'
    if (Test-Path -LiteralPath $foreignIcu) {
        $resolvedIcu = (Resolve-Path -LiteralPath $foreignIcu).Path
        if (-not $resolvedIcu.StartsWith($appInternal + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
            throw '拒绝删除应用目录之外的冲突 DLL'
        }
        Rename-Item -LiteralPath $resolvedIcu -NewName 'icuuc.dll.disabled'
    }

    $serverRuntime = Join-Path $projectRoot 'dist\ragstudio-server'
    $appRuntime = Join-Path $projectRoot 'dist\RAGStudio\server-runtime'
    if (Test-Path -LiteralPath $appRuntime) {
        throw '桌面程序构建目录仍含旧检索服务，请检查 PyInstaller 是否完成清理'
    }
    Copy-Item -LiteralPath $serverRuntime -Destination $appRuntime -Recurse
    $developerAssets = Join-Path $projectRoot 'dist\RAGStudio\developer-assets'
    New-Item -ItemType Directory -Path (Join-Path $developerAssets 'ragstudio') | Out-Null
    foreach ($name in @('__init__.py', 'embedding.py', 'project.py', 'search.py', 'service.py', 'serve.py', 'providers.py', 'credentials.py')) {
        Copy-Item -LiteralPath (Join-Path $projectRoot "ragstudio\$name") -Destination (Join-Path $developerAssets 'ragstudio')
    }
    Copy-Item -LiteralPath (Join-Path $projectRoot 'requirements.lock') -Destination $developerAssets
    Copy-Item -LiteralPath (Join-Path $projectRoot 'legal') -Destination (Join-Path $projectRoot 'dist\RAGStudio\legal') -Recurse -Force
    Copy-Item -LiteralPath (Join-Path $projectRoot 'THIRD_PARTY.txt') -Destination (Join-Path $projectRoot 'dist\RAGStudio\THIRD_PARTY.txt') -Force
    Copy-Item -LiteralPath (Join-Path $projectRoot 'docs\使用说明.md') -Destination (Join-Path $projectRoot 'dist\RAGStudio\使用说明.md') -Force
    Copy-Item -LiteralPath (Join-Path $projectRoot 'docs\模型接入.md') -Destination (Join-Path $projectRoot 'dist\RAGStudio\模型接入.md') -Force
    Write-Host "Windows 程序：$(Join-Path $projectRoot 'dist\RAGStudio\RAGStudio.exe')"
    Write-Host "检索服务：$(Join-Path $serverRuntime 'ragstudio-server.exe')"
} finally {
    Pop-Location
}
