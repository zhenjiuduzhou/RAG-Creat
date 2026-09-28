"""Standalone server entry point used by source and frozen exports."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from .service import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a RAG Studio knowledge base")
    parser.add_argument("--bundle", type=Path, default=Path.cwd())
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    token = os.getenv("RAG_STUDIO_TOKEN")
    if args.host not in ("127.0.0.1", "localhost", "::1") and not token:
        parser.error("非本机监听需要设置 RAG_STUDIO_TOKEN 环境变量")
    app = create_app(args.bundle, token=token)
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()

