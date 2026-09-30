import json
import sys
from dataclasses import asdict
from pathlib import Path


def main():
    if len(sys.argv) == 5 and sys.argv[1] == "--check-api":
        from ragstudio.api_check import check_api
        return check_api(sys.argv[2], sys.argv[3], Path(sys.argv[4]))
    if len(sys.argv) == 4 and sys.argv[1] == "--check-desktop":
        from ragstudio.desktop_check import check_desktop

        return check_desktop(Path(sys.argv[2]), Path(sys.argv[3]))
    if len(sys.argv) == 4 and sys.argv[1] == "--check-extraction":
        from ragstudio.extract import extract_file

        sections = extract_file(Path(sys.argv[2]))
        Path(sys.argv[3]).write_text(json.dumps([asdict(section) for section in sections], ensure_ascii=False),
                                     encoding="utf-8")
        return 0
    from ragstudio.app import main as desktop_main

    return desktop_main()

if __name__ == "__main__":
    raise SystemExit(main())

