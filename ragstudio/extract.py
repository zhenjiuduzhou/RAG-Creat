"""Safe text extraction from supported office, web, and image documents."""

from __future__ import annotations

import csv
import json
import io
import zipfile
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


MAX_FILE_SIZE = 100 * 1024 * 1024
MAX_UNCOMPRESSED_SIZE = 500 * 1024 * 1024


def _check_office_archive(path: Path) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) > 10000 or sum(row.file_size for row in entries) > MAX_UNCOMPRESSED_SIZE:
                raise ExtractionError(f"Office file '{path.name}' exceeds the expanded-size limit")
            if any(row.file_size > MAX_UNCOMPRESSED_SIZE for row in entries):
                raise ExtractionError(f"Office file '{path.name}' contains an oversized entry")
    except zipfile.BadZipFile as exc:
        raise ExtractionError(f"Office file '{path.name}' is damaged") from exc


@dataclass
class Section:
    text: str
    location: str
    method: str


class ExtractionError(Exception):
    """Raised when a document cannot safely be extracted."""


def _dependency(module: str, package: str | None = None) -> Any:
    try:
        return __import__(module, fromlist=["*"])
    except ImportError as exc:
        raise ExtractionError(f"Required package '{package or module}' is not installed") from exc


def _nonempty(text: str, location: str, method: str) -> Section | None:
    text = text.strip()
    return Section(text, location, method) if text else None


def _join(sections: Iterable[Section | None]) -> list[Section]:
    return [section for section in sections if section is not None]


def _extract_pdf(path: Path, tesseract_cmd: str | None) -> list[Section]:
    try:
        PdfReader = _dependency("pypdf").PdfReader
        doc = PdfReader(path, strict=False)
    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError(f"Cannot open PDF '{path.name}': {exc}") from exc
    if doc.is_encrypted:
        raise ExtractionError(f"PDF '{path.name}' is encrypted and requires a password")
    if len(doc.pages) > 2000:
        raise ExtractionError(f"PDF '{path.name}' exceeds the 2000-page limit")
    out: list[Section] = []
    renderer = None
    try:
        for index, page in enumerate(doc.pages, 1):
            try:
                text = page.extract_text() or ""
                if len(text.strip()) < 30:
                    if renderer is None:
                        renderer = _dependency("pypdfium2").PdfDocument(str(path))
                    render_page = renderer[index - 1]
                    try:
                        image = render_page.render(scale=2).to_pil()
                        out.extend(_ocr(image, f"page {index}", tesseract_cmd))
                    finally:
                        render_page.close()
                else:
                    section = _nonempty(text, f"page {index}", "text")
                    if section:
                        out.append(section)
            except ExtractionError:
                raise
            except Exception as exc:
                raise ExtractionError(f"Failed extracting PDF page {index}: {exc}") from exc
        return out
    finally:
        if renderer is not None:
            renderer.close()


@lru_cache(maxsize=1)
def _ocr_engine():
    RapidOCR = _dependency("rapidocr", "rapidocr").RapidOCR
    return RapidOCR()


def _ocr(image: Any, location: str, tesseract_cmd: str | None) -> list[Section]:
    try:
        if image.width * image.height > 50_000_000:
            raise ExtractionError("image exceeds the 50 megapixel OCR limit")
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        result = _ocr_engine()(buffer.getvalue())
        text = "\n".join(result.txts or ())
        section = _nonempty(text, location, "ocr")
        return [section] if section else []
    except ExtractionError as exc:
        raise ExtractionError(f"OCR failed for {location}: {exc}") from exc
    except Exception as exc:
        raise ExtractionError(f"OCR failed for {location}: {exc}") from exc


def _extract_docx(path: Path) -> list[Section]:
    docx = _dependency("docx", "python-docx")
    try:
        doc = docx.Document(path)
        sections: list[Section] = []
        heading = ""
        for i, paragraph in enumerate(doc.paragraphs, 1):
            if paragraph.style and paragraph.style.name.lower().startswith("heading"):
                heading = paragraph.text.strip()
            location = f"paragraph {i}" + (f" · {heading}" if heading else "")
            section = _nonempty(paragraph.text, location, "text")
            if section:
                sections.append(section)
        for ti, table in enumerate(doc.tables, 1):
            for ri, row in enumerate(table.rows, 1):
                section = _nonempty("\t".join(cell.text for cell in row.cells), f"table {ti}, row {ri}", "text")
                if section:
                    sections.append(section)
        return sections
    except Exception as exc:
        raise ExtractionError(f"Cannot extract DOCX '{path.name}': {exc}") from exc


def _extract_pptx(path: Path) -> list[Section]:
    pptx = _dependency("pptx", "python-pptx")
    try:
        presentation = pptx.Presentation(path)
        out = []
        for si, slide in enumerate(presentation.slides, 1):
            for shapei, shape in enumerate(slide.shapes, 1):
                if getattr(shape, "has_text_frame", False):
                    section = _nonempty(shape.text, f"slide {si}, shape {shapei}", "text")
                    if section:
                        out.append(section)
                if getattr(shape, "has_table", False):
                    for ri, row in enumerate(shape.table.rows, 1):
                        section = _nonempty("\t".join(cell.text for cell in row.cells), f"slide {si}, table row {ri}", "text")
                        if section:
                            out.append(section)
        return out
    except Exception as exc:
        raise ExtractionError(f"Cannot extract PPTX '{path.name}': {exc}") from exc


def _extract_xlsx(path: Path) -> list[Section]:
    openpyxl = _dependency("openpyxl")
    try:
        book = openpyxl.load_workbook(path, read_only=True, data_only=True, keep_links=False)
        try:
            out = []
            for sheet in book.worksheets:
                header = ""
                for ri, row in enumerate(sheet.iter_rows(values_only=True), 1):
                    values = ["" if value is None else str(value) for value in row]
                    content = "\t".join(values)
                    if ri == 1:
                        header = content
                    elif header:
                        content = header + "\n" + content
                    section = _nonempty(content, f"sheet {sheet.title}, row {ri}", "text")
                    if section:
                        out.append(section)
            return out
        finally:
            book.close()
    except Exception as exc:
        raise ExtractionError(f"Cannot extract XLSX '{path.name}': {exc}") from exc


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _extract_csv(path: Path) -> list[Section]:
    try:
        content = _read_text(path)
        rows = csv.reader(io.StringIO(content, newline=""))
        output = []
        header = ""
        for index, row in enumerate(rows, 1):
            content = "\t".join(row)
            if index == 1:
                header = content
            elif header:
                content = header + "\n" + content
            section = _nonempty(content, f"row {index}", "text")
            if section:
                output.append(section)
        return output
    except Exception as exc:
        raise ExtractionError(f"Cannot extract CSV '{path.name}': {exc}") from exc


def _extract_html(path: Path) -> list[Section]:
    bs4 = _dependency("bs4", "beautifulsoup4")
    try:
        soup = bs4.BeautifulSoup(_read_text(path), "html.parser")
        for node in soup(["script", "style", "noscript"]):
            node.decompose()
        section = _nonempty(soup.get_text(" ", strip=True), "document", "text")
        return [section] if section else []
    except Exception as exc:
        raise ExtractionError(f"Cannot extract HTML '{path.name}': {exc}") from exc


def _extract_json(path: Path) -> list[Section]:
    try:
        value = json.loads(_read_text(path))
        section = _nonempty(json.dumps(value, ensure_ascii=False, indent=2), "document", "text")
        return [section] if section else []
    except Exception as exc:
        raise ExtractionError(f"Cannot extract JSON '{path.name}': {exc}") from exc


def _extract_image(path: Path, tesseract_cmd: str | None) -> list[Section]:
    try:
        Image = _dependency("PIL.Image", "Pillow").open(path)
        Image.verify()
        with _dependency("PIL.Image", "Pillow").open(path) as image:
            return _ocr(image, "image", tesseract_cmd)
    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError(f"Cannot process image '{path.name}': {exc}") from exc


def extract_file(path: Path, tesseract_cmd: str | None = None) -> list[Section]:
    """Extract nonempty text sections with source locations from a supported file."""
    path = Path(path)
    if not path.exists() or not path.is_file():
        raise ExtractionError(f"File does not exist or is not a regular file: '{path}'")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ExtractionError(f"Cannot access '{path}': {exc}") from exc
    if size > MAX_FILE_SIZE:
        raise ExtractionError(f"File '{path.name}' exceeds the 100 MB limit")
    ext = path.suffix.lower()
    extractors = {
        ".pdf": lambda: _extract_pdf(path, tesseract_cmd),
        ".docx": lambda: _extract_docx(path),
        ".pptx": lambda: _extract_pptx(path),
        ".xlsx": lambda: _extract_xlsx(path),
        ".csv": lambda: _extract_csv(path),
        ".txt": lambda: [_nonempty(_read_text(path), "document", "text")],
        ".md": lambda: [_nonempty(_read_text(path), "document", "text")],
        ".html": lambda: _extract_html(path),
        ".json": lambda: _extract_json(path),
        ".png": lambda: _extract_image(path, tesseract_cmd),
        ".jpg": lambda: _extract_image(path, tesseract_cmd),
        ".jpeg": lambda: _extract_image(path, tesseract_cmd),
        ".tif": lambda: _extract_image(path, tesseract_cmd),
        ".tiff": lambda: _extract_image(path, tesseract_cmd),
    }
    if ext not in extractors:
        raise ExtractionError(f"Unsupported file format '{ext or '(no extension)'}'")
    if size == 0:
        raise ExtractionError(f"File '{path.name}' is empty")
    if ext in {".docx", ".pptx", ".xlsx"}:
        _check_office_archive(path)
    try:
        sections = _join(extractors[ext]())
        if not sections:
            raise ExtractionError(f"No text could be extracted from '{path.name}'")
        return sections
    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError(f"Failed extracting '{path.name}': {exc}") from exc
