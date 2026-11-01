"""Small PDFs made in the test, with pypdf: no third-party or personal content."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)


def _escape(text: str) -> bytes:
    data = text.encode("latin-1", "replace")
    slash = bytes([92])
    for char in (slash, b"(", b")"):
        data = data.replace(char, slash + char)
    return data


def _writer_with_text(pages: list[str]) -> PdfWriter:
    """A writer holding one page per string; an empty string is a blank page
    (what a scan looks like to a text extractor)."""
    writer = PdfWriter()
    font = writer._add_object(
        DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
    )
    for text in pages:
        page = writer.add_blank_page(612, 792)
        if not text:
            continue
        stream = DecodedStreamObject()
        stream.set_data(b"BT /F1 12 Tf 72 700 Td (" + _escape(text) + b") Tj ET")
        page[NameObject("/Contents")] = writer._add_object(stream)
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
            }
        )
    return writer


def make_text_pdf(path: Path, pages: list[str] | str = "Hello text") -> Path:
    """Write a PDF with the given text, one string per page."""
    if isinstance(pages, str):
        pages = [pages]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        _writer_with_text(pages).write(f)
    return path


def make_scan_pdf(path: Path, pages: int = 2) -> Path:
    """Write a PDF whose pages carry no text."""
    return make_text_pdf(path, [""] * pages)


def make_encrypted_pdf(path: Path, password: str = "secret") -> Path:
    """Write a PDF that needs a password to be read."""
    writer = _writer_with_text(["Locked text"])
    writer.encrypt(password)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        writer.write(f)
    return path


def make_broken_pdf(path: Path) -> Path:
    """Write a file that has the .pdf name and no PDF in it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4\nthis is not a pdf at all \x00\xff\xfe")
    return path
