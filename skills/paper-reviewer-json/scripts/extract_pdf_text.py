#!/usr/bin/env python3
"""
Extract text from a PDF for downstream review.

Preferred backend: pypdf (pure Python).
Optional backend: PyMuPDF (fitz) for PDFs where pypdf extraction is poor.

This script is read-only and writes only to --out.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def extract_with_pypdf(pdf_path: Path, max_pages: int | None, keep_page_breaks: bool) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "pypdf is required. Try: uv run --with pypdf -s scripts/extract_pdf_text.py --help"
        ) from e

    reader = PdfReader(str(pdf_path))
    out: list[str] = []
    n = len(reader.pages)
    limit = min(n, max_pages) if max_pages is not None else n
    for i in range(limit):
        txt = reader.pages[i].extract_text() or ""
        out.append(txt)
        if keep_page_breaks:
            out.append(f"\n\n----- PAGE {i+1} / {limit} -----\n\n")
    return "".join(out)


def extract_with_pymupdf(pdf_path: Path, max_pages: int | None, keep_page_breaks: bool) -> str:
    try:
        import fitz  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "PyMuPDF is required. Try: uv run --with PyMuPDF -s scripts/extract_pdf_text.py --backend pymupdf --help"
        ) from e

    doc = fitz.open(str(pdf_path))
    out: list[str] = []
    n = doc.page_count
    limit = min(n, max_pages) if max_pages is not None else n
    for i in range(limit):
        page = doc.load_page(i)
        out.append(page.get_text("text") or "")
        if keep_page_breaks:
            out.append(f"\n\n----- PAGE {i+1} / {limit} -----\n\n")
    return "".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Extract PDF text (read-only).")
    ap.add_argument("--pdf", required=True, help="Input PDF path.")
    ap.add_argument("--out", required=True, help="Output text path.")
    ap.add_argument("--max-pages", type=int, default=None, help="Max pages to extract.")
    ap.add_argument(
        "--keep-page-breaks",
        action="store_true",
        help="Insert page boundary markers in the output.",
    )
    ap.add_argument(
        "--backend",
        choices=("pypdf", "pymupdf"),
        default="pypdf",
        help="Extraction backend (default: pypdf).",
    )
    args = ap.parse_args(argv)

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        print(f"[ERROR] PDF not found: {pdf_path}")
        return 2

    try:
        if args.backend == "pymupdf":
            text = extract_with_pymupdf(pdf_path, args.max_pages, args.keep_page_breaks)
        else:
            text = extract_with_pypdf(pdf_path, args.max_pages, args.keep_page_breaks)
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}")
        return 3

    out_path = Path(args.out).expanduser().resolve()
    out_path.write_text(text, encoding="utf-8")
    print(f"[OK] Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
