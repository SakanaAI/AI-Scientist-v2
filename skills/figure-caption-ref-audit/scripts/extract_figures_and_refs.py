#!/usr/bin/env python3
"""
Extract figure-region screenshots, captions, and figref snippets from a PDF.

Requires: PyMuPDF (pip package: PyMuPDF; import name: fitz)

Safety:
- Read-only on the input PDF.
- Writes only under --out-dir.
- Does not overwrite existing outputs unless --overwrite is provided.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


CAPTION_RE = re.compile(
    r"^(?:Figure)\s+(?P<label>(?:\d+|[A-Za-z]+\.\d+|\(\s*[A-Za-z]+\s*\)\.\d+))(?:\.|:)",
    re.IGNORECASE,
)


@dataclass
class TextBlock:
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    text: str


def _safe_label(label: str) -> str:
    # For filenames
    s = label.strip()
    s = s.replace(" ", "")
    s = s.replace("(", "").replace(")", "")
    s = s.replace("/", "-")
    return s


def _collect_blocks(doc, max_pages: int | None) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    limit = min(doc.page_count, max_pages) if max_pages is not None else doc.page_count
    for i in range(limit):
        page = doc.load_page(i)
        try:
            raw = page.get_text("blocks")
        except Exception:
            continue
        for b in raw:
            # [x0, y0, x1, y1, text, block_no, ...]
            txt = (b[4] or "").strip()
            if not txt:
                continue
            blocks.append(TextBlock(i, float(b[0]), float(b[1]), float(b[2]), float(b[3]), txt))
    return blocks


def _find_caption_blocks(blocks: list[TextBlock]) -> list[TextBlock]:
    caps: list[TextBlock] = []
    for blk in blocks:
        if CAPTION_RE.match(blk.text):
            caps.append(blk)
    # Sort by page then vertical position
    caps.sort(key=lambda b: (b.page, b.y0))
    return caps


def _horiz_overlap_ratio(a: TextBlock, b: TextBlock) -> float:
    overlap = min(a.x1, b.x1) - max(a.x0, b.x0)
    if overlap <= 0:
        return 0.0
    width_min = min(a.x1 - a.x0, b.x1 - b.x0)
    return float(overlap) / float(width_min) if width_min > 0 else 0.0


def _pick_figure_bbox(page_rect, caption: TextBlock, page_blocks: list[TextBlock], margin: float, min_text_len: int, min_vertical_gap: float):
    """
    Heuristic:
    - Find the closest substantial text block above the caption that overlaps horizontally.
    - Use the region between that block's bottom and caption's top as the figure bbox.
    """
    above = [
        b
        for b in page_blocks
        if b.y1 < caption.y0 and (caption.y0 - b.y1) >= min_vertical_gap and len(b.text) >= min_text_len and _horiz_overlap_ratio(caption, b) >= 0.3
    ]
    if above:
        anchor = max(above, key=lambda b: b.y1)
        top = anchor.y1 + margin
    else:
        top = float(page_rect.y0) + margin

    bottom = caption.y0 - margin
    left = max(float(page_rect.x0), caption.x0 - margin)
    right = min(float(page_rect.x1), caption.x1 + margin)

    if bottom - top < 20:
        # Fallback: use most of page width.
        left = float(page_rect.x0) + margin
        right = float(page_rect.x1) - margin
        top = float(page_rect.y0) + margin
        bottom = caption.y0 - margin

    if bottom - top < 20:
        return None
    return (left, top, right, bottom)


def _build_figref_pattern(label: str) -> re.Pattern:
    label = label.strip()
    if label.isdigit():
        # Avoid matching 11 when label is 1.
        return re.compile(
            rf"\b(?:Figure|Fig\.?)(?:\s|\xA0)*{re.escape(label)}(?!\d)",
            re.IGNORECASE,
        )
    return re.compile(
        rf"\b(?:Figure|Fig\.?)(?:\s|\xA0)*{re.escape(label)}\b",
        re.IGNORECASE,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Extract figure images + captions + figrefs from a PDF.")
    ap.add_argument("--pdf", required=True, help="Input PDF path.")
    ap.add_argument("--out-dir", required=True, help="Output directory.")
    ap.add_argument("--max-pages", type=int, default=None, help="Max pages to scan.")
    ap.add_argument("--dpi", type=int, default=150, help="Render DPI for images.")
    ap.add_argument("--margin", type=float, default=6.0, help="Margin in PDF coordinates.")
    ap.add_argument("--min-text-length", type=int, default=50, help="Min chars for anchor text blocks.")
    ap.add_argument("--min-vertical-gap", type=float, default=30.0, help="Min vertical gap (points).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite outputs in out-dir.")
    args = ap.parse_args(argv)

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        print(f"[ERROR] PDF not found: {pdf_path}")
        return 2

    out_dir = Path(args.out_dir).expanduser().resolve()
    images_dir = out_dir / "images"
    figures_json = out_dir / "figures.json"

    if out_dir.exists():
        if not args.overwrite:
            print(f"[ERROR] out-dir exists (use --overwrite): {out_dir}")
            return 2
        # Safer than rmtree(out_dir): delete known outputs only.
        if images_dir.exists():
            shutil.rmtree(images_dir)
        if figures_json.exists():
            figures_json.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    try:
        import fitz  # type: ignore
    except Exception as e:
        print("[ERROR] PyMuPDF is required. Try: uv run --with PyMuPDF -s scripts/extract_figures_and_refs.py --help")
        print(f"Details: {e}")
        return 3

    doc = fitz.open(str(pdf_path))
    blocks = _collect_blocks(doc, max_pages=args.max_pages)
    captions = _find_caption_blocks(blocks)

    # Pre-group blocks by page for faster lookup
    blocks_by_page: dict[int, list[TextBlock]] = {}
    for b in blocks:
        blocks_by_page.setdefault(b.page, []).append(b)
    for page_blocks in blocks_by_page.values():
        page_blocks.sort(key=lambda b: b.y0)

    figures_out = []
    for cap in captions:
        m = CAPTION_RE.match(cap.text)
        if not m:
            continue
        label = m.group("label").strip()
        page = doc.load_page(cap.page)
        page_rect = page.rect
        bbox = _pick_figure_bbox(
            page_rect,
            caption=cap,
            page_blocks=blocks_by_page.get(cap.page, []),
            margin=args.margin,
            min_text_len=args.min_text_length,
            min_vertical_gap=args.min_vertical_gap,
        )
        notes = []
        if bbox is None:
            notes.append("failed_to_compute_figure_bbox")
            continue

        # Render bbox to image
        rect = fitz.Rect(*bbox)
        mat = fitz.Matrix(args.dpi / 72.0, args.dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
        img_name = f"figure_{_safe_label(label)}_p{cap.page+1}.png"
        img_path = images_dir / img_name
        pix.save(str(img_path))

        # Collect figrefs across scanned pages
        pat = _build_figref_pattern(label)
        figrefs = []
        for blk in blocks:
            # figrefs are meant to be in-text mentions; skip the caption block itself.
            if blk.page == cap.page and blk.text == cap.text:
                continue
            if pat.search(blk.text):
                figrefs.append({"page": int(blk.page + 1), "text": blk.text})

        figures_out.append(
            {
                "label": label,
                "page": int(cap.page + 1),
                "caption": cap.text,
                "caption_bbox": [cap.x0, cap.y0, cap.x1, cap.y1],
                "figure_bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
                "image_path": str(Path("images") / img_name),
                "figrefs": figrefs,
                "notes": ";".join(notes) if notes else "",
            }
        )

    out_obj = {
        "pdf": str(pdf_path),
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "images_dir": "images",
        "figures": figures_out,
    }
    figures_json.write_text(json.dumps(out_obj, indent=2), encoding="utf-8")
    print(f"[OK] Wrote: {figures_json}")
    print(f"[OK] Images: {images_dir} ({len(figures_out)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
