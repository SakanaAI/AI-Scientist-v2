#!/usr/bin/env python3
"""
Inventory .npy files under a directory.

Requires: numpy
Safe by default: uses mmap and avoids reading full arrays unless stats are requested.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any


def _walk(base: Path) -> list[Path]:
    out: list[Path] = []
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".venv", "venv"}]
        for fn in files:
            if fn.lower().endswith(".npy"):
                out.append(Path(root) / fn)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Inventory .npy files (safe, read-only).")
    ap.add_argument("--dir", required=True, help="Directory to scan for .npy files.")
    ap.add_argument("--out", required=True, help="Output inventory JSON path.")
    ap.add_argument(
        "--max-bytes",
        type=int,
        default=None,
        help="If set, skip .npy files larger than this many bytes.",
    )
    ap.add_argument(
        "--mmap",
        dest="mmap",
        action="store_true",
        help="Use mmap_mode='r' for numpy.load (default).",
    )
    ap.add_argument(
        "--no-mmap",
        dest="mmap",
        action="store_false",
        help="Disable mmap (may OOM on large arrays).",
    )
    ap.add_argument(
        "--stats",
        action="store_true",
        help="Compute lightweight stats for small arrays (bounded by --max-elements).",
    )
    ap.add_argument(
        "--max-elements",
        type=int,
        default=1_000_000,
        help="Max elements allowed for stats (default: 1,000,000).",
    )
    ap.set_defaults(mmap=True)
    args = ap.parse_args(argv)

    base = Path(args.dir).expanduser().resolve()
    if not base.exists() or not base.is_dir():
        print(f"[ERROR] Not a directory: {base}")
        return 2

    try:
        import numpy as np  # type: ignore
    except Exception as e:
        print("[ERROR] numpy is required. Try: uv run --with numpy -s scripts/npy_inventory.py --help")
        print(f"Details: {e}")
        return 3

    mmap_mode = "r" if args.mmap else None
    entries: list[dict[str, Any]] = []
    for path in sorted(_walk(base)):
        rel = str(path.relative_to(base))
        size_bytes = path.stat().st_size if path.exists() else None
        info: dict[str, Any] = {
            "rel_path": rel,
            "abs_path": str(path),
            "size_bytes": size_bytes,
        }
        if args.max_bytes is not None and size_bytes is not None and size_bytes > args.max_bytes:
            info["skipped"] = f"size_bytes {size_bytes} > max_bytes {args.max_bytes}"
            entries.append(info)
            continue
        try:
            arr = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
            info["dtype"] = str(arr.dtype)
            info["shape"] = list(arr.shape)
            info["ndim"] = int(arr.ndim)
            if args.stats:
                n = int(arr.size)
                if n <= max(1, args.max_elements):
                    try:
                        # Avoid materializing if possible; numpy may still scan the array.
                        finite = arr[np.isfinite(arr)]
                        info["stats"] = {
                            "min": float(np.min(finite)),
                            "max": float(np.max(finite)),
                            "mean": float(np.mean(finite)),
                        }
                    except Exception as e:
                        info["stats"] = {"error": str(e)}
                else:
                    info["stats"] = {"skipped": f"size {n} > max_elements"}
        except Exception as e:
            info["error"] = str(e)
        entries.append(info)

    out_obj = {
        "base_dir": str(base),
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "entries": entries,
    }
    Path(args.out).write_text(json.dumps(out_obj, indent=2), encoding="utf-8")
    print(f"[OK] Wrote inventory: {args.out} ({len(entries)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
