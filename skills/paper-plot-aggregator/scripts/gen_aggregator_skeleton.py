#!/usr/bin/env python3
"""
Generate a runnable plot aggregator skeleton from an inventory.json file.

This generator is intentionally dependency-light. The generated script will
require numpy + matplotlib to actually plot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TEMPLATE = '''#!/usr/bin/env python3
"""
Auto-generated plot aggregator skeleton.

This script is intended as a starting point. It will:
- Load existing .npy data only (no hallucinated numbers).
- Write figures into a figures directory.
- Generate a few \"quicklook\" plots for the first N arrays.

To run (recommended):
  uv run --with numpy --with matplotlib -s auto_plot_aggregator.py --figures-dir "__FIGURES_DIR__"
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np


DEFAULT_CLEAN = __DEFAULT_CLEAN__
DEFAULT_MAX_PLOTS = __MAX_PLOTS__

INVENTORY_JSON = r"""__INVENTORY_JSON__"""
INVENTORY = json.loads(INVENTORY_JSON)


def _safe_label(text: str) -> str:
    # Avoid underscores in labels (better for papers).
    return text.replace("_", " ")


def _resolve_path(entry: dict) -> Path:
    # Prefer absolute path when available, otherwise resolve relative to base_dir.
    if entry.get("abs_path"):
        p = Path(entry["abs_path"])
        if p.exists():
            return p
    base_dir = INVENTORY.get("base_dir") or os.getcwd()
    return Path(base_dir) / entry.get("rel_path", "")


def _ensure_figures_dir(figures_dir: Path, clean: bool) -> Path:
    # Resolve against CWD if relative.
    if not figures_dir.is_absolute():
        figures_dir = (Path(os.getcwd()) / figures_dir).resolve()
    else:
        figures_dir = figures_dir.resolve()

    # Safety: only allow cleaning inside CWD.
    if clean:
        cwd = Path(os.getcwd()).resolve()
        if cwd not in figures_dir.parents:
            raise RuntimeError(f\"Refusing to clean figures outside CWD: {figures_dir}\")
        if figures_dir.exists():
            shutil.rmtree(figures_dir)

    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def _quicklook_plot(entry: dict, figures_dir: Path, idx: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "matplotlib is required to plot. Install with: uv run --with matplotlib ..."
        ) from e

    path = _resolve_path(entry)
    if not path.exists():
        raise FileNotFoundError(str(path))

    arr = np.load(path, mmap_mode="r", allow_pickle=False)

    name = Path(entry.get("rel_path") or path.name).stem
    name = _safe_label(name)

    fig = plt.figure(figsize=(6, 4))
    if arr.ndim == 1:
        plt.plot(arr)
        plt.xlabel("index")
        plt.ylabel("value")
        plt.title(f\"{name} (1D)\")
    elif arr.ndim == 2:
        # Quicklook: show a heatmap for 2D arrays (may not be paper-ready).
        plt.imshow(arr, aspect="auto")
        plt.colorbar()
        plt.title(f\"{name} (2D heatmap)\")
    else:
        plt.text(0.1, 0.5, f\"ndim={arr.ndim} not plotted\", fontsize=12)
        plt.axis("off")
        plt.title(f\"{name} (skipped)\")

    out_name = f\"{idx:02d} - {name}.png\"
    out_path = figures_dir / out_name
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the auto plot aggregator.")
    ap.add_argument("--figures-dir", default="__FIGURES_DIR__", help="Output figures directory.")
    ap.add_argument(
        "--clean",
        dest=\"clean\",
        action=\"store_true\",
        default=DEFAULT_CLEAN,
        help="Clean figures-dir before plotting.",
    )
    ap.add_argument(
        "--no-clean",
        dest=\"clean\",
        action=\"store_false\",
        help="Do not clean figures-dir before plotting.",
    )
    ap.add_argument(
        "--max-plots",
        type=int,
        default=DEFAULT_MAX_PLOTS,
        help="Max quicklook plots to attempt.",
    )
    args = ap.parse_args()

    figures_dir = _ensure_figures_dir(Path(args.figures_dir), clean=bool(args.clean))

    entries = INVENTORY.get("entries", [])
    if not isinstance(entries, list):
        raise RuntimeError("INVENTORY.entries must be a list")

    n = max(0, min(int(args.max_plots), len(entries)))
    for i in range(n):
        try:
            _quicklook_plot(entries[i], figures_dir, i + 1)
        except Exception as e:
            print(f\"[WARN] Plot {i+1} failed: {e}\")

    print(f\"[OK] Wrote figures to: {figures_dir}\")
    return 0


if __name__ == \"__main__\":
    raise SystemExit(main())
'''


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate auto_plot_aggregator.py skeleton.")
    ap.add_argument("--inventory", required=True, help="Input inventory.json from npy_inventory.py.")
    ap.add_argument("--out", required=True, help="Output path for auto_plot_aggregator.py.")
    ap.add_argument("--figures-dir", default="figures", help="Figures directory (default: figures).")
    ap.add_argument(
        "--clean",
        action="store_true",
        help="Set generated script default to clean the figures dir before plotting (default: false).",
    )
    ap.add_argument(
        "--max-plots",
        type=int,
        default=12,
        help="Max quicklook plots the generated script will attempt (default: 12).",
    )
    args = ap.parse_args(argv)

    inv_path = Path(args.inventory).expanduser().resolve()
    try:
        inv = json.loads(inv_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[ERROR] Failed to read inventory JSON: {inv_path}: {e}")
        return 2

    entries = inv.get("entries", [])
    if not isinstance(entries, list):
        print("[ERROR] inventory.entries must be a list")
        return 2

    embedded = {
        "base_dir": inv.get("base_dir"),
        "generated_at": inv.get("generated_at"),
        "entries": [
            {
                "rel_path": e.get("rel_path"),
                "abs_path": e.get("abs_path"),
                "shape": e.get("shape"),
                "dtype": e.get("dtype"),
            }
            for e in entries
            if isinstance(e, dict) and (e.get("rel_path") or e.get("abs_path"))
        ],
    }

    inventory_json = json.dumps(embedded, indent=2)
    script = (
        TEMPLATE.replace("__FIGURES_DIR__", args.figures_dir)
        .replace("__DEFAULT_CLEAN__", "True" if args.clean else "False")
        .replace("__MAX_PLOTS__", str(int(args.max_plots)))
        .replace("__INVENTORY_JSON__", inventory_json)
    )

    out_path = Path(args.out).expanduser().resolve()
    out_path.write_text(script, encoding="utf-8")
    print(f"[OK] Wrote skeleton: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
