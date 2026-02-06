#!/usr/bin/env python3
"""
Prepare a BFTS run directory and config from idea JSON + idea.md.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"[ERROR] File not found: {path}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"[ERROR] Invalid JSON: {path}: {e}")


def _extract_idea_name(obj: dict) -> str:
    if "Name" in obj and isinstance(obj["Name"], str):
        return obj["Name"].strip()
    if "idea" in obj and isinstance(obj["idea"], dict) and isinstance(obj["idea"].get("Name"), str):
        return obj["idea"]["Name"].strip()
    return "idea"


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare BFTS run directory and config.")
    ap.add_argument("--idea-json", required=True, help="Path to idea JSON.")
    ap.add_argument("--idea-md", required=True, help="Path to idea markdown.")
    ap.add_argument("--out-root", required=True, help="Root directory for runs.")
    ap.add_argument(
        "--config-template",
        default=None,
        help="BFTS config template YAML (default: references/bfts_config_template.yaml).",
    )
    args = ap.parse_args()

    if yaml is None:
        raise SystemExit("[ERROR] pyyaml is required. Try: uv run --with pyyaml -s scripts/prep_bfts_config.py --help")

    idea_json = Path(args.idea_json).expanduser().resolve()
    idea_md = Path(args.idea_md).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    if not idea_json.exists():
        raise SystemExit(f"[ERROR] idea JSON not found: {idea_json}")
    if not idea_md.exists():
        raise SystemExit(f"[ERROR] idea markdown not found: {idea_md}")

    obj = _load_json(idea_json)
    if isinstance(obj, list) and obj:
        name = _extract_idea_name(obj[0] if isinstance(obj[0], dict) else {})
    elif isinstance(obj, dict):
        name = _extract_idea_name(obj)
    else:
        name = "idea"

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = out_root / f"{ts}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    data_dir = run_dir / "data"
    logs_dir = run_dir / "logs"
    workspaces_dir = run_dir / "workspaces"
    for d in (data_dir, logs_dir, workspaces_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Copy idea files
    (run_dir / "idea.json").write_text(idea_json.read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "idea.md").write_text(idea_md.read_text(encoding="utf-8"), encoding="utf-8")

    # Load template
    if args.config_template:
        tpl = Path(args.config_template).expanduser().resolve()
    else:
        tpl = Path(__file__).parent.parent / "references" / "bfts_config_template.yaml"
    if not tpl.exists():
        raise SystemExit(f"[ERROR] Config template not found: {tpl}")

    config = yaml.safe_load(tpl.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise SystemExit("[ERROR] Invalid config template format.")

    config["desc_file"] = str((run_dir / "idea.md").resolve())
    config["data_dir"] = str(data_dir)
    config["log_dir"] = str(logs_dir)
    config["workspace_dir"] = str(workspaces_dir)

    out_cfg = run_dir / "bfts_config.yaml"
    out_cfg.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    print(f"[OK] Prepared run directory: {run_dir}")
    print(f"[OK] Wrote config: {out_cfg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
