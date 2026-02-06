#!/usr/bin/env python3
"""Run BFTS experiments using the standalone runner package."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Run BFTS experiments (standalone).")
    ap.add_argument("--config", required=True, help="Path to bfts_config.yaml")
    ap.add_argument(
        "--online",
        action="store_true",
        help="Allow network calls to LLM providers (default: offline).",
    )
    args = ap.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}")
        return 2

    # Ensure package import
    scripts_dir = Path(__file__).parent.resolve()
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    # Default root for experiment data
    os.environ.setdefault("AI_SCIENTIST_SKILLS_ROOT", str(config_path.parent))
    os.environ.setdefault("AI_SCIENTIST_ROOT", str(config_path.parent))
    if args.online:
        os.environ["ASV2_ONLINE"] = "1"

    from asv2.treesearch.perform_experiments_bfts_with_agentmanager import (
        perform_experiments_bfts,
    )

    perform_experiments_bfts(str(config_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
