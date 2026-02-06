#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_records(path: Path):
    text = path.read_text(encoding="utf-8")
    text = text.strip()
    if not text:
        return []
    if text.startswith("{") or text.startswith("["):
        data = json.loads(text)
        if isinstance(data, dict) and "records" in data:
            return data["records"]
        if isinstance(data, list):
            return data
        return [data]
    # JSONL fallback
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize token and cost usage records.")
    ap.add_argument("--in", dest="in_path", required=True, help="Input JSON or JSONL file.")
    ap.add_argument("--out", required=True, help="Output JSON summary.")
    args = ap.parse_args()

    records = _load_records(Path(args.in_path))
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
        "by_model": {},
        "records": len(records),
    }

    for rec in records:
        if not isinstance(rec, dict):
            continue
        model = rec.get("model", "unknown")
        pt = int(rec.get("prompt_tokens", 0) or 0)
        ct = int(rec.get("completion_tokens", 0) or 0)
        tt = int(rec.get("total_tokens", pt + ct) or 0)
        cost = float(rec.get("cost", 0.0) or 0.0)
        totals["prompt_tokens"] += pt
        totals["completion_tokens"] += ct
        totals["total_tokens"] += tt
        totals["cost"] += cost
        by = totals["by_model"].setdefault(model, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0})
        by["prompt_tokens"] += pt
        by["completion_tokens"] += ct
        by["total_tokens"] += tt
        by["cost"] += cost

    Path(args.out).write_text(json.dumps(totals, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
