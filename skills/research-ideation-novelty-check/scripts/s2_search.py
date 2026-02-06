#!/usr/bin/env python3
"""
Semantic Scholar search helper (Graph API).

Default behavior is safe/offline unless you explicitly pass --online.
Reads API key from env var: S2_API_KEY (optional; improves rate limits).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
DEFAULT_FIELDS = "title,authors,venue,year,abstract,citationCount,url"


def _format_text(papers: list[dict]) -> str:
    lines: list[str] = []
    for i, p in enumerate(papers, start=1):
        authors = ", ".join(a.get("name", "Unknown") for a in p.get("authors", []) or [])
        lines.append(
            f"{i}: {p.get('title','')}\n"
            f"   Authors: {authors}\n"
            f"   Venue: {p.get('venue','')} ({p.get('year','')})\n"
            f"   Citations: {p.get('citationCount','')}\n"
            f"   URL: {p.get('url','')}\n"
            f"   Abstract: {p.get('abstract','')}\n"
        )
    return "\n".join(lines).strip() + "\n"


def _http_get_json(url: str, api_key: str | None, timeout: float) -> dict:
    req = urllib.request.Request(url, method="GET")
    req.add_header("User-Agent", "codex-s2-search/0.1")
    if api_key:
        req.add_header("X-API-KEY", api_key)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        raise RuntimeError(f"HTTP {e.code}: {body[:500]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error: {e}") from e
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise RuntimeError("Failed to parse JSON response") from e


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Search Semantic Scholar (Graph API).")
    p.add_argument("--query", required=True, help="Search query string.")
    p.add_argument("--limit", type=int, default=10, help="Max results (default: 10).")
    p.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text).",
    )
    p.add_argument("--out", help="Write output to this file (default: stdout).")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--online",
        action="store_true",
        help="Make a network request to Semantic Scholar (explicit opt-in).",
    )
    mode.add_argument(
        "--offline",
        action="store_true",
        help="Do not make network requests (default).",
    )
    p.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout seconds.")
    args = p.parse_args(argv)

    if not args.online:
        msg = (
            "Offline mode (default): no request sent.\n"
            f"Query: {args.query!r}\n"
            "Re-run with --online to query Semantic Scholar.\n"
        )
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(msg)
        else:
            sys.stdout.write(msg)
        return 0

    params = {
        "query": args.query,
        "limit": str(max(1, min(args.limit, 100))),
        "fields": DEFAULT_FIELDS,
    }
    url = API_URL + "?" + urllib.parse.urlencode(params)
    api_key = os.environ.get("S2_API_KEY")
    payload = _http_get_json(url, api_key=api_key, timeout=args.timeout)

    out_obj = {
        "query": args.query,
        "retrieved_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "total": payload.get("total"),
        "papers": payload.get("data", []),
    }

    if args.format == "json":
        out_text = json.dumps(out_obj, indent=2, ensure_ascii=False) + "\n"
    else:
        out_text = _format_text(out_obj["papers"])

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out_text)
    else:
        sys.stdout.write(out_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
