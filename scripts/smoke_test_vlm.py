#!/usr/bin/env python3
"""
One-shot smoke test: call the same VLM path as the paper pipeline
(create_client + get_response_from_vlm) with a single image.

Exit code 0 only if the API returns a non-empty text reply.
Does not run experiments or writeup.

Examples:
  export QWEN_API_KEY=...
  python scripts/smoke_test_vlm.py

  python scripts/smoke_test_vlm.py --model ollama/qwen3-vl:32b --image path/to/fig.png

Defaults use repo ``docs/logo_v1.png`` and ``qwen/qwen3-vl-plus`` (paths are resolved from repo root).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


# Stable image shipped with the repo (not tied to one experiment folder).
DEFAULT_SMOKE_IMAGE = "docs/logo_v1.png"
DEFAULT_SMOKE_MODEL = "qwen/qwen3-vl-plus"


def _default_image_path() -> str:
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        from PIL import Image

        img = Image.new("RGB", (128, 128), color=(90, 120, 200))
        img.save(path, format="PNG")
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test VLM (one image, one request).")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_SMOKE_MODEL,
        help="VLM id, or 'auto' (DashScope from QWEN_VLM_MODEL + QWEN_API_KEY).",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=DEFAULT_SMOKE_IMAGE,
        help=(
            "Image path (relative to repo root if not absolute). "
            "Default: docs/logo_v1.png. If that file is missing, a tiny synthetic PNG is used."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print request/response debug (same flag as get_response_from_vlm print_debug).",
    )
    args = parser.parse_args()

    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("smoke_test_vlm")

    os.environ.setdefault("AI_SCIENTIST_ROOT", str(root))

    from ai_scientist.vlm import create_client, get_response_from_vlm, resolve_vlm_model

    try:
        model = resolve_vlm_model(args.model)
    except Exception as e:
        log.error("resolve_vlm_model failed: %s", e)
        return 1

    cleanup: str | None = None
    img_arg = Path(args.image)
    if not img_arg.is_absolute():
        candidate = (root / img_arg).resolve()
    else:
        candidate = img_arg.resolve()

    if candidate.is_file():
        image_path = str(candidate)
        log.info("Using image: %s", image_path)
    else:
        log.warning("Image not found at %s — using a temporary synthetic PNG.", candidate)
        try:
            image_path = _default_image_path()
            cleanup = image_path
            log.info("Synthetic test image: %s", image_path)
        except Exception:
            log.exception("Could not create a temporary PNG (need Pillow).")
            return 1

    prompt = (
        "Describe this image in one short English sentence "
        "(main colors or content). Keep under 30 words."
    )

    try:
        client, model_resolved = create_client(model)
        text, _hist = get_response_from_vlm(
            prompt,
            image_path,
            client,
            model_resolved,
            system_message="You are a helpful vision assistant.",
            print_debug=args.verbose,
        )
    except KeyboardInterrupt:
        log.error("Interrupted.")
        return 130
    except Exception:
        log.exception("VLM request failed (network, auth, or API error).")
        return 1
    finally:
        if cleanup:
            try:
                os.unlink(cleanup)
            except OSError:
                pass

    if text is None or not str(text).strip():
        log.error("API returned success but message content is empty.")
        return 1

    print()
    print("SUCCESS: VLM call completed and non-empty text was received.")
    print("--- model:", model_resolved)
    print("--- reply (first 800 chars):")
    print(str(text).strip()[:800])
    print("---")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
