"""
PeerRead (allenai/peer_read) — loading reference for experiments.

Dataset card: https://huggingface.co/datasets/allenai/peer_read
Paper: Kang et al., NAACL 2018 (arXiv:1804.09635).

Authentication
--------------
This dataset is public (downloads Allen AI's GitHub archive). You do **not** need
``HF_TOKEN`` or Hugging Face login unless your org blocks anonymous hub access.

First-time setup
----------------
- Installs: ``pip install datasets`` (and network access).
- First ``load_dataset`` downloads ~1.2 GB zip; extracted + cached size is large
  (especially ``parsed_pdfs``). Set ``HF_HOME`` if you want the cache on a fast/large disk.

Loading script
--------------
The hub repo ships ``peer_read.py`` (not Parquet-only). On recent ``datasets``
versions you must pass ``trust_remote_code=True``.

Example
-------
>>> from datasets import load_dataset
>>> ds = load_dataset(
...     "allenai/peer_read",
...     "reviews",
...     trust_remote_code=True,
... )
>>> # Splits: "train", "validation", "test"  (NOT "dev")
>>> row = ds["train"][0]
>>> row["title"], row["accepted"], row["reviews"]
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

PEER_READ = "allenai/peer_read"
CFG_REVIEWS = "reviews"
CFG_PARSED_PDFS = "parsed_pdfs"


def load_peer_read(
    config: str = CFG_REVIEWS,
    *,
    trust_remote_code: bool = True,
    **kwargs: Any,
):
    """
    Load PeerRead. ``config`` is ``\"reviews\"`` (default) or ``\"parsed_pdfs\"``.

    Common kwargs: ``split="train"``, ``streaming=True`` (to avoid full RAM).
    """
    from datasets import load_dataset

    return load_dataset(
        PEER_READ,
        config,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )


def accepted_as_bool(example: dict) -> bool | None:
    """Normalize ``accepted``; hub JSON occasionally stringifies booleans."""
    v = example.get("accepted")
    if isinstance(v, bool):
        return v
    if v is None or v == "":
        return None
    s = str(v).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


def flatten_review_text(example: dict) -> str:
    """Concatenate paper fields and per-reviewer comments for text-only baselines."""
    chunks: list[str] = []
    for key in ("title", "abstract", "comments"):
        t = example.get(key)
        if t:
            chunks.append(str(t))
    for rev in example.get("reviews") or []:
        if isinstance(rev, dict):
            c = rev.get("comments")
            if c:
                chunks.append(str(c))
    return "\n\n".join(chunks)


if __name__ == "__main__":
    load_peer_read()
