"""
launch_scientist_bfts.py imports heavy/optional dependencies (torch, anthropic,
openai, transformers, ...) at module level that are not installed in every dev
environment. Stub them out before the module is imported so tests can exercise
its plain-Python functions (e.g. find_pdf_path_for_review) in isolation.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_STUB_MODULES = [
    "torch",
    "ai_scientist.llm",
    "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager",
    "ai_scientist.treesearch.bfts_utils",
    "ai_scientist.perform_plotting",
    "ai_scientist.perform_writeup",
    "ai_scientist.perform_icbinb_writeup",
    "ai_scientist.perform_llm_review",
    "ai_scientist.perform_vlm_review",
    "ai_scientist.utils.token_tracker",
]

for _mod in _STUB_MODULES:
    sys.modules.setdefault(_mod, MagicMock())
