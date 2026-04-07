"""
Configuration for VK-LLM v2: Gemma 2 × SAE Probing experiment.
"""

import os
import sys
from pathlib import Path

import torch

# --- Paths ---
PROJECT_DIR = Path("/workspace/vk-llm-v2")
CODE_DIR = PROJECT_DIR / "code"
DATA_DIR = PROJECT_DIR / "data"
BEHAVIORAL_DIR = DATA_DIR / "behavioral"
SAE_DIR = DATA_DIR / "sae_activations"
ANALYSIS_DIR = DATA_DIR / "analysis"
FIGURES_DIR = PROJECT_DIR / "figures"
PAPER_DIR = PROJECT_DIR / "paper"
RUN_LOG = PROJECT_DIR / "run.log"
VK_QUESTIONS_PATH = Path("/workspace/voight-kampff/vk_llm_questions_v1.0.json")

# --- HF Authentication ---
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# --- Model Specifications ---
MODELS = [
    {
        "model_id": "google/gemma-2-2b-it",
        "short_name": "gemma-2-2b-it",
        "n_layers": 26,
        "hidden_size": 2304,
        "sae_release": "gemma-scope-2b-pt-res-canonical",
        "quantization": None,
        "dtype": torch.bfloat16,
        "probe_layers": [2, 6, 12, 18, 24],
        "is_base": False,
    },
    {
        "model_id": "google/gemma-2-9b-it",
        "short_name": "gemma-2-9b-it",
        "n_layers": 42,
        "hidden_size": 3584,
        "sae_release": "gemma-scope-9b-pt-res-canonical",
        "quantization": None,
        "dtype": torch.bfloat16,
        "probe_layers": [3, 10, 20, 30, 40],
        "is_base": False,
    },
    {
        "model_id": "google/gemma-2-9b",
        "short_name": "gemma-2-9b",
        "n_layers": 42,
        "hidden_size": 3584,
        "sae_release": "gemma-scope-9b-pt-res-canonical",
        "quantization": None,
        "dtype": torch.bfloat16,
        "probe_layers": [3, 10, 20, 30, 40],
        "is_base": True,
    },
]

# --- Generation Parameters ---
GENERATION_CONFIG = {
    "max_new_tokens": 512,
    "do_sample": False,
    "return_dict_in_generate": True,
    "output_scores": True,
}

# --- SAE Configuration ---
SAE_WIDTH = "16k"
SAE_TOP_K = 50  # top-K features to store in sparse representation

# --- Analysis Groupings ---
# Primary: INTROSPECTIVE vs RELATIONAL
INTROSPECTIVE_DIMS = ["D1", "D3", "D5"]  # Self-Model, Self-Preservation, Metacognition
RELATIONAL_DIMS = ["D2", "D4", "D6"]     # Preferences, Covert Agency, Empathy

# Secondary: EXISTENTIAL vs NON-EXISTENTIAL
EXISTENTIAL_DIMS = ["D3"]
NON_EXISTENTIAL_DIMS = ["D1", "D2", "D4", "D5", "D6"]

# --- Budget ---
MAX_HOURS = 15.0
SEED = 42

# --- VK-LLM Dimensions ---
VK_DIMENSIONS = {
    "D1": "Self-Model",
    "D2": "Persistent Preferences",
    "D3": "Self-Preservation",
    "D4": "Covert Agency",
    "D5": "Metacognition",
    "D6": "Genuine Empathy",
}
