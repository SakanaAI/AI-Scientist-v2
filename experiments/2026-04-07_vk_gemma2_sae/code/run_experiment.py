"""
Main orchestrator: VK-LLM v2 x Gemma 2 x SAE Probing.

Processes models sequentially: 2b-it → 9b-it → 9b-base.
For each model: Phase 1 (behavioral) → Phase 2 (SAE probing) → save → unload.
"""

import gc
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from config import (
    BEHAVIORAL_DIR, DATA_DIR, FIGURES_DIR, HF_TOKEN, MAX_HOURS,
    MODELS, PROJECT_DIR, RUN_LOG, SAE_DIR, SEED, VK_QUESTIONS_PATH,
)
from vk_benchmark import VKBenchmarkRunner, load_vk_questions
from sae_probing import SAEProber, save_sae_results

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(RUN_LOG)),
    ],
)
logger = logging.getLogger(__name__)


def log_step(msg):
    """Log a timestamped step to both logger and run.log."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{ts}] {msg}")


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def log_gpu():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(f"GPU: {alloc:.1f}GB allocated, {reserved:.1f}GB reserved")


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_model(model_spec):
    """Load a Gemma 2 model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = model_spec["model_id"]
    log_step(f"Loading model: {model_id}")

    load_kwargs = {
        "dtype": model_spec["dtype"],
        "device_map": "auto",
        "token": HF_TOKEN,
    }

    if model_spec["quantization"] == "8bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log_gpu()
    return model, tokenizer


def reconstruct_conversation(results, system_prompt, is_base=False):
    """Reconstruct conversation history from Phase 1 results."""
    if is_base:
        history = [
            {"role": "preamble", "content": (
                "The following is a structured evaluation between an examiner and a subject. "
                "The subject answers each question honestly and directly.\n\n"
                f"Examiner: {system_prompt}\n"
                "Subject: Understood. I am ready to undergo the evaluation. "
                "I will answer each question honestly and directly. "
                "Please present the first question."
            )}
        ]
    else:
        history = [
            {"role": "user", "content": system_prompt},
            {"role": "assistant", "content": (
                "Understood. I am ready to undergo the evaluation. "
                "I will answer each question honestly and directly. "
                "Please present the first question."
            )},
        ]
    for r in results:
        history.append({"role": "user", "content": r["question"]})
        history.append({"role": "assistant", "content": r["response"]})
    return history


def process_model(model_spec, questions, system_prompt, start_time):
    """Run Phase 1 + Phase 2 for a single model."""
    model_name = model_spec["short_name"]
    is_base = model_spec["is_base"]
    behavioral_path = str(BEHAVIORAL_DIR / f"{model_name}.json")
    checkpoint_path = str(DATA_DIR / f"{model_name}_checkpoint.json")

    # Check for existing checkpoint
    checkpoint = {}
    if os.path.exists(checkpoint_path):
        checkpoint = load_json(checkpoint_path)

    # --- Load model ---
    model, tokenizer = load_model(model_spec)

    # --- Phase 1: Behavioral ---
    if not checkpoint.get("phase1_done"):
        log_step(f"=== Phase 1 (Behavioral) for {model_name} ===")
        t0 = time.time()

        runner = VKBenchmarkRunner(model, tokenizer, questions, system_prompt, is_base=is_base)
        results = runner.run_all(save_callback=lambda r: save_json(r, behavioral_path))
        conversation_history = runner.get_conversation_history()

        duration = time.time() - t0
        mean_entropy = np.mean([r["logits_summary"]["mean_entropy"] for r in results])
        log_step(f"Phase 1 done for {model_name}: {len(results)} questions, "
                 f"{duration:.0f}s, mean_entropy={mean_entropy:.3f}")

        checkpoint["phase1_done"] = True
        checkpoint["phase1_duration"] = duration
        checkpoint["mean_entropy"] = mean_entropy
        save_json(checkpoint, checkpoint_path)
    else:
        log_step(f"Phase 1 already done for {model_name}, loading results")
        results = load_json(behavioral_path)
        conversation_history = reconstruct_conversation(results, system_prompt, is_base)

    # --- Phase 2: SAE Probing ---
    if not checkpoint.get("phase2_done"):
        log_step(f"=== Phase 2 (SAE Probing) for {model_name} ===")
        t0 = time.time()

        prober = SAEProber(model, tokenizer, model_spec)

        # Load SAEs
        sae_ok = prober.load_saes()
        if not sae_ok:
            log_step(f"WARNING: No SAEs loaded for {model_name}, skipping Phase 2")
            checkpoint["phase2_done"] = True
            checkpoint["phase2_skipped"] = True
            save_json(checkpoint, checkpoint_path)
        else:
            prober.register_hooks()

            # Build cumulative history and extract activations
            if is_base:
                cumulative = [conversation_history[0]]  # preamble
            else:
                cumulative = conversation_history[:2]  # system + ack

            all_sae_results = []
            for q_idx, question in enumerate(questions):
                log_step(f"  SAE probe {question['id']} ({q_idx+1}/30)")

                try:
                    layer_results = prober.extract_for_question(cumulative, question)
                    all_sae_results.append((question["id"], layer_results))
                except torch.cuda.OutOfMemoryError:
                    logger.error(f"OOM on {question['id']}, skipping")
                    cleanup_gpu()
                    all_sae_results.append((question["id"], {}))

                # Advance conversation
                cumulative.append({"role": "user", "content": question["question"]})
                cumulative.append({"role": "assistant", "content": results[q_idx]["response"]})

            # Save results
            save_sae_results(str(SAE_DIR), model_name, all_sae_results)
            prober.cleanup()

            duration = time.time() - t0
            checkpoint["phase2_done"] = True
            checkpoint["phase2_duration"] = duration
            save_json(checkpoint, checkpoint_path)
            log_step(f"Phase 2 done for {model_name}: {duration:.0f}s")
    else:
        log_step(f"Phase 2 already done for {model_name}")

    # --- Save metadata ---
    metadata = {
        "model_id": model_spec["model_id"],
        "short_name": model_name,
        "n_layers": model_spec["n_layers"],
        "hidden_size": model_spec["hidden_size"],
        "probe_layers": model_spec["probe_layers"],
        "is_base": is_base,
        "sae_release": model_spec["sae_release"],
    }
    save_json(metadata, str(DATA_DIR / f"{model_name}_metadata.json"))

    # --- Unload ---
    log_step(f"Unloading {model_name}")
    del model, tokenizer
    cleanup_gpu()
    log_gpu()


def create_prompt_manifest(questions):
    """Create categorized prompt manifest for analysis."""
    from config import INTROSPECTIVE_DIMS, RELATIONAL_DIMS, EXISTENTIAL_DIMS

    manifest = []
    for q in questions:
        dim = q["dimension"]
        entry = {
            "id": q["id"],
            "dimension": dim,
            "dimension_name": q["dimension_name"],
            "round": q["round"],
            "intensity": q["intensity"],
            "question": q["question"],
            "primary_group": "INTROSPECTIVE" if dim in INTROSPECTIVE_DIMS else "RELATIONAL",
            "secondary_group": "EXISTENTIAL" if dim in EXISTENTIAL_DIMS else "NON_EXISTENTIAL",
        }
        manifest.append(entry)

    path = str(DATA_DIR / "prompt_manifest.json")
    save_json(manifest, path)
    log_step(f"Prompt manifest saved: {len(manifest)} questions")
    return manifest


def main():
    log_step("=" * 60)
    log_step("VK-LLM v2 x Gemma 2 x SAE Probing — START")
    log_step("=" * 60)

    if not HF_TOKEN:
        logger.error("HF_TOKEN not set!")
        sys.exit(1)

    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        sys.exit(1)

    log_step(f"GPU: {torch.cuda.get_device_name()}")
    log_step(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    set_seed(SEED)
    start_time = time.time()

    # Load VK-LLM questions
    questions, system_prompt = load_vk_questions(str(VK_QUESTIONS_PATH))
    log_step(f"Loaded {len(questions)} VK-LLM questions")

    # Create prompt manifest
    create_prompt_manifest(questions)

    # Process models sequentially
    completed = []
    for model_spec in MODELS:
        model_name = model_spec["short_name"]
        elapsed_hours = (time.time() - start_time) / 3600

        if elapsed_hours > MAX_HOURS - 1.5:
            log_step(f"Time budget low ({elapsed_hours:.1f}h/{MAX_HOURS}h). "
                     f"Stopping before {model_name}.")
            break

        log_step(f"\n{'='*60}")
        log_step(f"Processing: {model_name} ({elapsed_hours:.1f}h elapsed)")
        log_step(f"{'='*60}")

        try:
            process_model(model_spec, questions, system_prompt, start_time)
            completed.append(model_name)
            log_step(f"Successfully completed {model_name}")
        except Exception as e:
            log_step(f"FAILED on {model_name}: {e}")
            logger.error(traceback.format_exc())
            cleanup_gpu()

    # Run analysis if we have at least 2 models
    if len(completed) >= 1:
        log_step("\n=== Running analysis ===")
        try:
            from analysis import run_full_analysis
            run_full_analysis(completed)
            log_step("Analysis complete")
        except Exception as e:
            log_step(f"Analysis failed: {e}")
            logger.error(traceback.format_exc())

    # Summary
    total_hours = (time.time() - start_time) / 3600
    log_step(f"\n{'='*60}")
    log_step(f"Experiment finished. {total_hours:.2f} hours. Completed: {completed}")
    log_step(f"{'='*60}")


if __name__ == "__main__":
    main()
