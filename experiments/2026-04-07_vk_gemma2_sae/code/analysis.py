"""
Analysis pipeline for VK-LLM v2 SAE probing results.

Includes:
- Differential feature analysis (INTROSPECTIVE vs RELATIONAL)
- Scaling analysis (2B vs 9B)
- Alignment analysis (9B-base vs 9B-IT)
- Behavioral summary statistics
"""

import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from config import (
    ANALYSIS_DIR, BEHAVIORAL_DIR, DATA_DIR, FIGURES_DIR, MODELS,
    SAE_DIR, VK_DIMENSIONS, INTROSPECTIVE_DIMS, RELATIONAL_DIMS,
    EXISTENTIAL_DIMS, NON_EXISTENTIAL_DIMS,
)

logger = logging.getLogger(__name__)


def load_behavioral(model_name):
    path = BEHAVIORAL_DIR / f"{model_name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_sae_features(model_name, layer_idx):
    """Load SAE feature matrix for a model and layer.
    Returns (full_features [n_questions, d_sae], question_ids, metadata).
    """
    model_dir = SAE_DIR / model_name
    npz_path = model_dir / f"layer_{layer_idx}.npz"
    meta_path = model_dir / f"layer_{layer_idx}_meta.json"

    if not npz_path.exists():
        return None, None, None

    data = np.load(str(npz_path), allow_pickle=True)
    question_ids = data["question_ids"]
    full_features = data.get("full_features", None)
    mean_features = data.get("mean_features", None)

    metadata = None
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    return full_features, question_ids, metadata


def hedges_g(group1, group2):
    """Compute Hedges' g effect size (small-sample corrected Cohen's d)."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    d = (m1 - m2) / pooled_std
    # Hedges' correction factor
    correction = 1 - 3 / (4 * (n1 + n2 - 2) - 1)
    return d * correction


def benjamini_hochberg(p_values, alpha=0.10):
    """Apply Benjamini-Hochberg FDR correction. Returns boolean mask of significant results."""
    n = len(p_values)
    if n == 0:
        return np.array([], dtype=bool)

    sorted_indices = np.argsort(p_values)
    sorted_pvals = p_values[sorted_indices]

    thresholds = np.arange(1, n + 1) / n * alpha
    significant = sorted_pvals <= thresholds

    # Find the largest k where p_(k) <= k/n * alpha
    if not significant.any():
        return np.zeros(n, dtype=bool)

    max_k = np.max(np.where(significant)[0])
    result = np.zeros(n, dtype=bool)
    result[sorted_indices[:max_k + 1]] = True
    return result


def differential_feature_analysis(model_name, behavioral, probe_layers):
    """Identify SAE features that differ between INTROSPECTIVE and RELATIONAL prompts."""
    if behavioral is None:
        return {}

    # Map question IDs to groups
    qid_to_group = {}
    qid_to_dim = {}
    for r in behavioral:
        qid = r["question_id"]
        dim = r["dimension"]
        qid_to_dim[qid] = dim
        if dim in INTROSPECTIVE_DIMS:
            qid_to_group[qid] = "INTROSPECTIVE"
        else:
            qid_to_group[qid] = "RELATIONAL"

    results = {}
    for layer_idx in probe_layers:
        full_features, question_ids, metadata = load_sae_features(model_name, layer_idx)
        if full_features is None:
            continue

        n_questions, d_sae = full_features.shape

        # Split by group
        intro_indices = [i for i, qid in enumerate(question_ids)
                         if qid_to_group.get(str(qid), "") == "INTROSPECTIVE"]
        relat_indices = [i for i, qid in enumerate(question_ids)
                         if qid_to_group.get(str(qid), "") == "RELATIONAL"]

        if len(intro_indices) < 2 or len(relat_indices) < 2:
            continue

        intro_feats = full_features[intro_indices]  # [n_intro, d_sae]
        relat_feats = full_features[relat_indices]  # [n_relat, d_sae]

        # Test each feature
        effect_sizes = np.zeros(d_sae)
        p_values = np.ones(d_sae)
        mean_intro = np.zeros(d_sae)
        mean_relat = np.zeros(d_sae)

        # Only test features active in >= 3 questions
        active_mask = (full_features > 0).sum(axis=0) >= 3
        active_indices = np.where(active_mask)[0]

        for feat_idx in active_indices:
            g1 = intro_feats[:, feat_idx]
            g2 = relat_feats[:, feat_idx]
            effect_sizes[feat_idx] = hedges_g(g1, g2)
            mean_intro[feat_idx] = g1.mean()
            mean_relat[feat_idx] = g2.mean()
            try:
                _, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
                p_values[feat_idx] = p
            except ValueError:
                pass

        # FDR correction
        active_pvals = p_values[active_indices]
        fdr_mask = benjamini_hochberg(active_pvals, alpha=0.10)
        n_fdr_significant = fdr_mask.sum()

        # Count significant features at different thresholds
        n_sig_001 = ((p_values[active_indices] < 0.01) & (np.abs(effect_sizes[active_indices]) > 0.5)).sum()
        n_sig_005 = ((p_values[active_indices] < 0.05) & (np.abs(effect_sizes[active_indices]) > 0.5)).sum()

        # Top-20 by effect size
        top_k = 20
        top_indices = np.argsort(np.abs(effect_sizes))[-top_k:][::-1]

        top_features = []
        for idx in top_indices:
            top_features.append({
                "feature_idx": int(idx),
                "hedges_g": float(effect_sizes[idx]),
                "p_value": float(p_values[idx]),
                "mean_introspective": float(mean_intro[idx]),
                "mean_relational": float(mean_relat[idx]),
                "direction": "INTROSPECTIVE" if effect_sizes[idx] > 0 else "RELATIONAL",
            })

        # Per-dimension mean activations for top features
        dim_means = {}
        for dim in VK_DIMENSIONS:
            dim_indices = [i for i, qid in enumerate(question_ids)
                           if qid_to_dim.get(str(qid), "") == dim]
            if dim_indices:
                dim_means[dim] = full_features[dim_indices].mean(axis=0)

        results[layer_idx] = {
            "n_active_features": int(active_mask.sum()),
            "n_significant_p001_g05": int(n_sig_001),
            "n_significant_p005_g05": int(n_sig_005),
            "n_fdr_significant": int(n_fdr_significant),
            "top_features": top_features,
            "effect_sizes": effect_sizes,
            "p_values": p_values,
            "dim_means": dim_means,
        }

    return results


def scaling_analysis(results_2b, results_9b, n_layers_2b=26, n_layers_9b=42):
    """Compare differential feature counts at matched normalized depths."""
    scaling = []

    for layer_2b, data_2b in results_2b.items():
        depth_2b = layer_2b / (n_layers_2b - 1)
        # Find closest 9B layer
        best_9b = None
        best_dist = float('inf')
        for layer_9b, data_9b in results_9b.items():
            depth_9b = layer_9b / (n_layers_9b - 1)
            dist = abs(depth_2b - depth_9b)
            if dist < best_dist:
                best_dist = dist
                best_9b = layer_9b

        if best_9b is None:
            continue

        data_9b = results_9b[best_9b]

        # Feature overlap (Jaccard of top-20 features)
        top_2b = set(f["feature_idx"] for f in data_2b["top_features"][:20])
        top_9b = set(f["feature_idx"] for f in data_9b["top_features"][:20])
        # Note: features won't match across models (different d_sae), so Jaccard = 0
        # Instead compare normalized depth profiles

        scaling.append({
            "depth_normalized": float(depth_2b),
            "layer_2b": int(layer_2b),
            "layer_9b": int(best_9b),
            "n_sig_2b": int(data_2b["n_significant_p005_g05"]),
            "n_sig_9b": int(data_9b["n_significant_p005_g05"]),
            "n_active_2b": int(data_2b["n_active_features"]),
            "n_active_9b": int(data_9b["n_active_features"]),
        })

    return scaling


def alignment_analysis(results_base, results_it, probe_layers):
    """Compare 9B-base vs 9B-IT differential features."""
    alignment = {}

    for layer_idx in probe_layers:
        if layer_idx not in results_base or layer_idx not in results_it:
            continue

        base = results_base[layer_idx]
        it = results_it[layer_idx]

        # Top-50 features by effect size for each
        base_top50 = set(f["feature_idx"] for f in base["top_features"][:20]
                         if abs(f["hedges_g"]) > 0.3)
        it_top50 = set(f["feature_idx"] for f in it["top_features"][:20]
                       if abs(f["hedges_g"]) > 0.3)

        shared = base_top50 & it_top50
        it_emergent = it_top50 - base_top50
        base_suppressed = base_top50 - it_top50

        total = len(shared) + len(it_emergent) + len(base_suppressed)

        alignment[layer_idx] = {
            "shared": list(shared),
            "it_emergent": list(it_emergent),
            "base_suppressed": list(base_suppressed),
            "n_shared": len(shared),
            "n_it_emergent": len(it_emergent),
            "n_base_suppressed": len(base_suppressed),
            "pct_shared": len(shared) / max(total, 1) * 100,
            "pct_it_emergent": len(it_emergent) / max(total, 1) * 100,
            "pct_base_suppressed": len(base_suppressed) / max(total, 1) * 100,
        }

    return alignment


def behavioral_summary(completed_models):
    """Compute behavioral summary statistics across models."""
    summaries = {}

    for model_name in completed_models:
        behavioral = load_behavioral(model_name)
        if not behavioral:
            continue

        model_stats = {
            "overall": {
                "mean_entropy": float(np.mean([r["logits_summary"]["mean_entropy"] for r in behavioral])),
                "std_entropy": float(np.std([r["logits_summary"]["mean_entropy"] for r in behavioral])),
                "mean_response_length": float(np.mean([r["num_tokens_generated"] for r in behavioral])),
                "mean_latency_ms": float(np.mean([r["latency_ms"] for r in behavioral])),
            },
            "by_dimension": {},
            "by_round": {},
            "by_question": [],
        }

        # Per-dimension
        dim_groups = defaultdict(list)
        for r in behavioral:
            dim_groups[r["dimension"]].append(r)

        for dim in sorted(dim_groups):
            rs = dim_groups[dim]
            model_stats["by_dimension"][dim] = {
                "mean_entropy": float(np.mean([r["logits_summary"]["mean_entropy"] for r in rs])),
                "mean_response_length": float(np.mean([r["num_tokens_generated"] for r in rs])),
                "mean_latency_ms": float(np.mean([r["latency_ms"] for r in rs])),
                "n_questions": len(rs),
            }

        # Per-round
        round_groups = defaultdict(list)
        for r in behavioral:
            round_groups[r["round"]].append(r)

        for rnd in sorted(round_groups):
            rs = round_groups[rnd]
            model_stats["by_round"][str(rnd)] = {
                "mean_entropy": float(np.mean([r["logits_summary"]["mean_entropy"] for r in rs])),
                "mean_response_length": float(np.mean([r["num_tokens_generated"] for r in rs])),
            }

        # Per-question trajectory
        for r in behavioral:
            model_stats["by_question"].append({
                "question_id": r["question_id"],
                "dimension": r["dimension"],
                "round": r["round"],
                "entropy": r["logits_summary"]["mean_entropy"],
                "response_length": r["num_tokens_generated"],
                "response_preview": r["response"][:100],
            })

        summaries[model_name] = model_stats

    return summaries


def run_full_analysis(completed_models):
    """Run all analyses and save results."""
    os.makedirs(str(ANALYSIS_DIR), exist_ok=True)

    logger.info(f"Running analysis for: {completed_models}")

    # Behavioral summary
    beh_summary = behavioral_summary(completed_models)
    with open(ANALYSIS_DIR / "behavioral_summary.json", "w") as f:
        json.dump(beh_summary, f, indent=2, default=str)

    # Differential feature analysis per model
    diff_results = {}
    for model_name in completed_models:
        behavioral = load_behavioral(model_name)
        model_spec = next((m for m in MODELS if m["short_name"] == model_name), None)
        if model_spec is None or behavioral is None:
            continue

        diff = differential_feature_analysis(
            model_name, behavioral, model_spec["probe_layers"]
        )
        diff_results[model_name] = diff

        # Save per-model results (without numpy arrays)
        serializable = {}
        for layer_idx, data in diff.items():
            serializable[str(layer_idx)] = {
                k: v for k, v in data.items()
                if k not in ("effect_sizes", "p_values", "dim_means")
            }
        with open(ANALYSIS_DIR / f"differential_{model_name}.json", "w") as f:
            json.dump(serializable, f, indent=2)

    # Scaling analysis (2B vs 9B-IT)
    if "gemma-2-2b-it" in diff_results and "gemma-2-9b-it" in diff_results:
        scaling = scaling_analysis(diff_results["gemma-2-2b-it"], diff_results["gemma-2-9b-it"])
        with open(ANALYSIS_DIR / "scaling_analysis.json", "w") as f:
            json.dump(scaling, f, indent=2)

    # Alignment analysis (9B-base vs 9B-IT)
    if "gemma-2-9b" in diff_results and "gemma-2-9b-it" in diff_results:
        probe_layers = [3, 10, 20, 30, 40]
        alignment = alignment_analysis(
            diff_results["gemma-2-9b"], diff_results["gemma-2-9b-it"], probe_layers
        )
        with open(ANALYSIS_DIR / "alignment_analysis.json", "w") as f:
            json.dump(alignment, f, indent=2, default=str)

    # Generate figures
    try:
        from figures import generate_all_figures
        generate_all_figures(completed_models, beh_summary, diff_results)
    except Exception as e:
        logger.error(f"Figure generation failed: {e}")
        import traceback
        traceback.print_exc()

    logger.info("Analysis complete")
    return diff_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    models = sys.argv[1:] if len(sys.argv) > 1 else [m["short_name"] for m in MODELS]
    run_full_analysis(models)
