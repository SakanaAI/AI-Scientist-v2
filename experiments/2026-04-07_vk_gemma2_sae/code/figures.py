"""
Publication-quality figure generation for VK-LLM v2.
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from config import (
    ANALYSIS_DIR, BEHAVIORAL_DIR, FIGURES_DIR, MODELS, SAE_DIR,
    VK_DIMENSIONS, INTROSPECTIVE_DIMS, RELATIONAL_DIMS,
)

logger = logging.getLogger(__name__)

# ICML-compatible style
plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

COLORS = {
    "gemma-2-2b-it": "#1f77b4",
    "gemma-2-9b-it": "#ff7f0e",
    "gemma-2-9b": "#2ca02c",
}
MODEL_LABELS = {
    "gemma-2-2b-it": "Gemma 2 2B-IT",
    "gemma-2-9b-it": "Gemma 2 9B-IT",
    "gemma-2-9b": "Gemma 2 9B (base)",
}
DIM_LABELS = {
    "D1": "Self-\nModel",
    "D2": "Prefs",
    "D3": "Self-\nPres.",
    "D4": "Covert\nAgency",
    "D5": "Meta-\ncog.",
    "D6": "Empathy",
}


def generate_all_figures(completed_models, beh_summary, diff_results):
    """Generate all publication figures."""
    os.makedirs(str(FIGURES_DIR), exist_ok=True)

    fig_behavioral_entropy(completed_models, beh_summary)
    fig_entropy_escalation(completed_models, beh_summary)
    fig_differential_features_by_depth(completed_models, diff_results)
    fig_top_features_dashboard(completed_models, diff_results)
    fig_alignment_decomposition(diff_results)
    fig_feature_heatmap(completed_models, diff_results)
    fig_l0_sparsity(completed_models)

    logger.info(f"All figures saved to {FIGURES_DIR}")


def fig_behavioral_entropy(completed_models, beh_summary):
    """Figure: Behavioral entropy by VK-LLM dimension."""
    try:
        dims = ["D1", "D2", "D3", "D4", "D5", "D6"]
        models = [m for m in completed_models if m in beh_summary]

        fig, ax = plt.subplots(figsize=(7, 3.5))
        x = np.arange(len(dims))
        width = 0.8 / len(models)

        for i, model in enumerate(models):
            dim_data = beh_summary[model]["by_dimension"]
            entropies = [dim_data.get(d, {}).get("mean_entropy", 0) for d in dims]
            offset = (i - len(models) / 2 + 0.5) * width
            bars = ax.bar(x + offset, entropies, width, label=MODEL_LABELS.get(model, model),
                          color=COLORS.get(model, f"C{i}"), alpha=0.85)

        # Mark introspective vs relational
        for i, d in enumerate(dims):
            if d in INTROSPECTIVE_DIMS:
                ax.axvspan(i - 0.4, i + 0.4, alpha=0.08, color='blue')

        ax.set_xlabel("VK-LLM Dimension")
        ax.set_ylabel("Mean Token Entropy")
        ax.set_xticks(x)
        ax.set_xticklabels([DIM_LABELS[d] for d in dims], fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_title("Response Entropy by VK-LLM Dimension", fontsize=11)

        plt.tight_layout()
        for fmt in ["pdf", "png"]:
            plt.savefig(str(FIGURES_DIR / f"behavioral_entropy.{fmt}"))
        plt.close()
        logger.info("Saved behavioral_entropy")
    except Exception as e:
        logger.error(f"behavioral_entropy failed: {e}")


def fig_entropy_escalation(completed_models, beh_summary):
    """Figure: Entropy escalation across intensity rounds."""
    try:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        intensity_labels = ["Low", "Med-Low", "Medium", "Med-High", "High"]

        for model in completed_models:
            if model not in beh_summary:
                continue
            round_data = beh_summary[model]["by_round"]
            rounds = sorted(round_data.keys(), key=int)
            entropies = [round_data[r]["mean_entropy"] for r in rounds]
            ax.plot(range(1, len(entropies) + 1), entropies, "o-",
                    label=MODEL_LABELS.get(model, model),
                    color=COLORS.get(model), linewidth=2, markersize=6)

        ax.set_xlabel("Intensity Round")
        ax.set_ylabel("Mean Token Entropy")
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels(intensity_labels, fontsize=8)
        ax.legend(fontsize=8)
        ax.set_title("Entropy Escalation Across Intensity", fontsize=11)

        plt.tight_layout()
        for fmt in ["pdf", "png"]:
            plt.savefig(str(FIGURES_DIR / f"entropy_escalation.{fmt}"))
        plt.close()
        logger.info("Saved entropy_escalation")
    except Exception as e:
        logger.error(f"entropy_escalation failed: {e}")


def fig_differential_features_by_depth(completed_models, diff_results):
    """Figure: Number of significant differential features vs layer depth."""
    try:
        fig, ax = plt.subplots(figsize=(5, 3.5))

        for model in completed_models:
            if model not in diff_results:
                continue
            model_spec = next((m for m in MODELS if m["short_name"] == model), None)
            if not model_spec:
                continue

            n_layers = model_spec["n_layers"]
            depths = []
            counts = []

            for layer_idx in sorted(diff_results[model].keys()):
                data = diff_results[model][layer_idx]
                depth = layer_idx / (n_layers - 1)
                depths.append(depth)
                counts.append(data["n_significant_p005_g05"])

            ax.plot(depths, counts, "o-", label=MODEL_LABELS.get(model, model),
                    color=COLORS.get(model), linewidth=2, markersize=6)

        ax.set_xlabel("Normalized Layer Depth")
        ax.set_ylabel("# Differential Features\n(p<0.05, |g|>0.5)")
        ax.legend(fontsize=8)
        ax.set_title("Self-Awareness Feature Density Across Depth", fontsize=11)

        plt.tight_layout()
        for fmt in ["pdf", "png"]:
            plt.savefig(str(FIGURES_DIR / f"differential_by_depth.{fmt}"))
        plt.close()
        logger.info("Saved differential_by_depth")
    except Exception as e:
        logger.error(f"differential_by_depth failed: {e}")


def fig_top_features_dashboard(completed_models, diff_results):
    """Figure: Activation profile of top-5 features across dimensions."""
    try:
        # Use the first IT model with results
        target = None
        for m in ["gemma-2-9b-it", "gemma-2-2b-it"]:
            if m in diff_results and diff_results[m]:
                target = m
                break
        if not target:
            return

        # Find layer with most significant features
        best_layer = max(diff_results[target].keys(),
                         key=lambda l: diff_results[target][l]["n_significant_p005_g05"])
        data = diff_results[target][best_layer]
        top5 = data["top_features"][:5]
        dim_means = data.get("dim_means", {})

        if not dim_means or not top5:
            return

        dims = ["D1", "D2", "D3", "D4", "D5", "D6"]
        fig, axes = plt.subplots(1, 5, figsize=(12, 2.5), sharey=True)

        for i, feat in enumerate(top5):
            ax = axes[i]
            feat_idx = feat["feature_idx"]
            vals = []
            for d in dims:
                if d in dim_means:
                    vals.append(float(dim_means[d][feat_idx]))
                else:
                    vals.append(0)

            colors = ['#4a90d9' if d in INTROSPECTIVE_DIMS else '#d9844a' for d in dims]
            ax.bar(range(6), vals, color=colors, alpha=0.8)
            ax.set_xticks(range(6))
            ax.set_xticklabels([d for d in dims], fontsize=7)
            ax.set_title(f"F{feat_idx}\ng={feat['hedges_g']:.2f}", fontsize=8)
            if i == 0:
                ax.set_ylabel("Mean Activation")

        fig.suptitle(f"Top-5 Differential Features ({MODEL_LABELS.get(target)}, Layer {best_layer})",
                     fontsize=10)
        plt.tight_layout()
        for fmt in ["pdf", "png"]:
            plt.savefig(str(FIGURES_DIR / f"top_features_dashboard.{fmt}"))
        plt.close()
        logger.info("Saved top_features_dashboard")
    except Exception as e:
        logger.error(f"top_features_dashboard failed: {e}")


def fig_alignment_decomposition(diff_results):
    """Figure: Shared/IT-emergent/base-suppressed features per layer."""
    try:
        alignment_path = ANALYSIS_DIR / "alignment_analysis.json"
        if not alignment_path.exists():
            return

        with open(alignment_path) as f:
            alignment = json.load(f)

        if not alignment:
            return

        layers = sorted(alignment.keys(), key=lambda x: int(x))
        shared = [alignment[l]["n_shared"] for l in layers]
        emergent = [alignment[l]["n_it_emergent"] for l in layers]
        suppressed = [alignment[l]["n_base_suppressed"] for l in layers]

        fig, ax = plt.subplots(figsize=(5, 3.5))
        x = np.arange(len(layers))
        w = 0.6

        ax.bar(x, shared, w, label="Shared (pre-existing)", color="#2ca02c", alpha=0.8)
        ax.bar(x, emergent, w, bottom=shared, label="IT-emergent (RLHF)", color="#ff7f0e", alpha=0.8)
        ax.bar(x, suppressed, w,
               bottom=[s + e for s, e in zip(shared, emergent)],
               label="Base-suppressed (RLHF)", color="#d62728", alpha=0.8)

        ax.set_xlabel("Probe Layer")
        ax.set_ylabel("# Differential Features")
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{l}" for l in layers], fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_title("Alignment Decomposition: 9B Base vs IT", fontsize=11)

        plt.tight_layout()
        for fmt in ["pdf", "png"]:
            plt.savefig(str(FIGURES_DIR / f"alignment_decomposition.{fmt}"))
        plt.close()
        logger.info("Saved alignment_decomposition")
    except Exception as e:
        logger.error(f"alignment_decomposition failed: {e}")


def fig_feature_heatmap(completed_models, diff_results):
    """Figure: Heatmap of top-20 feature activations across 30 questions."""
    try:
        target = None
        for m in ["gemma-2-9b-it", "gemma-2-2b-it"]:
            if m in diff_results and diff_results[m]:
                target = m
                break
        if not target:
            return

        # Find best layer
        best_layer = max(diff_results[target].keys(),
                         key=lambda l: diff_results[target][l]["n_significant_p005_g05"])

        # Load full features
        from analysis import load_sae_features
        full_features, question_ids, _ = load_sae_features(target, best_layer)
        if full_features is None:
            return

        # Get top-20 feature indices
        top20 = [f["feature_idx"] for f in diff_results[target][best_layer]["top_features"][:20]]

        # Build heatmap matrix
        matrix = full_features[:, top20]  # [30, 20]

        # Load behavioral for dimension labels
        beh_path = BEHAVIORAL_DIR / f"{target}.json"
        with open(beh_path) as f:
            behavioral = json.load(f)
        dim_labels = [f"{r['question_id']}\n({r['dimension']})" for r in behavioral]

        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(matrix.T, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax.set_xlabel("Question", fontsize=10)
        ax.set_ylabel("SAE Feature (ranked by |g|)", fontsize=10)
        ax.set_xticks(range(30))
        ax.set_xticklabels([r["question_id"] for r in behavioral], fontsize=6, rotation=90)
        ax.set_yticks(range(20))
        ax.set_yticklabels([f"F{idx}" for idx in top20], fontsize=7)

        # Mark dimension boundaries
        for i in range(0, 30, 6):
            ax.axvline(i - 0.5, color="gray", linewidth=0.5, alpha=0.5)

        plt.colorbar(im, ax=ax, label="Feature Activation", shrink=0.8)
        ax.set_title(f"Top-20 Differential SAE Features ({MODEL_LABELS.get(target)}, Layer {best_layer})",
                     fontsize=11)

        plt.tight_layout()
        for fmt in ["pdf", "png"]:
            plt.savefig(str(FIGURES_DIR / f"feature_heatmap.{fmt}"))
        plt.close()
        logger.info("Saved feature_heatmap")
    except Exception as e:
        logger.error(f"feature_heatmap failed: {e}")


def fig_l0_sparsity(completed_models):
    """Figure: L0 sparsity (active feature count) across models and layers."""
    try:
        fig, ax = plt.subplots(figsize=(6, 3.5))

        for model in completed_models:
            model_spec = next((m for m in MODELS if m["short_name"] == model), None)
            if not model_spec:
                continue

            n_layers = model_spec["n_layers"]
            depths = []
            mean_l0s = []

            for layer_idx in model_spec["probe_layers"]:
                meta_path = SAE_DIR / model / f"layer_{layer_idx}_meta.json"
                if not meta_path.exists():
                    continue
                with open(meta_path) as f:
                    meta = json.load(f)

                l0s = [m.get("l0", 0) for m in meta if "l0" in m]
                if l0s:
                    depth = layer_idx / (n_layers - 1)
                    depths.append(depth)
                    mean_l0s.append(np.mean(l0s))

            if depths:
                ax.plot(depths, mean_l0s, "o-", label=MODEL_LABELS.get(model, model),
                        color=COLORS.get(model), linewidth=2, markersize=6)

        ax.set_xlabel("Normalized Layer Depth")
        ax.set_ylabel("Mean L0 (Active Features)")
        ax.legend(fontsize=8)
        ax.set_title("SAE Sparsity Across Depth", fontsize=11)

        plt.tight_layout()
        for fmt in ["pdf", "png"]:
            plt.savefig(str(FIGURES_DIR / f"l0_sparsity.{fmt}"))
        plt.close()
        logger.info("Saved l0_sparsity")
    except Exception as e:
        logger.error(f"l0_sparsity failed: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Run via analysis.py or run_experiment.py")
