"""
SAE Probing: Extract residual stream activations and apply Gemma Scope SAEs.

Uses PyTorch forward hooks + SAE.encode() from sae-lens.
Extracts activations at specified layers during VK-LLM prompts.
"""

import gc
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SAEProber:
    """Extract residual stream activations and SAE features for VK-LLM prompts."""

    def __init__(self, model, tokenizer, model_spec):
        self.model = model
        self.tokenizer = tokenizer
        self.model_spec = model_spec
        self.probe_layers = model_spec["probe_layers"]
        self.is_base = model_spec["is_base"]
        self.device = next(model.parameters()).device
        self._hooks = []
        self._activations = {}
        self.saes = {}

    def load_saes(self):
        """Load Gemma Scope SAEs for all probe layers."""
        from sae_lens import SAE

        release = self.model_spec["sae_release"]
        loaded = 0
        for layer_idx in self.probe_layers:
            sae_id = f"layer_{layer_idx}/width_16k/canonical"
            try:
                sae = SAE.from_pretrained(
                    release=release,
                    sae_id=sae_id,
                    device=str(self.device),
                )
                self.saes[layer_idx] = sae
                loaded += 1
                logger.info(f"Loaded SAE: {release}/{sae_id}")
            except Exception as e:
                logger.warning(f"Failed to load SAE for layer {layer_idx}: {e}")

        logger.info(f"SAEs loaded: {loaded}/{len(self.probe_layers)} layers")
        return loaded > 0

    def register_hooks(self):
        """Attach forward hooks to probe layers."""
        self.remove_hooks()
        layers = self._get_model_layers()
        for layer_idx in self.probe_layers:
            if layer_idx >= len(layers):
                logger.warning(f"Layer {layer_idx} >= {len(layers)}, skipping")
                continue
            hook = layers[layer_idx].register_forward_hook(self._make_hook(layer_idx))
            self._hooks.append(hook)
        logger.info(f"Hooks registered on layers: {self.probe_layers}")

    def remove_hooks(self):
        """Remove all hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _get_model_layers(self):
        """Get decoder layers for Gemma 2."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        raise ValueError(f"Cannot find layers in {type(self.model)}")

    def _make_hook(self, layer_idx):
        """Create hook that captures residual stream output."""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            self._activations[layer_idx] = hidden.detach().cpu().float()
        return hook_fn

    def _build_input_ids(self, conversation_history, question):
        """Build input tokens for forward pass."""
        if self.is_base:
            text = conversation_history[0]["content"]
            for entry in conversation_history[1:]:
                if entry["role"] == "user":
                    text += f"\nExaminer: {entry['content']}"
                elif entry["role"] == "assistant":
                    text += f"\nSubject: {entry['content']}"
            text += f"\nExaminer: {question['question']}\nSubject:"
            input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        else:
            messages = conversation_history + [
                {"role": "user", "content": question["question"]}
            ]
            input_ids = self.tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            ).to(self.device)
        return input_ids

    def extract_for_question(self, conversation_history, question):
        """Forward pass + SAE encode for one question.

        Returns dict[layer_idx] -> {
            "full_features": np.ndarray [d_sae],
            "top_indices": np.ndarray [top_k],
            "top_values": np.ndarray [top_k],
            "l0": int,
            "l1_norm": float,
            "raw_norm": float,
            "mean_features": np.ndarray [d_sae] (mean over all positions),
        }
        """
        input_ids = self._build_input_ids(conversation_history, question)

        self._activations = {}
        with torch.no_grad():
            self.model(input_ids=input_ids)

        results = {}
        for layer_idx, hidden in self._activations.items():
            # hidden: [1, seq_len, d_model]
            last_token_act = hidden[0, -1, :]  # [d_model]
            raw_norm = float(last_token_act.norm().item())

            result = {"raw_norm": raw_norm}

            if layer_idx in self.saes:
                sae = self.saes[layer_idx]

                # Encode last token
                with torch.no_grad():
                    features_raw = sae.encode(last_token_act.unsqueeze(0).to(sae.device))
                # Force to numpy-compatible tensor
                features = features_raw.squeeze(0).detach().clone().cpu()

                num_active = int((features > 0).sum().item())
                l1_norm = float(features.abs().sum().item())

                # Top-K sparse representation
                features_np = features.numpy().astype(np.float32)
                k = min(50, num_active, features.shape[0])
                if k > 0:
                    top_k_idx = np.argpartition(-np.abs(features_np), k)[:k]
                    top_k_idx = top_k_idx[np.argsort(-features_np[top_k_idx])]
                    result["top_indices"] = top_k_idx.astype(np.int32)
                    result["top_values"] = features_np[top_k_idx]
                else:
                    result["top_indices"] = np.array([], dtype=np.int32)
                    result["top_values"] = np.array([], dtype=np.float32)

                result["full_features"] = features_np
                result["l0"] = num_active
                result["l1_norm"] = l1_norm

                # Mean features across all positions (chunked for memory)
                seq_len = hidden.shape[1]
                chunk_size = 256
                mean_accum = torch.zeros(sae.cfg.d_sae, device='cpu')
                with torch.no_grad():
                    for start in range(0, seq_len, chunk_size):
                        end = min(start + chunk_size, seq_len)
                        chunk = hidden[0, start:end, :].to(sae.device)
                        chunk_feats = sae.encode(chunk)
                        mean_accum += chunk_feats.detach().cpu().sum(dim=0)
                mean_features = (mean_accum / seq_len).numpy().astype(np.float32)
                result["mean_features"] = mean_features

            results[layer_idx] = result

        self._activations = {}
        return results

    def cleanup(self):
        """Free SAE memory."""
        self.remove_hooks()
        self.saes = {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def save_sae_results(output_dir, model_name, all_results):
    """Save SAE probing results for a model.

    all_results: list of (question_id, layer_results_dict)
    """
    model_dir = Path(output_dir) / model_name
    os.makedirs(model_dir, exist_ok=True)

    # Save per-layer .npz files with full feature matrices
    layers = set()
    for _, layer_results in all_results:
        layers.update(layer_results.keys())

    for layer_idx in sorted(layers):
        question_ids = []
        full_features_list = []
        mean_features_list = []
        metadata = []

        for qid, layer_results in all_results:
            if layer_idx not in layer_results:
                continue
            r = layer_results[layer_idx]
            question_ids.append(qid)

            if "full_features" in r:
                full_features_list.append(r["full_features"])
                mean_features_list.append(r["mean_features"])
                metadata.append({
                    "question_id": qid,
                    "l0": r["l0"],
                    "l1_norm": r["l1_norm"],
                    "raw_norm": r["raw_norm"],
                    "top_indices": r["top_indices"].tolist(),
                    "top_values": r["top_values"].tolist(),
                })
            else:
                metadata.append({
                    "question_id": qid,
                    "raw_norm": r["raw_norm"],
                })

        # Save feature matrices as .npz
        save_dict = {"question_ids": np.array(question_ids, dtype=object)}
        if full_features_list:
            save_dict["full_features"] = np.stack(full_features_list)  # [n_questions, d_sae]
            save_dict["mean_features"] = np.stack(mean_features_list)
        np.savez_compressed(
            str(model_dir / f"layer_{layer_idx}.npz"),
            **save_dict,
        )

        # Save metadata JSON
        with open(model_dir / f"layer_{layer_idx}_meta.json", "w") as f:
            json.dump(metadata, f, indent=2)

    logger.info(f"SAE results saved to {model_dir}")
