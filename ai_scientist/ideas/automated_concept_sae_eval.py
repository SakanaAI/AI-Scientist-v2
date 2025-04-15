import warnings
from datetime import datetime
import numpy as np
import time  # Add at the top with other imports
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning)

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader

from datasets import load_dataset
from torchvision.models import resnet50
from huggingface_hub import login

login(token=os.environ["HF_TOKEN"])
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.decomposition import PCA # Using PCA as a placeholder for concept discovery
import numpy as np
from tqdm.auto import tqdm
import warnings
import os
import time
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
MODEL_NAME = "gpt2" # Base LLM (e.g., GPT-2 Small/Medium)
DATASET_NAME = "imdb" # Complex linguistic task (e.g., sentiment classification)
TASK_TYPE = "classification" # Helps guide evaluation later
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
NUM_SAMPLES_ACTIVATIONS = 1000 # Number of samples to collect activations from
NUM_SAMPLES_EVAL = 200      # Number of samples for evaluation
LAYER_HOOK_TARGET = "transformer.h.5" # Example layer to extract activations (GPT-2 Small Layer 6)
NUM_CONCEPTS = 50 # Number of concepts to discover
SAE_EXPANSION_FACTOR = 4 # SAE hidden dimension = expansion_factor * activation_dim
SAE_L1_COEFF = 1e-3
SAE_LEARNING_RATE = 1e-4
SAE_EPOCHS = 5
SAE_BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Set seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"Using device: {DEVICE}")

# --- 1. Setup: Load Model, Tokenizer, Dataset ---
print(f"Loading model: {MODEL_NAME}")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading dataset: {DATASET_NAME}")
# Load a subset for efficiency
dataset = load_dataset(DATASET_NAME, split=f'train[:{NUM_SAMPLES_ACTIVATIONS + NUM_SAMPLES_EVAL}]')
# Shuffle and split dataset
dataset = dataset.shuffle(seed=SEED)
activation_dataset = Subset(dataset, range(NUM_SAMPLES_ACTIVATIONS))
eval_dataset = Subset(dataset, range(NUM_SAMPLES_ACTIVATIONS, NUM_SAMPLES_ACTIVATIONS + NUM_SAMPLES_EVAL))

activation_loader = DataLoader(activation_dataset, batch_size=32, shuffle=False)
eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

print(f"Loaded {len(activation_dataset)} samples for activation collection.")
print(f"Loaded {len(eval_dataset)} samples for evaluation.")


# --- Helper Function to Get Activations ---
activation_cache = {}

def get_hook(layer_name):
    def hook(model, input, output):
        # Store the first element of the output tuple if it's a tuple
        act = output[0] if isinstance(output, tuple) else output
        activation_cache[layer_name] = act.detach().cpu()
    return hook

def get_activations(texts, target_layer):
    """Get activations from a specific layer for a batch of texts."""
    handle = None
    module = dict(model.named_modules())[target_layer]
    handle = module.register_forward_hook(get_hook(target_layer))

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        model(**inputs)

    handle.remove()
    # Return activations for the last token position (often representative for classification)
    # Shape: [batch_size, hidden_dim]
    last_token_activations = activation_cache[target_layer][:, -1, :]
    del activation_cache[target_layer] # Clear cache
    return last_token_activations

# --- 2. Collect Activations ---
print("Collecting activations...")
all_activations = []
for batch in tqdm(activation_loader, desc="Collecting Activations"):
    texts = batch[TEXT_COLUMN]
    activations = get_activations(texts, LAYER_HOOK_TARGET)
    all_activations.append(activations)

all_activations_tensor = torch.cat(all_activations, dim=0)
activation_dim = all_activations_tensor.shape[-1]
print(f"Collected activations shape: {all_activations_tensor.shape}") # Should be [NUM_SAMPLES_ACTIVATIONS, activation_dim]

# --- 3. Concept Discovery (using PCA as a placeholder) ---
print("Discovering concepts using PCA...")
# Note: SparsePCA might be computationally expensive; PCA is a simpler starting point.
# For SparsePCA: from sklearn.decomposition import SparsePCA
# discoverer = SparsePCA(n_components=NUM_CONCEPTS, random_state=SEED, n_jobs=-1)
discoverer = PCA(n_components=NUM_CONCEPTS, random_state=SEED)
discoverer.fit(all_activations_tensor.numpy())

concept_vectors = torch.tensor(discoverer.components_, dtype=torch.float32) # Shape: [NUM_CONCEPTS, activation_dim]
print(f"Discovered concept vectors shape: {concept_vectors.shape}")

# Get projections (concept activations) for each sample
concept_projections = torch.tensor(discoverer.transform(all_activations_tensor.numpy()), dtype=torch.float32) # Shape: [NUM_SAMPLES_ACTIVATIONS, NUM_CONCEPTS]
print(f"Concept projections shape: {concept_projections.shape}")

# --- 4. Compute Pseudo-Supervised Dictionary ---
# Adapting mean feature dictionary computation from Makelov et al.
print("Computing pseudo-supervised dictionary...")
pseudo_supervised_features = {} # Store features like {concept_idx: {bin_idx: feature_vector}}

# Calculate overall mean activation
mean_activation = all_activations_tensor.mean(dim=0)

for i in range(NUM_CONCEPTS):
    projections_for_concept = concept_projections[:, i]
    # Simple binary split based on median projection value for this concept
    median_proj = torch.median(projections_for_concept).item()
    
    # Bin 0: Activations with projection <= median
    # Bin 1: Activations with projection > median
    activations_bin0 = all_activations_tensor[projections_for_concept <= median_proj]
    activations_bin1 = all_activations_tensor[projections_for_concept > median_proj]

    mean_bin0 = activations_bin0.mean(dim=0) if len(activations_bin0) > 0 else torch.zeros_like(mean_activation)
    mean_bin1 = activations_bin1.mean(dim=0) if len(activations_bin1) > 0 else torch.zeros_like(mean_activation)
    
    # Feature = Conditional Mean - Overall Mean
    feature_bin0 = mean_bin0 - mean_activation
    feature_bin1 = mean_bin1 - mean_activation

    pseudo_supervised_features[i] = {0: feature_bin0, 1: feature_bin1}

print(f"Computed pseudo-supervised dictionary for {len(pseudo_supervised_features)} concepts.")


# --- 5. SAE Training (Placeholder) ---
print("Setting up SAE...")

class SAE(nn.Module):
    """Simple Vanilla SAE."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.decoder = nn.Linear(hidden_dim, input_dim)
        # Initialize decoder weights to be tied requires grad manipulation or careful init
        # Optional: Add bias terms

        # Normalize decoder weights (as in Makelov et al.)
        self.decoder.weight.data = nn.functional.normalize(self.decoder.weight.data, dim=0)

    def forward(self, x):
        encoded = self.relu(self.encoder(x)) # shape: [batch, hidden_dim]
        
        # Normalize decoder weights during forward pass if needed (or just at init)
        # with torch.no_grad():
        #     self.decoder.weight.data = nn.functional.normalize(self.decoder.weight.data, dim=0)
            
        decoded = self.decoder(encoded) # shape: [batch, input_dim]
        return decoded, encoded # Return reconstruction and latent activations

sae_hidden_dim = activation_dim * SAE_EXPANSION_FACTOR
sae = SAE(activation_dim, sae_hidden_dim).to(DEVICE)
sae_optimizer = optim.Adam(sae.parameters(), lr=SAE_LEARNING_RATE)

# MSE Loss + L1 Sparsity Loss
def sae_loss_fn(original_activations, reconstructed_activations, latent_activations, l1_coeff):
    mse_loss = nn.functional.mse_loss(reconstructed_activations, original_activations)
    l1_loss = l1_coeff * torch.norm(latent_activations, p=1, dim=1).mean()
    total_loss = mse_loss + l1_loss
    return total_loss, mse_loss, l1_loss

print(f"Training SAE (Hidden Dim: {sae_hidden_dim})...")
sae_dataset = torch.utils.data.TensorDataset(all_activations_tensor)
sae_loader = DataLoader(sae_dataset, batch_size=SAE_BATCH_SIZE, shuffle=True)

sae.train()
for epoch in range(SAE_EPOCHS):
    epoch_loss = 0
    for batch in tqdm(sae_loader, desc=f"SAE Epoch {epoch+1}/{SAE_EPOCHS}", leave=False):
        acts = batch[0].to(DEVICE)
        sae_optimizer.zero_grad()
        reconstructed, latents = sae(acts)
        total_loss, mse_loss, l1_loss = sae_loss_fn(acts, reconstructed, latents, SAE_L1_COEFF)
        total_loss.backward()
        sae_optimizer.step()
        
        # Re-normalize decoder weights after step
        with torch.no_grad():
            sae.decoder.weight.data = nn.functional.normalize(sae.decoder.weight.data, dim=0)
            
        epoch_loss += total_loss.item()
    avg_epoch_loss = epoch_loss / len(sae_loader)
    print(f"SAE Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")

sae.eval()
print("SAE Training finished.")


# --- 6. Evaluation (Sufficiency Placeholder) ---
print("Evaluating Sufficiency...")

def calculate_downstream_metric(texts, model, tokenizer, device):
    """Placeholder for task-specific metric (e.g., accuracy, logit diff)."""
    # Example for classification: predict class probabilities
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        # Assuming classification head exists or using logits of last token
        # This needs to be adapted based on the actual task head / evaluation method
        logits = outputs.logits[:, -1, :] # Example: Logits for last token
        # Calculate some metric, e.g., average probability of class 1
        probs = torch.softmax(logits, dim=-1)
        # Return a scalar metric for simplicity
        if probs.shape[1] > 1:
             return probs[:, 1].mean().item() # Example: Avg prob of class 1
        else:
             return probs[:, 0].mean().item() # Handle single class output

def evaluate_sufficiency(eval_loader, target_layer, dictionary_or_sae, is_sae, device):
    """Evaluate sufficiency by patching activations."""
    original_metric_sum = 0
    reconstructed_metric_sum = 0
    num_batches = 0

    original_forward_pass = dict(model.named_modules())[target_layer].forward

    for batch in tqdm(eval_loader, desc="Evaluating Sufficiency", leave=False):
        texts = batch[TEXT_COLUMN]
        labels = batch[LABEL_COLUMN] # Not used in this placeholder metric

        # --- Original Forward Pass ---
        original_metric = calculate_downstream_metric(texts, model, tokenizer, device)
        original_metric_sum += original_metric

        # --- Patched Forward Pass ---
        def patching_hook(module, input, output):
            original_activations = output[0] if isinstance(output, tuple) else output
            # Reconstruct using either the pseudo-supervised dictionary or the SAE
            if is_sae:
                # SAE Reconstruction
                with torch.no_grad():
                    reconstructed_activations, _ = dictionary_or_sae(original_activations.to(DEVICE))
                    reconstructed_activations = reconstructed_activations.cpu() # Move back if needed
            else:
                # Pseudo-Supervised Dictionary Reconstruction
                # Project onto concept vectors to determine bin membership
                projections_np = discoverer.transform(original_activations[:, -1, :].cpu().numpy()) # Use last token activation
                projections = torch.tensor(projections_np, dtype=torch.float32)

                reconstructed_activations = torch.zeros_like(original_activations[:, -1, :]) + mean_activation.cpu() # Start with mean
                for i in range(NUM_CONCEPTS):
                    proj_for_concept = projections[:, i]
                    median_proj = torch.median(projections_for_concept).values # Recompute median for consistency? Or use training median?
                    bin_indices = (proj_for_concept > median_proj).long() # 0 or 1
                    for sample_idx, bin_idx in enumerate(bin_indices):
                         reconstructed_activations[sample_idx] += dictionary_or_sae[i][bin_idx.item()].cpu()

                # We only reconstructed the last token activation, need to put it back
                # This simple patching of only last token might be insufficient!
                # A more robust approach would reconstruct all tokens or use hooks carefully
                patched_output_data = original_activations.clone()
                patched_output_data[:, -1, :] = reconstructed_activations


            # Ensure shape matches original output (handling tuples)
            if isinstance(output, tuple):
                 # Assume activation is the first element
                 return (patched_output_data.to(DEVICE),) + output[1:]
            else:
                 return patched_output_data.to(DEVICE)

        # Register the hook
        handle = dict(model.named_modules())[target_layer].register_forward_hook(patching_hook)

        # Run forward pass with the hook active
        reconstructed_metric = calculate_downstream_metric(texts, model, tokenizer, device)
        reconstructed_metric_sum += reconstructed_metric

        # Remove the hook
        handle.remove()

        num_batches += 1

    avg_original_metric = original_metric_sum / num_batches
    avg_reconstructed_metric = reconstructed_metric_sum / num_batches

    # Simple recovery metric (closer to 1 is better)
    # This assumes higher metric value is better. Needs adjustment based on actual metric.
    # Avoid division by zero
    recovery = (avg_reconstructed_metric / avg_original_metric) if abs(avg_original_metric) > 1e-6 else 0.0
    
    return avg_original_metric, avg_reconstructed_metric, recovery


# Evaluate Pseudo-Supervised Dictionary
ps_orig, ps_recon, ps_recovery = evaluate_sufficiency(eval_loader, LAYER_HOOK_TARGET, pseudo_supervised_features, is_sae=False, device=DEVICE)
print(f"Pseudo-Supervised Dict - Original Metric: {ps_orig:.4f}, Reconstructed Metric: {ps_recon:.4f}, Recovery: {ps_recovery:.4f}")

# Evaluate SAE
sae_orig, sae_recon, sae_recovery = evaluate_sufficiency(eval_loader, LAYER_HOOK_TARGET, sae, is_sae=True, device=DEVICE)
print(f"SAE - Original Metric: {sae_orig:.4f}, Reconstructed Metric: {sae_recon:.4f}, Recovery: {sae_recovery:.4f}")


# --- 7. Analysis (Placeholder) ---
print("Analysis...")
# TODO: Implement Necessity evaluation
# TODO: Implement Sparse Control evaluation (more complex)
# TODO: Compare results across different SAE types, concept discovery methods, layers, tasks.
# TODO: Validate discovered concepts more rigorously.

print("Script finished.")