"""
TTA Scaling Laws for Medical Image Classification — GPU-Accelerated
===================================================================
Uses kornia for fully GPU-native augmentations. All tensors stay on GPU.
Produces results + publication figures.
"""
import warnings
warnings.filterwarnings("ignore")

import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import kornia.augmentation as K
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import medmnist
from medmnist import PathMNIST, DermaMNIST, BloodMNIST

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tta_scaling_results")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

IMAGE_SIZE = 28
NUM_CHANNELS = 3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True  # optimize for fixed input sizes

# ==============================================================
# DATASETS
# ==============================================================

DATASETS = {
    "pathmnist":  (PathMNIST,  9, "Pathology"),
    "dermamnist": (DermaMNIST, 7, "Dermatology"),
    "bloodmnist": (BloodMNIST, 8, "Blood Cell"),
}


def load_dataset(name):
    """Load MedMNIST dataset, return GPU tensors."""
    cls, n_classes, label = DATASETS[name]
    from torchvision.transforms import ToTensor
    splits = {}
    for split in ["train", "val", "test"]:
        ds = cls(split=split, transform=ToTensor(), download=True)
        imgs, labs = [], []
        for img, lbl in ds:
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
            imgs.append(img)
            labs.append(int(lbl[0]) if hasattr(lbl, '__len__') else int(lbl))
        # Store on GPU immediately
        splits[split] = (
            torch.stack(imgs).to(DEVICE),
            torch.tensor(labs, dtype=torch.long).to(DEVICE),
        )
    return splits, n_classes, label


# ==============================================================
# MODELS
# ==============================================================

class LogReg(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.fc = nn.Linear(NUM_CHANNELS * IMAGE_SIZE**2, nc)
    def forward(self, x):
        return self.fc(x.flatten(1))

class MLP(nn.Module):
    def __init__(self, nc, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(NUM_CHANNELS * IMAGE_SIZE**2, h),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h, nc))
    def forward(self, x):
        return self.net(x)

class SmallCNN(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(128, nc)
    def forward(self, x):
        return self.fc(self.features(x).flatten(1))

class ResNet18Small(nn.Module):
    def __init__(self, nc):
        super().__init__()
        from torchvision.models import resnet18
        self.model = resnet18(weights=None, num_classes=nc)
        self.model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
    def forward(self, x):
        return self.model(x)

MODELS = {"LogReg": LogReg, "MLP": MLP, "SmallCNN": SmallCNN, "ResNet18": ResNet18Small}

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ==============================================================
# KORNIA GPU AUGMENTATIONS
# ==============================================================

def make_aug_pipeline(strategy="mixed"):
    """Create a kornia augmentation pipeline that runs entirely on GPU."""
    if strategy == "geometric":
        return K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomRotation(degrees=15, p=0.8),
            K.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.8, 1.0), p=0.8),
            data_keys=["input"],
        )
    elif strategy == "intensity":
        return K.AugmentationSequential(
            K.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, p=0.8),
            K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.01, 0.5), p=0.5),
            data_keys=["input"],
        )
    elif strategy == "mixed":
        return K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomRotation(degrees=15, p=0.8),
            K.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.8, 1.0), p=0.8),
            K.ColorJitter(brightness=0.2, contrast=0.2, p=0.6),
            data_keys=["input"],
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ==============================================================
# TRAINING
# ==============================================================

def train_model(model, train_imgs, train_labels, val_imgs, val_labels,
                epochs=20, bs=256, lr=1e-3):
    """Train with early stopping. Data already on GPU."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    ds = TensorDataset(train_imgs, train_labels)
    loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0)

    best_val = 0.0
    best_state = None
    hist = {"loss": [], "val_acc": []}

    for ep in range(epochs):
        model.train()
        total_loss = 0
        n = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
        scheduler.step()

        avg_loss = total_loss / n
        val_acc = evaluate(model, val_imgs, val_labels)
        hist["loss"].append(avg_loss)
        hist["val_acc"].append(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (ep + 1) % 5 == 0:
            print(f"    Ep {ep+1}/{epochs}: loss={avg_loss:.4f} val={val_acc:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return hist, best_val


# ==============================================================
# EVALUATION
# ==============================================================

@torch.no_grad()
def evaluate(model, imgs, labels, bs=512):
    """Standard single-pass accuracy. Data on GPU."""
    model.eval()
    correct = 0
    for i in range(0, len(imgs), bs):
        preds = model(imgs[i:i+bs]).argmax(1)
        correct += (preds == labels[i:i+bs]).sum().item()
    return correct / len(labels)


@torch.no_grad()
def get_probs(model, imgs, bs=512):
    """Get softmax probabilities. Data on GPU."""
    model.eval()
    all_p = []
    for i in range(0, len(imgs), bs):
        all_p.append(torch.softmax(model(imgs[i:i+bs]), dim=1))
    return torch.cat(all_p, dim=0)


@torch.no_grad()
def tta_evaluate(model, imgs, labels, n_views, aug_pipe, aggregation="mean", bs=512):
    """TTA evaluation — fully on GPU via kornia.

    Returns: (accuracy, probs_numpy, time_seconds)
    """
    model.eval()
    t0 = time.time()
    n = len(imgs)

    # Collect probs for each augmented view
    all_view_probs = []
    for v in range(n_views):
        view_probs = []
        for i in range(0, n, bs):
            xb = imgs[i:i+bs]
            xb_aug = aug_pipe(xb)  # GPU augmentation, returns tensor directly
            probs = torch.softmax(model(xb_aug), dim=1)
            view_probs.append(probs)
        all_view_probs.append(torch.cat(view_probs, dim=0))  # (N, C)

    # Stack: (n_views, N, C)
    stacked = torch.stack(all_view_probs, dim=0)

    if aggregation == "mean":
        avg_probs = stacked.mean(dim=0)
        preds = avg_probs.argmax(dim=1)
    elif aggregation == "vote":
        votes = stacked.argmax(dim=2)  # (n_views, N)
        preds = torch.mode(votes, dim=0).values
        avg_probs = stacked.mean(dim=0)
    elif aggregation == "confidence":
        max_confs = stacked.max(dim=2, keepdim=True).values
        weighted = stacked * max_confs
        avg_probs = weighted.sum(dim=0) / max_confs.sum(dim=0)
        preds = avg_probs.argmax(dim=1)
    else:
        raise ValueError(aggregation)

    acc = (preds == labels).float().mean().item()
    elapsed = time.time() - t0
    return acc, avg_probs.cpu().numpy(), elapsed


def compute_ece(probs, labels, n_bins=15):
    """Expected Calibration Error."""
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    confs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    accs = (preds == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confs > bins[i]) & (confs <= bins[i+1])
        if mask.sum() > 0:
            ece += mask.sum() * abs(accs[mask].mean() - confs[mask].mean())
    return ece / len(labels)


# ==============================================================
# MAIN EXPERIMENT
# ==============================================================

def run_experiments():
    """Run full experiment suite."""
    all_results = {}
    N_VIEWS = [1, 2, 5, 10, 25, 50, 100]
    STRATEGIES = ["geometric", "intensity", "mixed"]
    AGGREGATIONS = ["mean", "vote", "confidence"]
    EPOCHS = {"LogReg": 25, "MLP": 25, "SmallCNN": 30, "ResNet18": 30}

    for ds_name, (_, _, ds_label) in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"DATASET: {ds_name} ({ds_label})")
        print(f"{'='*60}")

        splits, n_classes, _ = load_dataset(ds_name)
        train_imgs, train_labels = splits["train"]
        val_imgs, val_labels = splits["val"]
        test_imgs, test_labels = splits["test"]
        print(f"  Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}, Classes={n_classes}")

        ds_results = {}

        for model_name, model_cls in MODELS.items():
            print(f"\n  --- {model_name} ---")
            model = model_cls(n_classes).to(DEVICE)
            n_params = count_params(model)
            print(f"  Params: {n_params:,}")

            t0 = time.time()
            hist, best_val = train_model(
                model, train_imgs, train_labels, val_imgs, val_labels,
                epochs=EPOCHS[model_name], bs=256, lr=1e-3
            )
            train_time = time.time() - t0

            # Baseline (no augmentation)
            base_acc = evaluate(model, test_imgs, test_labels)
            base_probs = get_probs(model, test_imgs).cpu().numpy()
            base_ece = compute_ece(base_probs, test_labels)
            print(f"  Base: acc={base_acc:.4f}, ECE={base_ece:.4f}, train={train_time:.1f}s")

            model_results = {
                "n_params": n_params,
                "train_time": train_time,
                "best_val_acc": best_val,
                "base_acc": base_acc,
                "base_ece": base_ece,
                "training_history": hist,
            }

            # --- TTA Scaling (mixed strategy, mean aggregation) ---
            print(f"  TTA Scaling (mixed/mean):")
            scaling = {}
            mixed_pipe = make_aug_pipeline("mixed")
            for nv in N_VIEWS:
                acc, probs, t = tta_evaluate(model, test_imgs, test_labels, nv, mixed_pipe)
                ece = compute_ece(probs, test_labels)
                delta = acc - base_acc
                print(f"    N={nv:3d}: acc={acc:.4f} ({delta:+.4f}), ECE={ece:.4f}, {t:.1f}s")
                scaling[nv] = {"acc": acc, "ece": ece, "time": t, "delta": delta}
            model_results["tta_scaling"] = scaling

            # --- Strategy Comparison (N=25, mean aggregation) ---
            print(f"  Strategies (N=25):")
            strats = {}
            for s in STRATEGIES:
                pipe = make_aug_pipeline(s)
                acc, probs, t = tta_evaluate(model, test_imgs, test_labels, 25, pipe)
                ece = compute_ece(probs, test_labels)
                print(f"    {s:12s}: acc={acc:.4f}, ECE={ece:.4f}, {t:.1f}s")
                strats[s] = {"acc": acc, "ece": ece, "time": t}
            model_results["strategy_comparison"] = strats

            # --- Aggregation Comparison (N=25, mixed strategy) ---
            print(f"  Aggregations (N=25):")
            aggs = {}
            for a in AGGREGATIONS:
                acc, probs, t = tta_evaluate(model, test_imgs, test_labels, 25, mixed_pipe, aggregation=a)
                ece = compute_ece(probs, test_labels)
                print(f"    {a:12s}: acc={acc:.4f}, ECE={ece:.4f}")
                aggs[a] = {"acc": acc, "ece": ece, "time": t}
            model_results["aggregation_comparison"] = aggs

            ds_results[model_name] = model_results

            # Free GPU memory
            del model
            torch.cuda.empty_cache()

        all_results[ds_name] = ds_results

    # Save raw results
    results_path = os.path.join(OUTPUT_DIR, "all_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    return all_results


# ==============================================================
# PLOTTING
# ==============================================================

def make_plots(results):
    """Generate publication-quality figures."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    colors = sns.color_palette("husl", 4)
    model_names = list(MODELS.keys())
    ds_names = list(DATASETS.keys())
    ds_labels = {k: v[2] for k, v in DATASETS.items()}

    # ---- Figure 1: TTA Scaling Curves (3 panels, one per dataset) ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    for idx, ds in enumerate(ds_names):
        ax = axes[idx]
        for mi, mn in enumerate(model_names):
            r = results[ds][mn]
            nviews = sorted([int(k) for k in r["tta_scaling"].keys()])
            accs = [r["tta_scaling"][str(nv) if str(nv) in r["tta_scaling"] else nv]["acc"] for nv in nviews]
            base = r["base_acc"]
            ax.axhline(base, color=colors[mi], linestyle=":", alpha=0.5, linewidth=1)
            ax.plot(nviews, accs, "o-", color=colors[mi], label=mn, linewidth=2, markersize=5)
        ax.set_xscale("log")
        ax.set_xlabel("Number of TTA Views")
        if idx == 0:
            ax.set_ylabel("Test Accuracy")
        ax.set_title(f"{ds_labels[ds]} ({ds})")
        ax.legend(fontsize=8)
    fig.suptitle("TTA Scaling: Accuracy vs. Number of Augmented Views", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_tta_scaling.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "fig1_tta_scaling.pdf"), bbox_inches="tight")
    plt.close()
    print("  Saved fig1_tta_scaling")

    # ---- Figure 2: Accuracy Gain (delta) vs Model Size ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for idx, ds in enumerate(ds_names):
        ax = axes[idx]
        for mi, mn in enumerate(model_names):
            r = results[ds][mn]
            n_params = r["n_params"]
            # Get delta at N=50
            key50 = "50" if "50" in r["tta_scaling"] else 50
            delta = r["tta_scaling"][key50]["acc"] - r["base_acc"]
            ax.scatter(n_params, delta * 100, s=120, color=colors[mi], label=mn, zorder=5)
        ax.set_xscale("log")
        ax.set_xlabel("Model Parameters")
        if idx == 0:
            ax.set_ylabel("Accuracy Gain from TTA (%)")
        ax.set_title(ds_labels[ds])
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)
    fig.suptitle("TTA Benefit vs. Model Capacity (N=50 views)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_gain_vs_params.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "fig2_gain_vs_params.pdf"), bbox_inches="tight")
    plt.close()
    print("  Saved fig2_gain_vs_params")

    # ---- Figure 3: ECE (Calibration) improvement with TTA ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for idx, ds in enumerate(ds_names):
        ax = axes[idx]
        for mi, mn in enumerate(model_names):
            r = results[ds][mn]
            nviews = sorted([int(k) for k in r["tta_scaling"].keys()])
            eces = [r["tta_scaling"][str(nv) if str(nv) in r["tta_scaling"] else nv]["ece"] for nv in nviews]
            base_ece = r["base_ece"]
            ax.axhline(base_ece, color=colors[mi], linestyle=":", alpha=0.5)
            ax.plot(nviews, eces, "s-", color=colors[mi], label=mn, linewidth=2, markersize=5)
        ax.set_xscale("log")
        ax.set_xlabel("Number of TTA Views")
        if idx == 0:
            ax.set_ylabel("Expected Calibration Error")
        ax.set_title(ds_labels[ds])
        ax.legend(fontsize=8)
    fig.suptitle("TTA Scaling: Calibration (ECE) vs. Number of Views", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_ece_scaling.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "fig3_ece_scaling.pdf"), bbox_inches="tight")
    plt.close()
    print("  Saved fig3_ece_scaling")

    # ---- Figure 4: Strategy comparison heatmap ----
    strategies = ["geometric", "intensity", "mixed"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for idx, ds in enumerate(ds_names):
        ax = axes[idx]
        data = np.zeros((len(model_names), len(strategies)))
        for mi, mn in enumerate(model_names):
            for si, s in enumerate(strategies):
                data[mi, si] = results[ds][mn]["strategy_comparison"][s]["acc"]
        sns.heatmap(data, ax=ax, annot=True, fmt=".3f", cmap="YlOrRd",
                    xticklabels=strategies, yticklabels=model_names if idx == 0 else [],
                    vmin=data.min() - 0.02, vmax=data.max() + 0.02)
        ax.set_title(ds_labels[ds])
    fig.suptitle("Augmentation Strategy Comparison (N=25)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_strategy_heatmap.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "fig4_strategy_heatmap.pdf"), bbox_inches="tight")
    plt.close()
    print("  Saved fig4_strategy_heatmap")

    # ---- Figure 5: Aggregation comparison ----
    agg_methods = ["mean", "vote", "confidence"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for idx, ds in enumerate(ds_names):
        ax = axes[idx]
        x = np.arange(len(model_names))
        w = 0.25
        for ai, a in enumerate(agg_methods):
            vals = [results[ds][mn]["aggregation_comparison"][a]["acc"] for mn in model_names]
            ax.bar(x + ai * w, vals, w, label=a, alpha=0.85)
        ax.set_xticks(x + w)
        ax.set_xticklabels(model_names, rotation=15)
        if idx == 0:
            ax.set_ylabel("Test Accuracy")
        ax.set_title(ds_labels[ds])
        ax.legend(fontsize=8)
    fig.suptitle("Aggregation Method Comparison (N=25, mixed)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_aggregation.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "fig5_aggregation.pdf"), bbox_inches="tight")
    plt.close()
    print("  Saved fig5_aggregation")

    # ---- Figure 6: Compute-Accuracy Tradeoff ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for idx, ds in enumerate(ds_names):
        ax = axes[idx]
        for mi, mn in enumerate(model_names):
            r = results[ds][mn]
            nviews = sorted([int(k) for k in r["tta_scaling"].keys()])
            times = [r["tta_scaling"][str(nv) if str(nv) in r["tta_scaling"] else nv]["time"] for nv in nviews]
            accs = [r["tta_scaling"][str(nv) if str(nv) in r["tta_scaling"] else nv]["acc"] for nv in nviews]
            ax.plot(times, accs, "o-", color=colors[mi], label=mn, linewidth=2, markersize=5)
            # Mark baseline
            ax.scatter([0], [r["base_acc"]], marker="x", s=100, color=colors[mi], zorder=5)
        ax.set_xlabel("Inference Time (seconds)")
        if idx == 0:
            ax.set_ylabel("Test Accuracy")
        ax.set_title(ds_labels[ds])
        ax.legend(fontsize=8)
    fig.suptitle("Compute-Accuracy Tradeoff: TTA Inference Time", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig6_compute_tradeoff.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "fig6_compute_tradeoff.pdf"), bbox_inches="tight")
    plt.close()
    print("  Saved fig6_compute_tradeoff")

    # ---- Summary Table (for paper) ----
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Dataset':<12} {'Model':<10} {'Params':<10} {'Base Acc':<10} {'TTA@50':<10} {'Delta':<10} {'ECE base':<10} {'ECE@50':<10}")
    print("-"*80)
    for ds in ds_names:
        for mn in model_names:
            r = results[ds][mn]
            key50 = "50" if "50" in r["tta_scaling"] else 50
            tta50 = r["tta_scaling"][key50]
            print(f"{ds:<12} {mn:<10} {r['n_params']:<10,} {r['base_acc']:<10.4f} {tta50['acc']:<10.4f} "
                  f"{tta50['acc']-r['base_acc']:<+10.4f} {r['base_ece']:<10.4f} {tta50['ece']:<10.4f}")

    print(f"\nAll figures saved to {FIG_DIR}/")


# ==============================================================
# ENTRY POINT
# ==============================================================

if __name__ == "__main__":
    print(f"\nStarting experiments at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    t_start = time.time()

    results = run_experiments()

    print(f"\n{'='*60}")
    print("GENERATING FIGURES")
    print(f"{'='*60}")
    make_plots(results)

    total = time.time() - t_start
    print(f"\nTotal runtime: {total/60:.1f} minutes")
    print("Done!")
