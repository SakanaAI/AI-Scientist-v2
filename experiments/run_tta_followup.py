"""
Follow-up experiment: TTA with vs without original image included.
Also tests BatchNorm adaptation at test time.
Quick run on PathMNIST + BloodMNIST with SmallCNN + ResNet18.
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
import seaborn as sns
import medmnist
from medmnist import PathMNIST, DermaMNIST, BloodMNIST

sys.stdout.reconfigure(line_buffering=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tta_scaling_results")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

IMAGE_SIZE = 28

DATASETS = {
    "pathmnist":  (PathMNIST,  9, "Pathology"),
    "bloodmnist": (BloodMNIST, 8, "Blood Cell"),
}

def load_dataset(name):
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
        splits[split] = (torch.stack(imgs).to(DEVICE), torch.tensor(labs, dtype=torch.long).to(DEVICE))
    return splits, n_classes, label


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

MODELS = {"SmallCNN": SmallCNN, "ResNet18": ResNet18Small}


def make_aug_pipeline():
    return K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        K.RandomRotation(degrees=15, p=0.8),
        K.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.8, 1.0), p=0.8),
        K.ColorJitter(brightness=0.2, contrast=0.2, p=0.6),
        data_keys=["input"],
    )


def train_model(model, train_imgs, train_labels, val_imgs, val_labels, epochs=30, bs=256, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    ds = TensorDataset(train_imgs, train_labels)
    loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0)
    best_val, best_state = 0.0, None
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()
        val_acc = evaluate(model, val_imgs, val_labels)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (ep+1) % 10 == 0:
            print(f"    Ep {ep+1}/{epochs}: val={val_acc:.4f}")
    if best_state:
        model.load_state_dict(best_state)
    return best_val


@torch.no_grad()
def evaluate(model, imgs, labels, bs=512):
    model.eval()
    correct = 0
    for i in range(0, len(imgs), bs):
        correct += (model(imgs[i:i+bs]).argmax(1) == labels[i:i+bs]).sum().item()
    return correct / len(labels)


@torch.no_grad()
def tta_evaluate(model, imgs, labels, n_aug_views, aug_pipe, include_original=False, bs=512):
    """TTA with option to include original unaugmented image."""
    model.eval()
    t0 = time.time()
    n = len(imgs)
    all_probs = []

    # Include original unaugmented prediction
    if include_original:
        orig_probs = []
        for i in range(0, n, bs):
            orig_probs.append(torch.softmax(model(imgs[i:i+bs]), dim=1))
        all_probs.append(torch.cat(orig_probs, dim=0))

    # Augmented views
    for _ in range(n_aug_views):
        view_probs = []
        for i in range(0, n, bs):
            xb_aug = aug_pipe(imgs[i:i+bs])
            view_probs.append(torch.softmax(model(xb_aug), dim=1))
        all_probs.append(torch.cat(view_probs, dim=0))

    stacked = torch.stack(all_probs, dim=0)
    avg_probs = stacked.mean(dim=0)
    preds = avg_probs.argmax(dim=1)
    acc = (preds == labels).float().mean().item()
    return acc, time.time() - t0


@torch.no_grad()
def tta_with_bn_adapt(model, imgs, labels, n_aug_views, aug_pipe, bs=512):
    """TTA with BatchNorm adaptation: re-estimate BN stats from augmented test batch."""
    # Save original BN state
    orig_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Set BN layers to training mode (recalculate stats) but keep rest in eval
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.train()  # This makes BN recompute running stats
            m.momentum = 1.0  # Use batch stats directly

    # Forward pass through augmented views to adapt BN
    n = len(imgs)
    for _ in range(min(n_aug_views, 10)):  # Adapt with up to 10 passes
        for i in range(0, n, bs):
            xb_aug = aug_pipe(imgs[i:i+bs])
            _ = model(xb_aug)

    # Now evaluate with adapted BN
    model.eval()
    all_probs = []
    # Include original
    orig_probs = []
    for i in range(0, n, bs):
        orig_probs.append(torch.softmax(model(imgs[i:i+bs]), dim=1))
    all_probs.append(torch.cat(orig_probs, dim=0))

    for _ in range(n_aug_views):
        view_probs = []
        for i in range(0, n, bs):
            xb_aug = aug_pipe(imgs[i:i+bs])
            view_probs.append(torch.softmax(model(xb_aug), dim=1))
        all_probs.append(torch.cat(view_probs, dim=0))

    stacked = torch.stack(all_probs, dim=0)
    avg_probs = stacked.mean(dim=0)
    preds = avg_probs.argmax(dim=1)
    acc = (preds == labels).float().mean().item()

    # Restore original BN state
    model.load_state_dict(orig_state)
    return acc


def main():
    print("="*60)
    print("FOLLOW-UP: TTA with Original + BN Adaptation")
    print("="*60)

    N_VIEWS = [1, 5, 10, 25, 50]
    results = {}
    aug_pipe = make_aug_pipeline()

    for ds_name in ["pathmnist", "bloodmnist"]:
        print(f"\n{'='*50}")
        print(f"DATASET: {ds_name}")
        print(f"{'='*50}")
        splits, nc, label = load_dataset(ds_name)
        train_imgs, train_labels = splits["train"]
        val_imgs, val_labels = splits["val"]
        test_imgs, test_labels = splits["test"]
        print(f"  Test: {len(test_imgs)} samples")

        ds_results = {}
        for model_name, model_cls in MODELS.items():
            print(f"\n  --- {model_name} ---")
            model = model_cls(nc).to(DEVICE)
            best_val = train_model(model, train_imgs, train_labels, val_imgs, val_labels)

            base_acc = evaluate(model, test_imgs, test_labels)
            print(f"  Base: {base_acc:.4f}")

            mr = {"base": base_acc, "tta_no_orig": {}, "tta_with_orig": {}, "tta_bn_adapt": {}}

            for nv in N_VIEWS:
                # Standard TTA (no original)
                acc_no, t1 = tta_evaluate(model, test_imgs, test_labels, nv, aug_pipe,
                                          include_original=False)
                # TTA with original
                acc_with, t2 = tta_evaluate(model, test_imgs, test_labels, nv, aug_pipe,
                                            include_original=True)
                # TTA with BN adaptation
                acc_bn = tta_with_bn_adapt(model, test_imgs, test_labels, nv, aug_pipe)

                d_no = acc_no - base_acc
                d_with = acc_with - base_acc
                d_bn = acc_bn - base_acc

                print(f"    N={nv:2d}: no_orig={acc_no:.4f}({d_no:+.4f}) "
                      f"with_orig={acc_with:.4f}({d_with:+.4f}) "
                      f"bn_adapt={acc_bn:.4f}({d_bn:+.4f})")

                mr["tta_no_orig"][nv] = acc_no
                mr["tta_with_orig"][nv] = acc_with
                mr["tta_bn_adapt"][nv] = acc_bn

            ds_results[model_name] = mr
            del model
            torch.cuda.empty_cache()

        results[ds_name] = ds_results

    # Save results
    with open(os.path.join(OUTPUT_DIR, "followup_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot: Figure 7 — TTA variant comparison
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for di, ds_name in enumerate(["pathmnist", "bloodmnist"]):
        for mi, mn in enumerate(["SmallCNN", "ResNet18"]):
            ax = axes[di][mi]
            r = results[ds_name][mn]
            nviews = sorted(r["tta_no_orig"].keys(), key=int)

            base = r["base"]
            ax.axhline(base, color="black", linestyle="--", linewidth=1.5, label=f"Baseline ({base:.3f})")

            ax.plot(nviews, [r["tta_no_orig"][n] for n in nviews],
                    "o-", label="TTA (aug only)", color="C0", linewidth=2)
            ax.plot(nviews, [r["tta_with_orig"][n] for n in nviews],
                    "s-", label="TTA (orig + aug)", color="C1", linewidth=2)
            ax.plot(nviews, [r["tta_bn_adapt"][n] for n in nviews],
                    "^-", label="TTA (BN adapt + orig)", color="C2", linewidth=2)

            ds_label = "Pathology" if ds_name == "pathmnist" else "Blood Cell"
            ax.set_title(f"{ds_label} — {mn}")
            ax.set_xlabel("Augmented Views")
            ax.set_ylabel("Test Accuracy")
            ax.legend(fontsize=8)

    fig.suptitle("Effect of Including Original Image and BatchNorm Adaptation on TTA",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig7_tta_variants.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "fig7_tta_variants.pdf"), bbox_inches="tight")
    plt.close()
    print(f"\nSaved fig7_tta_variants")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY: Does including original / BN adaptation help?")
    print("="*70)
    for ds_name in ["pathmnist", "bloodmnist"]:
        for mn in ["SmallCNN", "ResNet18"]:
            r = results[ds_name][mn]
            base = r["base"]
            best_no = max(r["tta_no_orig"].values())
            best_with = max(r["tta_with_orig"].values())
            best_bn = max(r["tta_bn_adapt"].values())
            print(f"{ds_name:12s} {mn:10s}: base={base:.4f} | "
                  f"aug_only={best_no:.4f}({best_no-base:+.4f}) | "
                  f"orig+aug={best_with:.4f}({best_with-base:+.4f}) | "
                  f"bn_adapt={best_bn:.4f}({best_bn-base:+.4f})")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")
