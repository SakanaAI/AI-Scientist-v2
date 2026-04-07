import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset
import medmnist
from medmnist import PathMNIST, DermaMNIST, BloodMNIST

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 28  # MedMNIST images are 28x28
NUM_CHANNELS = 3  # RGB


# ============================================================
# DATASET LOADING
# ============================================================

MEDMNIST_CLASSES = {
    "pathmnist": (PathMNIST, 9),
    "dermamnist": (DermaMNIST, 7),
    "bloodmnist": (BloodMNIST, 8),
}


def load_medmnist_dataset(subset_name="pathmnist"):
    """Load a MedMNIST v2 subset using the medmnist package.

    Available subsets: pathmnist (9 classes), dermamnist (7 classes),
    bloodmnist (8 classes).

    Returns dict with 'train', 'val', 'test' keys, each containing
    (images_tensor, labels_tensor), and num_classes.
    """
    dataset_cls, num_classes = MEDMNIST_CLASSES[subset_name]
    transform = T.Compose([T.ToTensor()])  # Converts PIL to (C, H, W) float [0,1]

    splits = {}
    for split_name in ["train", "val", "test"]:
        ds = dataset_cls(split=split_name if split_name != "val" else "val",
                         transform=transform, download=True)
        images = []
        labels = []
        for img, label in ds:
            # img is (C, H, W) tensor from ToTensor
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)  # grayscale -> RGB
            images.append(img)
            # label is numpy array like [0], extract the int
            labels.append(int(label[0]) if hasattr(label, '__len__') else int(label))

        images = torch.stack(images)  # (N, 3, 28, 28)
        labels = torch.tensor(labels, dtype=torch.long)
        splits[split_name] = (images, labels)

    return splits, num_classes


# ============================================================
# MODEL DEFINITIONS
# ============================================================

class LogisticRegressionModel(nn.Module):
    """Logistic regression on flattened pixel inputs."""
    def __init__(self, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(NUM_CHANNELS * IMAGE_SIZE * IMAGE_SIZE, num_classes)

    def forward(self, x):
        return self.linear(self.flatten(x))


class MLPModel(nn.Module):
    """2-layer MLP with ReLU activation."""
    def __init__(self, num_classes, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(NUM_CHANNELS * IMAGE_SIZE * IMAGE_SIZE, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class SmallCNN(nn.Module):
    """3-layer CNN suitable for 28x28 images."""
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class ResNet18Small(nn.Module):
    """ResNet-18 adapted for 28x28 inputs (smaller first conv, no initial pool)."""
    def __init__(self, num_classes):
        super().__init__()
        from torchvision.models import resnet18
        self.model = resnet18(weights=None, num_classes=num_classes)
        # Replace first conv for 28x28 images (smaller kernel, no stride)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()  # Remove initial pooling

    def forward(self, x):
        return self.model(x)


def get_model(model_name, num_classes):
    """Factory function for model creation."""
    models = {
        "logreg": LogisticRegressionModel,
        "mlp": MLPModel,
        "smallcnn": SmallCNN,
        "resnet18": ResNet18Small,
    }
    return models[model_name](num_classes).to(DEVICE)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# TRAINING
# ============================================================

def train_model(model, train_images, train_labels, val_images, val_labels,
                epochs=15, batch_size=128, lr=1e-3):
    """Train a model and return training history."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(train_images, train_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(train_ds)
        val_acc = evaluate_accuracy(model, val_images, val_labels)
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, val_acc={val_acc:.4f}")

    return history


# ============================================================
# EVALUATION (Standard + TTA)
# ============================================================

def evaluate_accuracy(model, images, labels, batch_size=256):
    """Standard single-pass evaluation."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            xb = images[i:i+batch_size].to(DEVICE)
            yb = labels[i:i+batch_size].to(DEVICE)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total


def get_augmentation_transforms(strategy="mixed"):
    """Get augmentation transforms for TTA.

    Strategies:
        - "geometric": flips, rotation, crop-resize
        - "intensity": brightness, contrast, Gaussian noise
        - "mixed": all combined
    """
    if strategy == "geometric":
        return T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(15),
            T.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        ])
    elif strategy == "intensity":
        return T.Compose([
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.GaussianBlur(kernel_size=3, sigma=(0.01, 0.5)),
        ])
    elif strategy == "mixed":
        return T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(15),
            T.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            T.ColorJitter(brightness=0.2, contrast=0.2),
        ])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def evaluate_with_tta(model, images, labels, n_views=10, strategy="mixed",
                      aggregation="mean", batch_size=256):
    """Evaluate with test-time augmentation.

    Args:
        model: trained model
        images: test images tensor (N, C, H, W) in [0, 1]
        labels: test labels tensor (N,)
        n_views: number of augmented views per image
        strategy: augmentation strategy ("geometric", "intensity", "mixed")
        aggregation: how to combine predictions ("mean", "vote", "confidence")
        batch_size: batch size for inference

    Returns:
        accuracy: float
        avg_probs: (N, num_classes) mean softmax probabilities
    """
    model.eval()
    aug_transform = get_augmentation_transforms(strategy)
    n_samples = len(images)

    # Collect softmax outputs for each view
    all_probs = []

    for view_idx in range(n_views):
        view_probs = []
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                xb = images[i:i+batch_size].clone()
                # Apply augmentation to each image in batch
                augmented = torch.stack([aug_transform(img) for img in xb])
                augmented = augmented.to(DEVICE)
                logits = model(augmented)
                probs = torch.softmax(logits, dim=1)
                view_probs.append(probs.cpu())
        all_probs.append(torch.cat(view_probs, dim=0))

    # Stack: (n_views, N, num_classes)
    all_probs = torch.stack(all_probs, dim=0)

    if aggregation == "mean":
        avg_probs = all_probs.mean(dim=0)
        preds = avg_probs.argmax(dim=1)
    elif aggregation == "vote":
        votes = all_probs.argmax(dim=2)  # (n_views, N)
        preds = torch.mode(votes, dim=0).values
        avg_probs = all_probs.mean(dim=0)
    elif aggregation == "confidence":
        # Weight each view by its max confidence
        max_confs = all_probs.max(dim=2, keepdim=True).values  # (n_views, N, 1)
        weighted = all_probs * max_confs
        avg_probs = weighted.sum(dim=0) / max_confs.sum(dim=0)
        preds = avg_probs.argmax(dim=1)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    accuracy = (preds == labels).float().mean().item()
    return accuracy, avg_probs.numpy()


def compute_ece(probs, labels, n_bins=15):
    """Compute Expected Calibration Error.

    Args:
        probs: (N, num_classes) softmax probabilities
        labels: (N,) true labels
        n_bins: number of calibration bins

    Returns:
        ece: float
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += mask.sum() * abs(bin_acc - bin_conf)

    return ece / len(labels)


# ============================================================
# RESULTS SAVING
# ============================================================

def save_results(results, filename):
    """Save results dict to .npy file."""
    np.save(filename, results, allow_pickle=True)
    print(f"Results saved to {filename}")


# ============================================================
# EXAMPLE USAGE (for reference by the tree search agent)
# ============================================================

if __name__ == "__main__":
    # This shows how to run a basic experiment.
    # The tree search agent will modify and extend this.

    DATASET = "pathmnist"
    MODEL_NAME = "smallcnn"
    EPOCHS = 10
    N_VIEWS_LIST = [1, 2, 5, 10, 25]

    print(f"Loading {DATASET}...")
    splits, num_classes = load_medmnist_dataset(DATASET)
    train_imgs, train_labels = splits["train"]
    val_imgs, val_labels = splits["val"]
    test_imgs, test_labels = splits["test"]
    print(f"  Train: {train_imgs.shape}, Val: {val_imgs.shape}, Test: {test_imgs.shape}")
    print(f"  Num classes: {num_classes}")

    print(f"\nTraining {MODEL_NAME}...")
    model = get_model(MODEL_NAME, num_classes)
    n_params = count_parameters(model)
    print(f"  Parameters: {n_params:,}")

    t0 = time.time()
    history = train_model(model, train_imgs, train_labels, val_imgs, val_labels, epochs=EPOCHS)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    # Single-pass baseline
    base_acc = evaluate_accuracy(model, test_imgs, test_labels)
    print(f"\nSingle-pass accuracy: {base_acc:.4f}")

    # TTA scaling
    results = {
        "dataset": DATASET,
        "model": MODEL_NAME,
        "n_params": n_params,
        "train_time": train_time,
        "base_accuracy": base_acc,
        "tta_results": {},
    }

    for n_views in N_VIEWS_LIST:
        t0 = time.time()
        tta_acc, tta_probs = evaluate_with_tta(
            model, test_imgs, test_labels, n_views=n_views, strategy="mixed"
        )
        tta_time = time.time() - t0
        ece = compute_ece(tta_probs, test_labels.numpy())
        print(f"  TTA(N={n_views:3d}): acc={tta_acc:.4f}, ECE={ece:.4f}, time={tta_time:.1f}s")
        results["tta_results"][n_views] = {
            "accuracy": tta_acc,
            "ece": ece,
            "inference_time": tta_time,
        }

    save_results(results, f"tta_results_{DATASET}_{MODEL_NAME}.npy")
    print("\nDone!")
