import warnings
from datetime import datetime
import numpy as np
import time  # Add at the top with other imports
import logging
logger = logging.getLogger(__name__)


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

## DATASET REFERENCE

# If you want to use crop-pest-and-disease-detection, you can refer to the following info:
# dataset_path = os.path.join(os.environ["AI_SCIENTIST_ROOT"], "sarscov2-ctscan-dataset")
# Categories: ['COVID', 'non-COVID']
# where each category is a folder in the dataset path.
# Make sure to use IMAGE_SIZE = 64 for this dataset.
# Make sure to split the dataset into train, val, and test.
# Data card:
# We build a public available SARS-CoV-2 CT scan dataset, containing 125 CT scans
# that are positive for SARS-CoV-2 infection (COVID-19) and 122 CT scans for patients
# non-infected by SARS-CoV-2, 247 CT scans in total.
# These data have been collected from real patients in hospitals from Sao Paulo, Brazil.
# The aim of this dataset is to encourage the research and development of artificial intelligent methods
# which are able to identify if a person is infected by SARS-CoV-2 through the analysis of his/her CT scans.
# The dataset is available at:
# www.kaggle.com/plameneduardo/sarscov2-ctscan-dataset
# Please cite:
# Soares, Eduardo, Angelov, Plamen, Biaso, Sarah, Higa Froes, Michele, and Kanda Abe, Daniel. "SARS-CoV-2 CT-scan dataset: A large dataset of real patients CT scans for SARS-CoV-2 identification." medRxiv (2020). doi: https://doi.org/10.1101/2020.04.24.20078584.
# Angelov, P., & Soares, E. (2020). Towards explainable deep neural networks (xDNN). Neural Networks, 130, 185-194.


## PRE-TRAINED MODELS REFERENCE

## Example: load a pre-trained model, use it to extract features from images, and calculate the similarity score between two images
from transformers import pipeline
from PIL import Image
import requests

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = pipeline(
    task="image-feature-extraction",
    model="google/vit-base-patch16-384",
    device=device,
    pool=True,
)
img_urls = [
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.jpeg",
]
image_real = Image.open(requests.get(img_urls[0], stream=True).raw).convert("RGB")
image_gen = Image.open(requests.get(img_urls[1], stream=True).raw).convert("RGB")
outputs = pipe([image_real, image_gen])
similarity_score = cosine_similarity(
    torch.Tensor(outputs[0]), torch.Tensor(outputs[1]), dim=1
)

# other image models:
pipe = pipeline(
    task="image-feature-extraction",
    model="facebook/dinov2-base",
    device=device,
    pool=True,
)
pipe = pipeline(
    task="image-feature-extraction",
    model="microsoft/rad-dino",
    device=device,
    pool=True,
)  # trained to encode chest X-rays

## Example: extract features from text
feature_extractor = pipeline(
    "feature-extraction", framework="pt", model="facebook/bart-base"
)
text = "Transformers is an awesome library!"
# Reducing along the first dimension to get a 768 dimensional array
embed = feature_extractor(text, return_tensors="pt")[0].numpy().mean(axis=0)


## MINI-IMAGENET REFERENCE

# If you want to use mini-imagenet, you can refer to the following code:

# 1. Configuration
BATCH_SIZE = 512  # Increased from 128 to utilize H100's memory
LEARNING_RATE = 3e-3  # Increased for faster convergence
WEIGHT_DECAY = 1e-2
IMAGE_SIZE = 84
NUM_WORKERS = 8  # Reduced as too many workers can cause overhead
DATASET_NAME = "mini-imagenet"
NUM_EPOCHS = 20  # Increased for better convergence
STEPS_TO_LOG = 25  # Reduced for more frequent feedback
NUM_TEST_BATCHES = 20  # Reduced while maintaining reasonable evaluation
DATASET = "timm/mini-imagenet"
WARMUP_EPOCHS = 2

transform = T.Compose(
    [
        T.Lambda(lambda img: img.convert("RGB")),
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
weights = None

# 2. Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Load the Hugging Face ImageNet dataset
train_dataset_hf = load_dataset(
    DATASET,  # or the appropriate dataset name
    split="train",
    trust_remote_code=True,  # Allow running custom code from the dataset
)

val_dataset_hf = load_dataset(
    DATASET,  # or the appropriate dataset name
    split="validation",
    trust_remote_code=True,  # Allow running custom code from the dataset
)

test_dataset_hf = load_dataset(
    DATASET,  # or the appropriate dataset name
    split="test",
    trust_remote_code=True,  # Allow running custom code from the dataset
)


# 4. Create a custom PyTorch Dataset to apply transforms on-the-fly
class HuggingFaceImageNet(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        # sample["image"] is a PIL Image
        img = sample["image"]
        label = sample["label"]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


train_dataset = HuggingFaceImageNet(train_dataset_hf, transform=transform)
val_dataset = HuggingFaceImageNet(val_dataset_hf, transform=transform)
test_dataset = HuggingFaceImageNet(test_dataset_hf, transform=transform)

# 5. Create DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,  # Decodes/transforms in parallel
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

# 6. Define the model, loss function, and optimizer
model = resnet50(weights=weights)
model = model.to(device)

start_time = time.time()
# Use torch.compile with safer settings
if torch.cuda.is_available():
    try:
        model = torch.compile(model)
    except Exception as e:
        logger.info(f"Warning: torch.compile failed, falling back to eager mode. Error: {e}")

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# SGD with momentum and nesterov
optimizer = optim.SGD(
    model.parameters(),
    lr=0.1,  # Higher initial LR for SGD
    momentum=0.9,
    weight_decay=WEIGHT_DECAY,  # Lower weight decay for SGD
    nesterov=True,  # Enable Nesterov momentum
)

# Modified learning rate schedule for SGD

scheduler = optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[
        optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS * len(train_loader)
        ),
        optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=(NUM_EPOCHS - WARMUP_EPOCHS) * len(train_loader),
            eta_min=1e-6,  # Lower minimum LR for SGD
        ),
    ],
    milestones=[WARMUP_EPOCHS * len(train_loader)],
)

# Add gradient clipping to prevent instability
GRAD_CLIP_NORM = 2.0


# Helper function to calculate accuracy
def calculate_accuracy(model, data_loader, device, max_batches=None):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            if max_batches and batch_idx >= max_batches:
                break
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


# Setup logging arrays and checkpoint directory before training loop
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"{DATASET_NAME}_training_log_{timestamp}.npy"
checkpoint_dir = f"{DATASET_NAME}_checkpoints_{timestamp}"
os.makedirs(checkpoint_dir, exist_ok=True)

# Initialize arrays to store metrics
metrics = {
    "epoch": [],
    "step": [],
    "loss": [],
    "train_accuracy": [],
    "val_accuracy": [],
    "test_accuracy": [],
}

# 9. Training Loop
model.train()
best_val_accuracy = 0.0
for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()
    running_loss = 0.0

    for step, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward + Optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        # Print progress and calculate accuracies every 1000 steps
        if (step + 1) % STEPS_TO_LOG == 0:
            elapsed = time.time() - start_time
            epoch_elapsed = time.time() - epoch_start_time
            avg_loss = running_loss / STEPS_TO_LOG
            # Calculate accuracies on a subset of data (5 batches) for efficiency
            train_accuracy = calculate_accuracy(
                model, train_loader, device, max_batches=NUM_TEST_BATCHES
            )
            val_accuracy = calculate_accuracy(
                model, val_loader, device, max_batches=NUM_TEST_BATCHES
            )
            test_accuracy = calculate_accuracy(
                model, test_loader, device, max_batches=NUM_TEST_BATCHES
            )

            # Log to console
            logger.info(
                f"Epoch [{epoch + 1}/{NUM_EPOCHS}], "
                f"Step [{step + 1}/{len(train_loader)}], "
                f"Loss: {loss.item():.4f}, "
                f"Total T: {elapsed:.2f}s, "
                f"Epoch T: {epoch_elapsed:.2f}s, "
                f"Train Acc: {train_accuracy:.2f}%, "
                f"Val Acc: {val_accuracy:.2f}%, "
                f"Test Acc: {test_accuracy:.2f}%"
            )

            # Store metrics
            metrics["epoch"].append(epoch + 1)
            metrics["step"].append(step + 1)
            metrics["loss"].append(avg_loss)
            metrics["train_accuracy"].append(train_accuracy)
            metrics["val_accuracy"].append(val_accuracy)
            metrics["test_accuracy"].append(test_accuracy)

            # Save metrics to numpy file
            np.save(log_file, metrics)

            # Save checkpoint if we have the best validation accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                perf_string = f"{val_accuracy:.2f}"
                perf_string = perf_string.replace(".", "_")
                checkpoint_path = os.path.join(
                    checkpoint_dir,
                    f"model_epoch{epoch+1}_step{step+1}_val{perf_string}.pt",
                )
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "step": step + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": avg_loss,
                        "val_accuracy": val_accuracy,
                    },
                    checkpoint_path,
                )
                np.save(log_file, metrics)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            running_loss = 0.0
            model.train()  # Set back to training mode

    # Reset epoch timer and print epoch summary
    epoch_time = time.time() - epoch_start_time
    epoch_start_time = time.time()
    logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")

    # Print total training time at the end
    total_time = time.time() - start_time
    hours = total_time // 3600
    minutes = (total_time % 3600) // 60
    seconds = total_time % 60
    logger.info(f"Total training time: {hours:.0f}h {minutes:.0f}m {seconds:.2f}s")

logger.info("Training finished!")

# Save final metrics and model
np.save(log_file, metrics)
final_checkpoint_path = os.path.join(checkpoint_dir, "model_final.pt")
torch.save(
    {
        "epoch": NUM_EPOCHS,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
        "val_accuracy": val_accuracy,
    },
    final_checkpoint_path,
)
