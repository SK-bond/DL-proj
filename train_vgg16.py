
# train_vgg16.py
# VGG16 transfer learning for 5-class diabetic retinopathy classification

import os
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import VGG16_Weights

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42
EPOCHS = 50
BATCH_SIZE = 4          # Small batch for 4 GB GPU
NUM_WORKERS = 0         # Safer for Windows
IMAGE_SIZE = 224
VALIDATION_SIZE = 750   # Target validation images, if available
PATIENCE = 8

LEARNING_RATE = 1e-4
NUM_CLASSES = 5

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "DL dataset" / "DR_balanced_split"

TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

OUTPUT_DIR = PROJECT_DIR / "vgg16_results"
MODEL_DIR = PROJECT_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "vgg16_dr_best.pth"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 65)
print("VGG16 DIABETIC RETINOPATHY CLASSIFICATION")
print("=" * 65)
print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True

if not TRAIN_DIR.exists() or not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folders not found.\n"
        f"Expected training folder: {TRAIN_DIR}\n"
        f"Expected testing folder: {TEST_DIR}"
    )

# ============================================================
# 2. TRANSFORMS
# ============================================================

# ImageNet normalization is required for pretrained VGG16.
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(
        brightness=0.10,
        contrast=0.10,
        saturation=0.05
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Read the folders without augmentation first.
base_dataset = datasets.ImageFolder(TRAIN_DIR)
test_dataset = datasets.ImageFolder(TEST_DIR)

class_names = base_dataset.classes
test_class_names = test_dataset.classes

if class_names != test_class_names:
    raise ValueError(
        f"Train/test classes differ.\n"
        f"Train: {class_names}\nTest: {test_class_names}"
    )

if len(class_names) != NUM_CLASSES:
    raise ValueError(
        f"Expected 5 class folders but found {len(class_names)}: "
        f"{class_names}"
    )

print("\nClass mapping:")
for index, name in enumerate(class_names):
    print(f"{index}: {name}")

print("\nImages in train folder:", len(base_dataset))
print("Images in test folder:", len(test_dataset))

# ============================================================
# 3. STRATIFIED TRAIN / VALIDATION SPLIT
# ============================================================

all_indices = np.arange(len(base_dataset))
all_labels = np.array(base_dataset.targets)

# Keep the test folder completely separate.
# Create validation data from the train folder.
val_count = min(VALIDATION_SIZE, max(1, len(base_dataset) // 5))

if val_count >= len(base_dataset):
    raise ValueError("Not enough training images to create validation data.")

train_indices, val_indices = train_test_split(
    all_indices,
    test_size=val_count,
    random_state=SEED,
    stratify=all_labels
)

# Separate dataset objects ensure that training augmentation
# is not applied to validation images.
train_augmented_dataset = datasets.ImageFolder(
    TRAIN_DIR, transform=train_transform
)
train_eval_dataset = datasets.ImageFolder(
    TRAIN_DIR, transform=eval_transform
)
test_dataset = datasets.ImageFolder(
    TEST_DIR, transform=eval_transform
)

train_dataset = Subset(train_augmented_dataset, train_indices)
val_dataset = Subset(train_eval_dataset, val_indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda")
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda")
)

print("\nData split:")
print("Training:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Testing:", len(test_dataset))

# Save the exact split indices for reproducibility.
pd.DataFrame({
    "train_index": pd.Series(train_indices),
    "validation_index": pd.Series(val_indices)
}).to_csv(OUTPUT_DIR / "data_split_indices.csv", index=False)

# ============================================================
# 4. LOAD PRETRAINED VGG16
# ============================================================

print("\nLoading pretrained VGG16 weights...")

# First run may download ImageNet weights.
model = models.vgg16(weights=VGG16_Weights.DEFAULT)

# Freeze convolutional layers initially.
for parameter in model.features.parameters():
    parameter.requires_grad = False

# Unfreeze the final convolutional block for fine-tuning.
# VGG16's final convolutional block starts at feature index 24.
for parameter in model.features[24:].parameters():
    parameter.requires_grad = True

# Replace the large original classifier with a smaller head.
# This reduces memory use and overfitting risk.
model.classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(512, 128),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    nn.Linear(128, NUM_CLASSES)
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
    min_lr=1e-7
)

# Mixed precision reduces GPU memory usage on CUDA.
amp_enabled = (device.type == "cuda")
scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

# ============================================================
# 5. TRAINING AND VALIDATION FUNCTIONS
# ============================================================

def run_epoch(loader, training):
    model.train(training)

    total_loss = 0.0
    all_targets = []
    all_predictions = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        all_targets.extend(labels.detach().cpu().numpy())
        all_predictions.extend(predictions.detach().cpu().numpy())

    average_loss = total_loss / len(loader.dataset)

    metrics = {
        "loss": average_loss,
        "accuracy": accuracy_score(all_targets, all_predictions),
        "precision": precision_score(
            all_targets, all_predictions,
            average="macro", zero_division=0
        ),
        "recall": recall_score(
            all_targets, all_predictions,
            average="macro", zero_division=0
        ),
        "f1": f1_score(
            all_targets, all_predictions,
            average="macro", zero_division=0
        )
    }

    return metrics


# ============================================================
# 6. TRAIN FOR UP TO 50 EPOCHS
# ============================================================

history = []
best_val_f1 = -1.0
best_epoch = 0
epochs_without_improvement = 0

for epoch in range(1, EPOCHS + 1):
    start_time = time.time()

    train_metrics = run_epoch(train_loader, training=True)

    with torch.no_grad():
        val_metrics = run_epoch(val_loader, training=False)

    scheduler.step(val_metrics["f1"])
    current_lr = optimizer.param_groups[0]["lr"]
    elapsed = time.time() - start_time

    row = {
        "epoch": epoch,
        "train_loss": train_metrics["loss"],
        "train_accuracy": train_metrics["accuracy"],
        "train_precision": train_metrics["precision"],
        "train_recall": train_metrics["recall"],
        "train_f1": train_metrics["f1"],
        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_f1": val_metrics["f1"],
        "learning_rate": current_lr,
        "epoch_time_seconds": elapsed
    }

    history.append(row)
    pd.DataFrame(history).to_csv(
        OUTPUT_DIR / "vgg16_training_history.csv",
        index=False
    )

    print(
        f"\nEpoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_metrics['loss']:.4f} | "
        f"Train Acc: {train_metrics['accuracy']:.4f} | "
        f"Val Loss: {val_metrics['loss']:.4f} | "
        f"Val Acc: {val_metrics['accuracy']:.4f} | "
        f"Val Macro F1: {val_metrics['f1']:.4f} | "
        f"LR: {current_lr:.2e} | "
        f"Time: {elapsed:.1f}s"
    )

    # Select the best checkpoint using validation Macro F1.
    if val_metrics["f1"] > best_val_f1:
        best_val_f1 = val_metrics["f1"]
        best_epoch = epoch
        epochs_without_improvement = 0

        torch.save({
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "class_names": class_names,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_f1
        }, BEST_MODEL_PATH)

        print("Best VGG16 model saved.")
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}.")
        break

print("\nTraining complete.")
print("Best epoch:", best_epoch)
print("Best validation Macro F1:", round(best_val_f1, 4))
print("Best model:", BEST_MODEL_PATH)

# ============================================================
# 7. LOAD BEST CHECKPOINT AND EVALUATE TEST SET
# ============================================================

checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location=device,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

test_targets = []
test_predictions = []
test_probabilities = []
test_loss_sum = 0.0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)

        probabilities = torch.softmax(outputs.float(), dim=1)
        predictions = outputs.argmax(dim=1)

        test_loss_sum += loss.item() * images.size(0)
        test_targets.extend(labels.cpu().numpy())
        test_predictions.extend(predictions.cpu().numpy())
        test_probabilities.extend(probabilities.cpu().numpy())

test_loss = test_loss_sum / len(test_dataset)

accuracy = accuracy_score(test_targets, test_predictions)
macro_precision = precision_score(
    test_targets, test_predictions,
    average="macro", zero_division=0
)
macro_recall = recall_score(
    test_targets, test_predictions,
    average="macro", zero_division=0
)
macro_f1 = f1_score(
    test_targets, test_predictions,
    average="macro", zero_division=0
)
weighted_f1 = f1_score(
    test_targets, test_predictions,
    average="weighted", zero_division=0
)

report = classification_report(
    test_targets,
    test_predictions,
    labels=list(range(NUM_CLASSES)),
    target_names=class_names,
    digits=4,
    zero_division=0
)

with open(
    OUTPUT_DIR / "vgg16_classification_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(report)

cm = confusion_matrix(
    test_targets,
    test_predictions,
    labels=list(range(NUM_CLASSES))
)

pd.DataFrame(
    cm, index=class_names, columns=class_names
).to_csv(OUTPUT_DIR / "vgg16_confusion_matrix.csv")

fig, ax = plt.subplots(figsize=(9, 7))
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
).plot(ax=ax, cmap="Blues", xticks_rotation=35, colorbar=False)
ax.set_title("VGG16 - Test Confusion Matrix")
fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "vgg16_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

predictions_df = pd.DataFrame({
    "true_label": [class_names[i] for i in test_targets],
    "predicted_label": [class_names[i] for i in test_predictions]
})

for i, class_name in enumerate(class_names):
    predictions_df[f"prob_{class_name}"] = [
        probabilities[i] for probabilities in test_probabilities
    ]

predictions_df.to_csv(
    OUTPUT_DIR / "vgg16_test_predictions.csv",
    index=False
)

metrics = {
    "model": "VGG16",
    "best_epoch": int(best_epoch),
    "best_validation_macro_f1": float(best_val_f1),
    "test_loss": float(test_loss),
    "test_accuracy": float(accuracy),
    "macro_precision": float(macro_precision),
    "macro_recall": float(macro_recall),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "train_images": int(len(train_dataset)),
    "validation_images": int(len(val_dataset)),
    "test_images": int(len(test_dataset)),
    "classes": class_names
}

with open(
    OUTPUT_DIR / "vgg16_evaluation_metrics.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(metrics, file, indent=4)

with open(
    OUTPUT_DIR / "vgg16_evaluation_metrics.txt",
    "w",
    encoding="utf-8"
) as file:
    for key, value in metrics.items():
        file.write(f"{key}: {value}\n")

print("\n" + "=" * 65)
print("VGG16 TEST RESULTS")
print("=" * 65)
print(f"Test Loss:       {test_loss:.4f}")
print(f"Accuracy:        {accuracy:.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall:    {macro_recall:.4f}")
print(f"Macro F1:        {macro_f1:.4f}")
print(f"Weighted F1:     {weighted_f1:.4f}")
print("\nClassification Report:\n")
print(report)

# ============================================================
# 8. GENERATE TRAINING GRAPHS
# ============================================================

history_df = pd.DataFrame(history)

graphs = [
    ("train_accuracy", "val_accuracy", "Accuracy", "vgg16_accuracy_graph.png"),
    ("train_loss", "val_loss", "Loss", "vgg16_loss_graph.png"),
    ("train_f1", "val_f1", "Macro F1-score", "vgg16_f1_graph.png"),
    ("train_precision", "val_precision", "Macro Precision", "vgg16_precision_graph.png"),
    ("train_recall", "val_recall", "Macro Recall", "vgg16_recall_graph.png")
]

for train_column, val_column, title, filename in graphs:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        history_df["epoch"],
        history_df[train_column],
        label=f"Training {title}"
    )
    ax.plot(
        history_df["epoch"],
        history_df[val_column],
        label=f"Validation {title}"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(title)
    ax.set_title(f"VGG16 {title} per Epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(history_df["epoch"], history_df["learning_rate"])
ax.set_xlabel("Epoch")
ax.set_ylabel("Learning rate")
ax.set_title("VGG16 Learning Rate Schedule")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "vgg16_learning_rate_graph.png", dpi=300)
plt.close(fig)

print("\nAll outputs saved in:", OUTPUT_DIR)
print("VGG16 training and evaluation finished successfully.")
