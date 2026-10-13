# ============================================================
# FINAL RESNET50 TEST EVALUATION
# Diabetic Retinopathy Severity Classification
# ============================================================

import os
import json
import time
import numpy as np
import pandas as pd
import torch

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Users\sheev\OneDrive\Desktop\DL proj"

TEST_DIR = os.path.join(
    BASE_DIR,
    "DL dataset",
    "DR_balanced_split",
    "test"
)

# IMPORTANT:
# Your actual .pth file is directly inside DL proj
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "resnet50_balanced_best.pth"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "final_evaluation_results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Image configuration
IMAGE_SIZE = 224
BATCH_SIZE = 8

# RTX 2050 has limited VRAM
NUM_WORKERS = 0

# ============================================================
# 2. DEVICE
# ============================================================

print("=" * 70)
print("FINAL RESNET50 TEST EVALUATION")
print("=" * 70)

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")

    print(f"GPU             : {torch.cuda.get_device_name(0)}")

    gpu_memory = (
        torch.cuda.get_device_properties(0).total_memory
        / (1024 ** 3)
    )

    print(f"GPU Memory      : {gpu_memory:.2f} GB")

else:
    DEVICE = torch.device("cpu")
    print("GPU             : Not available")

print(f"Device          : {DEVICE}")

print("=" * 70)

# ============================================================
# 3. CHECK PATHS
# ============================================================

print("\nChecking paths...")

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(
        f"\nTest directory not found:\n{TEST_DIR}"
    )

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\nModel file not found:\n{MODEL_PATH}"
    )

print("Test directory found:")
print(TEST_DIR)

print("\nModel found:")
print(MODEL_PATH)

# ============================================================
# 4. IMAGE TRANSFORMATION
# ============================================================

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# 5. LOAD TEST DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING TEST DATASET")
print("=" * 70)

test_dataset = datasets.ImageFolder(
    root=TEST_DIR,
    transform=test_transform
)

print(f"Total test images : {len(test_dataset)}")

# ============================================================
# 6. AUTOMATIC CLASS DETECTION
# ============================================================

class_names = test_dataset.classes
class_to_idx = test_dataset.class_to_idx

NUM_CLASSES = len(class_names)

print("\nDetected classes:")

for index, class_name in enumerate(class_names):
    print(f"{index}: {class_name}")

print(f"\nNumber of classes : {NUM_CLASSES}")

# Save class mapping
class_mapping = {
    "classes": class_names,
    "class_to_idx": class_to_idx
}

with open(
    os.path.join(
        OUTPUT_DIR,
        "class_mapping.json"
    ),
    "w"
) as f:
    json.dump(
        class_mapping,
        f,
        indent=4
    )

# ============================================================
# 7. CREATE TEST DATALOADER
# ============================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

print("\nTest DataLoader created.")

# ============================================================
# 8. LOAD RESNET50
# ============================================================

print("\n" + "=" * 70)
print("LOADING RESNET50")
print("=" * 70)

model = models.resnet50(weights=None)

# Replace final classification layer
model.fc = torch.nn.Linear(
    model.fc.in_features,
    NUM_CLASSES
)

# ============================================================
# 9. LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

# ------------------------------------------------------------
# Handle different checkpoint formats
# ------------------------------------------------------------

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint["state_dict"]

    else:

        # Assume dictionary itself is state_dict
        state_dict = checkpoint

else:

    state_dict = checkpoint

# Remove possible "module." prefix
clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("module."):
        clean_key = key.replace(
            "module.",
            "",
            1
        )
    else:
        clean_key = key

    clean_state_dict[clean_key] = value

# Load weights
try:

    model.load_state_dict(
        clean_state_dict,
        strict=True
    )

    print("Model weights loaded successfully.")

except RuntimeError as e:

    print("\nStrict loading failed.")
    print("Trying non-strict loading...\n")

    missing, unexpected = model.load_state_dict(
        clean_state_dict,
        strict=False
    )

    print("Missing keys:")
    print(missing)

    print("\nUnexpected keys:")
    print(unexpected)

# Move model to device
model = model.to(DEVICE)

model.eval()

print("\nModel architecture:")
print("ResNet50")

print(f"Output classes : {NUM_CLASSES}")

# ============================================================
# 10. EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("STARTING TEST EVALUATION")
print("=" * 70)

all_labels = []
all_predictions = []
all_probabilities = []

start_time = time.time()

# Disable gradients
with torch.no_grad():

    for batch_index, (images, labels) in enumerate(
        test_loader,
        start=1
    ):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        # Forward pass
        outputs = model(images)

        # Probabilities
        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        # Predicted class
        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        # Store results
        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )

        if batch_index % 10 == 0:

            processed = min(
                batch_index * BATCH_SIZE,
                len(test_dataset)
            )

            print(
                f"Processed {processed}/{len(test_dataset)} images"
            )

evaluation_time = time.time() - start_time

# Convert to NumPy
all_labels = np.array(
    all_labels
)

all_predictions = np.array(
    all_predictions
)

all_probabilities = np.array(
    all_probabilities
)

# ============================================================
# 11. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

# Weighted metrics
weighted_precision = precision_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

weighted_recall = recall_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

weighted_f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

# ============================================================
# 12. PRINT FINAL METRICS
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(
    f"Test Images       : {len(all_labels)}"
)

print(
    f"Accuracy          : {accuracy:.4f}"
)

print(
    f"Macro Precision   : {precision:.4f}"
)

print(
    f"Macro Recall      : {recall:.4f}"
)

print(
    f"Macro F1          : {f1:.4f}"
)

print(
    f"Weighted Precision: {weighted_precision:.4f}"
)

print(
    f"Weighted Recall   : {weighted_recall:.4f}"
)

print(
    f"Weighted F1       : {weighted_f1:.4f}"
)

print(
    f"Evaluation Time   : {evaluation_time / 60:.2f} minutes"
)

# ============================================================
# 13. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names,
    digits=4,
    zero_division=0
)

print(report)

# Save report
report_path = os.path.join(
    OUTPUT_DIR,
    "resnet50_test_classification_report.txt"
)

with open(
    report_path,
    "w"
) as f:

    f.write(
        "FINAL RESNET50 TEST EVALUATION\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Test Images : {len(all_labels)}\n"
    )

    f.write(
        f"Accuracy : {accuracy:.4f}\n"
    )

    f.write(
        f"Macro Precision : {precision:.4f}\n"
    )

    f.write(
        f"Macro Recall : {recall:.4f}\n"
    )

    f.write(
        f"Macro F1 : {f1:.4f}\n"
    )

    f.write(
        f"Weighted Precision : {weighted_precision:.4f}\n"
    )

    f.write(
        f"Weighted Recall : {weighted_recall:.4f}\n"
    )

    f.write(
        f"Weighted F1 : {weighted_f1:.4f}\n"
    )

    f.write(
        f"Evaluation Time : {evaluation_time:.2f} seconds\n\n"
    )

    f.write(
        "CLASSIFICATION REPORT\n"
    )

    f.write("=" * 70 + "\n")

    f.write(report)

# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)

# Save confusion matrix CSV
cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

cm_path = os.path.join(
    OUTPUT_DIR,
    "resnet50_test_confusion_matrix.csv"
)

cm_df.to_csv(
    cm_path
)

# ============================================================
# 15. SAVE METRICS JSON
# ============================================================

metrics = {
    "model": "ResNet50",
    "test_images": int(len(all_labels)),
    "number_of_classes": int(NUM_CLASSES),

    "accuracy": float(accuracy),

    "macro_precision": float(precision),
    "macro_recall": float(recall),
    "macro_f1": float(f1),

    "weighted_precision": float(
        weighted_precision
    ),

    "weighted_recall": float(
        weighted_recall
    ),

    "weighted_f1": float(
        weighted_f1
    ),

    "evaluation_time_seconds": float(
        evaluation_time
    ),

    "class_names": class_names
}

metrics_path = os.path.join(
    OUTPUT_DIR,
    "resnet50_test_metrics.json"
)

with open(
    metrics_path,
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )

# ============================================================
# 16. SAVE PREDICTIONS
# ============================================================

# Get image paths
image_paths = [
    sample[0]
    for sample in test_dataset.samples
]

prediction_data = []

for i in range(len(all_labels)):

    true_index = int(
        all_labels[i]
    )

    predicted_index = int(
        all_predictions[i]
    )

    prediction_data.append({

        "image_path":
            image_paths[i],

        "true_class":
            class_names[true_index],

        "predicted_class":
            class_names[predicted_index],

        "correct":
            bool(
                true_index == predicted_index
            ),

        "confidence":
            float(
                np.max(
                    all_probabilities[i]
                )
            )
    })

predictions_df = pd.DataFrame(
    prediction_data
)

predictions_path = os.path.join(
    OUTPUT_DIR,
    "resnet50_test_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)

# ============================================================
# 17. SAVE PER-CLASS METRICS
# ============================================================

per_class_report = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

per_class_rows = []

for class_name in class_names:

    row = per_class_report[class_name]

    per_class_rows.append({

        "class":
            class_name,

        "precision":
            row["precision"],

        "recall":
            row["recall"],

        "f1_score":
            row["f1-score"],

        "support":
            int(row["support"])
    })

per_class_df = pd.DataFrame(
    per_class_rows
)

per_class_path = os.path.join(
    OUTPUT_DIR,
    "resnet50_per_class_metrics.csv"
)

per_class_df.to_csv(
    per_class_path,
    index=False
)

# ============================================================
# 18. FINAL OUTPUT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFiles created:")

print(
    f"1. {report_path}"
)

print(
    f"2. {cm_path}"
)

print(
    f"3. {metrics_path}"
)

print(
    f"4. {predictions_path}"
)

print(
    f"5. {per_class_path}"
)

print(
    f"6. {os.path.join(OUTPUT_DIR, 'class_mapping.json')}"
)

print("\n" + "=" * 70)
print("FINAL METRICS")
print("=" * 70)

print(
    f"Accuracy       : {accuracy * 100:.2f}%"
)

print(
    f"Macro F1       : {f1 * 100:.2f}%"
)

print(
    f"Weighted F1    : {weighted_f1 * 100:.2f}%"
)

print("=" * 70)