# ============================================================
# DENSENET121 - BALANCED DIABETIC RETINOPATHY TRAINING
# RTX 2050 4-GB OPTIMIZED
# ============================================================

import os
import json
import time
import copy
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import ImageFile

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split

from torchvision import datasets, transforms, models
from torchvision.models import DenseNet121_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Users\sheev\OneDrive\Desktop\DL proj"

DATASET_ROOT = os.path.join(
    BASE_DIR,
    "DL dataset",
    "DR_balanced_split"
)

TRAIN_DIR = os.path.join(
    DATASET_ROOT,
    "train"
)

TEST_DIR = os.path.join(
    DATASET_ROOT,
    "test"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "densenet121_balanced_results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

NUM_CLASSES = 5

IMAGE_SIZE = 224

BATCH_SIZE = 8

GRADIENT_ACCUMULATION = 4

EFFECTIVE_BATCH_SIZE = (
    BATCH_SIZE * GRADIENT_ACCUMULATION
)

EPOCHS = 50

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

VAL_SPLIT = 0.15

RANDOM_SEED = 42

NUM_WORKERS = 0

PIN_MEMORY = True

USE_AMP = True


# ============================================================
# CLASS NAMES
# ============================================================

EXPECTED_CLASSES = [
    "Mild_DR",
    "Moderate_DR",
    "No_DR",
    "Proliferative_DR",
    "Severe_DR"
]


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Reproducibility
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_seed(RANDOM_SEED)


# ============================================================
# GPU CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("DENSENET121 - BALANCED DIABETIC RETINOPATHY TRAINING")
print("=" * 70)

print(f"PyTorch version : {torch.__version__}")

print(
    f"CUDA available  : {torch.cuda.is_available()}"
)

if torch.cuda.is_available():

    print(
        f"GPU             : "
        f"{torch.cuda.get_device_name(0)}"
    )

    total_memory = (
        torch.cuda.get_device_properties(0).total_memory
        / (1024 ** 3)
    )

    print(
        f"GPU Memory      : "
        f"{total_memory:.2f} GB"
    )

print(f"Device          : {DEVICE}")

print("=" * 70)


# ============================================================
# 2. CHECK DATASET
# ============================================================

print("\nChecking dataset...")

if not os.path.exists(DATASET_ROOT):

    raise FileNotFoundError(
        f"\nDataset root not found:\n{DATASET_ROOT}"
    )

if not os.path.exists(TRAIN_DIR):

    raise FileNotFoundError(
        f"\nTraining directory not found:\n{TRAIN_DIR}"
    )

if not os.path.exists(TEST_DIR):

    raise FileNotFoundError(
        f"\nTesting directory not found:\n{TEST_DIR}"
    )

print("Dataset directories found successfully.")

print(f"\nDataset root:")
print(DATASET_ROOT)

print(f"\nTraining directory:")
print(TRAIN_DIR)

print(f"\nTesting directory:")
print(TEST_DIR)


# ============================================================
# 3. DATA TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


val_test_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# 4. LOAD TRAINING DATASET
# ============================================================

print("\nLoading training dataset...")

full_train_dataset = datasets.ImageFolder(
    root=TRAIN_DIR,
    transform=train_transform
)

print(
    f"Total training images : "
    f"{len(full_train_dataset)}"
)

print("\nClasses:")

for idx, class_name in enumerate(
    full_train_dataset.classes
):
    print(
        f"{idx}: {class_name}"
    )

print(
    f"\nNumber of classes: "
    f"{len(full_train_dataset.classes)}"
)


# ============================================================
# CHECK CLASS ORDER
# ============================================================

if set(full_train_dataset.classes) != set(
    EXPECTED_CLASSES
):

    raise ValueError(
        "\nDataset classes do not match expected classes.\n"
        f"Found: {full_train_dataset.classes}\n"
        f"Expected: {EXPECTED_CLASSES}"
    )

# Keep ImageFolder's actual ordering
CLASS_NAMES = full_train_dataset.classes


# ============================================================
# 5. TRAIN / VALIDATION SPLIT
# ============================================================

total_samples = len(
    full_train_dataset
)

validation_size = int(
    total_samples * VAL_SPLIT
)

training_size = (
    total_samples - validation_size
)

generator = torch.Generator()

generator.manual_seed(
    RANDOM_SEED
)

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [
        training_size,
        validation_size
    ],
    generator=generator
)


# ============================================================
# IMPORTANT:
# Validation should not use training augmentation.
# ============================================================

val_base_dataset = datasets.ImageFolder(
    root=TRAIN_DIR,
    transform=val_test_transform
)

val_dataset = torch.utils.data.Subset(
    val_base_dataset,
    val_dataset.indices
)


print("\nDataset split:")

print(
    f"Training samples   : "
    f"{len(train_dataset)}"
)

print(
    f"Validation samples : "
    f"{len(val_dataset)}"
)


# ============================================================
# 6. LOAD TEST DATASET
# ============================================================

print("\nLoading test dataset...")

test_dataset = datasets.ImageFolder(
    root=TEST_DIR,
    transform=val_test_transform
)

print(
    f"Testing samples    : "
    f"{len(test_dataset)}"
)


# ============================================================
# CHECK TEST CLASSES
# ============================================================

if test_dataset.classes != CLASS_NAMES:

    raise ValueError(
        "\nTraining and testing class order does not match.\n"
        f"Training: {CLASS_NAMES}\n"
        f"Testing : {test_dataset.classes}"
    )


# ============================================================
# 7. DATA LOADERS
# ============================================================

print("\nCreating DataLoaders...")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

print("DataLoaders created successfully.")


# ============================================================
# 8. LOAD DENSENET121
# ============================================================

print("\nLoading DenseNet121...")

weights = DenseNet121_Weights.DEFAULT

model = models.densenet121(
    weights=weights
)


# ============================================================
# REPLACE CLASSIFIER
# ============================================================

input_features = model.classifier.in_features

model.classifier = nn.Linear(
    input_features,
    NUM_CLASSES
)

model = model.to(DEVICE)


print(
    "Pretrained DenseNet121 loaded."
)

print(
    f"Classifier input features : "
    f"{input_features}"
)

print(
    f"Output classes            : "
    f"{NUM_CLASSES}"
)


# ============================================================
# 9. LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 10. OPTIMIZER
# ============================================================

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# 11. LEARNING RATE SCHEDULER
# ============================================================

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)


# ============================================================
# 12. MIXED PRECISION
# ============================================================

if DEVICE.type == "cuda" and USE_AMP:

    scaler = torch.amp.GradScaler(
        "cuda"
    )

else:

    scaler = None


# ============================================================
# 13. TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    all_predictions = []

    all_labels = []

    optimizer.zero_grad(
        set_to_none=True
    )

    epoch_start = time.time()

    total_batches = len(
        train_loader
    )

    for batch_idx, (
        images,
        labels
    ) in enumerate(
        train_loader
    ):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Mixed Precision
        # ----------------------------------------------------

        if DEVICE.type == "cuda" and USE_AMP:

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                loss = (
                    loss /
                    GRADIENT_ACCUMULATION
                )

            scaler.scale(
                loss
            ).backward()

        else:

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss = (
                loss /
                GRADIENT_ACCUMULATION
            )

            loss.backward()


        # ----------------------------------------------------
        # Gradient Accumulation
        # ----------------------------------------------------

        if (
            (batch_idx + 1)
            % GRADIENT_ACCUMULATION
            == 0
        ):

            if DEVICE.type == "cuda" and USE_AMP:

                scaler.step(
                    optimizer
                )

                scaler.update()

            else:

                optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        actual_loss = (
            loss.item()
            * GRADIENT_ACCUMULATION
        )

        running_loss += (
            actual_loss
            * images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
        )


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            (batch_idx + 1) % 250 == 0
            or
            (batch_idx + 1)
            == total_batches
        ):

            print(
                f"Batch "
                f"{batch_idx + 1}/"
                f"{total_batches}",
                end="\r"
            )


    # --------------------------------------------------------
    # Handle remaining gradients
    # --------------------------------------------------------

    if (
        total_batches
        % GRADIENT_ACCUMULATION
        != 0
    ):

        if DEVICE.type == "cuda" and USE_AMP:

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            optimizer.step()

        optimizer.zero_grad(
            set_to_none=True
        )


    epoch_loss = (
        running_loss /
        len(train_dataset)
    )

    epoch_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    epoch_precision = precision_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    epoch_recall = recall_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    epoch_f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    epoch_time = (
        time.time()
        - epoch_start
    )

    return (
        epoch_loss,
        epoch_accuracy,
        epoch_precision,
        epoch_recall,
        epoch_f1,
        epoch_time
    )


# ============================================================
# 14. VALIDATION FUNCTION
# ============================================================

def validate():

    model.eval()

    running_loss = 0.0

    all_predictions = []

    all_labels = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            if (
                DEVICE.type == "cuda"
                and USE_AMP
            ):

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                ):

                    outputs = model(
                        images
                    )

                    loss = criterion(
                        outputs,
                        labels
                    )

            else:

                outputs = model(
                    images
                )

                loss = criterion(
                    outputs,
                    labels
                )


            running_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )


    val_loss = (
        running_loss /
        len(val_dataset)
    )

    val_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    val_precision = precision_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    val_recall = recall_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    val_f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    return (
        val_loss,
        val_accuracy,
        val_precision,
        val_recall,
        val_f1
    )


# ============================================================
# 15. TRAINING CONFIGURATION
# ============================================================

print("\n")
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

print(
    f"\nMaximum epochs : {EPOCHS}"
)

print(
    f"Batch size     : {BATCH_SIZE}"
)

print(
    f"Gradient accumulation : "
    f"{GRADIENT_ACCUMULATION}"
)

print(
    f"Effective batch size  : "
    f"{EFFECTIVE_BATCH_SIZE}"
)

print(
    f"Learning rate  : "
    f"{LEARNING_RATE}"
)

print(
    f"Weight decay   : "
    f"{WEIGHT_DECAY}"
)

print(
    f"Image size     : "
    f"{IMAGE_SIZE}x{IMAGE_SIZE}"
)

print(
    f"DataLoader workers : "
    f"{NUM_WORKERS}"
)

print(
    f"Mixed precision : "
    f"{USE_AMP}"
)

print("=" * 70)


# ============================================================
# 16. HISTORY
# ============================================================

history = []

best_val_f1 = -1.0

best_model_state = None


# ============================================================
# 17. TRAINING LOOP
# ============================================================

for epoch in range(
    1,
    EPOCHS + 1
):

    print("\n")

    print("=" * 70)

    print(
        f"EPOCH {epoch}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    (
        train_loss,
        train_accuracy,
        train_precision,
        train_recall,
        train_f1,
        epoch_time
    ) = train_one_epoch()


    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    (
        val_loss,
        val_accuracy,
        val_precision,
        val_recall,
        val_f1
    ) = validate()


    # --------------------------------------------------------
    # CURRENT LEARNING RATE
    # --------------------------------------------------------

    current_lr = (
        optimizer.param_groups[0]["lr"]
    )


    # --------------------------------------------------------
    # PRINT METRICS
    # --------------------------------------------------------

    print(
        f"\nTrain Loss      : "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy  : "
        f"{train_accuracy:.4f}"
    )

    print(
        f"Train Precision : "
        f"{train_precision:.4f}"
    )

    print(
        f"Train Recall    : "
        f"{train_recall:.4f}"
    )

    print(
        f"Train F1        : "
        f"{train_f1:.4f}"
    )

    print(
        f"\nVal Loss        : "
        f"{val_loss:.4f}"
    )

    print(
        f"Val Accuracy    : "
        f"{val_accuracy:.4f}"
    )

    print(
        f"Val Precision   : "
        f"{val_precision:.4f}"
    )

    print(
        f"Val Recall      : "
        f"{val_recall:.4f}"
    )

    print(
        f"Val F1          : "
        f"{val_f1:.4f}"
    )

    print(
        f"\nLearning Rate   : "
        f"{current_lr:.8f}"
    )

    print(
        f"Epoch Time      : "
        f"{epoch_time / 60:.2f} minutes"
    )


    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    history.append({

        "epoch": epoch,

        "train_loss": train_loss,

        "train_accuracy": train_accuracy,

        "train_precision": train_precision,

        "train_recall": train_recall,

        "train_f1": train_f1,

        "val_loss": val_loss,

        "val_accuracy": val_accuracy,

        "val_precision": val_precision,

        "val_recall": val_recall,

        "val_f1": val_f1,

        "learning_rate": current_lr,

        "epoch_time_seconds": epoch_time

    })


    # --------------------------------------------------------
    # SAVE CSV AFTER EVERY EPOCH
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            "densenet121_balanced_history.csv"
        ),
        index=False
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            {
                "model_state_dict":
                    best_model_state,

                "class_names":
                    CLASS_NAMES,

                "epoch":
                    epoch,

                "val_f1":
                    val_f1,

                "val_accuracy":
                    val_accuracy

            },

            os.path.join(
                RESULTS_DIR,
                "densenet121_balanced_best.pth"
            )
        )

        print(
            f"\nBEST MODEL SAVED - "
            f"Val F1: {val_f1:.4f}"
        )


    # --------------------------------------------------------
    # STEP SCHEDULER
    # --------------------------------------------------------

    scheduler.step()


    # --------------------------------------------------------
    # GPU MEMORY CLEANUP
    # --------------------------------------------------------

    if DEVICE.type == "cuda":

        torch.cuda.empty_cache()


# ============================================================
# 18. LOAD BEST MODEL
# ============================================================

print("\n")

print("=" * 70)

print("TRAINING COMPLETED")

print("=" * 70)

best_checkpoint_path = os.path.join(
    RESULTS_DIR,
    "densenet121_balanced_best.pth"
)

checkpoint = torch.load(
    best_checkpoint_path,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print(
    f"\nBest epoch : "
    f"{checkpoint['epoch']}"
)

print(
    f"Best validation F1 : "
    f"{checkpoint['val_f1']:.4f}"
)

print(
    f"Best validation accuracy : "
    f"{checkpoint['val_accuracy']:.4f}"
)


# ============================================================
# 19. SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(
    RESULTS_DIR,
    "densenet121_balanced_final.pth"
)

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "class_names":
            CLASS_NAMES,

        "epochs":
            EPOCHS

    },
    final_model_path
)

print(
    f"\nFinal model saved:"
)

print(
    final_model_path
)


# ============================================================
# 20. TEST EVALUATION
# ============================================================

print("\n")

print("=" * 70)

print("STARTING TEST EVALUATION")

print("=" * 70)


model.eval()

test_predictions = []

test_labels = []

test_probabilities = []

test_running_loss = 0.0

test_start = time.time()


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )


        if (
            DEVICE.type == "cuda"
            and USE_AMP
        ):

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                outputs = model(
                    images
                )

                loss = criterion(
                    outputs,
                    labels
                )

        else:

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )


        test_running_loss += (
            loss.item()
            * images.size(0)
        )


        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )


        test_predictions.extend(
            predictions.cpu().numpy()
        )

        test_labels.extend(
            labels.cpu().numpy()
        )

        test_probabilities.extend(
            probabilities.cpu().numpy()
        )


test_time = (
    time.time()
    - test_start
)


# ============================================================
# 21. TEST METRICS
# ============================================================

test_loss = (
    test_running_loss /
    len(test_dataset)
)

test_accuracy = accuracy_score(
    test_labels,
    test_predictions
)

test_precision = precision_score(
    test_labels,
    test_predictions,
    average="weighted",
    zero_division=0
)

test_recall = recall_score(
    test_labels,
    test_predictions,
    average="weighted",
    zero_division=0
)

test_f1 = f1_score(
    test_labels,
    test_predictions,
    average="weighted",
    zero_division=0
)

test_macro_precision = precision_score(
    test_labels,
    test_predictions,
    average="macro",
    zero_division=0
)

test_macro_recall = recall_score(
    test_labels,
    test_predictions,
    average="macro",
    zero_division=0
)

test_macro_f1 = f1_score(
    test_labels,
    test_predictions,
    average="macro",
    zero_division=0
)


# ============================================================
# 22. PRINT TEST RESULTS
# ============================================================

print("\n")

print("=" * 70)

print("DENSENET121 TEST RESULTS")

print("=" * 70)

print(
    f"\nTest Loss       : "
    f"{test_loss:.4f}"
)

print(
    f"Accuracy        : "
    f"{test_accuracy:.4f}"
)

print(
    f"Precision       : "
    f"{test_precision:.4f}"
)

print(
    f"Recall          : "
    f"{test_recall:.4f}"
)

print(
    f"F1 Score        : "
    f"{test_f1:.4f}"
)

print(
    f"Macro Precision : "
    f"{test_macro_precision:.4f}"
)

print(
    f"Macro Recall    : "
    f"{test_macro_recall:.4f}"
)

print(
    f"Macro F1        : "
    f"{test_macro_f1:.4f}"
)

print(
    f"Evaluation Time : "
    f"{test_time / 60:.2f} minutes"
)


# ============================================================
# 23. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_labels,
    test_predictions
)

print("\n")

print("=" * 70)

print("CONFUSION MATRIX")

print("=" * 70)

print(cm)


cm_path = os.path.join(
    RESULTS_DIR,
    "densenet121_balanced_confusion_matrix.csv"
)

pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
).to_csv(
    cm_path
)


# ============================================================
# 24. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    test_labels,
    test_predictions,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

print("\n")

print("=" * 70)

print("CLASSIFICATION REPORT")

print("=" * 70)

print(report)


report_path = os.path.join(
    RESULTS_DIR,
    "densenet121_balanced_classification_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(report)


# ============================================================
# 25. SAVE TEST METRICS
# ============================================================

metrics = {

    "model":
        "DenseNet121",

    "dataset":
        "DR_balanced_split",

    "test_samples":
        len(test_dataset),

    "training_samples":
        len(train_dataset),

    "validation_samples":
        len(val_dataset),

    "num_classes":
        NUM_CLASSES,

    "epochs":
        EPOCHS,

    "batch_size":
        BATCH_SIZE,

    "gradient_accumulation":
        GRADIENT_ACCUMULATION,

    "effective_batch_size":
        EFFECTIVE_BATCH_SIZE,

    "accuracy":
        float(test_accuracy),

    "accuracy_percent":
        float(test_accuracy * 100),

    "weighted_precision":
        float(test_precision),

    "weighted_recall":
        float(test_recall),

    "weighted_f1":
        float(test_f1),

    "weighted_f1_percent":
        float(test_f1 * 100),

    "macro_precision":
        float(test_macro_precision),

    "macro_recall":
        float(test_macro_recall),

    "macro_f1":
        float(test_macro_f1),

    "macro_f1_percent":
        float(test_macro_f1 * 100),

    "test_loss":
        float(test_loss),

    "evaluation_time_seconds":
        float(test_time),

    "class_names":
        CLASS_NAMES

}


metrics_path = os.path.join(
    RESULTS_DIR,
    "densenet121_balanced_metrics.json"
)

with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


# ============================================================
# 26. SAVE TEST PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({

    "true_label":
        [
            CLASS_NAMES[x]
            for x in test_labels
        ],

    "predicted_label":
        [
            CLASS_NAMES[x]
            for x in test_predictions
        ],

    "correct":
        [
            true == pred
            for true, pred
            in zip(
                test_labels,
                test_predictions
            )
        ]

})


prediction_path = os.path.join(
    RESULTS_DIR,
    "densenet121_balanced_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# 27. TRAINING ACCURACY GRAPH
# ============================================================

history_df = pd.DataFrame(
    history
)

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    history_df["epoch"],
    history_df["train_accuracy"],
    linewidth=2,
    label="Training Accuracy"
)

plt.plot(
    history_df["epoch"],
    history_df["val_accuracy"],
    linewidth=2,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title(
    "DenseNet121 Training vs Validation Accuracy"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "densenet121_accuracy_graph.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 28. TRAINING LOSS GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    history_df["epoch"],
    history_df["train_loss"],
    linewidth=2,
    label="Training Loss"
)

plt.plot(
    history_df["epoch"],
    history_df["val_loss"],
    linewidth=2,
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "DenseNet121 Training vs Validation Loss"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "densenet121_loss_graph.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 29. F1 GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    history_df["epoch"],
    history_df["train_f1"],
    linewidth=2,
    label="Training F1"
)

plt.plot(
    history_df["epoch"],
    history_df["val_f1"],
    linewidth=2,
    label="Validation F1"
)

plt.xlabel("Epoch")

plt.ylabel("F1 Score")

plt.title(
    "DenseNet121 Training vs Validation F1"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "densenet121_f1_graph.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 30. FINAL SUMMARY
# ============================================================

print("\n")

print("=" * 70)

print("DENSENET121 TRAINING AND EVALUATION COMPLETED")

print("=" * 70)

print("\nResults saved in:")

print(RESULTS_DIR)

print("\nGenerated files:")

print(
    "  - densenet121_balanced_best.pth"
)

print(
    "  - densenet121_balanced_final.pth"
)

print(
    "  - densenet121_balanced_history.csv"
)

print(
    "  - densenet121_balanced_metrics.json"
)

print(
    "  - densenet121_balanced_classification_report.txt"
)

print(
    "  - densenet121_balanced_confusion_matrix.csv"
)

print(
    "  - densenet121_balanced_predictions.csv"
)

print(
    "  - densenet121_accuracy_graph.png"
)

print(
    "  - densenet121_loss_graph.png"
)

print(
    "  - densenet121_f1_graph.png"
)

print("\n")

print(
    f"Final Test Accuracy : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Final Weighted F1   : "
    f"{test_f1 * 100:.2f}%"
)

print(
    f"Final Macro F1      : "
    f"{test_macro_f1 * 100:.2f}%"
)

print("=" * 70)