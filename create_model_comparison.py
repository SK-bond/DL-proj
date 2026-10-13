import os
import pandas as pd

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\sheev\OneDrive\Desktop\DL proj"

RESNET_DIR = os.path.join(
    BASE_DIR,
    "resnet50_balanced_results"
)

DENSENET_DIR = os.path.join(
    BASE_DIR,
    "densenet121_balanced_results"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "model_comparison_results"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "model_comparison.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FIND HISTORY FILES
# ============================================================

def find_history_file(folder):
    if not os.path.exists(folder):
        return None

    files = os.listdir(folder)

    history_files = [
        f for f in files
        if "history" in f.lower() and f.lower().endswith(".csv")
    ]

    if not history_files:
        return None

    return os.path.join(folder, history_files[0])


resnet_history = find_history_file(RESNET_DIR)
densenet_history = find_history_file(DENSENET_DIR)


print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(f"\nResNet50 history:")
print(resnet_history)

print(f"\nDenseNet121 history:")
print(densenet_history)


# ============================================================
# CHECK FILES
# ============================================================

if resnet_history is None:
    raise FileNotFoundError(
        f"\nResNet50 history file not found in:\n{RESNET_DIR}"
    )

if densenet_history is None:
    raise FileNotFoundError(
        f"\nDenseNet121 history file not found in:\n{DENSENET_DIR}"
    )


# ============================================================
# LOAD HISTORIES
# ============================================================

resnet = pd.read_csv(resnet_history)
densenet = pd.read_csv(densenet_history)


# ============================================================
# GET BEST VALIDATION RESULTS
# ============================================================

resnet_best = resnet.loc[
    resnet["val_f1"].idxmax()
]

densenet_best = densenet.loc[
    densenet["val_f1"].idxmax()
]


# ============================================================
# CREATE COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame({

    "Model": [
        "ResNet50",
        "DenseNet121"
    ],

    "Best Epoch": [
        int(resnet_best["epoch"]),
        int(densenet_best["epoch"])
    ],

    "Train Accuracy": [
        resnet_best["train_accuracy"],
        densenet_best["train_accuracy"]
    ],

    "Validation Accuracy": [
        resnet_best["val_accuracy"],
        densenet_best["val_accuracy"]
    ],

    "Validation Precision": [
        resnet_best["val_precision"],
        densenet_best["val_precision"]
    ],

    "Validation Recall": [
        resnet_best["val_recall"],
        densenet_best["val_recall"]
    ],

    "Validation F1": [
        resnet_best["val_f1"],
        densenet_best["val_f1"]
    ],

    "Validation Loss": [
        resnet_best["val_loss"],
        densenet_best["val_loss"]
    ]
})


# ============================================================
# ROUND VALUES
# ============================================================

numeric_columns = comparison.columns[2:]

comparison[numeric_columns] = comparison[numeric_columns].round(4)


# ============================================================
# SAVE CSV
# ============================================================

comparison.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON CREATED")
print("=" * 70)

print(comparison.to_string(index=False))

print("\nSaved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)