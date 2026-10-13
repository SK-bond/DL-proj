
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# FINAL RESNET50 RESULTS PLOTTING
# ============================================================

BASE_DIR = r"C:\Users\sheev\OneDrive\Desktop\DL proj"

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "final_evaluation_results"
)

PER_CLASS_FILE = os.path.join(
    RESULTS_DIR,
    "resnet50_per_class_metrics.csv"
)

METRICS_FILE = os.path.join(
    RESULTS_DIR,
    "resnet50_test_metrics.json"
)

OUTPUT_DIR = os.path.join(
    RESULTS_DIR,
    "plots"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("FINAL RESNET50 RESULTS PLOTTING")
print("=" * 70)

# ============================================================
# CHECK FILES
# ============================================================

print("\nChecking files...")

if not os.path.isfile(PER_CLASS_FILE):
    raise FileNotFoundError(
        f"\nPer-class metrics file not found:\n{PER_CLASS_FILE}"
    )

if not os.path.isfile(METRICS_FILE):
    raise FileNotFoundError(
        f"\nMetrics JSON file not found:\n{METRICS_FILE}"
    )

print("Required files found.")

# ============================================================
# LOAD PER-CLASS METRICS
# ============================================================

df = pd.read_csv(PER_CLASS_FILE)

print("\nColumns detected:")
print(df.columns.tolist())

# ============================================================
# FLEXIBLE COLUMN DETECTION
# ============================================================

def find_column(dataframe, possible_names, description):
    """
    Find a column using case-insensitive matching.
    """

    normalized = {
        str(col).strip().lower(): col
        for col in dataframe.columns
    }

    for name in possible_names:
        key = name.strip().lower()

        if key in normalized:
            return normalized[key]

    raise KeyError(
        f"\nRequired column for '{description}' was not found.\n"
        f"Expected one of:\n{possible_names}\n\n"
        f"Available columns:\n{dataframe.columns.tolist()}"
    )


# ------------------------------------------------------------
# CLASS COLUMN
# ------------------------------------------------------------

CLASS_COL = find_column(
    df,
    [
        "class",
        "class_name",
        "classname",
        "label",
        "category"
    ],
    "class name"
)

# ------------------------------------------------------------
# PRECISION
# ------------------------------------------------------------

PRECISION_COL = find_column(
    df,
    [
        "precision",
        "Precision"
    ],
    "precision"
)

# ------------------------------------------------------------
# RECALL
# ------------------------------------------------------------

RECALL_COL = find_column(
    df,
    [
        "recall",
        "Recall"
    ],
    "recall"
)

# ------------------------------------------------------------
# F1
# ------------------------------------------------------------

F1_COL = find_column(
    df,
    [
        "f1",
        "f1_score",
        "f1-score",
        "f1 score"
    ],
    "F1 score"
)

# ============================================================
# PRINT DETECTED MAPPING
# ============================================================

print("\nDetected column mapping:")

print(f"Class     : {CLASS_COL}")
print(f"Precision : {PRECISION_COL}")
print(f"Recall    : {RECALL_COL}")
print(f"F1        : {F1_COL}")

# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    CLASS_COL,
    PRECISION_COL,
    RECALL_COL,
    F1_COL
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    print("\nERROR: Missing columns:")
    for col in missing_columns:
        print(f"  - {col}")

    print("\nAvailable columns:")
    for col in df.columns:
        print(f"  - {col}")

    raise ValueError(
        "\nCannot create plots because required columns are missing."
    )

print("\nAll required columns are present.")

# ============================================================
# CONVERT METRICS TO NUMERIC
# ============================================================

for col in [PRECISION_COL, RECALL_COL, F1_COL]:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# Check for invalid values

if df[[PRECISION_COL, RECALL_COL, F1_COL]].isnull().any().any():

    print("\nWARNING: Some metric values could not be converted.")

    print(
        df[
            [
                CLASS_COL,
                PRECISION_COL,
                RECALL_COL,
                F1_COL
            ]
        ]
    )

    raise ValueError(
        "\nInvalid or missing metric values detected."
    )

# ============================================================
# DISPLAY DATA
# ============================================================

print("\nPer-class metrics:")
print(
    df[
        [
            CLASS_COL,
            PRECISION_COL,
            RECALL_COL,
            F1_COL
        ]
    ].to_string(index=False)
)

# ============================================================
# PLOT 1: PRECISION / RECALL / F1
# ============================================================

plt.figure(figsize=(12, 7))

x = range(len(df))

plt.plot(
    x,
    df[PRECISION_COL],
    marker="o",
    linewidth=2,
    label="Precision"
)

plt.plot(
    x,
    df[RECALL_COL],
    marker="o",
    linewidth=2,
    label="Recall"
)

plt.plot(
    x,
    df[F1_COL],
    marker="o",
    linewidth=2,
    label="F1 Score"
)

plt.xticks(
    x,
    df[CLASS_COL],
    rotation=20
)

plt.ylim(0, 1.05)

plt.xlabel("Diabetic Retinopathy Class")
plt.ylabel("Score")

plt.title(
    "ResNet50 Per-Class Precision, Recall and F1 Score"
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plot1 = os.path.join(
    OUTPUT_DIR,
    "resnet50_per_class_metrics.png"
)

plt.savefig(
    plot1,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"\nSaved:")
print(plot1)

# ============================================================
# PLOT 2: F1 SCORE
# ============================================================

plt.figure(figsize=(11, 7))

plt.bar(
    df[CLASS_COL],
    df[F1_COL]
)

plt.ylim(0, 1.05)

plt.xlabel("Diabetic Retinopathy Class")
plt.ylabel("F1 Score")

plt.title(
    "ResNet50 F1 Score by Diabetic Retinopathy Class"
)

plt.xticks(
    rotation=20
)

plt.grid(
    axis="y",
    alpha=0.3
)

# Add values above bars

for i, value in enumerate(df[F1_COL]):

    plt.text(
        i,
        value + 0.02,
        f"{value:.3f}",
        ha="center",
        fontsize=10
    )

plt.tight_layout()

plot2 = os.path.join(
    OUTPUT_DIR,
    "resnet50_per_class_f1.png"
)

plt.savefig(
    plot2,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved:")
print(plot2)

# ============================================================
# PLOT 3: OVERALL METRICS
# ============================================================

with open(
    METRICS_FILE,
    "r",
    encoding="utf-8"
) as f:

    metrics = json.load(f)

print("\nMetrics JSON loaded.")

# ============================================================
# FIND OVERALL METRICS
# ============================================================

def get_metric(dictionary, possible_names):

    normalized = {
        str(k).strip().lower(): v
        for k, v in dictionary.items()
    }

    for name in possible_names:

        key = name.strip().lower()

        if key in normalized:
            return float(normalized[key])

    return None


accuracy = get_metric(
    metrics,
    [
        "accuracy",
        "test_accuracy"
    ]
)

precision = get_metric(
    metrics,
    [
        "macro_precision",
        "precision",
        "test_precision"
    ]
)

recall = get_metric(
    metrics,
    [
        "macro_recall",
        "recall",
        "test_recall"
    ]
)

f1 = get_metric(
    metrics,
    [
        "macro_f1",
        "f1",
        "test_f1"
    ]
)

# ============================================================
# CHECK OVERALL METRICS
# ============================================================

overall_metrics = {
    "Accuracy": accuracy,
    "Macro Precision": precision,
    "Macro Recall": recall,
    "Macro F1": f1
}

missing_overall = [
    name
    for name, value in overall_metrics.items()
    if value is None
]

if missing_overall:

    print("\nWARNING: Missing overall metrics:")

    for metric in missing_overall:
        print(f"  - {metric}")

    print("\nAvailable JSON keys:")
    print(list(metrics.keys()))

else:

    # ========================================================
    # PLOT OVERALL METRICS
    # ========================================================

    names = list(overall_metrics.keys())
    values = list(overall_metrics.values())

    plt.figure(figsize=(10, 7))

    plt.bar(
        names,
        values
    )

    plt.ylim(0, 1.05)

    plt.ylabel("Score")

    plt.title(
        "ResNet50 Overall Test Performance"
    )

    plt.xticks(
        rotation=15
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    for i, value in enumerate(values):

        plt.text(
            i,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            fontsize=11
        )

    plt.tight_layout()

    plot3 = os.path.join(
        OUTPUT_DIR,
        "resnet50_overall_metrics.png"
    )

    plt.savefig(
        plot3,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"\nSaved:")
    print(plot3)

# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("PLOTTING COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")

for filename in os.listdir(OUTPUT_DIR):

    print(
        os.path.join(
            OUTPUT_DIR,
            filename
        )
    )

print("=" * 70)
