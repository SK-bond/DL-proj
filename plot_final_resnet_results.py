import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Users\sheev\OneDrive\Desktop\DL proj"

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "final_evaluation_results"
)

METRICS_FILE = os.path.join(
    RESULTS_DIR,
    "resnet50_test_metrics.json"
)

PER_CLASS_FILE = os.path.join(
    RESULTS_DIR,
    "resnet50_per_class_metrics.csv"
)

OUTPUT_DIR = RESULTS_DIR

# ============================================================
# START
# ============================================================

print("=" * 70)
print("FINAL RESNET50 RESULT VISUALIZATION")
print("=" * 70)

# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(METRICS_FILE):
    raise FileNotFoundError(
        f"\nMetrics file not found:\n{METRICS_FILE}"
    )

if not os.path.exists(PER_CLASS_FILE):
    raise FileNotFoundError(
        f"\nPer-class metrics file not found:\n{PER_CLASS_FILE}"
    )

print("\nRequired files found.")

# ============================================================
# LOAD OVERALL METRICS
# ============================================================

with open(METRICS_FILE, "r") as f:
    metrics = json.load(f)

print("\nOverall metrics loaded.")

# ============================================================
# LOAD PER-CLASS METRICS
# ============================================================

df = pd.read_csv(PER_CLASS_FILE)

print("\nPer-class metrics columns:")
print(df.columns.tolist())

# ============================================================
# REQUIRED COLUMN CHECK
# ============================================================

required_columns = {
    "class",
    "precision",
    "recall",
    "f1_score",
    "support"
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        "\nMissing required columns:\n"
        + "\n".join(sorted(missing_columns))
    )

print("\nAll required columns found.")

# ============================================================
# CLEAN DATA
# ============================================================

df = df.dropna(
    subset=[
        "class",
        "precision",
        "recall",
        "f1_score"
    ]
).copy()

print("\nClasses detected:")
for class_name in df["class"]:
    print(f" - {class_name}")

# ============================================================
# CHART 1
# PER-CLASS PRECISION, RECALL AND F1
# ============================================================

plt.figure(figsize=(12, 7))

x = list(range(len(df)))

width = 0.25

plt.bar(
    [i - width for i in x],
    df["precision"],
    width=width,
    label="Precision"
)

plt.bar(
    x,
    df["recall"],
    width=width,
    label="Recall"
)

plt.bar(
    [i + width for i in x],
    df["f1_score"],
    width=width,
    label="F1-score"
)

plt.xticks(
    x,
    df["class"],
    rotation=20
)

plt.ylim(0, 1.05)

plt.xlabel("Diabetic Retinopathy Class")
plt.ylabel("Score")

plt.title(
    "ResNet50 Per-Class Performance"
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

performance_path = os.path.join(
    OUTPUT_DIR,
    "resnet50_per_class_performance.png"
)

plt.savefig(
    performance_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"\nSaved:\n{performance_path}"
)

# ============================================================
# GET OVERALL METRICS
# ============================================================

def get_metric(*possible_names):

    for name in possible_names:

        if name in metrics:
            return float(metrics[name])

    raise KeyError(
        "None of these metrics were found: "
        + ", ".join(possible_names)
    )


accuracy = get_metric(
    "accuracy"
)

macro_precision = get_metric(
    "macro_precision",
    "Macro Precision"
)

macro_recall = get_metric(
    "macro_recall",
    "Macro Recall"
)

macro_f1 = get_metric(
    "macro_f1",
    "Macro F1"
)

# ============================================================
# CHART 2
# OVERALL PERFORMANCE
# ============================================================

metric_names = [
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1"
]

metric_values = [
    accuracy,
    macro_precision,
    macro_recall,
    macro_f1
]

plt.figure(figsize=(10, 6))

bars = plt.bar(
    metric_names,
    metric_values
)

plt.ylim(0, 1.05)

plt.xlabel("Metric")
plt.ylabel("Score")

plt.title(
    "ResNet50 Overall Test Performance"
)

plt.grid(
    axis="y",
    alpha=0.3
)

# Add values above bars

for bar, value in zip(
    bars,
    metric_values
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.3f}",
        ha="center",
        fontweight="bold"
    )

plt.tight_layout()

overall_path = os.path.join(
    OUTPUT_DIR,
    "resnet50_overall_performance.png"
)

plt.savefig(
    overall_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved:\n{overall_path}"
)

# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFINAL TEST METRICS")

print(
    f"Accuracy        : {accuracy * 100:.2f}%"
)

print(
    f"Macro Precision : {macro_precision * 100:.2f}%"
)

print(
    f"Macro Recall    : {macro_recall * 100:.2f}%"
)

print(
    f"Macro F1        : {macro_f1 * 100:.2f}%"
)

print("\nGenerated files:")

print(
    "1. resnet50_per_class_performance.png"
)

print(
    "2. resnet50_overall_performance.png"
)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("=" * 70)