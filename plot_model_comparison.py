
import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\sheev\OneDrive\Desktop\DL proj"

COMPARISON_FILE = os.path.join(
    BASE_DIR,
    "model_comparison_results",
    "model_comparison.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "model_comparison_results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CHECK FILE
# ============================================================

if not os.path.exists(COMPARISON_FILE):
    raise FileNotFoundError(
        f"""
Comparison file not found:

{COMPARISON_FILE}

Run create_model_comparison.py first.
"""
    )


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(COMPARISON_FILE)

print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print("\nModels detected:")
print(df["Model"].tolist())

print("\nComparison data:")
print(df.to_string(index=False))


# ============================================================
# 1. ACCURACY COMPARISON
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    df["Model"],
    df["Validation Accuracy"]
)

plt.title("Validation Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)

for i, value in enumerate(df["Validation Accuracy"]):
    plt.text(
        i,
        value + 0.02,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()

accuracy_file = os.path.join(
    OUTPUT_DIR,
    "model_comparison_accuracy.png"
)

plt.savefig(
    accuracy_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nSaved:")
print(accuracy_file)


# ============================================================
# 2. PRECISION COMPARISON
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    df["Model"],
    df["Validation Precision"]
)

plt.title("Validation Precision Comparison")
plt.xlabel("Model")
plt.ylabel("Precision")
plt.ylim(0, 1)

for i, value in enumerate(df["Validation Precision"]):
    plt.text(
        i,
        value + 0.02,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()

precision_file = os.path.join(
    OUTPUT_DIR,
    "model_comparison_precision.png"
)

plt.savefig(
    precision_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nSaved:")
print(precision_file)


# ============================================================
# 3. RECALL COMPARISON
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    df["Model"],
    df["Validation Recall"]
)

plt.title("Validation Recall Comparison")
plt.xlabel("Model")
plt.ylabel("Recall")
plt.ylim(0, 1)

for i, value in enumerate(df["Validation Recall"]):
    plt.text(
        i,
        value + 0.02,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()

recall_file = os.path.join(
    OUTPUT_DIR,
    "model_comparison_recall.png"
)

plt.savefig(
    recall_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nSaved:")
print(recall_file)


# ============================================================
# 4. F1 SCORE COMPARISON
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    df["Model"],
    df["Validation F1"]
)

plt.title("Validation F1 Score Comparison")
plt.xlabel("Model")
plt.ylabel("F1 Score")
plt.ylim(0, 1)

for i, value in enumerate(df["Validation F1"]):
    plt.text(
        i,
        value + 0.02,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()

f1_file = os.path.join(
    OUTPUT_DIR,
    "model_comparison_f1.png"
)

plt.savefig(
    f1_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nSaved:")
print(f1_file)


# ============================================================
# 5. ALL METRICS COMPARISON
# ============================================================

metrics = [
    "Validation Accuracy",
    "Validation Precision",
    "Validation Recall",
    "Validation F1"
]

x = range(len(df["Model"]))

width = 0.18

plt.figure(figsize=(11, 7))

for i, metric in enumerate(metrics):

    values = df[metric].values

    positions = [
        p + (i - 1.5) * width
        for p in x
    ]

    plt.bar(
        positions,
        values,
        width=width,
        label=metric.replace("Validation ", "")
    )

    for position, value in zip(positions, values):

        plt.text(
            position,
            value + 0.015,
            f"{value:.2f}",
            ha="center",
            fontsize=8
        )


plt.xticks(
    list(x),
    df["Model"]
)

plt.ylabel("Score")
plt.xlabel("Model")

plt.title(
    "ResNet50 vs DenseNet121 - Performance Comparison"
)

plt.ylim(0, 1.1)

plt.legend()

plt.tight_layout()

all_metrics_file = os.path.join(
    OUTPUT_DIR,
    "model_comparison_all_metrics.png"
)

plt.savefig(
    all_metrics_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nSaved:")
print(all_metrics_file)


# ============================================================
# 6. LOSS COMPARISON
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    df["Model"],
    df["Validation Loss"]
)

plt.title("Validation Loss Comparison")
plt.xlabel("Model")
plt.ylabel("Validation Loss")

for i, value in enumerate(df["Validation Loss"]):

    plt.text(
        i,
        value + 0.02,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()

loss_file = os.path.join(
    OUTPUT_DIR,
    "model_comparison_loss.png"
)

plt.savefig(
    loss_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nSaved:")
print(loss_file)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("COMPARISON COMPLETED")
print("=" * 70)

print("\nBest model based on Validation F1:")

best_index = df["Validation F1"].idxmax()

print(
    f"Model : {df.loc[best_index, 'Model']}"
)

print(
    f"F1    : {df.loc[best_index, 'Validation F1']:.4f}"
)

print(
    f"Epoch : {int(df.loc[best_index, 'Best Epoch'])}"
)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")

print("1. model_comparison_accuracy.png")
print("2. model_comparison_precision.png")
print("3. model_comparison_recall.png")
print("4. model_comparison_f1.png")
print("5. model_comparison_all_metrics.png")
print("6. model_comparison_loss.png")

print("=" * 70)
