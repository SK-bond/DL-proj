import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# RESNET50 - TRAINING GRAPHS
# ============================================================

BASE_DIR = r"C:\Users\sheev\OneDrive\Desktop\DL proj"

CSV_PATH = os.path.join(
    BASE_DIR,
    "resnet50_balanced_results",
    "resnet50_balanced_history.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "resnet50_balanced_results"
)

print("=" * 70)
print("RESNET50 - TRAINING GRAPH GENERATION")
print("=" * 70)

# ============================================================
# LOAD HISTORY
# ============================================================

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"\nHistory file not found:\n{CSV_PATH}"
    )

history = pd.read_csv(CSV_PATH)

print("\nHistory loaded successfully.")
print(f"Total epochs recorded: {len(history)}")

print("\nColumns:")
print(history.columns.tolist())

# ============================================================
# 1. ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    history["epoch"],
    history["train_accuracy"],
    marker="o",
    linewidth=2,
    label="Training Accuracy"
)

plt.plot(
    history["epoch"],
    history["val_accuracy"],
    marker="o",
    linewidth=2,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ResNet50 Training vs Validation Accuracy")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

accuracy_path = os.path.join(
    OUTPUT_DIR,
    "resnet50_accuracy_graph.png"
)

plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nAccuracy graph saved:")
print(accuracy_path)

# ============================================================
# 2. LOSS GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    history["epoch"],
    history["train_loss"],
    marker="o",
    linewidth=2,
    label="Training Loss"
)

plt.plot(
    history["epoch"],
    history["val_loss"],
    marker="o",
    linewidth=2,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ResNet50 Training vs Validation Loss")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

loss_path = os.path.join(
    OUTPUT_DIR,
    "resnet50_loss_graph.png"
)

plt.savefig(
    loss_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nLoss graph saved:")
print(loss_path)

# ============================================================
# 3. F1 SCORE GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    history["epoch"],
    history["train_f1"],
    marker="o",
    linewidth=2,
    label="Training F1"
)

plt.plot(
    history["epoch"],
    history["val_f1"],
    marker="o",
    linewidth=2,
    label="Validation F1"
)

plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("ResNet50 Training vs Validation F1 Score")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

f1_path = os.path.join(
    OUTPUT_DIR,
    "resnet50_f1_graph.png"
)

plt.savefig(
    f1_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nF1 graph saved:")
print(f1_path)

# ============================================================
# 4. PRECISION GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    history["epoch"],
    history["train_precision"],
    marker="o",
    linewidth=2,
    label="Training Precision"
)

plt.plot(
    history["epoch"],
    history["val_precision"],
    marker="o",
    linewidth=2,
    label="Validation Precision"
)

plt.xlabel("Epoch")
plt.ylabel("Precision")
plt.title("ResNet50 Training vs Validation Precision")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

precision_path = os.path.join(
    OUTPUT_DIR,
    "resnet50_precision_graph.png"
)

plt.savefig(
    precision_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nPrecision graph saved:")
print(precision_path)

# ============================================================
# 5. RECALL GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    history["epoch"],
    history["train_recall"],
    marker="o",
    linewidth=2,
    label="Training Recall"
)

plt.plot(
    history["epoch"],
    history["val_recall"],
    marker="o",
    linewidth=2,
    label="Validation Recall"
)

plt.xlabel("Epoch")
plt.ylabel("Recall")
plt.title("ResNet50 Training vs Validation Recall")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

recall_path = os.path.join(
    OUTPUT_DIR,
    "resnet50_recall_graph.png"
)

plt.savefig(
    recall_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nRecall graph saved:")
print(recall_path)

# ============================================================
# 6. LEARNING RATE GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    history["epoch"],
    history["learning_rate"],
    marker="o",
    linewidth=2
)

plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("ResNet50 Learning Rate Schedule")

plt.grid(True, alpha=0.3)
plt.tight_layout()

lr_path = os.path.join(
    OUTPUT_DIR,
    "resnet50_learning_rate_graph.png"
)

plt.savefig(
    lr_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nLearning-rate graph saved:")
print(lr_path)

# ============================================================
# SUMMARY
# ============================================================

best_epoch = history.loc[
    history["val_f1"].idxmax()
]

print("\n" + "=" * 70)
print("GRAPH GENERATION COMPLETED")
print("=" * 70)

print(f"\nBest Validation F1 : {best_epoch['val_f1']:.4f}")
print(f"Best Epoch         : {int(best_epoch['epoch'])}")
print(f"Validation Accuracy: {best_epoch['val_accuracy']:.4f}")

print("\nGenerated files:")
print("  - resnet50_accuracy_graph.png")
print("  - resnet50_loss_graph.png")
print("  - resnet50_f1_graph.png")
print("  - resnet50_precision_graph.png")
print("  - resnet50_recall_graph.png")
print("  - resnet50_learning_rate_graph.png")

print("\nSaved in:")
print(OUTPUT_DIR)

print("=" * 70)