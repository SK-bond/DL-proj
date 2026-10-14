
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Project result folders
BASE_DIR = Path(__file__).resolve().parent
RESNET_FILE = BASE_DIR / "final_evaluation_results" / "resnet50_test_metrics.json"
VGG_FILE = BASE_DIR / "vgg16_results" / "vgg16_evaluation_metrics.json"

OUTPUT_DIR = BASE_DIR / "resnet50_vs_vgg16_comparison"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_metrics(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


resnet = load_metrics(RESNET_FILE)
vgg = load_metrics(VGG_FILE)

# Print keys once if your JSON uses different metric names
print("ResNet50 metric keys:", list(resnet.keys()))
print("VGG16 metric keys:", list(vgg.keys()))

# Adapt these names if your JSON files use different keys
metrics = {
    "Accuracy": ("accuracy", "accuracy"),
    "Macro Precision": ("macro_precision", "macro_precision"),
    "Macro Recall": ("macro_recall", "macro_recall"),
    "Macro F1-score": ("macro_f1", "macro_f1"),
}

# Read metric values
names = []
resnet_values = []
vgg_values = []

for label, (resnet_key, vgg_key) in metrics.items():
    if resnet_key in resnet and vgg_key in vgg:
        names.append(label)
        resnet_values.append(float(resnet[resnet_key]))
        vgg_values.append(float(vgg[vgg_key]))

if not names:
    raise ValueError(
        "Metric keys did not match. Check the printed JSON keys "
        "and update the metrics dictionary."
    )

# Convert fractions to percentages if necessary
if max(resnet_values + vgg_values) <= 1:
    resnet_values = [x * 100 for x in resnet_values]
    vgg_values = [x * 100 for x in vgg_values]

# Save comparison data
with open(OUTPUT_DIR / "comparison_metrics.csv", "w", encoding="utf-8") as f:
    f.write("Metric,ResNet50 (%),VGG16 (%)\n")
    for name, r, v in zip(names, resnet_values, vgg_values):
        f.write(f"{name},{r:.2f},{v:.2f}\n")

# Plot grouped bars
x = np.arange(len(names))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width / 2, resnet_values, width, label="ResNet50")
plt.bar(x + width / 2, vgg_values, width, label="VGG16")

plt.xticks(x, names, rotation=10)
plt.ylabel("Score (%)")
plt.title("ResNet50 vs VGG16 — Evaluation Metrics")
plt.ylim(0, 100)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "resnet50_vs_vgg16.png", dpi=300)
plt.show()

print("\nComparison saved to:", OUTPUT_DIR)
print("CSV: comparison_metrics.csv")
print("Graph: resnet50_vs_vgg16.png")
