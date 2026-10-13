
# compare_final_models.py
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent
OUT = PROJECT / "final_model_comparison"
OUT.mkdir(exist_ok=True)

FILES = {
    "ResNet50": [
        PROJECT / "evaluation_results" / "evaluation_metrics.json",
        PROJECT / "resnet50_balanced_results" / "resnet50_balanced_metrics.json",
    ],
    "VGG16": [
        PROJECT / "vgg16_results" / "vgg16_evaluation_metrics.json",
    ],
    "DenseNet-121": [
        PROJECT / "densenet121_results" / "densenet121_evaluation_metrics.json",
        PROJECT / "densenet_results" / "densenet121_evaluation_metrics.json",
        PROJECT / "densenet121_results" / "evaluation_metrics.json",
    ],
}

KEYS = {
    "Accuracy (%)": ["test_accuracy", "accuracy"],
    "Macro Precision (%)": ["macro_precision", "test_macro_precision"],
    "Macro Recall (%)": ["macro_recall", "test_macro_recall"],
    "Macro F1 (%)": ["macro_f1", "test_macro_f1"],
    "Weighted F1 (%)": ["weighted_f1", "test_weighted_f1"],
}

rows = []

for model, paths in FILES.items():
    data = None
    source = None

    for path in paths:
        if path.exists():
            try:
                candidate = json.loads(path.read_text(encoding="utf-8"))
                if any(k in candidate for ks in KEYS.values() for k in ks):
                    data, source = candidate, path
                    break
            except (OSError, json.JSONDecodeError):
                pass

    if data is None:
        print(f"Missing evaluation metrics: {model}")
        continue

    row = {"Model": model, "Metrics file": str(source)}

    for metric, keys in KEYS.items():
        value = next(
            (data[k] for k in keys
             if isinstance(data.get(k), (int, float))),
            None
        )
        if value is not None:
            value = value * 100 if 0 <= value <= 1 else value
        row[metric] = round(value, 2) if value is not None else None

    rows.append(row)
    print(f"Loaded: {model}")

if not rows:
    raise FileNotFoundError("No model evaluation JSON files were found.")

df = pd.DataFrame(rows)
df.to_csv(OUT / "three_model_comparison.csv", index=False)

print("\nTHREE-MODEL COMPARISON")
print(df.drop(columns="Metrics file").to_string(index=False, na_rep="N/A"))

metrics = list(KEYS.keys())

# One combined grouped bar chart.
plot_df = df.set_index("Model")[metrics]
plot_df = plot_df.dropna(axis=1, how="all")

ax = plot_df.plot(kind="bar", figsize=(11, 6))
ax.set_title("ResNet50 vs VGG16 vs DenseNet-121")
ax.set_ylabel("Score (%)")
ax.set_ylim(0, 100)
ax.set_xlabel("Model")
ax.grid(axis="y", alpha=0.25)
plt.xticks(rotation=0)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(OUT / "three_model_comparison.png", dpi=300)
plt.close()

# Individual metric charts.
for metric in metrics:
    available = df[["Model", metric]].dropna()
    if available.empty:
        continue

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(available["Model"], available[metric])
    ax.set_title(metric + " Comparison")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    for i, value in enumerate(available[metric]):
        ax.text(i, value + 1, f"{value:.2f}%", ha="center")
    fig.tight_layout()
    filename = metric.lower().replace(" ", "_").replace("(", "").replace(")", "")
    fig.savefig(OUT / f"{filename}.png", dpi=300)
    plt.close(fig)

print("\nSaved comparison files to:", OUT)
