# ============================================================
# compare_models.py
# FINAL TEST-SET COMPARISON
# ResNet50 vs DenseNet121 vs Vision Transformer
# Diabetic Retinopathy Classification
# ============================================================

import os
import json
import re
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = r"C:\Users\sheev\OneDrive\Desktop\DL proj"

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "model_comparison_results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# MODEL RESULT DIRECTORIES
# ============================================================

MODEL_DIRS = {
    "ResNet50": os.path.join(
        BASE_DIR,
        "resnet50_balanced_results"
    ),

    "DenseNet121": os.path.join(
        BASE_DIR,
        "densenet121_balanced_results"
    ),

    "ViT": os.path.join(
        BASE_DIR,
        "vit_balanced_results"
    )
}


# ============================================================
# SEARCH FOR FILE
# ============================================================

def find_file(folder, keywords):

    if not os.path.exists(folder):
        return None

    files = os.listdir(folder)

    for filename in files:

        lower_name = filename.lower()

        if all(
            keyword.lower() in lower_name
            for keyword in keywords
        ):
            return os.path.join(folder, filename)

    return None


# ============================================================
# FIND CLASSIFICATION REPORT
# ============================================================

def find_classification_report(folder):

    for filename in os.listdir(folder):

        lower_name = filename.lower()

        if (
            "classification" in lower_name
            and "report" in lower_name
            and lower_name.endswith(".txt")
        ):
            return os.path.join(folder, filename)

    return None


# ============================================================
# PARSE CLASSIFICATION REPORT
# ============================================================

def parse_classification_report(filepath):

    if filepath is None:
        return None

    with open(
        filepath,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        text = f.read()

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = None

    accuracy_match = re.search(
        r"accuracy\s+([0-9]*\.?[0-9]+)",
        text,
        re.IGNORECASE
    )

    if accuracy_match:

        accuracy = float(
            accuracy_match.group(1)
        )

    # --------------------------------------------------------
    # Macro average
    # --------------------------------------------------------

    macro_match = re.search(
        r"macro avg\s+"
        r"([0-9]*\.?[0-9]+)\s+"
        r"([0-9]*\.?[0-9]+)\s+"
        r"([0-9]*\.?[0-9]+)",
        text,
        re.IGNORECASE
    )

    precision = None
    recall = None
    f1 = None

    if macro_match:

        precision = float(
            macro_match.group(1)
        )

        recall = float(
            macro_match.group(2)
        )

        f1 = float(
            macro_match.group(3)
        )

    # --------------------------------------------------------
    # Weighted average fallback
    # --------------------------------------------------------

    if f1 is None:

        weighted_match = re.search(
            r"weighted avg\s+"
            r"([0-9]*\.?[0-9]+)\s+"
            r"([0-9]*\.?[0-9]+)\s+"
            r"([0-9]*\.?[0-9]+)",
            text,
            re.IGNORECASE
        )

        if weighted_match:

            precision = float(
                weighted_match.group(1)
            )

            recall = float(
                weighted_match.group(2)
            )

            f1 = float(
                weighted_match.group(3)
            )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# READ METRICS JSON
# ============================================================

def read_json_metrics(folder):

    json_file = find_file(
        folder,
        ["metrics"]
    )

    if json_file is None:
        return {}

    try:

        with open(
            json_file,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        return data

    except Exception as e:

        print(
            f"Could not read JSON: "
            f"{json_file}"
        )

        print(e)

        return {}


# ============================================================
# FIND TEST LOSS
# ============================================================

def find_test_loss(data):

    possible_keys = [
        "test_loss",
        "Test Loss",
        "testLoss",
        "loss",
        "Loss"
    ]

    # Direct keys
    for key in possible_keys:

        if key in data:

            value = data[key]

            if isinstance(value, (int, float)):

                return float(value)

    # Recursive search
    if isinstance(data, dict):

        for key, value in data.items():

            if isinstance(value, dict):

                result = find_test_loss(value)

                if result is not None:
                    return result

    return None


# ============================================================
# PROCESS MODELS
# ============================================================

results = []


print("=" * 75)
print("DIABETIC RETINOPATHY MODEL COMPARISON")
print("TEST SET COMPARISON")
print("=" * 75)


for model_name, folder in MODEL_DIRS.items():

    print("\n" + "-" * 75)
    print(f"Processing: {model_name}")
    print(f"Folder: {folder}")
    print("-" * 75)

    if not os.path.exists(folder):

        print(
            f"ERROR: Result folder not found for "
            f"{model_name}"
        )

        continue

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report_file = find_classification_report(
        folder
    )

    if report_file:

        print(
            f"Classification report found:\n"
            f"{os.path.basename(report_file)}"
        )

        report_metrics = parse_classification_report(
            report_file
        )

    else:

        print(
            "WARNING: Classification report not found."
        )

        report_metrics = None

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    json_data = read_json_metrics(folder)

    test_loss = find_test_loss(
        json_data
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    if report_metrics:

        accuracy = report_metrics["accuracy"]
        precision = report_metrics["precision"]
        recall = report_metrics["recall"]
        f1 = report_metrics["f1"]

    else:

        accuracy = None
        precision = None
        recall = None
        f1 = None

    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    results.append({

        "Model": model_name,

        "Accuracy": (
            accuracy * 100
            if accuracy is not None
            else None
        ),

        "Macro Precision": (
            precision * 100
            if precision is not None
            else None
        ),

        "Macro Recall": (
            recall * 100
            if recall is not None
            else None
        ),

        "Macro F1": (
            f1 * 100
            if f1 is not None
            else None
        ),

        "Test Loss": test_loss
    })


# ============================================================
# CREATE DATAFRAME
# ============================================================

if not results:

    print(
        "\nNo model results were found."
    )

    raise SystemExit


df = pd.DataFrame(results)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 75)
print("FINAL TEST-SET COMPARISON")
print("=" * 75)

print(
    df.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ============================================================
# SAVE CSV
# ============================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "final_model_comparison.csv"
)

df.to_csv(
    csv_path,
    index=False
)

print(
    f"\nSaved:\n{csv_path}"
)


# ============================================================
# FIND BEST MODEL
# ============================================================

best_f1_model = None
best_accuracy_model = None

if df["Macro F1"].notna().any():

    best_f1_index = df[
        "Macro F1"
    ].idxmax()

    best_f1_model = df.loc[
        best_f1_index,
        "Model"
    ]

    best_f1_value = df.loc[
        best_f1_index,
        "Macro F1"
    ]

else:

    best_f1_value = None


if df["Accuracy"].notna().any():

    best_accuracy_index = df[
        "Accuracy"
    ].idxmax()

    best_accuracy_model = df.loc[
        best_accuracy_index,
        "Model"
    ]

    best_accuracy_value = df.loc[
        best_accuracy_index,
        "Accuracy"
    ]

else:

    best_accuracy_value = None


# ============================================================
# TEXT REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "final_model_comparison_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "=" * 75 + "\n"
    )

    f.write(
        "DIABETIC RETINOPATHY MODEL COMPARISON\n"
    )

    f.write(
        "FINAL TEST-SET RESULTS\n"
    )

    f.write(
        "=" * 75 + "\n\n"
    )

    f.write(
        df.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

    f.write("\n\n")
    f.write("=" * 75 + "\n")

    if best_f1_model:

        f.write(
            f"Best Model by Macro F1: "
            f"{best_f1_model} "
            f"({best_f1_value:.2f}%)\n"
        )

    if best_accuracy_model:

        f.write(
            f"Best Model by Accuracy: "
            f"{best_accuracy_model} "
            f"({best_accuracy_value:.2f}%)\n"
        )


print(
    f"Report saved:\n{report_path}"
)


# ============================================================
# GRAPH 1
# ALL PERFORMANCE METRICS
# ============================================================

metric_columns = [
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1"
]

plot_df = df.dropna(
    subset=metric_columns,
    how="all"
)

if len(plot_df) > 0:

    ax = plot_df.set_index(
        "Model"
    )[metric_columns].plot(
        kind="bar",
        figsize=(12, 7)
    )

    plt.title(
        "Diabetic Retinopathy Model Comparison"
    )

    plt.xlabel(
        "Model"
    )

    plt.ylabel(
        "Performance (%)"
    )

    plt.ylim(
        0,
        100
    )

    plt.xticks(
        rotation=0
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.legend(
        title="Metrics"
    )

    plt.tight_layout()

    graph_path = os.path.join(
        OUTPUT_DIR,
        "final_model_performance_comparison.png"
    )

    plt.savefig(
        graph_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Performance graph saved:\n"
        f"{graph_path}"
    )


# ============================================================
# GRAPH 2
# MACRO F1 COMPARISON
# ============================================================

f1_df = df.dropna(
    subset=["Macro F1"]
)

if len(f1_df) > 0:

    plt.figure(
        figsize=(9, 6)
    )

    bars = plt.bar(
        f1_df["Model"],
        f1_df["Macro F1"]
    )

    for bar, value in zip(
        bars,
        f1_df["Macro F1"]
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 1,
            f"{value:.2f}%",
            ha="center",
            fontsize=10
        )

    plt.title(
        "Macro F1 Score Comparison"
    )

    plt.xlabel(
        "Model"
    )

    plt.ylabel(
        "Macro F1 Score (%)"
    )

    plt.ylim(
        0,
        100
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    f1_graph_path = os.path.join(
        OUTPUT_DIR,
        "macro_f1_comparison.png"
    )

    plt.savefig(
        f1_graph_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"F1 graph saved:\n"
        f"{f1_graph_path}"
    )


# ============================================================
# GRAPH 3
# TEST LOSS
# ============================================================

loss_df = df.dropna(
    subset=["Test Loss"]
)

if len(loss_df) > 0:

    plt.figure(
        figsize=(9, 6)
    )

    bars = plt.bar(
        loss_df["Model"],
        loss_df["Test Loss"]
    )

    for bar, value in zip(
        bars,
        loss_df["Test Loss"]
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.02,
            f"{value:.4f}",
            ha="center",
            fontsize=10
        )

    plt.title(
        "Test Loss Comparison"
    )

    plt.xlabel(
        "Model"
    )

    plt.ylabel(
        "Test Loss"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    loss_graph_path = os.path.join(
        OUTPUT_DIR,
        "test_loss_comparison.png"
    )

    plt.savefig(
        loss_graph_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Loss graph saved:\n"
        f"{loss_graph_path}"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 75)
print("COMPARISON COMPLETED")
print("=" * 75)

print(
    f"""
Output directory:

{OUTPUT_DIR}

Files generated:

1. final_model_comparison.csv
2. final_model_comparison_report.txt
3. final_model_performance_comparison.png
4. macro_f1_comparison.png
5. test_loss_comparison.png
"""
)

if best_f1_model:

    print(
        f"BEST MODEL BY MACRO F1: "
        f"{best_f1_model} "
        f"({best_f1_value:.2f}%)"
    )

print("=" * 75)