# ============================================================
# DIABETIC RETINOPATHY CLASSIFICATION
# DATA PREPARATION AND DATASET SPLITTING
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# 1. LOAD DATASET
# ============================================================

csv_path = r"DL dataset\DR_grading.csv"

df = pd.read_csv(csv_path)

print("============================================")
print("DATASET INFORMATION")
print("============================================")

print("Total images :", len(df))
print("Columns      :", list(df.columns))


# ============================================================
# 2. CHECK CLASS DISTRIBUTION
# ============================================================

print("\n============================================")
print("ORIGINAL CLASS DISTRIBUTION")
print("============================================")

print(df["diagnosis"].value_counts().sort_index())


# ============================================================
# 3. STRATIFIED TRAIN-TEST SPLIT
#    70% Training
#    30% Temporary
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["diagnosis"],
    random_state=42
)


# ============================================================
# 4. STRATIFIED VALIDATION-TEST SPLIT
#    15% Validation
#    15% Testing
# ============================================================

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["diagnosis"],
    random_state=42
)


# ============================================================
# 5. DISPLAY SPLIT INFORMATION
# ============================================================

print("\n============================================")
print("DATASET SPLIT")
print("============================================")

print("Total images      :", len(df))
print("Training images   :", len(train_df))
print("Validation images :", len(validation_df))
print("Testing images    :", len(test_df))


# ============================================================
# 6. DISPLAY CLASS DISTRIBUTION
# ============================================================

print("\n============================================")
print("TRAINING CLASS DISTRIBUTION")
print("============================================")

print(train_df["diagnosis"].value_counts().sort_index())


print("\n============================================")
print("VALIDATION CLASS DISTRIBUTION")
print("============================================")

print(validation_df["diagnosis"].value_counts().sort_index())


print("\n============================================")
print("TESTING CLASS DISTRIBUTION")
print("============================================")

print(test_df["diagnosis"].value_counts().sort_index())


# ============================================================
# 7. SAVE SPLIT DATASETS
# ============================================================

train_df.to_csv(
    r"DL dataset\train.csv",
    index=False
)

validation_df.to_csv(
    r"DL dataset\validation.csv",
    index=False
)

test_df.to_csv(
    r"DL dataset\test.csv",
    index=False
)


# ============================================================
# 8. FINAL MESSAGE
# ============================================================

print("\n============================================")
print("SUCCESS")
print("============================================")

print("Training CSV   : DL dataset\\train.csv")
print("Validation CSV : DL dataset\\validation.csv")
print("Testing CSV    : DL dataset\\test.csv")

print("\nDataset preparation completed successfully.")