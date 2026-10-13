# ============================================================
# DIABETIC RETINOPATHY CLASSIFICATION
# IMAGE PREPROCESSING AND AUGMENTATION
# ============================================================

import os
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

DATASET_DIR = r"DL dataset\DR_grading\DR_grading"

TRAIN_CSV = r"DL dataset\train.csv"
VALIDATION_CSV = r"DL dataset\validation.csv"
TEST_CSV = r"DL dataset\test.csv"


# ============================================================
# 2. IMAGE CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16


# ============================================================
# 3. LOAD CSV FILES
# ============================================================

train_df = pd.read_csv(TRAIN_CSV)
validation_df = pd.read_csv(VALIDATION_CSV)
test_df = pd.read_csv(TEST_CSV)

# Convert diagnosis into string because
# Keras flow_from_dataframe uses categorical labels.
train_df["diagnosis"] = train_df["diagnosis"].astype(str)
validation_df["diagnosis"] = validation_df["diagnosis"].astype(str)
test_df["diagnosis"] = test_df["diagnosis"].astype(str)


# ============================================================
# 4. CREATE FULL IMAGE PATH
# ============================================================

train_df["image_path"] = train_df["id_code"].apply(
    lambda x: os.path.join(DATASET_DIR, x)
)

validation_df["image_path"] = validation_df["id_code"].apply(
    lambda x: os.path.join(DATASET_DIR, x)
)

test_df["image_path"] = test_df["id_code"].apply(
    lambda x: os.path.join(DATASET_DIR, x)
)


# ============================================================
# 5. VERIFY IMAGE PATHS
# ============================================================

print("============================================")
print("IMAGE PATH VERIFICATION")
print("============================================")

missing_train = sum(
    not os.path.exists(path)
    for path in train_df["image_path"]
)

missing_validation = sum(
    not os.path.exists(path)
    for path in validation_df["image_path"]
)

missing_test = sum(
    not os.path.exists(path)
    for path in test_df["image_path"]
)

print("Missing training images   :", missing_train)
print("Missing validation images :", missing_validation)
print("Missing testing images    :", missing_test)


# ============================================================
# 6. DATA AUGMENTATION
# ============================================================

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,

    rotation_range=15,
    width_shift_range=0.10,
    height_shift_range=0.10,

    shear_range=0.10,
    zoom_range=0.10,

    horizontal_flip=True,

    fill_mode="nearest"
)


# ============================================================
# 7. VALIDATION PREPROCESSING
# ============================================================

validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)


# ============================================================
# 8. TEST PREPROCESSING
# ============================================================

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)


# ============================================================
# 9. CREATE TRAINING GENERATOR
# ============================================================

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,

    x_col="image_path",
    y_col="diagnosis",

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    classes=["0", "1", "2", "3", "4"],

    shuffle=True,

    seed=42
)


# ============================================================
# 10. CREATE VALIDATION GENERATOR
# ============================================================

validation_generator = validation_datagen.flow_from_dataframe(
    dataframe=validation_df,

    x_col="image_path",
    y_col="diagnosis",

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    classes=["0", "1", "2", "3", "4"],

    shuffle=False
)


# ============================================================
# 11. CREATE TEST GENERATOR
# ============================================================

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,

    x_col="image_path",
    y_col="diagnosis",

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    classes=["0", "1", "2", "3", "4"],

    shuffle=False
)


# ============================================================
# 12. CALCULATE CLASS WEIGHTS
# ============================================================

class_labels = np.array([0, 1, 2, 3, 4])

class_weights_values = compute_class_weight(
    class_weight="balanced",
    classes=class_labels,
    y=train_df["diagnosis"].astype(int)
)

class_weights = dict(
    zip(class_labels, class_weights_values)
)


# ============================================================
# 13. DISPLAY CLASS WEIGHTS
# ============================================================

print("\n============================================")
print("CLASS WEIGHTS")
print("============================================")

for class_id, weight in class_weights.items():
    print(f"Class {class_id}: {weight:.4f}")


# ============================================================
# 14. DISPLAY GENERATOR INFORMATION
# ============================================================

print("\n============================================")
print("PREPROCESSING INFORMATION")
print("============================================")

print("Image size       :", IMAGE_SIZE)
print("Batch size       :", BATCH_SIZE)

print("Training samples :", train_generator.samples)
print("Validation samples :", validation_generator.samples)
print("Testing samples  :", test_generator.samples)

print("\nClass mapping:")
print(train_generator.class_indices)


# ============================================================
# 15. FINAL MESSAGE
# ============================================================

print("\n============================================")
print("SUCCESS")
print("============================================")

print("Preprocessing and augmentation setup completed.")