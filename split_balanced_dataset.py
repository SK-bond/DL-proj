
import os
import shutil
import random
from pathlib import Path

# ============================================================
# BALANCED DATASET TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# Base project directory
BASE_DIR = Path(r"C:\Users\sheev\OneDrive\Desktop\DL proj")

# Balanced dataset created earlier
SOURCE_DIR = BASE_DIR / "DL dataset" / "DR_balanced_1000"

# Output directory
OUTPUT_DIR = BASE_DIR / "DL dataset" / "DR_balanced_split"

# Class names
CLASSES = [
    "No_DR",
    "Mild_DR",
    "Moderate_DR",
    "Severe_DR",
    "Proliferative_DR"
]

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Reproducibility
RANDOM_SEED = 42

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================
# CHECK RATIOS
# ============================================================

if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-6:
    raise ValueError("Train, validation and test ratios must sum to 1.0")


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for split in ["train", "val", "test"]:
    for class_name in CLASSES:
        (OUTPUT_DIR / split / class_name).mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(RANDOM_SEED)


# ============================================================
# PROCESS EACH CLASS
# ============================================================

print("=" * 70)
print("BALANCED DATASET SPLITTING")
print("=" * 70)

total_train = 0
total_val = 0
total_test = 0

for class_name in CLASSES:

    source_class_dir = SOURCE_DIR / class_name

    if not source_class_dir.exists():
        raise FileNotFoundError(
            f"\nClass folder not found:\n{source_class_dir}"
        )

    # Get all images
    images = [
        file for file in source_class_dir.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    # Shuffle images
    random.shuffle(images)

    total_images = len(images)

    if total_images == 0:
        raise ValueError(
            f"No images found in:\n{source_class_dir}"
        )

    # ========================================================
    # CALCULATE SPLIT SIZES
    # ========================================================

    train_count = int(total_images * TRAIN_RATIO)
    val_count = int(total_images * VAL_RATIO)

    test_count = total_images - train_count - val_count

    # ========================================================
    # SPLIT
    # ========================================================

    train_images = images[:train_count]

    val_images = images[
        train_count:
        train_count + val_count
    ]

    test_images = images[
        train_count + val_count:
    ]

    # ========================================================
    # COPY FILES
    # ========================================================

    for image in train_images:
        shutil.copy2(
            image,
            OUTPUT_DIR / "train" / class_name / image.name
        )

    for image in val_images:
        shutil.copy2(
            image,
            OUTPUT_DIR / "val" / class_name / image.name
        )

    for image in test_images:
        shutil.copy2(
            image,
            OUTPUT_DIR / "test" / class_name / image.name
        )

    total_train += len(train_images)
    total_val += len(val_images)
    total_test += len(test_images)

    print()
    print(f"Class: {class_name}")
    print(f"  Total : {total_images}")
    print(f"  Train : {len(train_images)}")
    print(f"  Val   : {len(val_images)}")
    print(f"  Test  : {len(test_images)}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("SPLIT COMPLETED")
print("=" * 70)

print(f"Training images   : {total_train}")
print(f"Validation images : {total_val}")
print(f"Testing images    : {total_test}")
print(f"Total images      : {total_train + total_val + total_test}")

print()
print("Expected distribution:")
print("  Train      : 3,500 images")
print("  Validation :   750 images")
print("  Test       :   750 images")
print("  Total      : 5,000 images")

print()
print("Output directory:")
print(OUTPUT_DIR)

print("=" * 70)
