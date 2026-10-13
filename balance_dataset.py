# ============================================================
# balance_dataset.py
# Diabetic Retinopathy Dataset Balancing + Augmentation
# Target: 1000 images per class
# ============================================================

import os
import shutil
import random
import cv2
import numpy as np
import pandas as pd

from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(r"C:\Users\sheev\OneDrive\Desktop\DL proj")

# Original image dataset
IMAGE_DIR = PROJECT_DIR / "DL dataset" / "DR_grading" / "DR_grading"

# CSV containing image IDs and diagnosis labels
CSV_PATH = PROJECT_DIR / "DL dataset" / "DR_grading.csv"

# Output balanced dataset
OUTPUT_DIR = PROJECT_DIR / "DL dataset" / "DR_balanced_1000"

TARGET_IMAGES = 1000

# Reproducibility
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# 2. CLASS INFORMATION
# ============================================================

CLASS_NAMES = {
    0: "No_DR",
    1: "Mild_DR",
    2: "Moderate_DR",
    3: "Severe_DR",
    4: "Proliferative_DR"
}


# ============================================================
# 3. PRINT CONFIGURATION
# ============================================================

print("=" * 70)
print("DIABETIC RETINOPATHY DATASET BALANCING")
print("=" * 70)

print(f"Original image directory:")
print(IMAGE_DIR)

print(f"\nCSV file:")
print(CSV_PATH)

print(f"\nOutput directory:")
print(OUTPUT_DIR)

print(f"\nTarget images per class: {TARGET_IMAGES}")

print("=" * 70)


# ============================================================
# 4. CHECK PATHS
# ============================================================

if not IMAGE_DIR.exists():
    raise FileNotFoundError(
        f"\nImage directory not found:\n{IMAGE_DIR}"
    )

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"""
CSV file not found:

{CSV_PATH}

Please check the CSV filename.
"""
    )


# ============================================================
# 5. LOAD CSV
# ============================================================

print("\nLoading CSV...")

df = pd.read_csv(CSV_PATH)

print("\nCSV columns:")
print(df.columns.tolist())

print(f"\nTotal CSV records: {len(df)}")


# ============================================================
# 6. CHECK REQUIRED COLUMNS
# ============================================================

if "id_code" not in df.columns:
    raise ValueError(
        "\n'id_code' column not found in CSV."
    )

if "diagnosis" not in df.columns:
    raise ValueError(
        "\n'diagnosis' column not found in CSV."
    )


# ============================================================
# 7. NORMALIZE IMAGE IDs
# ============================================================

df["id_code"] = df["id_code"].astype(str)

# Remove .jpg if already present
df["image_name"] = df["id_code"].apply(
    lambda x: x if x.lower().endswith(".jpg") else x + ".jpg"
)


# ============================================================
# 8. VERIFY IMAGES
# ============================================================

print("\nChecking image files...")

existing_records = []

missing_count = 0

for _, row in df.iterrows():

    image_path = IMAGE_DIR / row["image_name"]

    if image_path.exists():
        existing_records.append(row)
    else:
        missing_count += 1


df = pd.DataFrame(existing_records)

print(f"Images found : {len(df)}")
print(f"Images missing: {missing_count}")


# ============================================================
# 9. ORIGINAL CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("ORIGINAL CLASS DISTRIBUTION")
print("=" * 70)

class_counts = df["diagnosis"].value_counts().sort_index()

for class_id in range(5):

    count = int(class_counts.get(class_id, 0))

    print(
        f"Class {class_id} - "
        f"{CLASS_NAMES[class_id]:20s}: "
        f"{count} images"
    )


# ============================================================
# 10. CREATE OUTPUT DIRECTORIES
# ============================================================

if OUTPUT_DIR.exists():

    print("\nOutput directory already exists.")

    response = input(
        "Delete existing balanced dataset? (y/n): "
    ).strip().lower()

    if response == "y":

        shutil.rmtree(OUTPUT_DIR)

    else:

        print(
            "\nOperation cancelled."
        )

        exit()


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 11. IMAGE AUGMENTATION
# ============================================================

datagen = ImageDataGenerator(

    rotation_range=15,

    width_shift_range=0.10,

    height_shift_range=0.10,

    shear_range=0.10,

    zoom_range=0.15,

    horizontal_flip=True,

    brightness_range=(0.85, 1.15),

    fill_mode="nearest"
)


# ============================================================
# 12. FUNCTION TO LOAD IMAGE
# ============================================================

def load_image(image_path):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        return None

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return image


# ============================================================
# 13. FUNCTION TO SAVE IMAGE
# ============================================================

def save_image(image, output_path):

    image = np.clip(
        image,
        0,
        255
    ).astype(np.uint8)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        str(output_path),
        image
    )


# ============================================================
# 14. PROCESS EACH CLASS
# ============================================================

summary = []

for class_id in range(5):

    class_name = CLASS_NAMES[class_id]

    print("\n" + "=" * 70)

    print(
        f"PROCESSING CLASS {class_id}: {class_name}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Create class output directory
    # --------------------------------------------------------

    class_output_dir = (
        OUTPUT_DIR / class_name
    )

    class_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Get images belonging to current class
    # --------------------------------------------------------

    class_df = df[
        df["diagnosis"] == class_id
    ].copy()

    original_count = len(class_df)

    print(
        f"Original images: {original_count}"
    )

    # --------------------------------------------------------
    # CASE 1: More than 1000
    # Randomly select 1000
    # --------------------------------------------------------

    if original_count >= TARGET_IMAGES:

        print(
            f"Selecting {TARGET_IMAGES} original images..."
        )

        selected_df = class_df.sample(
            n=TARGET_IMAGES,
            random_state=RANDOM_SEED
        )

        for _, row in selected_df.iterrows():

            source_path = (
                IMAGE_DIR /
                row["image_name"]
            )

            destination_path = (
                class_output_dir /
                row["image_name"]
            )

            shutil.copy2(
                source_path,
                destination_path
            )

        augmented_count = 0

    # --------------------------------------------------------
    # CASE 2: Less than 1000
    # Copy originals + generate augmented images
    # --------------------------------------------------------

    else:

        print(
            f"Copying {original_count} original images..."
        )

        # Copy all original images

        for _, row in class_df.iterrows():

            source_path = (
                IMAGE_DIR /
                row["image_name"]
            )

            destination_path = (
                class_output_dir /
                row["image_name"]
            )

            shutil.copy2(
                source_path,
                destination_path
            )

        # Number of images required

        required = (
            TARGET_IMAGES -
            original_count
        )

        print(
            f"Augmented images required: {required}"
        )

        augmented_count = 0

        # ----------------------------------------------------
        # Generate augmented images
        # ----------------------------------------------------

        while augmented_count < required:

            # Randomly select an original image

            row = class_df.sample(
                n=1
            ).iloc[0]

            image_path = (
                IMAGE_DIR /
                row["image_name"]
            )

            image = load_image(
                image_path
            )

            if image is None:
                continue

            # Add batch dimension

            image_batch = np.expand_dims(
                image,
                axis=0
            )

            # Generate augmentation

            generator = datagen.flow(
                image_batch,
                batch_size=1
            )

            augmented_image = next(
                generator
            )[0]

            # Create filename

            original_stem = Path(
                row["image_name"]
            ).stem

            augmented_filename = (
                f"{original_stem}_aug_"
                f"{augmented_count + 1:04d}.jpg"
            )

            output_path = (
                class_output_dir /
                augmented_filename
            )

            save_image(
                augmented_image,
                output_path
            )

            augmented_count += 1

            # Progress

            if (
                augmented_count % 100 == 0
                or augmented_count == required
            ):

                print(
                    f"Generated: "
                    f"{augmented_count}/{required}"
                )


    # ========================================================
    # 15. SUMMARY
    # ========================================================

    final_count = len(
        list(
            class_output_dir.glob("*.jpg")
        )
    )

    print(
        f"\nFinal images: {final_count}"
    )

    summary.append({

        "class_id": class_id,

        "class_name": class_name,

        "original_images": original_count,

        "augmented_images": augmented_count,

        "final_images": final_count

    })


# ============================================================
# 16. SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary
)

summary_path = (
    OUTPUT_DIR /
    "balance_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# 17. FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL BALANCED DATASET")
print("=" * 70)

all_balanced = True

for class_id in range(5):

    class_name = CLASS_NAMES[class_id]

    class_dir = (
        OUTPUT_DIR /
        class_name
    )

    count = len(
        list(
            class_dir.glob("*.jpg")
        )
    )

    print(
        f"{class_name:20s}: "
        f"{count} images"
    )

    if count != TARGET_IMAGES:

        all_balanced = False


# ============================================================
# 18. RESULT
# ============================================================

print("\n" + "=" * 70)

if all_balanced:

    print(
        "SUCCESS: Dataset balanced successfully!"
    )

    print(
        f"Each class contains exactly "
        f"{TARGET_IMAGES} images."
    )

else:

    print(
        "WARNING: Dataset is not perfectly balanced."
    )


print("=" * 70)

print(
    f"\nBalanced dataset location:\n"
    f"{OUTPUT_DIR}"
)

print(
    f"\nSummary file:\n"
    f"{summary_path}"
)

print("\n" + "=" * 70)
print("BALANCING COMPLETED")
print("=" * 70)