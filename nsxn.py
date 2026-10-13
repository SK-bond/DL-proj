import pandas as pd
import os

# ============================================================
# STEP 3B: VERIFY CSV AND IMAGE DATASET
# ============================================================

csv_path = r"DL dataset\DR_grading.csv"
image_root = r"DL dataset\DR_grading"

# Load CSV
df = pd.read_csv(csv_path)

# Get all image filenames recursively
image_files = []

for root, dirs, files in os.walk(image_root):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_files.append(file)

# Convert to sets
csv_images = set(df["id_code"])
folder_images = set(image_files)

# Check matching filenames
matching = csv_images.intersection(folder_images)

print("============================================")
print("DDR DATASET VERIFICATION")
print("============================================")

print("Images in CSV       :", len(csv_images))
print("Images in folders   :", len(folder_images))
print("Matching filenames  :", len(matching))

print("\nExample CSV filenames:")
for x in list(csv_images)[:5]:
    print(x)

print("\nExample actual image filenames:")
for x in list(folder_images)[:5]:
    print(x)

print("\n============================================")

if len(matching) == len(csv_images):
    print("SUCCESS: CSV and images match completely.")
else:
    print("WARNING: CSV and images DO NOT match.")
    print("Do NOT train the model yet.")