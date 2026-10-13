# ============================================================
# gradcam_resnet50.py
# Explainable AI - Grad-CAM for ResNet50
# Diabetic Retinopathy Classification
# ============================================================

import os
import json
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Users\sheev\OneDrive\Desktop\DL proj"

TEST_DIR = os.path.join(
    BASE_DIR,
    "DL dataset",
    "DR_balanced_split",
    "test"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "resnet50_balanced_best.pth"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "gradcam_results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Number of examples to generate per class
IMAGES_PER_CLASS = 3

# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("RESNET50 GRAD-CAM EXPLAINABILITY")
print("DIABETIC RETINOPATHY")
print("=" * 70)

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")

print(f"Device          : {DEVICE}")

# ============================================================
# CHECK PATHS
# ============================================================

print("\nChecking paths...")

if not os.path.isdir(TEST_DIR):
    raise FileNotFoundError(
        f"\nTest directory not found:\n{TEST_DIR}"
    )

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"\nModel file not found:\n{MODEL_PATH}"
    )

print("Test directory found:")
print(TEST_DIR)

print("\nModel found:")
print(MODEL_PATH)

# ============================================================
# LOAD DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING TEST DATASET")
print("=" * 70)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = ImageFolder(
    root=TEST_DIR,
    transform=transform
)

class_names = dataset.classes
num_classes = len(class_names)

print(f"Total test images : {len(dataset)}")

print("\nDetected classes:")

for idx, name in enumerate(class_names):
    print(f"{idx}: {name}")

print(f"\nNumber of classes : {num_classes}")

# ============================================================
# LOAD RESNET50
# ============================================================

print("\n" + "=" * 70)
print("LOADING RESNET50")
print("=" * 70)

model = models.resnet50(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

# Handle different checkpoint formats
if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

else:
    state_dict = checkpoint

# Remove DataParallel prefix if present
clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("module."):
        key = key.replace("module.", "", 1)

    clean_state_dict[key] = value

model.load_state_dict(
    clean_state_dict,
    strict=True
)

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully.")
print("Architecture : ResNet50")
print(f"Output classes : {num_classes}")

# ============================================================
# GRAD-CAM CLASS
# ============================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(
            self.save_activation
        )

        self.backward_handle = target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(self, module, input, output):

        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):

        self.gradients = grad_output[0].detach()

    def generate(self, image, target_class):

        self.model.zero_grad()

        output = self.model(image)

        score = output[:, target_class]

        score.backward()

        gradients = self.gradients
        activations = self.activations

        # Global average pooling of gradients
        weights = torch.mean(
            gradients,
            dim=(2, 3),
            keepdim=True
        )

        cam = torch.sum(
            weights * activations,
            dim=1
        )

        cam = torch.relu(cam)

        cam = cam[0].cpu().numpy()

        # Normalize
        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max - cam_min > 1e-8:

            cam = (
                cam - cam_min
            ) / (
                cam_max - cam_min
            )

        else:

            cam = np.zeros_like(cam)

        return cam, output

    def close(self):

        self.forward_handle.remove()
        self.backward_handle.remove()


# ============================================================
# TARGET LAYER
# ============================================================

# Last convolutional layer of ResNet50
target_layer = model.layer4[-1].conv3

gradcam = GradCAM(
    model=model,
    target_layer=target_layer
)

print("\nGrad-CAM target layer:")
print("model.layer4[-1].conv3")

# ============================================================
# IMAGE TRANSFORM FOR DISPLAY
# ============================================================

display_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
])

# ============================================================
# SELECT IMAGES
# ============================================================

print("\nSelecting images...")

selected_images = []

for class_index, class_name in enumerate(class_names):

    class_folder = os.path.join(
        TEST_DIR,
        class_name
    )

    image_files = []

    for extension in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:

        image_files.extend(
            glob.glob(
                os.path.join(
                    class_folder,
                    extension
                )
            )
        )

    image_files = sorted(image_files)

    if len(image_files) == 0:
        print(
            f"WARNING: No images found for {class_name}"
        )
        continue

    selected = image_files[:IMAGES_PER_CLASS]

    for image_path in selected:

        selected_images.append(
            (
                class_index,
                class_name,
                image_path
            )
        )

print(
    f"Selected {len(selected_images)} images "
    f"for Grad-CAM."
)

# ============================================================
# GENERATE GRAD-CAM
# ============================================================

print("\n" + "=" * 70)
print("GENERATING GRAD-CAM VISUALIZATIONS")
print("=" * 70)

results = []

for image_number, (
    true_class_index,
    true_class_name,
    image_path
) in enumerate(selected_images, start=1):

    print(
        f"\nProcessing {image_number}/"
        f"{len(selected_images)}"
    )

    print(
        f"True class : {true_class_name}"
    )

    # --------------------------------------------------------
    # Load original image
    # --------------------------------------------------------

    original_image = Image.open(
        image_path
    ).convert("RGB")

    display_image = display_transform(
        original_image
    )

    # --------------------------------------------------------
    # Prepare tensor
    # --------------------------------------------------------

    input_tensor = transform(
        original_image
    ).unsqueeze(0).to(DEVICE)

    # --------------------------------------------------------
    # First prediction
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(
            input_tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_class
        ].item()

    predicted_class_name = class_names[
        predicted_class
    ]

    print(
        f"Predicted class : "
        f"{predicted_class_name}"
    )

    print(
        f"Confidence : "
        f"{confidence * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Generate Grad-CAM for predicted class
    # --------------------------------------------------------

    cam, _ = gradcam.generate(
        input_tensor,
        predicted_class
    )

    # --------------------------------------------------------
    # Resize CAM
    # --------------------------------------------------------

    cam_image = Image.fromarray(
        np.uint8(cam * 255)
    )

    cam_image = cam_image.resize(
        (
            display_image.width,
            display_image.height
        ),
        Image.Resampling.BILINEAR
    )

    cam_array = np.asarray(
        cam_image
    ) / 255.0

    # --------------------------------------------------------
    # Create output filename
    # --------------------------------------------------------

    base_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    output_name = (
        f"{image_number:02d}_"
        f"{true_class_name}_"
        f"pred_{predicted_class_name}_"
        f"{base_name}"
    )

    # --------------------------------------------------------
    # Create visualization
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    # Original
    axes[0].imshow(
        display_image
    )

    axes[0].set_title(
        f"Original\n"
        f"True: {true_class_name}"
    )

    axes[0].axis("off")

    # Heatmap
    axes[1].imshow(
        cam_array,
        cmap="jet"
    )

    axes[1].set_title(
        "Grad-CAM Heatmap"
    )

    axes[1].axis("off")

    # Overlay
    axes[2].imshow(
        display_image
    )

    axes[2].imshow(
        cam_array,
        cmap="jet",
        alpha=0.45
    )

    axes[2].set_title(
        f"Prediction: {predicted_class_name}\n"
        f"Confidence: {confidence * 100:.2f}%"
    )

    axes[2].axis("off")

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        output_name + ".png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    results.append({
        "image": os.path.basename(image_path),
        "true_class": true_class_name,
        "predicted_class": predicted_class_name,
        "confidence": confidence,
        "correct": (
            true_class_index == predicted_class
        ),
        "output_file": output_path
    })

    print(
        f"Saved: {output_path}"
    )

# ============================================================
# SAVE RESULTS CSV
# ============================================================

results_df = pd.DataFrame(
    results
)

results_csv = os.path.join(
    OUTPUT_DIR,
    "gradcam_results.csv"
)

results_df.to_csv(
    results_csv,
    index=False
)

# ============================================================
# SAVE CLASS MAPPING
# ============================================================

class_mapping_path = os.path.join(
    OUTPUT_DIR,
    "class_mapping.json"
)

with open(
    class_mapping_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            str(index): name
            for index, name
            in enumerate(class_names)
        },
        f,
        indent=4
    )

# ============================================================
# SUMMARY
# ============================================================

correct_count = sum(
    result["correct"]
    for result in results
)

total_count = len(results)

print("\n" + "=" * 70)
print("GRAD-CAM COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    f"Images processed : {total_count}"
)

print(
    f"Correct predictions : {correct_count}"
)

if total_count > 0:

    print(
        f"Selected-image accuracy : "
        f"{correct_count / total_count * 100:.2f}%"
    )

print("\nFiles created:")

print(
    f"1. {OUTPUT_DIR}"
)

print(
    f"2. {results_csv}"
)

print(
    f"3. {class_mapping_path}"
)

print("=" * 70)

gradcam.close()