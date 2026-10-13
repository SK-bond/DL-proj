# ============================================================
# GRAD-CAM EXPLAINABILITY FOR RESNET50
# DIABETIC RETINOPATHY CLASSIFICATION
# ============================================================

import os
import json
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Users\sheev\OneDrive\Desktop\DL proj"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "resnet50_balanced_best.pth"
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "DL dataset",
    "DR_balanced_split",
    "test"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "gradcam_results"
)

IMAGE_SIZE = 224

SEED = 42

# Number of images to generate per class
IMAGES_PER_CLASS = 1


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("GRAD-CAM RESNET50 - DIABETIC RETINOPATHY")
print("=" * 70)

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")

print(f"Device          : {DEVICE}")

print("=" * 70)


# ============================================================
# CHECK PATHS
# ============================================================

print("\nChecking paths...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(
        f"\nTest directory not found:\n{TEST_DIR}"
    )

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print("Model found:")
print(MODEL_PATH)

print("\nTest directory found:")
print(TEST_DIR)

print("\nOutput directory:")
print(OUTPUT_DIR)


# ============================================================
# DETECT CLASSES AUTOMATICALLY
# ============================================================

class_names = sorted(
    [
        directory
        for directory in os.listdir(TEST_DIR)
        if os.path.isdir(
            os.path.join(TEST_DIR, directory)
        )
    ]
)

if len(class_names) == 0:
    raise ValueError(
        "No class folders found inside test directory."
    )

NUM_CLASSES = len(class_names)

print("\nDetected classes:")

for index, class_name in enumerate(class_names):
    print(f"{index}: {class_name}")

print(f"\nNumber of classes: {NUM_CLASSES}")


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],

            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ]
)


# ============================================================
# LOAD RESNET50
# ============================================================

print("\n" + "=" * 70)
print("LOADING RESNET50")
print("=" * 70)

model = models.resnet50(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    NUM_CLASSES
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("\nLoading trained weights...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)


# Handle different checkpoint formats
if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        # Assume checkpoint itself is state_dict
        state_dict = checkpoint

else:

    state_dict = checkpoint


# Remove "module." prefix if present
clean_state_dict = {}

for key, value in state_dict.items():

    new_key = key

    if new_key.startswith("module."):
        new_key = new_key[
            len("module.") :
        ]

    clean_state_dict[new_key] = value


model.load_state_dict(
    clean_state_dict,
    strict=True
)

model = model.to(DEVICE)

model.eval()

print("Model loaded successfully.")

print("\nModel:")
print("ResNet50")

print(
    f"Output classes: {NUM_CLASSES}"
)


# ============================================================
# GRAD-CAM CLASS
# ============================================================

class GradCAM:

    def __init__(
        self,
        model,
        target_layer
    ):

        self.model = model

        self.target_layer = target_layer

        self.activations = None

        self.gradients = None

        # Forward hook
        self.forward_handle = (
            target_layer.register_forward_hook(
                self.save_activation
            )
        )

        # Backward hook
        self.backward_handle = (
            target_layer.register_full_backward_hook(
                self.save_gradient
            )
        )


    def save_activation(
        self,
        module,
        input,
        output
    ):

        self.activations = output


    def save_gradient(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0]


    def generate(
        self,
        image_tensor,
        target_class
    ):

        self.model.zero_grad()

        output = self.model(
            image_tensor
        )

        score = output[
            0,
            target_class
        ]

        score.backward()

        activations = self.activations

        gradients = self.gradients

        if activations is None:
            raise RuntimeError(
                "Activations were not captured."
            )

        if gradients is None:
            raise RuntimeError(
                "Gradients were not captured."
            )

        # Global average pooling
        weights = torch.mean(
            gradients,
            dim=(2, 3),
            keepdim=True
        )

        # Weighted activation maps
        cam = torch.sum(
            weights * activations,
            dim=1
        )

        cam = torch.relu(cam)

        cam = cam[0]

        # Normalize
        cam -= cam.min()

        max_value = cam.max()

        if max_value > 0:
            cam /= max_value

        cam = cam.detach().cpu().numpy()

        return cam, output


    def remove_hooks(self):

        self.forward_handle.remove()

        self.backward_handle.remove()


# ============================================================
# TARGET LAYER
# ============================================================

# Last convolutional layer of ResNet50
target_layer = model.layer4[-1].conv3

print(
    "\nGrad-CAM target layer:"
)

print(
    "model.layer4[-1].conv3"
)


# ============================================================
# CREATE GRAD-CAM
# ============================================================

gradcam = GradCAM(
    model,
    target_layer
)


# ============================================================
# FIND IMAGE FILES
# ============================================================

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
)


def get_images_from_class(
    class_name
):

    class_dir = os.path.join(
        TEST_DIR,
        class_name
    )

    files = []

    for filename in os.listdir(
        class_dir
    ):

        if filename.lower().endswith(
            VALID_EXTENSIONS
        ):

            files.append(
                os.path.join(
                    class_dir,
                    filename
                )
            )

    return sorted(files)


# ============================================================
# COLLECT IMAGES
# ============================================================

selected_images = []

print(
    "\n" + "=" * 70
)

print(
    "SELECTING TEST IMAGES"
)

print(
    "=" * 70
)

for class_name in class_names:

    images = get_images_from_class(
        class_name
    )

    if len(images) == 0:

        print(
            f"WARNING: No images found for {class_name}"
        )

        continue

    # Random but reproducible selection
    random.seed(
        SEED + class_names.index(
            class_name
        )
    )

    selected = random.sample(
        images,
        min(
            IMAGES_PER_CLASS,
            len(images)
        )
    )

    for image_path in selected:

        selected_images.append(
            (
                class_name,
                image_path
            )
        )

    print(
        f"{class_name}: "
        f"{len(selected)} image(s)"
    )


print(
    f"\nTotal images selected: "
    f"{len(selected_images)}"
)


# ============================================================
# DENORMALIZE IMAGE
# ============================================================

def denormalize(
    tensor
):

    mean = torch.tensor(
        [
            0.485,
            0.456,
            0.406
        ]
    ).view(
        3,
        1,
        1
    )

    std = torch.tensor(
        [
            0.229,
            0.224,
            0.225
        ]
    ).view(
        3,
        1,
        1
    )

    image = (
        tensor.cpu() * std
    ) + mean

    image = torch.clamp(
        image,
        0,
        1
    )

    return image.permute(
        1,
        2,
        0
    ).numpy()


# ============================================================
# PROCESS IMAGES
# ============================================================

results = []

print(
    "\n" + "=" * 70
)

print(
    "GENERATING GRAD-CAM VISUALIZATIONS"
)

print(
    "=" * 70
)


for image_number, (
    true_class,
    image_path
) in enumerate(
    selected_images,
    start=1
):

    print(
        f"\nProcessing "
        f"{image_number}/{len(selected_images)}"
    )

    print(
        f"Actual class: {true_class}"
    )

    print(
        f"Image: {os.path.basename(image_path)}"
    )


    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    original_image = Image.open(
        image_path
    ).convert(
        "RGB"
    )


    # --------------------------------------------------------
    # TRANSFORM
    # --------------------------------------------------------

    image_tensor = transform(
        original_image
    ).unsqueeze(
        0
    ).to(
        DEVICE
    )

    image_tensor.requires_grad = True


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with torch.no_grad():

        prediction_output = model(
            image_tensor
        )

        probabilities = torch.softmax(
            prediction_output,
            dim=1
        )

        predicted_class_index = (
            torch.argmax(
                probabilities,
                dim=1
            ).item()
        )

        confidence = probabilities[
            0,
            predicted_class_index
        ].item()


    predicted_class = class_names[
        predicted_class_index
    ]


    print(
        f"Predicted class: "
        f"{predicted_class}"
    )

    print(
        f"Confidence: "
        f"{confidence * 100:.2f}%"
    )


    # --------------------------------------------------------
    # GRAD-CAM
    # --------------------------------------------------------

    cam, _ = gradcam.generate(
        image_tensor,
        predicted_class_index
    )


    # --------------------------------------------------------
    # RESIZE CAM
    # --------------------------------------------------------

    cam_image = Image.fromarray(
        np.uint8(
            cam * 255
        )
    )

    cam_image = cam_image.resize(
        original_image.size,
        Image.Resampling.BILINEAR
    )

    cam_array = np.array(
        cam_image
    ) / 255.0


    # --------------------------------------------------------
    # DENORMALIZED IMAGE
    # --------------------------------------------------------

    display_image = denormalize(
        image_tensor[0].detach()
    )


    # --------------------------------------------------------
    # CREATE FIGURE
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5)
    )


    # --------------------------------------------------------
    # ORIGINAL
    # --------------------------------------------------------

    axes[0].imshow(
        display_image
    )

    axes[0].set_title(
        f"Original Image\n"
        f"Actual: {true_class}"
    )

    axes[0].axis(
        "off"
    )


    # --------------------------------------------------------
    # HEATMAP
    # --------------------------------------------------------

    axes[1].imshow(
        display_image
    )

    axes[1].imshow(
        cam_array,
        cmap="jet",
        alpha=0.5
    )

    axes[1].set_title(
        "Grad-CAM Heatmap"
    )

    axes[1].axis(
        "off"
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    axes[2].imshow(
        display_image
    )

    axes[2].imshow(
        cam_array,
        cmap="jet",
        alpha=0.5
    )

    axes[2].set_title(
        f"Prediction\n"
        f"{predicted_class}\n"
        f"Confidence: "
        f"{confidence * 100:.2f}%"
    )

    axes[2].axis(
        "off"
    )


    # --------------------------------------------------------
    # OVERALL TITLE
    # --------------------------------------------------------

    correct = (
        true_class ==
        predicted_class
    )

    status = (
        "Correct"
        if correct
        else "Incorrect"
    )

    fig.suptitle(
        f"ResNet50 Grad-CAM - "
        f"Diabetic Retinopathy\n"
        f"{status}",
        fontsize=15,
        fontweight="bold"
    )


    plt.tight_layout()


    # --------------------------------------------------------
    # OUTPUT FILENAME
    # --------------------------------------------------------

    safe_true = (
        true_class
        .replace(" ", "_")
        .replace("/", "_")
    )

    safe_pred = (
        predicted_class
        .replace(" ", "_")
        .replace("/", "_")
    )

    image_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    output_filename = (
        f"{image_number:02d}_"
        f"{safe_true}_"
        f"pred_{safe_pred}_"
        f"{image_name}.png"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        output_filename
    )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    print(
        f"Saved: {output_path}"
    )


    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    results.append(
        {
            "image": os.path.basename(
                image_path
            ),

            "actual_class": true_class,

            "predicted_class":
                predicted_class,

            "confidence":
                confidence,

            "correct":
                correct,

            "gradcam_file":
                output_filename
        }
    )


# ============================================================
# REMOVE HOOKS
# ============================================================

gradcam.remove_hooks()


# ============================================================
# SAVE RESULTS CSV
# ============================================================

results_df = pd.DataFrame(
    results
)

csv_path = os.path.join(
    OUTPUT_DIR,
    "gradcam_results.csv"
)

results_df.to_csv(
    csv_path,
    index=False
)


# ============================================================
# SAVE CLASS MAPPING
# ============================================================

mapping_path = os.path.join(
    OUTPUT_DIR,
    "class_mapping.json"
)

with open(
    mapping_path,
    "w"
) as f:

    json.dump(
        {
            str(index): class_name
            for index, class_name
            in enumerate(class_names)
        },
        f,
        indent=4
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

correct_count = sum(
    result["correct"]
    for result in results
)

total_count = len(
    results
)

if total_count > 0:

    sample_accuracy = (
        correct_count /
        total_count
    )

else:

    sample_accuracy = 0


print(
    "\n" + "=" * 70
)

print(
    "GRAD-CAM COMPLETED SUCCESSFULLY"
)

print(
    "=" * 70
)

print(
    f"\nImages processed : "
    f"{total_count}"
)

print(
    f"Correct predictions: "
    f"{correct_count}"
)

print(
    f"Sample accuracy   : "
    f"{sample_accuracy * 100:.2f}%"
)

print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)

print(
    "\nGenerated:"
)

print(
    "1. Grad-CAM visualization PNG files"
)

print(
    "2. gradcam_results.csv"
)

print(
    "3. class_mapping.json"
)

print(
    "\n" + "=" * 70
)