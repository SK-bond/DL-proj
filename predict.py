# ============================================================
# SINGLE IMAGE DIABETIC RETINOPATHY PREDICTION
# ResNet50 - Experiment 2 Improved
# ============================================================

import os
import sys
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models, transforms

import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Users\sheev\OneDrive\Desktop\DL proj"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "resnet50_exp2_improved_best.pth"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "prediction_results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. SETTINGS
# ============================================================

IMAGE_SIZE = 224
NUM_CLASSES = 5

CLASS_NAMES = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR"
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 3. SYSTEM INFORMATION
# ============================================================

print("=" * 70)
print("DIABETIC RETINOPATHY - SINGLE IMAGE PREDICTION")
print("=" * 70)

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(
        f"GPU             : "
        f"{torch.cuda.get_device_name(0)}"
    )

print(f"Device          : {DEVICE}")

print("=" * 70)


# ============================================================
# 4. CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )


# ============================================================
# 5. IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 6. LOAD RESNET50
# ============================================================

print("\nLoading ResNet50 model...")

model = models.resnet50(
    weights=None
)

model.fc = nn.Sequential(
    nn.Dropout(
        p=0.30
    ),

    nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )
)


# ============================================================
# 7. LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)


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

        state_dict = checkpoint

else:

    state_dict = checkpoint


# ============================================================
# 8. REMOVE module. PREFIX IF PRESENT
# ============================================================

clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("module."):

        key = key[7:]

    clean_state_dict[key] = value


model.load_state_dict(
    clean_state_dict,
    strict=True
)

model = model.to(DEVICE)

model.eval()

print("Model loaded successfully.")


# ============================================================
# 9. LOAD IMAGE
# ============================================================

def load_image(image_path):

    if not os.path.exists(image_path):

        raise FileNotFoundError(
            f"\nImage not found:\n{image_path}"
        )

    image = Image.open(
        image_path
    ).convert("RGB")

    original_image = image.copy()

    tensor = transform(image)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(DEVICE)

    return original_image, tensor


# ============================================================
# 10. PREDICT IMAGE
# ============================================================

def predict(image_tensor):

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = F.softmax(
            outputs,
            dim=1
        )[0]

        predicted_class = torch.argmax(
            probabilities
        ).item()

    return (
        predicted_class,
        probabilities.cpu().numpy()
    )


# ============================================================
# 11. GRAD-CAM
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

        self.forward_handle = (
            target_layer.register_forward_hook(
                self.save_activation
            )
        )

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

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights * activations
        ).sum(
            dim=1,
            keepdim=True
        )

        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            mode="bilinear",
            align_corners=False
        )

        cam = cam[
            0,
            0
        ].detach().cpu().numpy()

        cam -= cam.min()

        if cam.max() > 0:

            cam /= cam.max()

        return cam


    def close(self):

        self.forward_handle.remove()

        self.backward_handle.remove()


# ============================================================
# 12. SAVE RESULT
# ============================================================

def save_result(
    original_image,
    cam,
    predicted_class,
    probabilities,
    image_path
):

    original_image = original_image.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    )

    original_array = np.array(
        original_image
    )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5)
    )

    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    axes[0].imshow(
        original_array
    )

    axes[0].set_title(
        "Original Fundus Image",
        fontsize=12
    )

    axes[0].axis("off")

    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------

    axes[1].imshow(
        cam,
        cmap="jet"
    )

    axes[1].set_title(
        "Grad-CAM Heatmap",
        fontsize=12
    )

    axes[1].axis("off")

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    axes[2].imshow(
        original_array
    )

    axes[2].imshow(
        cam,
        cmap="jet",
        alpha=0.45
    )

    axes[2].set_title(
        "Grad-CAM Overlay",
        fontsize=12
    )

    axes[2].axis("off")

    # --------------------------------------------------------
    # Main title
    # --------------------------------------------------------

    confidence = (
        probabilities[predicted_class]
        * 100
    )

    fig.suptitle(
        f"Prediction: "
        f"{CLASS_NAMES[predicted_class]} "
        f"({confidence:.2f}%)",
        fontsize=15
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # Output filename
    # --------------------------------------------------------

    image_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{image_name}_prediction.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    return output_path


# ============================================================
# 13. GET IMAGE PATH
# ============================================================

if len(sys.argv) < 2:

    print("\nUsage:")

    print(
        'python predict.py "path_to_image.jpg"'
    )

    print("\nExample:")

    print(
        r'python predict.py "C:\Users\sheev\Downloads\eye.jpg"'
    )

    sys.exit(1)


IMAGE_PATH = sys.argv[1]


# ============================================================
# 14. LOAD IMAGE
# ============================================================

print("\nLoading image:")

print(IMAGE_PATH)

original_image, image_tensor = load_image(
    IMAGE_PATH
)


# ============================================================
# 15. PREDICTION
# ============================================================

print("\nRunning prediction...")

predicted_class, probabilities = predict(
    image_tensor
)


# ============================================================
# 16. DISPLAY RESULT
# ============================================================

print("\n" + "=" * 70)

print("PREDICTION RESULT")

print("=" * 70)

print(
    f"Image          : "
    f"{os.path.basename(IMAGE_PATH)}"
)

print(
    f"Predicted Class: "
    f"{CLASS_NAMES[predicted_class]}"
)

print(
    f"Confidence     : "
    f"{probabilities[predicted_class] * 100:.2f}%"
)

print("\nClass probabilities:")

for class_id in range(NUM_CLASSES):

    print(
        f"  {CLASS_NAMES[class_id]:20s} : "
        f"{probabilities[class_id] * 100:.2f}%"
    )


# ============================================================
# 17. GENERATE GRAD-CAM
# ============================================================

print("\nGenerating Grad-CAM...")

target_layer = model.layer4[-1]

gradcam = GradCAM(
    model,
    target_layer
)

cam = gradcam.generate(
    image_tensor,
    predicted_class
)


# ============================================================
# 18. SAVE RESULT
# ============================================================

output_path = save_result(
    original_image,
    cam,
    predicted_class,
    probabilities,
    IMAGE_PATH
)

gradcam.close()


# ============================================================
# 19. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)

print("PREDICTION COMPLETED")

print("=" * 70)

print(
    f"Prediction : "
    f"{CLASS_NAMES[predicted_class]}"
)

print(
    f"Confidence : "
    f"{probabilities[predicted_class] * 100:.2f}%"
)

print(
    f"\nGrad-CAM result saved to:"
)

print(output_path)

print("=" * 70)