
# ============================================================
# Grad-CAM for ResNet50 and VGG16
# Diabetic Retinopathy Severity Classification
# ============================================================

import csv
import random
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet50_Weights, VGG16_Weights
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

BASE_DIR = Path(r"C:\Users\sheev\OneDrive\Desktop\DL proj")

TEST_DIR = BASE_DIR / "DL dataset" / "DR_balanced_split" / "test"

RESNET_PATH = BASE_DIR / "models" / "resnet50_balanced_best.pth"
VGG_PATH = BASE_DIR / "models" / "vgg16_dr_best.pth"

OUTPUT_DIR = BASE_DIR / "gradcam_both_results"

IMAGE_SIZE = 224
IMAGES_PER_CLASS = 2
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet normalization used by the trained models.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"Device: {DEVICE}")
print(f"Test directory: {TEST_DIR}")


# ------------------------------------------------------------
# 2. CHECK PATHS AND LOAD TEST DATASET
# ------------------------------------------------------------

if not TEST_DIR.exists():
    raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")

if not RESNET_PATH.exists():
    raise FileNotFoundError(f"ResNet50 checkpoint not found: {RESNET_PATH}")

if not VGG_PATH.exists():
    raise FileNotFoundError(f"VGG16 checkpoint not found: {VGG_PATH}")

dataset = ImageFolder(TEST_DIR, transform=TRANSFORM)
class_names = dataset.classes
num_classes = len(class_names)

if num_classes != 5:
    raise ValueError(
        f"Expected 5 classes, but found {num_classes}: {class_names}"
    )

print("Class mapping:", dataset.class_to_idx)
print("Test images:", len(dataset))


# ------------------------------------------------------------
# 3. CHECKPOINT LOADING HELPER
# ------------------------------------------------------------

def load_checkpoint(path):
    """
    Load a trusted, locally saved PyTorch checkpoint.

    Supports either:
    - A raw state_dict
    - A dictionary containing model_state_dict
    - A dictionary containing state_dict
    """
    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(path, map_location="cpu")

    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key], checkpoint

    if isinstance(checkpoint, dict) and checkpoint:
        if all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint, {}

    raise ValueError(
        f"Unsupported checkpoint format in {path}. "
        "Inspect the checkpoint keys before proceeding."
    )


def remove_data_parallel_prefix(state_dict):
    """Remove the 'module.' prefix if saved from DataParallel."""
    if state_dict and all(k.startswith("module.") for k in state_dict):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


# ------------------------------------------------------------
# 4. BUILD RESNET50
# ------------------------------------------------------------

def build_resnet50():
    # No pretrained download is required here.
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    state_dict, metadata = load_checkpoint(RESNET_PATH)
    state_dict = remove_data_parallel_prefix(state_dict)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    saved_classes = metadata.get("class_names")
    if saved_classes is not None and list(saved_classes) != class_names:
        raise ValueError(
            "ResNet50 checkpoint class order differs from the test dataset.\n"
            f"Checkpoint: {saved_classes}\nDataset: {class_names}"
        )

    return model.to(DEVICE)


# ------------------------------------------------------------
# 5. BUILD VGG16
# ------------------------------------------------------------

def build_vgg16():
    # Match the custom classifier used by the training script.
    model = models.vgg16(weights=None)

    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes),
    )

    state_dict, metadata = load_checkpoint(VGG_PATH)
    state_dict = remove_data_parallel_prefix(state_dict)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    saved_classes = metadata.get("class_names")
    if saved_classes is not None and list(saved_classes) != class_names:
        raise ValueError(
            "VGG16 checkpoint class order differs from the test dataset.\n"
            f"Checkpoint: {saved_classes}\nDataset: {class_names}"
        )

    return model.to(DEVICE)


# ------------------------------------------------------------
# 6. GRAD-CAM CALCULATION
# ------------------------------------------------------------

def calculate_gradcam(model, image_tensor, target_layer):
    """
    Calculate Grad-CAM for the model's predicted class.

    A forward hook captures activations and registers a gradient
    hook directly on the layer output. This avoids the
    register_full_backward_hook / in-place ReLU error.
    """
    activations = {}
    gradients = {}

    def forward_hook(module, inputs, output):
        activations["value"] = output

        # Capture the gradient of this output during backpropagation.
        output.register_hook(
            lambda grad: gradients.__setitem__("value", grad)
        )

    handle = target_layer.register_forward_hook(forward_hook)

    try:
        model.zero_grad(set_to_none=True)

        input_batch = image_tensor.unsqueeze(0).to(DEVICE)

        # IMPORTANT: do not wrap this forward pass in torch.no_grad().
        with torch.enable_grad():
            logits = model(input_batch)
            probabilities = torch.softmax(logits, dim=1)

            predicted_class = int(logits.argmax(dim=1).item())
            confidence = float(
                probabilities[0, predicted_class].detach().cpu().item()
            )

            score = logits[0, predicted_class]
            score.backward()

        if "value" not in activations or "value" not in gradients:
            raise RuntimeError("Grad-CAM activation or gradient was not captured.")

        activation = activations["value"].detach()
        gradient = gradients["value"].detach()

        # Average gradients spatially to get one weight per channel.
        weights = gradient.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of feature maps.
        cam = (weights * activation).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0].cpu().numpy()

        # Normalize the heatmap to [0, 1].
        cam_min = float(cam.min())
        cam_max = float(cam.max())

        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam, dtype=np.float32)

        probabilities = probabilities[0].detach().cpu().numpy()

        return cam, predicted_class, confidence, probabilities

    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


# ------------------------------------------------------------
# 7. IMAGE DISPLAY AND SAVE
# ------------------------------------------------------------

def denormalize_image(tensor):
    """Convert a normalized tensor back to an RGB image."""
    image = tensor.detach().cpu().permute(1, 2, 0).numpy()
    mean = np.array(MEAN).reshape(1, 1, 3)
    std = np.array(STD).reshape(1, 1, 3)

    image = image * std + mean
    return np.clip(image, 0, 1)


def save_visualization(
    image_tensor,
    cam,
    model_name,
    image_path,
    true_label,
    predicted_label,
    confidence,
    output_dir,
):
    original = denormalize_image(image_tensor)

    # Apply a heatmap and blend it with the original image.
    heatmap_rgb = plt.get_cmap("jet")(cam)[..., :3]
    overlay = np.clip(0.55 * original + 0.45 * heatmap_rgb, 0, 1)

    safe_stem = image_path.stem.replace(" ", "_")
    output_path = output_dir / (
        f"{safe_stem}_true-{true_label}_pred-{predicted_label}.png"
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].imshow(original)
    axes[0].set_title(f"Original\nTrue: {true_label}")

    axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM heatmap")

    axes[2].imshow(overlay)
    axes[2].set_title(
        f"{model_name} overlay\nPred: {predicted_label} ({confidence:.1%})"
    )

    for axis in axes:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return output_path


# ------------------------------------------------------------
# 8. SELECT THE SAME TEST IMAGES FOR BOTH MODELS
# ------------------------------------------------------------

indices_by_class = {name: [] for name in class_names}

for index, (_, label_index) in enumerate(dataset.samples):
    indices_by_class[class_names[label_index]].append(index)

selected_indices = []

for class_name in class_names:
    candidates = indices_by_class[class_name].copy()
    random.shuffle(candidates)
    selected_indices.extend(candidates[:IMAGES_PER_CLASS])

if not selected_indices:
    raise RuntimeError("No test images were selected.")

print(f"Images selected for each model: {len(selected_indices)}")


# ------------------------------------------------------------
# 9. RUN GRAD-CAM FOR EACH MODEL
# ------------------------------------------------------------

def run_model(model_name, model_builder, target_layer_getter):
    output_dir = OUTPUT_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading {model_name}...")
    model = model_builder()
    target_layer = target_layer_getter(model)

    csv_path = output_dir / "gradcam_predictions.csv"
    rows = []

    try:
        for position, index in enumerate(selected_indices, start=1):
            image_tensor, true_index = dataset[index]
            image_path = Path(dataset.samples[index][0])

            true_label = class_names[true_index]

            cam, predicted_index, confidence, probabilities = calculate_gradcam(
                model, image_tensor, target_layer
            )

            predicted_label = class_names[predicted_index]

            saved_path = save_visualization(
                image_tensor=image_tensor,
                cam=cam,
                model_name=model_name,
                image_path=image_path,
                true_label=true_label,
                predicted_label=predicted_label,
                confidence=confidence,
                output_dir=output_dir,
            )

            row = {
                "model": model_name,
                "image": image_path.name,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "confidence": round(confidence, 6),
                "correct": predicted_index == true_index,
                "visualization": str(saved_path),
            }

            for class_index, class_name in enumerate(class_names):
                row[f"probability_{class_name}"] = round(
                    float(probabilities[class_index]), 6
                )

            rows.append(row)

            print(
                f"[{position}/{len(selected_indices)}] "
                f"{image_path.name} -> {predicted_label} "
                f"({confidence:.1%}) | true={true_label}"
            )

    finally:
        # Free model memory before loading the next model.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    correct = sum(row["correct"] for row in rows)
    accuracy = correct / len(rows) if rows else 0.0

    print(f"\n{model_name} completed.")
    print(f"Visualizations: {output_dir}")
    print(f"Prediction CSV: {csv_path}")
    print(f"Accuracy on selected examples: {accuracy:.2%}")


# ------------------------------------------------------------
# 10. MAIN
# ------------------------------------------------------------

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ResNet50: final convolutional block.
    run_model(
        model_name="ResNet50",
        model_builder=build_resnet50,
        target_layer_getter=lambda model: model.layer4[-1],
    )

    # VGG16: last convolutional layer.
    run_model(
        model_name="VGG16",
        model_builder=build_vgg16,
        target_layer_getter=lambda model: model.features[28],
    )

    print("\nAll Grad-CAM runs finished.")
    print(f"Results saved in: {OUTPUT_DIR}")
