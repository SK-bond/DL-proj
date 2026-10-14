
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "resnet50_balanced_best.pth"

# Keep the same class order used during training.
CLASS_NAMES = [
    "Mild_DR",
    "Moderate_DR",
    "No_DR",
    "Proliferative_DR",
    "Severe_DR",
]

# ImageNet normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_model():
    """Load the trained ResNet50 checkpoint."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Recreate the architecture used for training.
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    # Support common checkpoint dictionary formats.
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise TypeError("Unsupported checkpoint format.")

    # Handle checkpoints saved using DataParallel.
    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, device


def predict_image(image_path, model, device):
    """Predict the diabetic retinopathy severity of one image."""
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

        predicted_index = probabilities.argmax().item()
        confidence = probabilities[predicted_index].item() * 100

    return CLASS_NAMES[predicted_index], confidence, probabilities


def main():
    parser = argparse.ArgumentParser(
        description="Predict diabetic retinopathy severity from a fundus image."
    )
    parser.add_argument(
        "image",
        help="Path to the retinal fundus image",
    )
    args = parser.parse_args()

    try:
        model, device = load_model()
        label, confidence, probabilities = predict_image(
            args.image, model, device
        )

        print("\n--- Diabetic Retinopathy Prediction ---")
        print(f"Image: {args.image}")
        print(f"Device: {device}")
        print(f"Predicted severity: {label}")
        print(f"Model confidence: {confidence:.2f}%")

        print("\nClass probabilities:")
        for name, probability in zip(CLASS_NAMES, probabilities.tolist()):
            print(f"{name}: {probability * 100:.2f}%")

        print("\nNote: This is an experimental model prediction, not a medical diagnosis.")

    except Exception as error:
        print(f"Prediction failed: {error}")


if __name__ == "__main__":
    main()
