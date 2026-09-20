import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, models
from torch.utils.data import DataLoader


# =========================================================
# 1. Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

TEST_DIR = PROJECT_ROOT /"food-101-20"/ "test"

B3_PATH = PROJECT_ROOT / "models" / "best_food_model.pth"
VIT_PATH = PROJECT_ROOT / "models" / "best_food_model1.pth"


# =========================================================
# 2. Settings
# =========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32

# Windows: avoid multiprocessing problems during analysis
NUM_WORKERS = 0

print(f"Device: {DEVICE}")
print(f"Test dataset: {TEST_DIR}")


# =========================================================
# 3. Check paths
# =========================================================

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test directory not found:\n{TEST_DIR}"
    )

if not B3_PATH.exists():
    raise FileNotFoundError(
        f"B3 model not found:\n{B3_PATH}"
    )

if not VIT_PATH.exists():
    raise FileNotFoundError(
        f"ViT model not found:\n{VIT_PATH}"
    )


# =========================================================
# 4. Official transforms
# =========================================================

b3_weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1
vit_weights = models.ViT_B_16_Weights.IMAGENET1K_V1

b3_transform = b3_weights.transforms()
vit_transform = vit_weights.transforms()


# =========================================================
# 5. Test datasets
# =========================================================

b3_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=b3_transform
)

vit_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=vit_transform
)

class_names = b3_dataset.classes

print("\nClasses:")

for i, name in enumerate(class_names):
    print(f"{i:2d}: {name}")

print(f"\nNumber of classes: {len(class_names)}")


# =========================================================
# 6. DataLoaders
# =========================================================

b3_loader = DataLoader(
    b3_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

vit_loader = DataLoader(
    vit_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# =========================================================
# 7. Helper: load checkpoint
# =========================================================

def load_checkpoint(model, path):

    checkpoint = torch.load(
        path,
        map_location=DEVICE
    )

    # Case 1: checkpoint contains model_state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

        state_dict = checkpoint["model_state_dict"]

    # Case 2: checkpoint contains state_dict
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:

        state_dict = checkpoint["state_dict"]

    # Case 3: checkpoint itself is state_dict
    else:

        state_dict = checkpoint

    model.load_state_dict(state_dict)

    return model


# =========================================================
# 8. Build EfficientNet-B3
# =========================================================

print("\nLoading EfficientNet-B3...")

b3_model = models.efficientnet_b3(
    weights=None
)

b3_model.classifier[1] = nn.Linear(
    in_features=b3_model.classifier[1].in_features,
    out_features=len(class_names)
)

b3_model = load_checkpoint(
    b3_model,
    B3_PATH
)

b3_model = b3_model.to(DEVICE)
b3_model.eval()


# =========================================================
# 9. Build ViT-B/16
# =========================================================

print("Loading ViT-B/16...")

vit_model = models.vit_b_16(
    weights=None
)

vit_model.heads.head = nn.Linear(
    in_features=vit_model.heads.head.in_features,
    out_features=len(class_names)
)

vit_model = load_checkpoint(
    vit_model,
    VIT_PATH
)

vit_model = vit_model.to(DEVICE)
vit_model.eval()


print("\n✅ Both models loaded successfully.")


# =========================================================
# 10. Parameter count
# =========================================================

def count_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


b3_params = count_parameters(b3_model)
vit_params = count_parameters(vit_model)


# =========================================================
# 11. Model file sizes
# =========================================================

b3_size_mb = B3_PATH.stat().st_size / (1024 ** 2)
vit_size_mb = VIT_PATH.stat().st_size / (1024 ** 2)


# =========================================================
# 12. Evaluation function
# =========================================================

def evaluate(model, dataloader):

    all_preds = []
    all_labels = []

    total_inference_time = 0.0

    with torch.inference_mode():

        for images, labels in dataloader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Make CUDA timing accurate
            if DEVICE == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            outputs = model(images)

            if DEVICE == "cuda":
                torch.cuda.synchronize()

            total_inference_time += (
                time.perf_counter() - start_time
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_preds.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

    correct = sum(
        prediction == label
        for prediction, label
        in zip(all_preds, all_labels)
    )

    total = len(all_labels)

    accuracy = correct / total

    return (
        accuracy,
        total_inference_time,
        all_preds,
        all_labels
    )


# =========================================================
# 13. Evaluate B3
# =========================================================

print("\nEvaluating EfficientNet-B3...")

b3_accuracy, b3_time, b3_preds, b3_labels = evaluate(
    b3_model,
    b3_loader
)


# =========================================================
# 14. Evaluate ViT
# =========================================================

print("Evaluating ViT-B/16...")

vit_accuracy, vit_time, vit_preds, vit_labels = evaluate(
    vit_model,
    vit_loader
)


# =========================================================
# 15. Per-class accuracy
# =========================================================

def calculate_per_class_accuracy(
    labels,
    predictions
):

    results = {}

    for class_index, class_name in enumerate(class_names):

        total_class = 0
        correct_class = 0

        for label, prediction in zip(
            labels,
            predictions
        ):

            if label == class_index:

                total_class += 1

                if prediction == class_index:
                    correct_class += 1

        if total_class > 0:

            accuracy = (
                correct_class / total_class
            )

        else:

            accuracy = 0.0

        results[class_name] = accuracy

    return results


b3_class_accuracy = calculate_per_class_accuracy(
    b3_labels,
    b3_preds
)

vit_class_accuracy = calculate_per_class_accuracy(
    vit_labels,
    vit_preds
)


# =========================================================
# 16. Overall comparison
# =========================================================

print("\n")
print("=" * 75)
print("MODEL COMPARISON")
print("=" * 75)

print(
    f"{'Metric':<25}"
    f"{'EfficientNet-B3':<25}"
    f"{'ViT-B/16':<25}"
)

print("-" * 75)

print(
    f"{'Accuracy':<25}"
    f"{b3_accuracy * 100:>10.2f}%"
    f"{vit_accuracy * 100:>15.2f}%"
)

print(
    f"{'Parameters':<25}"
    f"{b3_params / 1e6:>10.2f}M"
    f"{vit_params / 1e6:>15.2f}M"
)

print(
    f"{'Model Size':<25}"
    f"{b3_size_mb:>10.2f} MB"
    f"{vit_size_mb:>15.2f} MB"
)

print(
    f"{'Inference Time':<25}"
    f"{b3_time:>10.3f} sec"
    f"{vit_time:>15.3f} sec"
)

print("=" * 75)


# =========================================================
# 17. Per-class comparison
# =========================================================

print("\n")
print("=" * 75)
print("PER-CLASS ACCURACY")
print("=" * 75)

print(
    f"{'Class':<25}"
    f"{'B3':>12}"
    f"{'ViT':>12}"
)

print("-" * 75)

for class_name in class_names:

    b3_acc = (
        b3_class_accuracy[class_name] * 100
    )

    vit_acc = (
        vit_class_accuracy[class_name] * 100
    )

    print(
        f"{class_name:<25}"
        f"{b3_acc:>10.2f}%"
        f"{vit_acc:>10.2f}%"
    )


# =========================================================
# 18. Accuracy difference
# =========================================================

print("\n")
print("=" * 75)
print("ACCURACY DIFFERENCE (ViT - B3)")
print("=" * 75)

for class_name in class_names:

    difference = (
        vit_class_accuracy[class_name]
        - b3_class_accuracy[class_name]
    ) * 100

    print(
        f"{class_name:<25}"
        f"{difference:+.2f}%"
    )


# =========================================================
# 19. Final message
# =========================================================

print("\n")
print("✅ Comparison complete!")
