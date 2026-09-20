import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# 1. Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from going_modular.data_setup import create_dataloaders


# =========================================================
# 2. Settings
# =========================================================

TEST_DIR = PROJECT_ROOT / "data" / "test"
MODEL_PATH = PROJECT_ROOT / "models" / "best_food_model1.pth"

BATCH_SIZE = 16
NUM_WORKERS = 0

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")


# =========================================================
# 3. ViT official transforms
# =========================================================

weights = models.ViT_B_16_Weights.IMAGENET1K_V1
transform = weights.transforms()


# =========================================================
# 4. DataLoader
# =========================================================

_, test_dataloader, class_names = create_dataloaders(
    train_dir="food-101-20/train",
    test_dir="food-101-20/test",
    transform=transform,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS
)

class_names = test_dataloader.dataset.classes

print("\nClasses:")
for i, name in enumerate(class_names):
    print(f"{i:2d}: {name}")

print(f"\nNumber of classes: {len(class_names)}")


# =========================================================
# 5. Build ViT
# =========================================================

model = models.vit_b_16(weights=None)

model.heads.head = nn.Linear(
    in_features=model.heads.head.in_features,
    out_features=len(class_names)
)


# =========================================================
# 6. Load best checkpoint
# =========================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

# Handle different checkpoint formats
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

print("\n✅ Best ViT model loaded.")


# =========================================================
# 7. Predictions
# =========================================================

all_preds = []
all_labels = []

with torch.inference_mode():

    for images, labels in test_dataloader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


# =========================================================
# 8. Overall Accuracy
# =========================================================

correct = sum(
    pred == label
    for pred, label in zip(all_preds, all_labels)
)

total = len(all_labels)

accuracy = correct / total

print("\n" + "=" * 60)
print("OVERALL RESULTS")
print("=" * 60)

print(f"Correct predictions : {correct}")
print(f"Total images        : {total}")
print(f"Accuracy            : {accuracy * 100:.2f}%")


# =========================================================
# 9. Confusion Matrix
# =========================================================

cm = confusion_matrix(
    all_labels,
    all_preds,
    labels=list(range(len(class_names)))
)

plt.figure(figsize=(16, 14))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("ViT-B/16 Confusion Matrix")

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()

plt.savefig(
    PROJECT_ROOT / "vit_confusion_matrix.png",
    dpi=300
)

plt.show()


# =========================================================
# 10. Per-Class Accuracy
# =========================================================

print("\n" + "=" * 60)
print("PER-CLASS ACCURACY")
print("=" * 60)

for i, class_name in enumerate(class_names):

    total_class = cm[i].sum()

    if total_class == 0:
        class_acc = 0
    else:
        class_acc = cm[i, i] / total_class

    print(
        f"{class_name:25s} : "
        f"{class_acc * 100:6.2f}% "
        f"({cm[i, i]}/{total_class})"
    )


# =========================================================
# 11. Classification Report
# =========================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    all_labels,
    all_preds,
    target_names=class_names,
    digits=4,
    zero_division=0
)

print(report)


# =========================================================
# 12. Top Confusions
# =========================================================

print("\n" + "=" * 60)
print("TOP CONFUSIONS")
print("=" * 60)

confusions = []

for true_idx in range(len(class_names)):

    for pred_idx in range(len(class_names)):

        if true_idx != pred_idx:

            count = cm[true_idx, pred_idx]

            if count > 0:

                confusions.append(
                    (
                        count,
                        class_names[true_idx],
                        class_names[pred_idx]
                    )
                )


confusions.sort(reverse=True)


print("\nMost common mistakes:\n")

for rank, (count, true_class, pred_class) in enumerate(
    confusions[:15],
    start=1
):

    print(
        f"{rank:2d}. "
        f"{true_class:25s} → "
        f"{pred_class:25s} "
        f"({count} images)"
    )


# =========================================================
# 13. Save report
# =========================================================

with open(
    PROJECT_ROOT / "vit_classification_report.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("ViT-B/16 Food Classifier Analysis\n")
    f.write("=" * 60 + "\n\n")

    f.write(
        f"Overall Accuracy: {accuracy * 100:.2f}%\n\n"
    )

    f.write("PER-CLASS ACCURACY\n")
    f.write("=" * 60 + "\n")

    for i, class_name in enumerate(class_names):

        total_class = cm[i].sum()

        if total_class == 0:
            class_acc = 0
        else:
            class_acc = cm[i, i] / total_class

        f.write(
            f"{class_name:25s} : "
            f"{class_acc * 100:6.2f}% "
            f"({cm[i, i]}/{total_class})\n"
        )

    f.write("\n\nTOP CONFUSIONS\n")
    f.write("=" * 60 + "\n")

    for rank, (count, true_class, pred_class) in enumerate(
        confusions[:15],
        start=1
    ):

        f.write(
            f"{rank:2d}. "
            f"{true_class} -> {pred_class} "
            f"({count} images)\n"
        )


print("\n✅ Analysis complete!")

print("\nSaved files:")

print("📊 vit_confusion_matrix.png")
print("📄 vit_classification_report.txt")