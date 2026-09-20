import torch
import torchvision
import torch.nn as nn
import gradio as gr


# ==========================================
# 1. Device
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")


# ==========================================
# 2. Class names
# ==========================================

class_names = [
    'apple_pie',
    'caesar_salad',
    'cheesecake',
    'chicken_wings',
    'chocolate_cake',
    'donuts',
    'french_fries',
    'fried_rice',
    'greek_salad',
    'hamburger',
    'hot_dog',
    'ice_cream',
    'lasagna',
    'macarons',
    'pancakes',
    'pizza',
    'ramen',
    'spaghetti_carbonara',
    'sushi',
    'waffles'
]


# ==========================================
# 3. Create EfficientNet-B3
# ==========================================

weights = torchvision.models.EfficientNet_B3_Weights.DEFAULT

model = torchvision.models.efficientnet_b3(
    weights=weights
)


# ==========================================
# 4. Replace classifier
# ==========================================

model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(
        in_features=1536,
        out_features=len(class_names)
    )
)


# ==========================================
# 5. Load trained model
# ==========================================

model.load_state_dict(
    torch.load(
        "models/best_food_model.pth",
        map_location=device,
        weights_only=True
    )
)


# ==========================================
# 6. Prepare model
# ==========================================

model = model.to(device)

model.eval()

print("✅ Model loaded successfully!")


# ==========================================
# 7. Image preprocessing
# ==========================================

transform = weights.transforms()


# ==========================================
# 8. Prediction function
# ==========================================

def predict_image(image):

    image = image.convert("RGB")

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.inference_mode():

        logits = model(image_tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

    probabilities = probabilities[0]

    top_probs, top_indices = torch.topk(
        probabilities,
        k=5
    )

    results = {}

    for probability, index in zip(
        top_probs,
        top_indices
    ):
        results[class_names[index.item()]] = probability.item()

    return results


# ==========================================
# 9. Create Gradio interface
# ==========================================

demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(
        type="pil",
        label="Upload Food Image"
    ),
    outputs=gr.Label(
        num_top_classes=5,
        label="Prediction"
    ),
    title="🍔 20 Food Classifier",
    description=(
        "Upload an image and the EfficientNet-B3 model "
        "will predict the food class."
    )
)


# ==========================================
# 10. Run app
# ==========================================

if __name__ == "__main__":
    demo.launch()