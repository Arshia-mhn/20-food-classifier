# 🍔 FoodVision-20

A deep learning image classification project that recognizes **20 food categories** using PyTorch and a fine-tuned Vision Transformer (ViT-B/16).

> 🚧 **Deployment:** Coming soon — an interactive web application will be added after the deployment stage.

---

## 📌 Overview

FoodVision-20 is a practical computer vision project built to explore transfer learning, model comparison, fine-tuning, and error analysis.

The project started by evaluating pretrained CNN architectures and was later extended to a Vision Transformer.

The final model achieved **90.53% test accuracy** on a 20-class food classification task.

---

## 🧠 Model

The final model is based on:

**Vision Transformer (ViT-B/16)** pretrained on ImageNet.

### Fine-tuning strategy

Instead of training the entire model from scratch, the pretrained model was fine-tuned by:

- Keeping most of the pretrained backbone frozen
- Unfreezing the **last Transformer encoder block**
- Training the classification head
- Using ImageNet pretrained weights

### Training configuration

| Setting | Value |
|---|---|
| Architecture | ViT-B/16 |
| Pretrained weights | ImageNet |
| Input size | 224 × 224 |
| Batch size | 32 |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Loss | Cross Entropy Loss |
| Fine-tuned layers | Last Transformer block + Head |
| Best epoch | 2 |
| Test accuracy | **90.53%** |

---

## 📊 Model Comparison

Three model configurations were evaluated:

| Model | Test Accuracy |
|---|---:|
| EfficientNet-B3 | 86.00% |
| ViT-B/16 — Frozen | 89.10% |
| ViT-B/16 — Fine-tuned | **90.53%** |

Fine-tuning the final Transformer block improved the ViT model over the frozen-backbone version.

---

## 📈 Training & Overfitting

The best test performance was achieved at **Epoch 2**.

After that point, training accuracy continued increasing while test performance gradually decreased.

This behavior indicates **overfitting** and is why the best checkpoint from Epoch 2 was selected rather than the final training epoch.

This project therefore uses **best-checkpoint selection** instead of simply using the model from the last epoch.

---

## 🔍 Evaluation

The model was evaluated using several analysis methods:

- Confusion Matrix
- Per-class accuracy
- Classification report
- Top confusion pairs
- Misclassified image visualization
- Training and test loss curves
- Training and test accuracy curves

These evaluations help identify not only the overall accuracy but also which food categories are difficult for the model to distinguish.

---

## 🍕 Food Categories

The model classifies images into the following 20 categories:

1. Apple Pie
2. Caesar Salad
3. Cheesecake
4. Chicken Wings
5. Chocolate Cake
6. Donuts
7. French Fries
8. Fried Rice
9. Greek Salad
10. Hamburger
11. Hot Dog
12. Ice Cream
13. Lasagna
14. Macarons
15. Pancakes
16. Pizza
17. Ramen
18. Spaghetti Carbonara
19. Sushi
20. Waffles

---

## 🛠️ Tech Stack

- Python
- PyTorch
- Torchvision
- Scikit-learn
- NumPy
- Matplotlib
- Jupyter Notebook

---

## 📁 Project Structure

```text
20-food-classifier/
│
├── FoodVision-20.ipynb
├── data_setup1.py
├── engine.py
├── requirements.txt
├── .gitignore
└── README.md