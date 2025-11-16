import os
import random
import numpy as np
from tqdm import tqdm
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from sklearn.metrics import precision_score, recall_score, f1_score

from config import *

# =======================
#  Config
# =======================
DATA_DIR = "raw-img"   # folder with 10 subfolders (cats, dogs...)
BATCH_SIZE = 32
LR = 1e-3
NUM_EPOCHS = 20
VAL_SPLIT = 0.2
RANDOM_SEED = 42
IMG_SIZE = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# =======================
#  Reproducibility
# =======================
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# =======================
#  Data Transforms
# =======================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# =======================
#  Dataset & loaders
# =======================
full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transform)

num_classes = len(full_dataset.classes)
print("Classes:", full_dataset.classes)
print("Num images:", len(full_dataset))

n_total = len(full_dataset)
n_val = int(VAL_SPLIT * n_total)
n_train = n_total - n_val

train_dataset, val_dataset = random_split(
    full_dataset,
    [n_train, n_val],
    generator=torch.Generator().manual_seed(RANDOM_SEED),
)

val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=0)


# =======================
#  Model
# =======================
class AnimalCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * (IMG_SIZE // 16) * (IMG_SIZE // 16), 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = AnimalCNN(num_classes=num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# =======================
#  Training one epoch
# =======================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0
    correct = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()

    return {
        "loss": running_loss / len(loader.dataset),
        "acc": correct / len(loader.dataset)
    }


# =======================
#  Evaluation (with metrics)
# =======================
def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0
    correct = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Val", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    avg_acc = correct / len(loader.dataset)

    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {
        "loss": avg_loss,
        "acc": avg_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "labels": all_labels,  # useful later
        "preds": all_preds
    }


# =======================
#  Main training loop
# =======================
if __name__ == "__main__":

    best_val_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)

        # ===========================
        #   LOG TO W&B (using run from config.py)
        # ===========================
        run.log({
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],

            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],

            "precision": val_metrics["precision"],
            "recall": val_metrics["recall"],
            "f1": val_metrics["f1"],
        })

        print(
            f"Train loss: {train_metrics['loss']:.4f} | Train acc: {train_metrics['acc']:.4f}\n"
            f"Val loss: {val_metrics['loss']:.4f} | Val acc: {val_metrics['acc']:.4f} | "
            f"Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f}"
        )

        # ===========================
        #   BEST MODEL SAVE + LOGGING
        # ===========================
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]

            torch.save(model.state_dict(), "checkpoints/best_animals10_cnn.pt")

            run.log({"best_val_acc": best_val_acc})

            print(f"✅ Saved new best model! ({best_val_acc:.4f})")

    run.finish()
    print("\nTraining complete.")
