import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import random
from config import *
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# MODEL DEFINITION ABOVE main
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

    def forward(self, x):
        return self.block(x)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.c1 = ConvBlock(3, 32)
        self.c2 = ConvBlock(32, 64)
        self.c3 = ConvBlock(64, 128)
        self.c4 = ConvBlock(128, 256)

        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x


# ============================================================
# SAFE ENTRY POINT
# ============================================================

if __name__ == "__main__":
    set_seed(SEED)

    wandb.init(
        project="10-Animals",
        name="baseline_cnn",
        config={
            "seed": SEED,
            "lr": 1e-4,
            "batch_size": 32,
            "epochs": 25
        }
    )

    # ============================================================
    # 1) AUGMENTATIONS
    # ============================================================

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.2),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor()
    ])

    test_tf = transforms.Compose([transforms.ToTensor()])


    # ============================================================
    # 2) LOAD FULL DATASET
    # ============================================================

    root_path = "./filtered-img"
    full_ds = datasets.ImageFolder(root_path)

    num_classes = len(full_ds.classes)
    dataset_size = len(full_ds)

    all_labels = [label for _, label in full_ds.samples]
    class_counts = np.bincount(all_labels)

    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)


    # SPLIT
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_ds.dataset.transform = train_tf
    val_ds.dataset.transform = test_tf

    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True,
        num_workers=4, persistent_workers=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=32, shuffle=False,
        num_workers=4, persistent_workers=True
    )


    # ============================================================
    # 3) MODEL, OPTIMIZER, LOSS
    # ============================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3
    )


    # ============================================================
    # 4) TRAIN LOOP
    # ============================================================

    def evaluate(model, loader):
        model.eval()
        all_preds, all_labels = [], []
        loss_sum = 0

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                loss_sum += loss.item()

                preds = out.argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
        precision = precision_score(all_labels, all_preds, average="weighted")
        recall = recall_score(all_labels, all_preds, average="weighted")
        f1 = f1_score(all_labels, all_preds, average="weighted")

        return loss_sum / len(loader), accuracy, precision, recall, f1


    best_f1 = 0
    patience = 6
    patience_counter = 0

    for epoch in range(25):
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        val_loss, val_acc, precision, recall, f1 = evaluate(model, val_loader)
        scheduler.step(val_loss)

        print(f"[{epoch+1}/25] TrainLoss={train_loss:.4f} TrainAcc={train_acc:.4f} "
              f"ValLoss={val_loss:.4f} ValAcc={val_acc:.4f} F1={f1:.4f}")

        run.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

        # Early stopping
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            torch.save(model.state_dict(), "best_cnn_wandb.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping.")
                break

    print(f"\nBest F1: {best_f1:.4f}")
