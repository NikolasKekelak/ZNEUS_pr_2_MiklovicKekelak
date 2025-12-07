import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
import wandb

from models import get_model     # from your models.py
from config import SEED          # from config.py

import random



class Agent:
    def __init__(self, config):
        self.config = config
        self.seed = config.seed if hasattr(config, "seed") else SEED
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Confirm device selection
        if torch.cuda.is_available():
            print(f" Using CUDA/GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("  CUDA not available - using CPU (training will be slower)")
            print("   Install CUDA-enabled PyTorch to use GPU")

        self._set_seed(self.seed)
        self._create_transforms()
        self._load_data()
        self._init_model()
        self._init_training_components()


    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _create_transforms(self):
        self.train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor()
        ])

        self.test_tf = transforms.Compose([transforms.ToTensor()])

    def _load_data(self):
        root = {}
        if self.config.image_size == 64:
            root = "./filtered-img-64"
        elif self.config.image_size == 128:
            root = "./filtered-img-128"
        elif self.config.image_size == 224:
            root = "./filtered-img-224"
        elif self.config.image_size == 256:
            root = "./filtered-img-256"
        else:
            raise ValueError(f"Unsupported image_size: {self.config.image_size}")

        full_ds = datasets.ImageFolder(root)

        all_labels = [lbl for _, lbl in full_ds.samples]
        class_counts = np.bincount(all_labels)

        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * len(class_counts)

        self.class_weights = torch.tensor(class_weights, dtype=torch.float)


        size = len(full_ds)
        train_size = int(0.7 * size)
        val_size = size - train_size

        self.train_ds, self.val_ds = torch.utils.data.random_split(
            full_ds, [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed)
        )

        self.train_ds.dataset.transform = self.train_tf
        self.val_ds.dataset.transform = self.test_tf

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4
        )

        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4
        )

        self.num_classes = len(full_ds.classes)

    def _init_model(self):
       self.model = get_model(self.config.model_name, num_classes=self.num_classes).to(self.device)

    def _init_training_components(self):
        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights.to(self.device)
        )

        lr = self.config.learning_rate if hasattr(self.config, "learning_rate") else self.config.lr

        if hasattr(self.config, "optimizer") and self.config.optimizer == "rms":
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=0.5,
            patience=3
        )

    def evaluate(self):
        self.model.eval()
        preds_all, labels_all = [], []
        running_loss = 0

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)

                out = self.model(x)
                loss = self.criterion(out, y)
                running_loss += loss.item()

                preds = out.argmax(dim=1)

                preds_all.extend(preds.cpu().numpy())
                labels_all.extend(y.cpu().numpy())

        acc = (np.array(preds_all) == np.array(labels_all)).mean()
        precision = precision_score(labels_all, preds_all, average="weighted")
        recall = recall_score(labels_all, preds_all, average="weighted")
        f1 = f1_score(labels_all, preds_all, average="weighted")

        return running_loss / len(self.val_loader), acc, precision, recall, f1

    def train(self, run):
        best_f1 = 0
        best_model_path = None
        
        # Create unique model filename based on run ID to avoid overwriting
        model_filename = f"best_model_{run.id}.pth"
        
        for epoch in range(self.config.epochs):
            self.model.train()
            running_loss = 0
            correct = 0
            total = 0

            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                out = self.model(x)
                loss = self.criterion(out, y)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                preds = out.argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)

            train_loss = running_loss / len(self.train_loader)
            train_acc = correct / total

            val_loss, val_acc, precision, recall, f1 = self.evaluate()

            self.scheduler.step(val_loss)

            run.log({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            })

            print(f"[{epoch+1}/{self.config.epochs}] "
                  f"TrainLoss={train_loss:.4f} ValLoss={val_loss:.4f} "
                  f"ValAcc={val_acc:.4f} F1={f1:.4f}")

            # save best model (unique filename per run)
            if f1 > best_f1:
                best_f1 = f1
                best_model_path = model_filename
                torch.save(self.model.state_dict(), model_filename)
                print(f"Saved best model (F1={best_f1:.4f}) to {model_filename}")

        # Save best model to wandb as artifact
        if best_model_path:
            artifact = wandb.Artifact(f"model-{run.id}", type="model")
            artifact.add_file(best_model_path)
            run.log_artifact(artifact)
            print(f"Uploaded best model to wandb: {best_model_path}")

        return best_f1
