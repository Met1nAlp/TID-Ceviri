"""Train the digit selection model on GPU."""

import argparse
import json
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.digit_selection.config import (
    BEST_MODEL_PATH,
    BATCH_SIZE,
    CLASS_NAMES,
    EARLY_STOPPING_PATIENCE,
    LABELS_PATH,
    LATEST_MODEL_PATH,
    LEARNING_RATE,
    LOG_DIR,
    MIN_LR,
    NUM_EPOCHS,
    WEIGHT_DECAY,
)
from src.digit_selection.dataset import get_class_counts, get_dataloaders
from src.digit_selection.model import DigitSelectionMLP


class DigitSelectionTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        device,
        experiment_name=None,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.experiment_name = experiment_name or datetime.now().strftime(
            "digit_selection_%Y%m%d_%H%M%S"
        )
        self.writer = SummaryWriter(LOG_DIR / self.experiment_name)
        self.amp_enabled = self.device.type == "cuda"
        self.scaler = GradScaler(device="cuda", enabled=self.amp_enabled)

        class_counts = torch.tensor(get_class_counts("train"), dtype=torch.float32)
        class_weights = class_counts.sum() / torch.clamp(class_counts, min=1.0)
        class_weights = class_weights / class_weights.mean()
        self.class_weights = class_weights.to(self.device)

        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=3,
            min_lr=MIN_LR,
        )

        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "test_acc": [],
            "lr": [],
        }

    def _run_epoch(self, loader, train: bool):
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0

        desc = "Train" if train else "Eval"
        pbar = tqdm(loader, desc=desc, leave=False)

        for features, labels in pbar:
            batch_count += 1
            features = features.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if train:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(train):
                with autocast(device_type=self.device.type, enabled=self.amp_enabled):
                    logits = self.model(features)
                    loss = self.criterion(logits, labels)

            if train:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            pbar.set_postfix(
                {
                    "loss": f"{total_loss / batch_count:.4f}",
                    "acc": f"{100.0 * correct / max(1, total):.2f}%",
                }
            )

        avg_loss = total_loss / max(1, len(loader))
        acc = 100.0 * correct / max(1, total)
        return avg_loss, acc

    @torch.no_grad()
    def evaluate_test(self):
        _, test_acc = self._run_epoch(self.test_loader, train=False)
        return test_acc

    def save_checkpoint(self, epoch, is_best):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_acc": self.best_val_acc,
            "class_names": CLASS_NAMES,
            "history": self.history,
        }
        torch.save(checkpoint, LATEST_MODEL_PATH)
        if is_best:
            torch.save(checkpoint, BEST_MODEL_PATH)

    def train(self, epochs):
        print("=" * 60)
        print("Digit Selection Training")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"AMP: {'enabled' if self.amp_enabled else 'disabled'}")
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Classes: {CLASS_NAMES}")
        print(f"Class weights: {self.class_weights.tolist()}")
        print("=" * 60)

        for epoch in range(epochs):
            start_time = time.time()
            train_loss, train_acc = self._run_epoch(self.train_loader, train=True)
            val_loss, val_acc = self._run_epoch(self.val_loader, train=False)
            test_acc = self.evaluate_test()
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["test_acc"].append(test_acc)
            self.history["lr"].append(current_lr)

            self.writer.add_scalars(
                "Loss",
                {"train": train_loss, "val": val_loss},
                epoch,
            )
            self.writer.add_scalars(
                "Accuracy",
                {"train": train_acc, "val": val_acc, "test": test_acc},
                epoch,
            )
            self.writer.add_scalar("LearningRate", current_lr, epoch)

            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch + 1}/{epochs} ({elapsed:.1f}s) | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% | "
                f"test_acc={test_acc:.2f}% | lr={current_lr:.6f}"
            )

            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            self.save_checkpoint(epoch, is_best)

            if self.epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        self.writer.close()
        print(f"\nBest validation accuracy: {self.best_val_acc:.2f}%")
        print(f"Best checkpoint: {BEST_MODEL_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Train the digit selection model.")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"{LABELS_PATH} not found. Run digit preprocessing before training."
        )

    labels_payload = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    print(f"Loaded labels: {labels_payload['class_names']}")

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=args.batch_size)
    model = DigitSelectionMLP()
    trainer = DigitSelectionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=args.device,
        experiment_name=args.name,
    )
    trainer.train(args.epochs)


if __name__ == "__main__":
    main()
