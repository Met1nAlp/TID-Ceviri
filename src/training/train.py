"""
Training Script for TID Recognition System
Supports mixed precision training for RTX 3070
Optimized for high accuracy
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.training.config import (
    DEVICE, NUM_CLASSES, NUM_EPOCHS, BATCH_SIZE, 
    LEARNING_RATE, WEIGHT_DECAY, USE_AMP,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA,
    MODEL_DIR, LOG_DIR, SAVE_BEST_ONLY, CHECKPOINT_EVERY,
    WARMUP_EPOCHS, MIN_LR
)
from src.data.dataset import get_dataloaders
from src.models.hybrid_model import get_model
from src.models.ultra_simple import get_ultra_simple_model
from src.training.focus import build_weighted_training_config, load_focus_bundle

try:
    from src.models.simple_model import get_simple_model
except ImportError:
    get_simple_model = None


class Trainer:
    """Training class with mixed precision and early stopping"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: str = DEVICE,
        use_amp: bool = USE_AMP,
        experiment_name: str = None,
        criterion: nn.Module | None = None,
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        focus_class_ids: list[int] | None = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_amp = use_amp
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.focus_class_ids = sorted(set(focus_class_ids or []))
        
        # Experiment name
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        
        # Loss function (no label smoothing for better initial learning)
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler - reduce on plateau (based on val loss)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=MIN_LR
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler(device='cuda') if use_amp else None
        
        # Tensorboard
        self.writer = SummaryWriter(LOG_DIR / experiment_name)
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_focus_acc': [],
            'lr': []
        }
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Handle both landmark-only and hybrid data
            if len(batch) == 2:
                landmarks, labels = batch
                frames = None
            else:
                landmarks, frames, labels = batch
                frames = frames.to(self.device)
            
            landmarks = landmarks.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast(device_type='cuda'):
                    outputs = self.model(landmarks, frames)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(landmarks, frames)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss/(batch_idx+1):.4f}",
                'acc': f"{100.*correct/total:.2f}%"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self, epoch: int):
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # For top-3 accuracy
        top3_correct = 0
        focus_correct = 0
        focus_total = 0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")
        
        for batch_idx, batch in enumerate(pbar):
            if len(batch) == 2:
                landmarks, labels = batch
                frames = None
            else:
                landmarks, frames, labels = batch
                frames = frames.to(self.device)
            
            landmarks = landmarks.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            if self.use_amp:
                with autocast(device_type='cuda'):
                    outputs = self.model(landmarks, frames)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(landmarks, frames)
                loss = self.criterion(outputs, labels)
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Top-3 accuracy
            _, top3_pred = outputs.topk(3, dim=1)
            for i in range(labels.size(0)):
                if labels[i] in top3_pred[i]:
                    top3_correct += 1

            if self.focus_class_ids:
                focus_mask = torch.zeros_like(labels, dtype=torch.bool)
                for class_id in self.focus_class_ids:
                    focus_mask |= labels == class_id
                if focus_mask.any():
                    focus_total += int(focus_mask.sum().item())
                    focus_correct += int(predicted[focus_mask].eq(labels[focus_mask]).sum().item())
            
            pbar.set_postfix({
                'loss': f"{total_loss/(batch_idx+1):.4f}",
                'acc': f"{100.*correct/total:.2f}%"
            })
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        top3_accuracy = 100. * top3_correct / total
        
        focus_accuracy = 100. * focus_correct / focus_total if focus_total else None
        return avg_loss, accuracy, top3_accuracy, focus_accuracy
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'focus_class_ids': self.focus_class_ids,
        }
        
        # Save latest
        latest_path = MODEL_DIR / "latest_checkpoint.pth"
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = MODEL_DIR / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"✓ New best model saved! Val Acc: {self.best_val_acc:.2f}%")
        
        # Save periodic checkpoint
        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            epoch_path = MODEL_DIR / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']
        
        # Override LR with current config value (for fine-tuning)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate
        print(f"Overriding LR to {self.learning_rate} from current run config")
        
        return checkpoint['epoch']
    
    def train(self, num_epochs: int = NUM_EPOCHS, resume_from: str = None):
        """Main training loop"""
        start_epoch = 0
        total_epochs = num_epochs
        
        # Resume from checkpoint if specified
        if resume_from and Path(resume_from).exists():
            start_epoch = self.load_checkpoint(resume_from) + 1
            total_epochs = start_epoch + num_epochs  # Add epochs on top of checkpoint
            print(f"Resumed from epoch {start_epoch}, will train to epoch {total_epochs}")
        
        print("=" * 60)
        print(f"Training on {self.device} with {'mixed precision' if self.use_amp else 'full precision'}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("=" * 60)
        
        for epoch in range(start_epoch, total_epochs):
            epoch_start = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_acc, top3_acc, focus_acc = self.validate(epoch)
            
            # Update learning rate (ReduceLROnPlateau needs val_loss)
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_focus_acc'].append(focus_acc)
            self.history['lr'].append(current_lr)
            
            # Tensorboard logging
            self.writer.add_scalars('Loss', {
                'train': train_loss,
                'val': val_loss
            }, epoch)
            self.writer.add_scalars('Accuracy', {
                'train': train_acc,
                'val': val_acc,
                'val_top3': top3_acc
            }, epoch)
            if focus_acc is not None:
                self.writer.add_scalar('Accuracy/val_focus', focus_acc, epoch)
            self.writer.add_scalar('Learning Rate', current_lr, epoch)
            
            # Epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, Top-3: {top3_acc:.2f}%")
            if focus_acc is not None:
                print(f"  Focus - Val Acc: {focus_acc:.2f}% on {len(self.focus_class_ids)} class(es)")
            print(f"  LR: {current_lr:.6f}")
            
            # Check for improvement
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            if not SAVE_BEST_ONLY or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if self.epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        self.writer.close()
        print("\n" + "=" * 60)
        print(f"Training complete! Best validation accuracy: {self.best_val_acc:.2f}%")
        print("=" * 60)
        
        return self.history


def main():
    parser = argparse.ArgumentParser(description="Train TID Recognition Model")
    parser.add_argument("--model", type=str, default="landmark_only",
                       choices=["landmark_only", "hybrid", "simple", "mlp", "lstm"],
                       help="Model type to train")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                       help="Batch size")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--name", type=str, default=None,
                       help="Experiment name")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                       help="Learning rate (use a lower one for fine-tuning)")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                       help="Weight decay")
    parser.add_argument("--focus-analysis-report", type=str, default=None,
                       help="Path to analyze_benchmark_report JSON output")
    parser.add_argument("--focus-actions", type=str, nargs="+", default=["targeted_finetune"],
                       help="Recommended actions to include from the analysis report")
    parser.add_argument("--focus-top-n", type=int, default=12,
                       help="How many high-priority classes to focus")
    parser.add_argument("--include-confusion-partners", action="store_true",
                       help="Also boost the dominant confusion partners for each focus class")
    parser.add_argument("--focus-sample-boost", type=float, default=3.0,
                       help="Weighted sampler multiplier for focus classes")
    parser.add_argument("--partner-sample-boost", type=float, default=1.75,
                       help="Weighted sampler multiplier for confusion partners")
    parser.add_argument("--focus-loss-boost", type=float, default=2.0,
                       help="Cross-entropy class-weight multiplier for focus classes")
    parser.add_argument("--partner-loss-boost", type=float, default=1.35,
                       help="Cross-entropy class-weight multiplier for confusion partners")
    parser.add_argument("--focus-only-train", action="store_true",
                       help="Restrict the training split to focus classes and selected partners")
    parser.add_argument("--auto-resume-best", action="store_true",
                       help="If no --resume is given, continue from models/best_model.pth when it exists")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("TID Recognition System - Training")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 60 + "\n")

    focus_bundle = None
    train_allowed_class_ids = None
    train_sample_weight_map = None
    criterion = None

    if args.focus_analysis_report:
        focus_bundle = load_focus_bundle(
            analysis_report_path=args.focus_analysis_report,
            actions=args.focus_actions,
            top_n=args.focus_top_n,
            include_confusion_partners=args.include_confusion_partners,
        )
        weighted_config = build_weighted_training_config(
            num_classes=NUM_CLASSES,
            focus_class_ids=focus_bundle["focus_class_ids"],
            partner_class_ids=focus_bundle["partner_class_ids"],
            focus_sample_boost=args.focus_sample_boost,
            partner_sample_boost=args.partner_sample_boost,
            focus_loss_boost=args.focus_loss_boost,
            partner_loss_boost=args.partner_loss_boost,
            device=DEVICE,
        )
        train_sample_weight_map = weighted_config["sample_weight_map"]
        criterion = nn.CrossEntropyLoss(weight=weighted_config["loss_weights"])

        if args.focus_only_train:
            train_allowed_class_ids = set(focus_bundle["focus_class_ids"]) | set(
                focus_bundle["partner_class_ids"]
            )

        print("Focus fine-tune aktif")
        print(f"  Kaynak rapor: {focus_bundle['report_path']}")
        print(f"  Aksiyonlar : {', '.join(focus_bundle['actions'])}")
        print(f"  Hedef sinif: {len(focus_bundle['focus_class_ids'])}")
        print(f"  Partner    : {len(focus_bundle['partner_class_ids'])}")

    resume_path = args.resume
    if resume_path is None and (args.auto_resume_best or args.focus_analysis_report):
        best_model_path = MODEL_DIR / "best_model.pth"
        if best_model_path.exists():
            resume_path = str(best_model_path)
            print(f"Resume otomatik ayarlandi: {resume_path}")

    # Get dataloaders
    mode = "landmarks" if args.model in ["landmark_only", "simple", "mlp", "lstm"] else "hybrid"
    train_loader, val_loader, _ = get_dataloaders(
        mode=mode,
        batch_size=args.batch_size,
        train_allowed_class_ids=train_allowed_class_ids,
        train_sample_weight_map=train_sample_weight_map,
    )
    
    # Get model
    if args.model == "simple":
        if get_simple_model is None:
            raise ImportError("src.models.simple_model bulunamadi. simple modeli kullanmayin.")
        model = get_simple_model()
    elif args.model == "mlp":
        model = get_ultra_simple_model("mlp")
    elif args.model == "lstm":
        model = get_ultra_simple_model("lstm")
    else:
        model = get_model(args.model)
    
    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        experiment_name=args.name,
        criterion=criterion,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        focus_class_ids=(focus_bundle["focus_class_ids"] if focus_bundle else None),
    )
    
    trainer.train(num_epochs=args.epochs, resume_from=resume_path)


if __name__ == "__main__":
    main()
