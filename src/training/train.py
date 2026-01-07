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
from src.models.simple_model import get_simple_model


class Trainer:
    """Training class with mixed precision and early stopping"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: str = DEVICE,
        use_amp: bool = USE_AMP,
        experiment_name: str = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_amp = use_amp
        
        # Experiment name
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        
        # Loss function (no label smoothing for better initial learning)
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
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
            
            pbar.set_postfix({
                'loss': f"{total_loss/(batch_idx+1):.4f}",
                'acc': f"{100.*correct/total:.2f}%"
            })
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        top3_accuracy = 100. * top3_correct / total
        
        return avg_loss, accuracy, top3_accuracy
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history
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
            val_loss, val_acc, top3_acc = self.validate(epoch)
            
            # Update learning rate (ReduceLROnPlateau needs val_loss)
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
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
            self.writer.add_scalar('Learning Rate', current_lr, epoch)
            
            # Epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, Top-3: {top3_acc:.2f}%")
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
                       choices=["landmark_only", "hybrid", "simple"],
                       help="Model type to train")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                       help="Batch size")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--name", type=str, default=None,
                       help="Experiment name")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("TID Recognition System - Training")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60 + "\n")
    
    # Get dataloaders
    mode = "landmarks" if args.model in ["landmark_only", "simple"] else "hybrid"
    train_loader, val_loader, _ = get_dataloaders(
        mode=mode,
        batch_size=args.batch_size
    )
    
    # Get model
    if args.model == "simple":
        model = get_simple_model()
    else:
        model = get_model(args.model)
    
    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        experiment_name=args.name
    )
    
    trainer.train(num_epochs=args.epochs, resume_from=args.resume)


if __name__ == "__main__":
    main()
