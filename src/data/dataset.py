"""
PyTorch Dataset for TID Recognition
Supports both landmarks-only and hybrid (landmarks + frames) modes
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from typing import Optional, Tuple, List
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.training.config import (
    PROCESSED_DIR, DATA_DIR, SEQUENCE_LENGTH, IMG_SIZE, 
    NUM_CLASSES, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
    AUGMENTATION
)


class TIDLandmarkDataset(Dataset):
    """
    Dataset for landmark-based sign language recognition
    """
    
    def __init__(
        self, 
        split: str = "train",
        augment: bool = True,
        landmarks_dir: Optional[Path] = None
    ):
        """
        Args:
            split: 'train', 'val', or 'test'
            augment: Whether to apply data augmentation
            landmarks_dir: Directory containing processed landmarks
        """
        self.split = split
        self.augment = augment and split == "train"
        
        # Set paths
        self.landmarks_dir = Path(landmarks_dir) if landmarks_dir else (PROCESSED_DIR / split)
        
        # Load metadata and filter to only existing files
        metadata = pd.read_csv(self.landmarks_dir / "metadata.csv")
        
        # Filter to only include files that exist
        valid_indices = []
        for idx, row in metadata.iterrows():
            landmarks_path = self.landmarks_dir / f"{row['video']}.npy"
            if landmarks_path.exists():
                valid_indices.append(idx)
        
        self.metadata = metadata.iloc[valid_indices].reset_index(drop=True)
        
        print(f"Loaded {len(self.metadata)} samples for {split} (filtered from {len(metadata)})")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        video_name = row['video']
        label = int(row['label'])
        
        # Load landmarks
        landmarks_path = self.landmarks_dir / f"{video_name}.npy"
        landmarks = np.load(landmarks_path).astype(np.float32)
        
        # Apply augmentation
        if self.augment:
            landmarks = self._augment(landmarks)
        
        # Convert to tensor
        landmarks = torch.from_numpy(landmarks)
        label = torch.tensor(label, dtype=torch.long)
        
        return landmarks, label
    
    def _augment(self, landmarks: np.ndarray) -> np.ndarray:
        """Apply data augmentation to landmarks"""
        
        # Time stretching (simulate different speeds)
        if np.random.random() < 0.3:
            landmarks = self._time_stretch(landmarks)
        
        # Add Gaussian noise to landmarks
        if np.random.random() < 0.5:
            noise_std = AUGMENTATION["landmark_noise"]["std"]
            noise = np.random.normal(0, noise_std, landmarks.shape)
            landmarks = landmarks + noise.astype(np.float32)
        
        # Random landmark dropout
        if np.random.random() < 0.3:
            dropout_rate = AUGMENTATION["landmark_dropout"]["rate"]
            mask = np.random.random(landmarks.shape) > dropout_rate
            landmarks = landmarks * mask.astype(np.float32)
        
        # Time masking (mask consecutive frames)
        if np.random.random() < 0.3:
            landmarks = self._time_mask(landmarks)
        
        return landmarks
    
    def _time_stretch(self, landmarks: np.ndarray) -> np.ndarray:
        """Apply time stretching augmentation"""
        min_rate = AUGMENTATION["time_stretch"]["min_rate"]
        max_rate = AUGMENTATION["time_stretch"]["max_rate"]
        rate = np.random.uniform(min_rate, max_rate)
        
        num_frames = landmarks.shape[0]
        target_frames = int(num_frames * rate)
        
        if target_frames == num_frames:
            return landmarks
        
        # Interpolate to new length
        indices = np.linspace(0, num_frames - 1, target_frames)
        stretched = np.zeros((target_frames, landmarks.shape[1]), dtype=np.float32)
        
        for i, idx in enumerate(indices):
            lower = int(np.floor(idx))
            upper = min(int(np.ceil(idx)), num_frames - 1)
            weight = idx - lower
            
            if lower == upper:
                stretched[i] = landmarks[lower]
            else:
                stretched[i] = (1 - weight) * landmarks[lower] + weight * landmarks[upper]
        
        # Resize back to original or pad/truncate
        if target_frames < SEQUENCE_LENGTH:
            # Pad with last frame
            pad_amount = SEQUENCE_LENGTH - target_frames
            padding = np.tile(stretched[-1:], (pad_amount, 1))
            stretched = np.vstack([stretched, padding])
        elif target_frames > SEQUENCE_LENGTH:
            # Uniform sampling
            indices = np.linspace(0, target_frames - 1, SEQUENCE_LENGTH).astype(int)
            stretched = stretched[indices]
        
        return stretched
    
    def _time_mask(self, landmarks: np.ndarray) -> np.ndarray:
        """Mask consecutive frames with zeros"""
        max_frames = AUGMENTATION["time_mask"]["max_frames"]
        mask_length = np.random.randint(1, max_frames + 1)
        start_idx = np.random.randint(0, landmarks.shape[0] - mask_length)
        
        landmarks[start_idx:start_idx + mask_length] = 0
        
        return landmarks


class TIDHybridDataset(Dataset):
    """
    Hybrid dataset that provides both landmarks and video frames
    For maximum accuracy with CNN + GRU fusion
    """
    
    def __init__(
        self,
        split: str = "train",
        augment: bool = True,
        landmarks_dir: Optional[Path] = None,
        videos_dir: Optional[Path] = None,
        sample_frames: int = 16  # Subsample frames for CNN
    ):
        self.split = split
        self.augment = augment and split == "train"
        self.sample_frames = sample_frames
        
        # Set paths
        self.landmarks_dir = Path(landmarks_dir) if landmarks_dir else (PROCESSED_DIR / split)
        self.videos_dir = Path(videos_dir) if videos_dir else (DATA_DIR / split)
        
        # Load metadata and filter to only existing files
        metadata = pd.read_csv(self.landmarks_dir / "metadata.csv")
        
        # Filter to only include files that exist
        valid_indices = []
        for idx, row in metadata.iterrows():
            landmarks_path = self.landmarks_dir / f"{row['video']}.npy"
            if landmarks_path.exists():
                valid_indices.append(idx)
        
        self.metadata = metadata.iloc[valid_indices].reset_index(drop=True)
        
        print(f"Loaded {len(self.metadata)} samples for {split} hybrid (filtered from {len(metadata)})")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        video_name = row['video']
        label = int(row['label'])
        
        # Load landmarks
        landmarks_path = self.landmarks_dir / f"{video_name}.npy"
        landmarks = np.load(landmarks_path).astype(np.float32)
        
        # Load and sample video frames
        video_path = self.videos_dir / f"{video_name}.mp4"
        frames = self._load_video_frames(str(video_path))
        
        # Apply augmentation
        if self.augment:
            landmarks = self._augment_landmarks(landmarks)
            # Note: frames augmentation can be added here too
        
        # Convert to tensors
        landmarks = torch.from_numpy(landmarks)
        frames = torch.from_numpy(frames)  # (T, H, W, C) -> need to permute later
        label = torch.tensor(label, dtype=torch.long)
        
        # Permute frames to (T, C, H, W) for CNN
        frames = frames.permute(0, 3, 1, 2)
        
        return landmarks, frames, label
    
    def _load_video_frames(self, video_path: str) -> np.ndarray:
        """Load and sample frames from video"""
        cap = cv2.VideoCapture(video_path)
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Resize frame
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            # Return black frames if video couldn't be loaded
            return np.zeros((self.sample_frames, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        
        # Uniform sampling
        frames = np.array(frames)
        indices = np.linspace(0, len(frames) - 1, self.sample_frames).astype(int)
        sampled = frames[indices]
        
        # Normalize to [0, 1]
        sampled = sampled.astype(np.float32) / 255.0
        
        return sampled
    
    def _augment_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Same augmentation as TIDLandmarkDataset"""
        # Add noise
        if np.random.random() < 0.5:
            noise = np.random.normal(0, 0.01, landmarks.shape)
            landmarks = landmarks + noise.astype(np.float32)
        
        # Dropout
        if np.random.random() < 0.3:
            mask = np.random.random(landmarks.shape) > 0.05
            landmarks = landmarks * mask.astype(np.float32)
        
        return landmarks


def get_dataloaders(
    mode: str = "landmarks",  # "landmarks" or "hybrid"
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get train, val, and test dataloaders
    
    Args:
        mode: "landmarks" for GRU-only, "hybrid" for CNN+GRU
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        pin_memory: Pin memory for GPU training
    
    Returns:
        train_loader, val_loader, test_loader
    """
    if mode == "landmarks":
        DatasetClass = TIDLandmarkDataset
    else:
        DatasetClass = TIDHybridDataset
    
    train_dataset = DatasetClass(split="train", augment=True)
    val_dataset = DatasetClass(split="val", augment=False)
    test_dataset = DatasetClass(split="test", augment=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset
    print("Testing TIDLandmarkDataset...")
    dataset = TIDLandmarkDataset(split="train")
    
    if len(dataset) > 0:
        landmarks, label = dataset[0]
        print(f"Landmarks shape: {landmarks.shape}")
        print(f"Label: {label}")
    else:
        print("Dataset is empty. Run preprocessing first.")
