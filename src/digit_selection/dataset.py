"""Dataset utilities for the digit selection model."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.digit_selection.config import (
    BATCH_SIZE,
    CLASS_NAMES,
    LANDMARK_DIR,
    METADATA_PATH,
    NUM_WORKERS,
    PIN_MEMORY,
)


class DigitSelectionDataset(Dataset):
    """Loads pre-extracted hand landmark vectors for digit selection."""

    def __init__(self, split: str = "train", augment: bool = True):
        if not METADATA_PATH.exists():
            raise FileNotFoundError(
                f"{METADATA_PATH} not found. Run digit preprocessing first."
            )

        metadata = pd.read_csv(METADATA_PATH)
        self.metadata = metadata[metadata["split"] == split].reset_index(drop=True)
        self.split = split
        self.augment = augment and split == "train"

        if len(self.metadata) == 0:
            raise ValueError(f"No samples found for split='{split}'.")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        landmark_path = LANDMARK_DIR / f"{row['sample_id']}.npy"
        features = np.load(landmark_path).astype(np.float32)

        if self.augment:
            features = self._augment(features)

        mean = features.mean()
        std = features.std() + 1e-8
        features = (features - mean) / std

        label = int(row["label"])
        return torch.from_numpy(features), torch.tensor(label, dtype=torch.long)

    def _augment(self, features: np.ndarray) -> np.ndarray:
        augmented = features.copy()

        if np.random.random() < 0.7:
            noise = np.random.normal(0.0, 0.01, augmented.shape).astype(np.float32)
            augmented += noise

        if np.random.random() < 0.5:
            scale = np.random.uniform(0.95, 1.05)
            augmented *= np.float32(scale)

        if np.random.random() < 0.3:
            dropout_mask = np.random.random(augmented.shape) > 0.05
            augmented *= dropout_mask.astype(np.float32)

        return augmented


def get_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
):
    train_dataset = DigitSelectionDataset(split="train", augment=True)
    val_dataset = DigitSelectionDataset(split="val", augment=False)
    test_dataset = DigitSelectionDataset(split="test", augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def get_class_counts(split: str = "train"):
    metadata = pd.read_csv(METADATA_PATH)
    filtered = metadata[metadata["split"] == split]
    counts = filtered["label"].value_counts().to_dict()
    return [counts.get(idx, 0) for idx in range(len(CLASS_NAMES))]
