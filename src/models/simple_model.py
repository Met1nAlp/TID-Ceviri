"""
Simple LSTM Model for debugging
Much simpler architecture to verify training works
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.training.config import (
    SEQUENCE_LENGTH, LANDMARK_FEATURES, NUM_CLASSES
)


class SimpleSignModel(nn.Module):
    """
    Simple LSTM model for sign language recognition
    Simpler than the hybrid model, easier to train
    """
    
    def __init__(
        self,
        input_size: int = LANDMARK_FEATURES,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Batch normalization for input
        self.input_bn = nn.BatchNorm1d(input_size)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x, frames=None):
        """
        Args:
            x: (batch_size, seq_len, input_size)
            frames: ignored, for API compatibility
        """
        batch_size = x.size(0)
        
        # Apply batch norm (need to permute for BatchNorm1d)
        x = x.permute(0, 2, 1)  # (B, C, T)
        x = self.input_bn(x)
        x = x.permute(0, 2, 1)  # (B, T, C)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (B, T, H*2)
        
        # Use last timestep
        last_output = lstm_out[:, -1, :]  # (B, H*2)
        
        # Classify
        logits = self.classifier(last_output)
        
        return logits


def get_simple_model(**kwargs) -> nn.Module:
    """Get simple model for debugging"""
    return SimpleSignModel(**kwargs)


if __name__ == "__main__":
    model = SimpleSignModel()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    x = torch.randn(4, SEQUENCE_LENGTH, LANDMARK_FEATURES)
    out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
