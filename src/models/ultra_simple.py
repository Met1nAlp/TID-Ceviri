"""
Ultra Simple Model for debugging training
Just a simple MLP to verify the pipeline works
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.training.config import (
    SEQUENCE_LENGTH, LANDMARK_FEATURES, NUM_CLASSES
)


class UltraSimpleModel(nn.Module):
    """
    Ultra simple model - just MLP for debugging
    If this doesn't learn, the problem is data/labels
    If this learns, the problem is model architecture
    """
    
    def __init__(
        self,
        input_size: int = LANDMARK_FEATURES,
        seq_length: int = SEQUENCE_LENGTH,
        hidden_size: int = 512,
        num_classes: int = NUM_CLASSES
    ):
        super().__init__()
        
        self.flatten = nn.Flatten()
        
        # Simple MLP with residual-like connections
        self.net = nn.Sequential(
            nn.Linear(seq_length * input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, frames=None):
        """
        Args:
            x: (batch_size, seq_len, input_size)
            frames: ignored, for API compatibility
        """
        x = self.flatten(x)  # (B, seq_len * input_size)
        return self.net(x)


class SimpleLSTM(nn.Module):
    """
    LSTM model with strong regularization to reduce overfitting
    - 2 layer bidirectional LSTM
    - LayerNorm + high dropout
    - Attention pooling (better than just last hidden state)
    """
    
    def __init__(
        self,
        input_size: int = LANDMARK_FEATURES,
        hidden_size: int = 256,
        num_classes: int = NUM_CLASSES
    ):
        super().__init__()
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_size)
        
        # 2-layer bidirectional LSTM with dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.4,
            bidirectional=True
        )
        
        # Layer norm after LSTM
        self.lstm_norm = nn.LayerNorm(hidden_size * 2)
        
        # Attention pooling (weighted average over time)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Classifier with strong dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x, frames=None):
        # Normalize input
        x = self.input_norm(x)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (B, T, H*2)
        lstm_out = self.lstm_norm(lstm_out)
        
        # Attention pooling over all timesteps
        attn_weights = self.attention(lstm_out)  # (B, T, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (B, H*2)
        
        return self.classifier(context)


def get_ultra_simple_model(model_type="mlp", **kwargs):
    """Get ultra simple model for debugging"""
    if model_type == "mlp":
        return UltraSimpleModel(**kwargs)
    else:
        return SimpleLSTM(**kwargs)


if __name__ == "__main__":
    # Test
    model = UltraSimpleModel()
    print(f"MLP Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    x = torch.randn(4, SEQUENCE_LENGTH, LANDMARK_FEATURES)
    out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    
    model2 = SimpleLSTM()
    print(f"LSTM Parameters: {sum(p.numel() for p in model2.parameters()):,}")
    out2 = model2(x)
    print(f"Input: {x.shape} -> Output: {out2.shape}")
