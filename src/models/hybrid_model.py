"""
Hybrid Model for Sign Language Recognition
Combines GRU for temporal landmark sequences with CNN for visual features
Optimized for high accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.training.config import (
    SEQUENCE_LENGTH, LANDMARK_FEATURES, NUM_CLASSES,
    GRU_HIDDEN_SIZE, GRU_NUM_LAYERS, GRU_DROPOUT, GRU_BIDIRECTIONAL,
    CNN_FEATURE_DIM, FUSION_DIM, FUSION_DROPOUT
)


class LandmarkEncoder(nn.Module):
    """
    GRU-based encoder for landmark sequences
    Bidirectional for better temporal understanding
    """
    
    def __init__(
        self,
        input_size: int = LANDMARK_FEATURES,
        hidden_size: int = GRU_HIDDEN_SIZE,
        num_layers: int = GRU_NUM_LAYERS,
        dropout: float = GRU_DROPOUT,
        bidirectional: bool = GRU_BIDIRECTIONAL
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Attention mechanism for weighted temporal pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            output: (batch_size, hidden_size)
        """
        batch_size = x.size(0)
        
        # Project input
        x = self.input_proj(x)  # (B, T, H)
        
        # GRU
        gru_out, _ = self.gru(x)  # (B, T, H*num_directions)
        
        # Project output
        gru_out = self.output_proj(gru_out)  # (B, T, H)
        
        # Attention-weighted pooling
        attn_weights = self.attention(gru_out)  # (B, T, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Weighted sum
        output = torch.sum(gru_out * attn_weights, dim=1)  # (B, H)
        
        return output


class VisualEncoder(nn.Module):
    """
    CNN-based encoder for video frames
    Uses pre-trained EfficientNet for feature extraction
    """
    
    def __init__(
        self,
        output_dim: int = GRU_HIDDEN_SIZE,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        # Load pretrained EfficientNet-B0
        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        
        # Remove classifier
        self.backbone.classifier = nn.Identity()
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Temporal pooling (average across frames)
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # Feature projection
        self.proj = nn.Sequential(
            nn.Linear(CNN_FEATURE_DIM, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(GRU_DROPOUT)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_frames, channels, height, width)
        Returns:
            output: (batch_size, output_dim)
        """
        batch_size, num_frames = x.shape[:2]
        
        # Reshape for batch processing
        x = x.view(batch_size * num_frames, *x.shape[2:])  # (B*T, C, H, W)
        
        # Extract features
        features = self.backbone(x)  # (B*T, CNN_FEATURE_DIM)
        
        # Reshape back
        features = features.view(batch_size, num_frames, -1)  # (B, T, F)
        
        # Temporal pooling
        features = features.permute(0, 2, 1)  # (B, F, T)
        features = self.temporal_pool(features).squeeze(-1)  # (B, F)
        
        # Project
        output = self.proj(features)  # (B, output_dim)
        
        return output


class HybridModel(nn.Module):
    """
    Hybrid model combining landmark and visual features
    Uses late fusion for combining modalities
    """
    
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        landmark_size: int = LANDMARK_FEATURES,
        hidden_size: int = GRU_HIDDEN_SIZE,
        fusion_dim: int = FUSION_DIM,
        dropout: float = FUSION_DROPOUT,
        use_visual: bool = True
    ):
        super().__init__()
        
        self.use_visual = use_visual
        
        # Landmark encoder (always used)
        self.landmark_encoder = LandmarkEncoder(
            input_size=landmark_size,
            hidden_size=hidden_size
        )
        
        # Visual encoder (optional)
        if use_visual:
            self.visual_encoder = VisualEncoder(
                output_dim=hidden_size
            )
            fusion_input_dim = hidden_size * 2
        else:
            fusion_input_dim = hidden_size
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Linear(fusion_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/He initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, landmarks, frames=None):
        """
        Args:
            landmarks: (batch_size, seq_len, landmark_features)
            frames: (batch_size, num_frames, channels, height, width) - optional
        Returns:
            logits: (batch_size, num_classes)
        """
        # Encode landmarks
        landmark_features = self.landmark_encoder(landmarks)  # (B, H)
        
        # Encode visual features if available
        if self.use_visual and frames is not None:
            visual_features = self.visual_encoder(frames)  # (B, H)
            combined = torch.cat([landmark_features, visual_features], dim=-1)
        else:
            combined = landmark_features
        
        # Fusion
        fused = self.fusion(combined)  # (B, fusion_dim)
        
        # Classification
        logits = self.classifier(fused)  # (B, num_classes)
        
        return logits


class LandmarkOnlyModel(nn.Module):
    """
    Simpler model using only landmarks (faster training and inference)
    Still achieves high accuracy with proper training
    """
    
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        landmark_size: int = LANDMARK_FEATURES,
        hidden_size: int = GRU_HIDDEN_SIZE,
        dropout: float = FUSION_DROPOUT
    ):
        super().__init__()
        
        # Landmark encoder
        self.encoder = LandmarkEncoder(
            input_size=landmark_size,
            hidden_size=hidden_size
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, landmarks, frames=None):
        """Forward pass (frames ignored, kept for API compatibility)"""
        features = self.encoder(landmarks)
        logits = self.classifier(features)
        return logits


def get_model(model_type: str = "hybrid", **kwargs) -> nn.Module:
    """
    Factory function to get model
    
    Args:
        model_type: "hybrid" or "landmark_only"
        **kwargs: Additional model arguments
    
    Returns:
        Initialized model
    """
    if model_type == "hybrid":
        return HybridModel(**kwargs)
    elif model_type == "landmark_only":
        return LandmarkOnlyModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test models
    print("Testing LandmarkOnlyModel...")
    model = LandmarkOnlyModel()
    batch = torch.randn(4, SEQUENCE_LENGTH, LANDMARK_FEATURES)
    output = model(batch)
    print(f"Input shape: {batch.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
