"""
TID Recognition System - Configuration
High accuracy priority configuration for hybrid model
"""

import os
from pathlib import Path

# ============================================
# PATHS
# ============================================
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "AUTSL"
PROCESSED_DIR = BASE_DIR / "processed_data"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

# Create directories
PROCESSED_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# ============================================
# DATASET
# ============================================
NUM_CLASSES = 226
TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"
TEST_CSV = DATA_DIR / "test.csv"
CLASS_MAP_CSV = DATA_DIR / "SignList_ClassId_TR_EN.csv"

# ============================================
# VIDEO PROCESSING
# ============================================
SEQUENCE_LENGTH = 48  # Number of frames per sequence
IMG_SIZE = 224  # For CNN feature extraction
FPS_TARGET = 24  # Target FPS for normalization

# ============================================
# MEDIAPIPE LANDMARKS
# ============================================
# Pose: 33 landmarks * 4 (x, y, z, visibility) = 132
# Left Hand: 21 landmarks * 3 (x, y, z) = 63
# Right Hand: 21 landmarks * 3 (x, y, z) = 63
# Total: 258 features (or 225 if excluding visibility)
POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
LANDMARK_FEATURES = 258  # Using x, y, z + visibility for pose

# ============================================
# MODEL ARCHITECTURE (HYBRID)
# ============================================
# GRU for landmark sequences
GRU_HIDDEN_SIZE = 512
GRU_NUM_LAYERS = 3
GRU_DROPOUT = 0.4
GRU_BIDIRECTIONAL = True

# CNN feature extractor
CNN_BACKBONE = "efficientnet_b0"  # Lightweight but effective
CNN_FEATURE_DIM = 1280  # EfficientNet-B0 output

# Fusion
FUSION_DIM = 512
FUSION_DROPOUT = 0.5

# ============================================
# TRAINING - HIGH ACCURACY CONFIG
# ============================================
BATCH_SIZE = 16  # RTX 3070 8GB VRAM
NUM_EPOCHS = 100
LEARNING_RATE = 3e-3  # 10x higher for faster learning
WEIGHT_DECAY = 1e-5   # Less regularization initially
WARMUP_EPOCHS = 0     # No warmup
MIN_LR = 1e-6

# Mixed precision for faster training
USE_AMP = True

# Learning rate scheduler
LR_SCHEDULER = "plateau"  # Reduce on plateau instead of cosine

# Early stopping
EARLY_STOPPING_PATIENCE = 20  # More patience
EARLY_STOPPING_MIN_DELTA = 0.001
EARLY_STOPPING_MIN_DELTA = 0.001

# ============================================
# DATA AUGMENTATION
# ============================================
AUGMENTATION = {
    "time_stretch": {"min_rate": 0.8, "max_rate": 1.2},
    "time_mask": {"max_frames": 5},
    "landmark_noise": {"std": 0.01},
    "landmark_dropout": {"rate": 0.05},
    "horizontal_flip": False,  # Not applicable for sign language
}

# ============================================
# INFERENCE
# ============================================
SLIDING_WINDOW_SIZE = 48  # Frames
SLIDING_WINDOW_STRIDE = 24  # 50% overlap
CONFIDENCE_THRESHOLD = 0.7
TOP_K_PREDICTIONS = 3

# ============================================
# DEVICE
# ============================================
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4 if DEVICE == "cuda" else 0
PIN_MEMORY = DEVICE == "cuda"

# ============================================
# LOGGING
# ============================================
SAVE_BEST_ONLY = True
CHECKPOINT_EVERY = 1  # Save checkpoint every epoch
TENSORBOARD_LOG = True
