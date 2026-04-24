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
GRU_HIDDEN_SIZE = 256
GRU_NUM_LAYERS = 2
GRU_DROPOUT = 0.3
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
BATCH_SIZE = 64  # Larger batch for stable gradients
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3  # Higher LR for faster learning
WEIGHT_DECAY = 1e-4   # Slight regularization
WARMUP_EPOCHS = 5     # Longer warmup for stability
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
    "landmark_noise": {"std": 0.05},          # 5x artirildi - Android MediaPipe farklarina karsi
    "landmark_dropout": {"rate": 0.15},        # 3x artirildi
    "horizontal_flip": False,
    "coord_scale": {"min": 0.90, "max": 1.10},  # Koordinat olcekleme (±10%)
    "coord_shift": {"max": 0.03},               # Koordinat kaydirma (±3%)
}

# ============================================
# INFERENCE
# ============================================
SLIDING_WINDOW_SIZE = 48  # Frames
SLIDING_WINDOW_STRIDE = 24  # 50% overlap
CONFIDENCE_THRESHOLD = 0.4
TOP_K_PREDICTIONS = 3

# Motion detection thresholds — tum pipeline'larda ortak (web, desktop, mobile)
MOTION_THRESHOLD    = 0.008   # el hareketi bu degerin uzerindeyse -> signing baslar
IDLE_THRESHOLD      = 0.006   # bu degerin altina duserse -> idle sayilir
MIN_SIGN_FRAMES     = 15      # gecerli isaret icin minimum frame sayisi
IDLE_FRAMES_TO_STOP = 10      # signing'i bitirmek icin gereken ardisik idle frame
START_FRAMES        = 2       # signing'i baslatmak icin gereken ardisik hareket frame

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
