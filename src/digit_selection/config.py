"""
Configuration for the digit selection model.

This model is separate from the main 226-class sign recognizer and is intended
to confirm one of the top-3 candidate words by recognizing digit gestures.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"
EXTERNAL_DATA_DIR = BASE_DIR / "external_data" / "Sign-Language-Digits-Dataset"
RAW_DATASET_DIR = EXTERNAL_DATA_DIR / "Dataset"
PROCESSED_DIR = BASE_DIR / "processed_digit_data"
LANDMARK_DIR = PROCESSED_DIR / "landmarks"
MEDIAPIPE_MODEL_DIR = BASE_DIR / "src" / "data" / "models"

MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)
LANDMARK_DIR.mkdir(exist_ok=True)
MEDIAPIPE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Target classes for selection mode.
TARGET_DIGITS = ("1", "2", "3")
OTHER_DIGITS = ("0", "4", "5", "6", "7", "8", "9")
CLASS_NAMES = ["digit_1", "digit_2", "digit_3", "other_digit"]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
SOURCE_TO_TARGET = {
    "1": "digit_1",
    "2": "digit_2",
    "3": "digit_3",
    "0": "other_digit",
    "4": "other_digit",
    "5": "other_digit",
    "6": "other_digit",
    "7": "other_digit",
    "8": "other_digit",
    "9": "other_digit",
}

# Input features: 21 hand landmarks * (x, y, z)
INPUT_FEATURES = 63
NUM_CLASSES = len(CLASS_NAMES)

# Split ratios
RANDOM_SEED = 42
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# MediaPipe extraction
MIN_HAND_DETECTION_CONFIDENCE = 0.3
MIN_HAND_PRESENCE_CONFIDENCE = 0.3
NUM_HANDS = 1

# Training
BATCH_SIZE = 128
NUM_EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 8
MIN_LR = 1e-6

DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "-1" else "cpu"
NUM_WORKERS = 0 if os.name == "nt" else 4
PIN_MEMORY = DEVICE == "cuda"

# Filenames
METADATA_PATH = PROCESSED_DIR / "metadata.csv"
LABELS_PATH = PROCESSED_DIR / "labels.json"
BEST_MODEL_PATH = MODEL_DIR / "digit_selection_best.pth"
LATEST_MODEL_PATH = MODEL_DIR / "digit_selection_latest.pth"
MOBILE_MODEL_PATH = MODEL_DIR / "digit_selection_mobile.ptl"
