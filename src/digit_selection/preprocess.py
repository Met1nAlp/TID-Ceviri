"""Preprocess the digit image dataset into MediaPipe hand landmarks."""

import json
from collections import Counter
from pathlib import Path
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe import Image as mp_Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.digit_selection.config import (
    CLASS_TO_INDEX,
    CLASS_NAMES,
    LABELS_PATH,
    LANDMARK_DIR,
    MEDIAPIPE_MODEL_DIR,
    METADATA_PATH,
    MIN_HAND_DETECTION_CONFIDENCE,
    MIN_HAND_PRESENCE_CONFIDENCE,
    NUM_HANDS,
    RANDOM_SEED,
    RAW_DATASET_DIR,
    SOURCE_TO_TARGET,
    TEST_RATIO,
    VAL_RATIO,
)

HAND_LANDMARKER_NAME = "hand_landmarker.task"
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


def ensure_hand_landmarker():
    model_path = MEDIAPIPE_MODEL_DIR / HAND_LANDMARKER_NAME
    if not model_path.exists():
        print(f"Downloading {HAND_LANDMARKER_NAME}...")
        urllib.request.urlretrieve(HAND_LANDMARKER_URL, model_path)
    return model_path


def create_hand_landmarker(model_path: Path):
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
    )
    return vision.HandLandmarker.create_from_options(options)


def canonicalize_hand_landmarks(hand_landmarks, handedness: str):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)

    wrist = coords[0].copy()
    coords = coords - wrist

    scale = np.linalg.norm(coords[:, :2], axis=1).max()
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    coords = coords / scale

    if handedness == "Right":
        coords[:, 0] *= -1.0

    return coords.reshape(-1)


def build_split_map(metadata: pd.DataFrame):
    rng = np.random.default_rng(RANDOM_SEED)
    split_map = {}

    for label_idx in sorted(metadata["label"].unique()):
        indices = np.array(
            metadata.index[metadata["label"] == label_idx].to_list(),
            dtype=np.int64,
        )
        rng.shuffle(indices)

        total = len(indices)
        test_count = max(1, int(round(total * TEST_RATIO)))
        val_count = max(1, int(round(total * VAL_RATIO)))
        train_count = total - test_count - val_count

        if train_count <= 0:
            train_count = max(1, total - test_count - val_count)

        train_indices = indices[:train_count]
        val_indices = indices[train_count:train_count + val_count]
        test_indices = indices[train_count + val_count:]

        for idx in train_indices:
            split_map[idx] = "train"
        for idx in val_indices:
            split_map[idx] = "val"
        for idx in test_indices:
            split_map[idx] = "test"

    return split_map


def preprocess_dataset():
    if not RAW_DATASET_DIR.exists():
        raise FileNotFoundError(
            f"{RAW_DATASET_DIR} not found. Download the digit dataset first."
        )

    model_path = ensure_hand_landmarker()
    hand_landmarker = create_hand_landmarker(model_path)

    samples = []
    skipped_counter = Counter()

    image_paths = sorted(RAW_DATASET_DIR.glob("*/*.JPG"))
    if not image_paths:
        image_paths = sorted(RAW_DATASET_DIR.glob("*/*.jpg"))

    print(f"Processing {len(image_paths)} digit images...")

    for image_path in image_paths:
        source_digit = image_path.parent.name
        target_label = SOURCE_TO_TARGET.get(source_digit)
        if target_label is None:
            skipped_counter["unknown_label"] += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            skipped_counter["unreadable_image"] += 1
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp_Image(image_format=ImageFormat.SRGB, data=rgb)
        result = hand_landmarker.detect(mp_image)

        if not result.hand_landmarks or not result.handedness:
            skipped_counter["no_hand_detected"] += 1
            continue

        hand_landmarks = result.hand_landmarks[0]
        handedness = result.handedness[0][0].category_name
        features = canonicalize_hand_landmarks(hand_landmarks, handedness)

        sample_id = f"{source_digit}_{image_path.stem}"
        np.save(LANDMARK_DIR / f"{sample_id}.npy", features.astype(np.float32))

        samples.append(
            {
                "sample_id": sample_id,
                "source_digit": source_digit,
                "target_label": target_label,
                "label": CLASS_TO_INDEX[target_label],
                "handedness": handedness,
                "image_path": str(image_path.relative_to(RAW_DATASET_DIR.parent.parent)),
            }
        )

    metadata = pd.DataFrame(samples)
    if metadata.empty:
        raise RuntimeError("No samples were extracted. Check MediaPipe detection.")

    split_map = build_split_map(metadata)
    metadata["split"] = metadata.index.map(split_map)
    metadata = metadata.sort_values(["split", "label", "sample_id"]).reset_index(drop=True)
    metadata.to_csv(METADATA_PATH, index=False)

    labels_payload = {
        "class_names": CLASS_NAMES,
        "class_to_index": CLASS_TO_INDEX,
        "source_to_target": SOURCE_TO_TARGET,
    }
    LABELS_PATH.write_text(json.dumps(labels_payload, indent=2), encoding="utf-8")

    print("\nExtraction complete.")
    print(f"Saved metadata: {METADATA_PATH}")
    print(f"Saved labels:   {LABELS_PATH}")

    print("\nClass summary:")
    summary = metadata.groupby(["split", "target_label"]).size().unstack(fill_value=0)
    print(summary)

    if skipped_counter:
        print("\nSkipped samples:")
        for reason, count in skipped_counter.items():
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    preprocess_dataset()
