"""Export the digit selection model to a mobile-friendly .ptl file."""

import json

import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

from src.digit_selection.config import (
    BEST_MODEL_PATH,
    INPUT_FEATURES,
    LABELS_PATH,
    MOBILE_MODEL_PATH,
)
from src.digit_selection.model import DigitSelectionMLP


def main():
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{BEST_MODEL_PATH} not found. Train the digit selection model first."
        )

    checkpoint = torch.load(BEST_MODEL_PATH, map_location="cpu")
    model = DigitSelectionMLP()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sample_input = torch.zeros(1, INPUT_FEATURES)
    with torch.no_grad():
        traced = torch.jit.trace(model, sample_input)
    optimized = optimize_for_mobile(traced)
    optimized._save_for_lite_interpreter(str(MOBILE_MODEL_PATH))

    size_mb = MOBILE_MODEL_PATH.stat().st_size / 1024 / 1024
    print(f"Saved mobile model: {MOBILE_MODEL_PATH} ({size_mb:.2f} MB)")

    if LABELS_PATH.exists():
        labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
        print(f"Labels: {labels['class_names']}")


if __name__ == "__main__":
    main()
