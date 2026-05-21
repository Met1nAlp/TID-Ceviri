"""Digit-based candidate selection for the web/desktop flow."""

import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from src.digit_selection.model import DigitSelectionMLP


class DigitSelectionPredictor:
    """Recognize 1/2/3 gestures to select one of the top-3 candidates."""

    def __init__(
        self,
        model_path="models/digit_selection_best.pth",
        labels_path="processed_digit_data/labels.json",
        device="cuda",
        confidence_threshold=0.8,
        selection_timeout_seconds=3.0,
        selection_arm_delay_seconds=0.7,
        stable_frames=3,
        vote_history_size=5,
    ):
        repo_root = Path(__file__).parent.parent
        model_path = repo_root / model_path
        labels_path = repo_root / labels_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self.selection_timeout_seconds = selection_timeout_seconds
        self.selection_arm_delay_seconds = selection_arm_delay_seconds
        self.selection_interrupt_grace_seconds = max(selection_arm_delay_seconds, 1.2)
        self.stable_frames = stable_frames
        self.vote_history = deque(maxlen=vote_history_size)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = DigitSelectionMLP()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        labels_payload = json.loads(Path(labels_path).read_text(encoding="utf-8"))
        self.class_names = checkpoint.get("class_names", labels_payload["class_names"])

        self.active = False
        self.candidates = []
        self.started_at = 0.0
        self.expires_at = 0.0
        self.last_prediction = None
        self.last_event = "idle"
        self.last_reason = ""
        self.last_selected = None
        self.selection_serial = 0

    def start_selection(self, candidates):
        self.active = True
        self.candidates = [dict(candidate) for candidate in candidates[:3]]
        self.started_at = time.monotonic()
        self.expires_at = self.started_at + self.selection_timeout_seconds
        self.vote_history.clear()
        self.last_prediction = None
        self.last_event = "armed"
        self.last_reason = "awaiting_digit"
        self.last_selected = None
        self.selection_serial += 1

    def cancel(self, reason="cancelled"):
        self.active = False
        self.vote_history.clear()
        self.last_event = "cancelled"
        self.last_reason = reason

    def is_arming(self):
        return self.active and time.monotonic() < (self.started_at + self.selection_arm_delay_seconds)

    def is_interrupt_guard_active(self):
        return self.active and time.monotonic() < (
            self.started_at + self.selection_interrupt_grace_seconds
        )

    def has_digit_evidence(self):
        if self.vote_history:
            return True
        prediction = self.last_prediction or {}
        return prediction.get("digit_value") is not None

    def _canonicalize_hand_landmarks(self, hand_landmarks, handedness):
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)

        wrist = coords[0].copy()
        coords = coords - wrist

        scale = np.linalg.norm(coords[:, :2], axis=1).max()
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0
        coords = coords / scale

        if handedness == "Right":
            coords[:, 0] *= -1.0

        mean = coords.mean()
        std = coords.std() + 1e-8
        coords = (coords - mean) / std
        return coords.reshape(-1).astype(np.float32)

    @torch.no_grad()
    def _classify_hand_result(self, hand_result):
        if not hand_result.hand_landmarks or not hand_result.handedness:
            return None

        hand_landmarks = hand_result.hand_landmarks[0]
        handedness = hand_result.handedness[0][0].category_name
        features = self._canonicalize_hand_landmarks(hand_landmarks, handedness)
        tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(tensor), dim=1)[0].cpu().numpy()

        top_idx = int(np.argmax(probs))
        top_label = self.class_names[top_idx]
        confidence = float(probs[top_idx])
        digit_value = None
        if top_label.startswith("digit_"):
            digit_value = int(top_label.split("_", maxsplit=1)[1])

        return {
            "class_index": top_idx,
            "label": top_label,
            "digit_value": digit_value,
            "confidence": confidence,
            "handedness": handedness,
        }

    def process_hand_result(self, hand_result):
        if not self.active:
            return None

        now = time.monotonic()
        if now >= self.expires_at:
            self.active = False
            self.vote_history.clear()
            self.last_event = "timeout"
            self.last_reason = "selection_timeout"
            return {"event": "timeout"}

        if now < self.started_at + self.selection_arm_delay_seconds:
            self.last_event = "waiting"
            self.last_reason = "selection_arm_delay"
            return None

        prediction = self._classify_hand_result(hand_result)
        self.last_prediction = prediction

        if prediction is None:
            self.last_event = "waiting"
            self.last_reason = "no_hand_detected"
            return None

        if prediction["digit_value"] is None or prediction["confidence"] < self.confidence_threshold:
            self.last_event = "waiting"
            self.last_reason = "low_confidence_digit"
            return None

        self.vote_history.append(prediction["digit_value"])
        self.last_event = "digit_seen"
        self.last_reason = f"digit_{prediction['digit_value']}"

        recent_votes = list(self.vote_history)[-self.stable_frames :]
        if len(recent_votes) < self.stable_frames or len(set(recent_votes)) != 1:
            return None

        digit_value = recent_votes[-1]
        candidate_index = digit_value - 1
        if candidate_index < 0 or candidate_index >= len(self.candidates):
            self.last_event = "waiting"
            self.last_reason = "digit_without_candidate"
            return None

        chosen_candidate = dict(self.candidates[candidate_index])
        chosen_payload = {
            "digit_value": digit_value,
            "candidate_index": candidate_index,
            "candidate": chosen_candidate,
            "confidence": round(prediction["confidence"] * 100.0, 1),
        }

        self.active = False
        self.vote_history.clear()
        self.last_event = "selected"
        self.last_reason = "stable_digit_match"
        self.last_selected = chosen_payload
        self.selection_serial += 1
        return {"event": "selected", **chosen_payload}

    def get_status(self):
        remaining_ms = 0
        if self.active:
            remaining_ms = max(0, int((self.expires_at - time.monotonic()) * 1000))

        stable_digit = None
        if self.vote_history:
            counts = {}
            for digit_value in self.vote_history:
                counts[digit_value] = counts.get(digit_value, 0) + 1
            stable_digit = max(counts.items(), key=lambda item: item[1])[0]

        prediction = self.last_prediction or {}
        return {
            "active": self.active,
            "remaining_ms": remaining_ms,
            "candidates": self.candidates,
            "last_event": self.last_event,
            "last_reason": self.last_reason,
            "last_digit_value": prediction.get("digit_value"),
            "last_digit_label": prediction.get("label"),
            "last_confidence": round(float(prediction.get("confidence", 0.0)) * 100.0, 1),
            "stable_digit": stable_digit,
            "stable_votes": len(self.vote_history),
            "required_stable_frames": self.stable_frames,
            "last_selected": self.last_selected,
            "selection_serial": self.selection_serial,
        }
