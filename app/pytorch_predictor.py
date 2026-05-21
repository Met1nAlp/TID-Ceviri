"""
PyTorch predictor for the web application.
"""

import json
import sys
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe import Image as mp_Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

sys.path.append(str(Path(__file__).parent.parent))

from src.models.ultra_simple import SimpleLSTM
from src.training.config import (
    CONFIDENCE_THRESHOLD,
    IDLE_FRAMES_TO_STOP,
    IDLE_THRESHOLD,
    LANDMARK_FEATURES,
    MIN_SIGN_FRAMES,
    MOTION_THRESHOLD,
    NUM_CLASSES,
    SEQUENCE_LENGTH,
    START_FRAMES,
)


class PyTorchPredictor:
    """Predictor using PyTorch model with MediaPipe Tasks API."""

    def __init__(
        self,
        model_path="models/best_model.pth",
        device="cuda",
        enable_temporal_smoothing=False,
        use_video_landmarkers=False,
        swap_handedness=True,
        motion_threshold=None,
        idle_threshold=None,
        min_sign_frames=None,
        idle_frames_to_stop=None,
        start_frames=None,
        confidence_threshold=None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = SimpleLSTM(input_size=LANDMARK_FEATURES, num_classes=NUM_CLASSES)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        self.class_labels = self._load_class_labels()
        self.enable_temporal_smoothing = enable_temporal_smoothing
        self.use_video_landmarkers = use_video_landmarkers
        self.swap_handedness = swap_handedness
        self.timestamp_ms = 0

        print("Initializing MediaPipe...")
        self._download_models()
        self._init_mediapipe()
        print("MediaPipe initialized.")

        self.prev_landmarks = None
        self.state = "idle"
        self.sign_frames = []
        self.idle_frames = 0
        self.signing_frames = 0

        self.MOTION_THRESHOLD = MOTION_THRESHOLD if motion_threshold is None else motion_threshold
        self.IDLE_THRESHOLD = IDLE_THRESHOLD if idle_threshold is None else idle_threshold
        self.MIN_SIGN_FRAMES = MIN_SIGN_FRAMES if min_sign_frames is None else min_sign_frames
        self.MIN_DECISION_FRAMES = self.MIN_SIGN_FRAMES
        self.MAX_SIGN_FRAMES = max(SEQUENCE_LENGTH + 24, 72)
        self.IDLE_FRAMES_TO_STOP = (
            IDLE_FRAMES_TO_STOP if idle_frames_to_stop is None else idle_frames_to_stop
        )
        self.START_FRAMES = START_FRAMES if start_frames is None else start_frames
        self.CONFIDENCE_THRESHOLD = (
            CONFIDENCE_THRESHOLD if confidence_threshold is None else confidence_threshold
        )

        self.TEMPERATURE = 1.5
        self.MARGIN_THRESHOLD = 0.15
        self.MIN_HAND_FRAMES_DIVISOR = 8
        self.PRE_BUFFER_SIZE = 8
        self.pre_buffer = deque(maxlen=self.PRE_BUFFER_SIZE)
        self.VOTE_HISTORY_SIZE = 3
        self.prediction_history = []
        self.COOLDOWN_FRAMES = 20
        self.TRAILING_IDLE_KEEP_FRAMES = 2
        self.cooldown_counter = 0
        self.last_debug = {
            "state": self.state,
            "motion": 0.0,
            "signing_frames": 0,
            "idle_frames": 0,
            "collected_frames": 0,
            "cooldown_counter": 0,
            "hand_visible": False,
            "hand_frames": 0,
            "min_hand_frames": 0,
            "last_event": "waiting",
            "last_reason": "",
            "last_label": "-",
            "last_confidence": 0.0,
            "last_ambiguous": False,
            "last_low_confidence": False,
        }
        self.pose_motion_indices = (11, 12, 13, 14, 15, 16)

    def _download_models(self):
        self.model_dir = Path(__file__).parent.parent / "src" / "data" / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        models = {
            "pose_landmarker_heavy.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
            "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        }

        for filename, url in models.items():
            filepath = self.model_dir / filename
            if not filepath.exists():
                print(f"  Downloading {filename}...")
                urllib.request.urlretrieve(url, filepath)

    def _init_mediapipe(self):
        if self.use_video_landmarkers:
            pose_options = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=str(self.model_dir / "pose_landmarker_heavy.task")
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.3,
                min_pose_presence_confidence=0.3,
                min_tracking_confidence=0.3,
            )
            hand_options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=str(self.model_dir / "hand_landmarker.task")
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.3,
                min_hand_presence_confidence=0.3,
                min_tracking_confidence=0.3,
            )
        else:
            pose_options = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=str(self.model_dir / "pose_landmarker_heavy.task")
                ),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.3,
                min_pose_presence_confidence=0.3,
            )
            hand_options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=str(self.model_dir / "hand_landmarker.task")
                ),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.3,
                min_hand_presence_confidence=0.3,
            )

        self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

    def _load_class_labels(self):
        class_map_path = Path("class_mapping.json")
        if class_map_path.exists():
            with open(class_map_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
                return {int(k): (v, v) for k, v in mapping.items()}

        corrected_csv_path = Path("AUTSL/SignList_ClassId_TR_EN_corrected.csv")
        csv_path = corrected_csv_path if corrected_csv_path.exists() else Path("AUTSL/SignList_ClassId_TR_EN.csv")
        if csv_path.exists():
            import pandas as pd

            df = pd.read_csv(csv_path)
            return {row["ClassId"]: (row["TR"], row["EN"]) for _, row in df.iterrows()}

        return {i: (f"class_{i}", f"class_{i}") for i in range(NUM_CLASSES)}

    def extract_landmarks(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp_Image(image_format=ImageFormat.SRGB, data=rgb_frame)

        if self.use_video_landmarkers:
            self.timestamp_ms += 33
            pose_result = self.pose_landmarker.detect_for_video(mp_image, self.timestamp_ms)
            hand_result = self.hand_landmarker.detect_for_video(mp_image, self.timestamp_ms)
        else:
            pose_result = self.pose_landmarker.detect(mp_image)
            hand_result = self.hand_landmarker.detect(mp_image)

        landmarks = []

        if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
            for lm in pose_result.pose_landmarks[0]:
                visibility = lm.visibility if getattr(lm, "visibility", None) is not None else 1.0
                landmarks.extend([lm.x, lm.y, lm.z, visibility])
        else:
            landmarks.extend([0.0] * 132)

        left_hand_found = False
        right_hand_found = False
        left_coords = []
        right_coords = []

        if hand_result.hand_landmarks and hand_result.handedness:
            for hand_landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
                hand_label = handedness[0].category_name
                target_label = hand_label

                if self.swap_handedness:
                    if hand_label == "Left":
                        target_label = "Right"
                    elif hand_label == "Right":
                        target_label = "Left"

                if target_label == "Left" and not left_hand_found:
                    left_hand_found = True
                    for lm in hand_landmarks:
                        left_coords.extend([lm.x, lm.y, lm.z])
                elif target_label == "Right" and not right_hand_found:
                    right_hand_found = True
                    for lm in hand_landmarks:
                        right_coords.extend([lm.x, lm.y, lm.z])

        landmarks.extend(left_coords if left_hand_found else [0.0] * 63)
        landmarks.extend(right_coords if right_hand_found else [0.0] * 63)

        return np.array(landmarks, dtype=np.float32), (pose_result, hand_result)

    def _build_prediction(self, class_id, prob, ambiguous=False, low_confidence=False):
        tr_label, en_label = self.class_labels.get(int(class_id), (f"class_{class_id}", f"class_{class_id}"))
        return {
            "label_tr": tr_label,
            "label_en": en_label,
            "confidence": round(float(prob) * 100, 1),
            "ambiguous": ambiguous,
            "low_confidence": low_confidence,
            "class_id": int(class_id),
        }

    def _has_hand_landmarks(self, frame):
        return np.abs(frame[132:]).sum() > 0.1

    def get_debug_status(self):
        return dict(self.last_debug)

    def reset_stream_state(self):
        self.prev_landmarks = None
        self.state = "idle"
        self.sign_frames = []
        self.idle_frames = 0
        self.signing_frames = 0
        self.pre_buffer.clear()
        self.prediction_history.clear()
        self.cooldown_counter = 0
        self.last_debug.update(
            {
                "state": self.state,
                "motion": 0.0,
                "signing_frames": 0,
                "idle_frames": 0,
                "collected_frames": 0,
                "cooldown_counter": 0,
                "hand_visible": False,
                "hand_frames": 0,
                "min_hand_frames": 0,
                "last_event": "reset",
                "last_reason": "",
                "last_label": "-",
                "last_confidence": 0.0,
                "last_ambiguous": False,
                "last_low_confidence": False,
            }
        )

    def _record_last_event(
        self,
        event,
        reason="",
        prediction=None,
        hand_frames=0,
        min_hand_frames=0,
    ):
        self.last_debug.update(
            {
                "last_event": event,
                "last_reason": reason,
                "hand_frames": int(hand_frames),
                "min_hand_frames": int(min_hand_frames),
            }
        )

        if prediction is None:
            self.last_debug.update(
                {
                    "last_label": "-",
                    "last_confidence": 0.0,
                    "last_ambiguous": False,
                    "last_low_confidence": False,
                }
            )
            return

        self.last_debug.update(
            {
                "last_label": prediction.get("label_tr", "-"),
                "last_confidence": round(float(prediction.get("confidence", 0.0)), 1),
                "last_ambiguous": bool(prediction.get("ambiguous", False)),
                "last_low_confidence": bool(prediction.get("low_confidence", False)),
            }
        )

    def _update_debug_snapshot(self, motion, landmarks):
        self.last_debug.update(
            {
                "state": self.state,
                "motion": round(float(motion), 6),
                "signing_frames": int(self.signing_frames),
                "idle_frames": int(self.idle_frames),
                "collected_frames": int(len(self.sign_frames)),
                "cooldown_counter": int(self.cooldown_counter),
                "hand_visible": bool(self._has_hand_landmarks(landmarks)),
            }
        )

    @torch.no_grad()
    def predict(self, sequence, return_probs=False):
        mean = sequence.mean()
        std = sequence.std() + 1e-8
        sequence = (sequence - mean) / std

        x = torch.from_numpy(sequence).unsqueeze(0).to(self.device)
        logits = self.model(x)
        scaled_logits = logits / self.TEMPERATURE
        probs = torch.softmax(scaled_logits, dim=1)[0].cpu().numpy()

        top_indices = np.argsort(probs)[-3:][::-1]
        predictions = [self._build_prediction(int(idx), float(probs[idx])) for idx in top_indices]

        if len(predictions) >= 2:
            margin = (predictions[0]["confidence"] - predictions[1]["confidence"]) / 100.0
            if margin < self.MARGIN_THRESHOLD:
                for prediction in predictions:
                    prediction["ambiguous"] = True

        if return_probs:
            return predictions, probs
        return predictions

    def _compute_motion(self, landmarks):
        motion = self.preview_motion(landmarks)
        self.prev_landmarks = landmarks.copy()
        return motion

    def preview_motion(self, landmarks, previous_landmarks=None):
        reference_landmarks = self.prev_landmarks if previous_landmarks is None else previous_landmarks
        if reference_landmarks is None:
            return 0.0

        curr_hands = landmarks[132:]
        prev_hands = reference_landmarks[132:]
        left_motion = np.mean(np.abs(curr_hands[:63] - prev_hands[:63]))
        right_motion = np.mean(np.abs(curr_hands[63:] - prev_hands[63:]))
        hand_motion = max(left_motion, right_motion)

        pose_diffs = []
        for landmark_index in self.pose_motion_indices:
            base = landmark_index * 4
            pose_diffs.extend(
                [
                    abs(landmarks[base] - reference_landmarks[base]),
                    abs(landmarks[base + 1] - reference_landmarks[base + 1]),
                    abs(landmarks[base + 2] - reference_landmarks[base + 2]),
                ]
            )
        pose_motion = float(np.mean(pose_diffs)) if pose_diffs else 0.0

        return max(hand_motion, pose_motion * 0.6)

    def process_frame(self, frame):
        landmarks, results = self.extract_landmarks(frame)
        return self.process_landmarks(landmarks, results)

    def process_landmarks(self, landmarks, results=None):
        self.pre_buffer.append(landmarks.copy())
        motion = self._compute_motion(landmarks)

        if self.enable_temporal_smoothing and self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self._update_debug_snapshot(motion, landmarks)
            return [], results, self.state

        predictions = []

        if self.state == "idle":
            if motion > self.MOTION_THRESHOLD:
                self.signing_frames += 1
                if self.signing_frames >= self.START_FRAMES:
                    self.state = "signing"
                    if self.enable_temporal_smoothing:
                        self.sign_frames = [buffered.copy() for buffered in self.pre_buffer]
                        self.prediction_history.clear()
                    else:
                        self.sign_frames = []
                    self.idle_frames = 0
                    self.signing_frames = 0
            else:
                self.signing_frames = 0

        elif self.state == "signing":
            self.sign_frames.append(landmarks.copy())

            if motion < self.IDLE_THRESHOLD:
                self.idle_frames += 1
                if self.idle_frames >= self.IDLE_FRAMES_TO_STOP:
                    if len(self.sign_frames) >= self.MIN_DECISION_FRAMES:
                        predictions = self._predict_sign()
                        if self.enable_temporal_smoothing and predictions:
                            self.cooldown_counter = self.COOLDOWN_FRAMES
                    self.state = "idle"
                    self.sign_frames = []
                    self.idle_frames = 0
            else:
                self.idle_frames = 0

            if len(self.sign_frames) >= self.MAX_SIGN_FRAMES:
                predictions = self._predict_sign()
                if self.enable_temporal_smoothing and predictions:
                    self.cooldown_counter = self.COOLDOWN_FRAMES
                self.state = "idle"
                self.sign_frames = []
                self.idle_frames = 0

        self._update_debug_snapshot(motion, landmarks)
        return predictions, results, self.state

    def _predict_sign(self):
        effective_frames = self.sign_frames
        trailing_idle_trim = max(0, self.idle_frames - self.TRAILING_IDLE_KEEP_FRAMES)
        if trailing_idle_trim > 0 and len(self.sign_frames) - trailing_idle_trim >= 1:
            effective_frames = self.sign_frames[:-trailing_idle_trim]

        n = len(effective_frames)
        if n == 0:
            self._record_last_event("skipped", "empty_sequence")
            return []

        valid_frames = sum(1 for frame in effective_frames if np.mean(np.abs(frame)) > 0.01)
        if valid_frames < n * 0.5:
            self._record_last_event("skipped", "too_many_empty_frames")
            return []

        hand_frames = 0
        min_required_hand_frames = 0
        if self.enable_temporal_smoothing:
            hand_frames = sum(1 for frame in effective_frames if self._has_hand_landmarks(frame))
            min_required_hand_frames = max(3, n // self.MIN_HAND_FRAMES_DIVISOR)
            if hand_frames < min_required_hand_frames:
                self.prediction_history.clear()
                self._record_last_event(
                    "skipped",
                    "insufficient_hand_frames",
                    hand_frames=hand_frames,
                    min_hand_frames=min_required_hand_frames,
                )
                return []

        indices = np.linspace(0, n - 1, SEQUENCE_LENGTH)
        frames = []
        for idx in indices:
            lower = int(np.floor(idx))
            upper = min(int(np.ceil(idx)), n - 1)
            weight = idx - lower
            if lower == upper:
                frames.append(effective_frames[lower].copy())
            else:
                interp = (1 - weight) * effective_frames[lower] + weight * effective_frames[upper]
                frames.append(interp.astype(np.float32))

        sequence = np.array(frames, dtype=np.float32)
        predictions, probs = self.predict(sequence, return_probs=True)

        for prediction in predictions:
            if prediction["confidence"] < self.CONFIDENCE_THRESHOLD * 100:
                prediction["low_confidence"] = True

        if not predictions:
            self._record_last_event(
                "skipped",
                "no_prediction",
                hand_frames=hand_frames,
                min_hand_frames=min_required_hand_frames,
            )
            return predictions

        if not self.enable_temporal_smoothing:
            self._record_last_event(
                "predicted",
                "raw_prediction",
                prediction=predictions[0],
                hand_frames=hand_frames,
                min_hand_frames=min_required_hand_frames,
            )
            return predictions

        top_class_id = predictions[0]["class_id"]
        self.prediction_history.append(top_class_id)
        if len(self.prediction_history) > self.VOTE_HISTORY_SIZE:
            self.prediction_history.pop(0)

        voted_class_id = top_class_id
        if len(self.prediction_history) >= 2:
            counts = {}
            for class_id in self.prediction_history:
                counts[class_id] = counts.get(class_id, 0) + 1
            candidate = max(counts.items(), key=lambda item: item[1])[0]
            if candidate != top_class_id and counts[candidate] >= 2:
                voted_class_id = candidate

        ambiguous = any(prediction.get("ambiguous", False) for prediction in predictions)
        ordered_class_ids = [voted_class_id]
        for idx in np.argsort(probs)[::-1]:
            idx = int(idx)
            if idx != voted_class_id:
                ordered_class_ids.append(idx)
            if len(ordered_class_ids) == 3:
                break

        final_predictions = [
            self._build_prediction(
                idx,
                float(probs[idx]),
                ambiguous=ambiguous,
                low_confidence=float(probs[idx]) < self.CONFIDENCE_THRESHOLD,
            )
            for idx in ordered_class_ids
        ]

        top_prediction = final_predictions[0] if final_predictions else None
        event_reason = "predicted"
        if top_prediction and top_prediction.get("low_confidence"):
            event_reason = "low_confidence"
        elif top_prediction and top_prediction.get("ambiguous"):
            event_reason = "ambiguous"

        self._record_last_event(
            "predicted",
            event_reason,
            prediction=top_prediction,
            hand_frames=hand_frames,
            min_hand_frames=min_required_hand_frames,
        )
        return final_predictions

    def draw_landmarks(self, frame, results):
        pose_result, hand_result = results

        if pose_result.pose_landmarks:
            for lm in pose_result.pose_landmarks[0]:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 3, (245, 117, 66), -1)

        if hand_result.hand_landmarks:
            for hand_landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
                color = (121, 22, 76) if handedness[0].category_name == "Left" else (80, 110, 10)
                for lm in hand_landmarks:
                    x = int(lm.x * frame.shape[1])
                    y = int(lm.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 3, color, -1)

        return frame
