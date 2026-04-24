"""
Real-time Sign Language Recognition
Uses SimpleLSTM model with MediaPipe Tasks API
Motion-based sign segmentation (matches web app logic)
"""

import cv2
import numpy as np
import torch
import urllib.request
from collections import deque
from pathlib import Path
import pandas as pd
from typing import List, Tuple, Dict
import time
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.training.config import (
    DEVICE, SEQUENCE_LENGTH, LANDMARK_FEATURES, NUM_CLASSES,
    CONFIDENCE_THRESHOLD, TOP_K_PREDICTIONS, DATA_DIR, MODEL_DIR,
    MOTION_THRESHOLD, IDLE_THRESHOLD, MIN_SIGN_FRAMES, IDLE_FRAMES_TO_STOP, START_FRAMES
)
from src.models.ultra_simple import SimpleLSTM

# MediaPipe Tasks API
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class RealTimePredictor:
    """
    Real-time sign language recognition using motion-based segmentation
    Matches web app (pytorch_predictor.py) logic exactly
    """

    def __init__(
        self,
        model_path: str = None,
        device: str = DEVICE,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        motion_threshold: float = MOTION_THRESHOLD,
        idle_threshold: float = IDLE_THRESHOLD,
        min_sign_frames: int = MIN_SIGN_FRAMES,
        idle_frames_to_stop: int = IDLE_FRAMES_TO_STOP,
        start_frames: int = START_FRAMES,
    ):
        self.device = device
        self.confidence_threshold = confidence_threshold

        # Load model (SimpleLSTM - same as web app)
        self.model = self._load_model(model_path)
        self.model.eval()

        # Load class labels
        self.class_labels = self._load_class_labels()

        # Initialize MediaPipe Tasks API (separate landmarkers like training)
        self._download_models()
        self._init_mediapipe()

        # Motion-based segmentation state (matches web app)
        self.state = "idle"
        self.sign_frames = []
        self.idle_frames = 0
        self.signing_frames = 0
        self.prev_landmarks = None

        # Thresholds — config.py'den geliyor, tum platformlarda ayni
        self.MOTION_THRESHOLD    = motion_threshold
        self.IDLE_THRESHOLD      = idle_threshold
        self.MIN_SIGN_FRAMES     = min_sign_frames
        self.MAX_SIGN_FRAMES     = SEQUENCE_LENGTH
        self.IDLE_FRAMES_TO_STOP = idle_frames_to_stop
        self.START_FRAMES        = start_frames

        # Last predictions for display
        self.last_predictions = []
        self.TEMPERATURE = 1.5
        self.MARGIN_THRESHOLD = 0.15
        self.MIN_HAND_FRAMES_DIVISOR = 8
        self.POSE_START_THRESHOLD = 0.0045
        self.PRE_BUFFER_SIZE = 8
        self.pre_buffer = deque(maxlen=self.PRE_BUFFER_SIZE)
        self.VOTE_HISTORY_SIZE = 3
        self.prediction_history = []
        self.COOLDOWN_FRAMES = 20
        self.cooldown_counter = 0
        self.timestamp_ms = 0

    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load trained SimpleLSTM model"""
        model = SimpleLSTM(
            input_size=LANDMARK_FEATURES,
            num_classes=NUM_CLASSES
        )

        if model_path is None:
            model_path = MODEL_DIR / "best_model.pth"

        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            acc = checkpoint.get('best_val_acc', 'N/A')
            print(f"✓ Model loaded from {model_path} (val acc: {acc})")
        else:
            print(f"⚠ Model not found at {model_path}, using untrained model")

        return model.to(self.device)

    def _download_models(self):
        """Download MediaPipe model files if not present"""
        self.mp_model_dir = Path(__file__).parent.parent / "data" / "models"
        self.mp_model_dir.mkdir(parents=True, exist_ok=True)

        models = {
            "pose_landmarker_heavy.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
            "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        }

        for filename, url in models.items():
            filepath = self.mp_model_dir / filename
            if not filepath.exists():
                print(f"  Downloading {filename}...")
                urllib.request.urlretrieve(url, filepath)

    def _init_mediapipe(self):
        """Initialize separate PoseLandmarker + HandLandmarker in VIDEO mode."""
        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(self.mp_model_dir / "pose_landmarker_heavy.task")
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
            min_tracking_confidence=0.3
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(self.mp_model_dir / "hand_landmarker.task")
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

        print("MediaPipe Tasks API initialized (IMAGE mode — matches training pipeline)")

    def _load_class_labels(self) -> Dict[int, Tuple[str, str]]:
        """Load class ID to label mapping"""
        import json

        # Try class_mapping.json first
        class_map_path = Path("class_mapping.json")
        if class_map_path.exists():
            with open(class_map_path, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
                return {int(k): (v, v) for k, v in mapping.items()}

        # Try CSV
        csv_path = DATA_DIR / "SignList_ClassId_TR_EN.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            return {row['ClassId']: (row['TR'], row['EN'])
                    for _, row in df.iterrows()}

        return {i: (f"Class_{i}", f"Class_{i}") for i in range(NUM_CLASSES)}

    def extract_landmarks(self, frame) -> Tuple[np.ndarray, tuple]:
        """Extract landmarks using MediaPipe Tasks API - returns 258 features."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        self.timestamp_ms += 33
        pose_result = self.pose_landmarker.detect_for_video(mp_image, self.timestamp_ms)
        hand_result = self.hand_landmarker.detect_for_video(mp_image, self.timestamp_ms)

        landmarks = []

        # Pose landmarks (33 * 4 = 132 features)
        if pose_result.pose_landmarks:
            for lm in pose_result.pose_landmarks[0]:
                landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            landmarks.extend([0.0] * 132)

        # Hand landmarks (21 * 3 = 63 per hand)
        left_hand_found = False
        right_hand_found = False
        left_coords = []
        right_coords = []

        if hand_result.hand_landmarks and hand_result.handedness:
            for hand_landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
                hand_label = handedness[0].category_name

                if hand_label == "Left" and not left_hand_found:
                    left_hand_found = True
                    for lm in hand_landmarks:
                        left_coords.extend([lm.x, lm.y, lm.z])
                elif hand_label == "Right" and not right_hand_found:
                    right_hand_found = True
                    for lm in hand_landmarks:
                        right_coords.extend([lm.x, lm.y, lm.z])

        # Add left hand
        if left_hand_found:
            landmarks.extend(left_coords)
        else:
            landmarks.extend([0.0] * 63)

        # Add right hand
        if right_hand_found:
            landmarks.extend(right_coords)
        else:
            landmarks.extend([0.0] * 63)

        return np.array(landmarks, dtype=np.float32), (pose_result, hand_result)

    def _compute_motion(self, landmarks: np.ndarray) -> float:
        """Compute motion magnitude from hand landmarks."""
        if self.prev_landmarks is None:
            self.prev_landmarks = landmarks.copy()
            return 0.0

        # Focus on hand landmarks (132-258)
        curr_hands = landmarks[132:]
        prev_hands = self.prev_landmarks[132:]
        
        # Sol el (0-62) ve sağ el (63-125) ayrı ayrı kontrol et
        left_motion = np.mean(np.abs(curr_hands[:63] - prev_hands[:63]))
        right_motion = np.mean(np.abs(curr_hands[63:] - prev_hands[63:]))
        
        # En yüksek hareketi al (tek el yeterli)
        self.prev_landmarks = landmarks.copy()
        return max(left_motion, right_motion)

    def _build_prediction(self, class_id: int, prob: float,
                          ambiguous: bool = False,
                          low_confidence: bool = False) -> Dict:
        tr_label, en_label = self.class_labels.get(int(class_id), (f"Class_{class_id}", f"Class_{class_id}"))
        return {
            'label_tr': tr_label,
            'label_en': en_label,
            'confidence': round(float(prob) * 100, 1),
            'ambiguous': ambiguous,
            'low_confidence': low_confidence,
            'class_id': int(class_id),
        }

    def _has_hand_landmarks(self, frame: np.ndarray) -> bool:
        return np.abs(frame[132:]).sum() > 0.1

    @torch.no_grad()
    def predict(self, sequence: np.ndarray, return_probs: bool = False):
        """Make prediction (same normalization as web + training)"""
        # Per-sample normalization (matches dataset.py and pytorch_predictor.py)
        mean = sequence.mean()
        std = sequence.std() + 1e-8
        sequence = (sequence - mean) / std

        x = torch.from_numpy(sequence).unsqueeze(0).to(self.device)
        logits = self.model(x)
        scaled_logits = logits / self.TEMPERATURE
        probs = torch.softmax(scaled_logits, dim=1)[0].cpu().numpy()

        top_indices = np.argsort(probs)[-TOP_K_PREDICTIONS:][::-1]
        predictions = [self._build_prediction(int(idx), float(probs[idx])) for idx in top_indices]

        if len(predictions) >= 2:
            margin = (predictions[0]['confidence'] - predictions[1]['confidence']) / 100.0
            if margin < self.MARGIN_THRESHOLD:
                for prediction in predictions:
                    prediction['ambiguous'] = True

        if return_probs:
            return predictions, probs

        return predictions

    def _predict_sign(self) -> List[Dict]:
        """Predict sign from collected frames.
        Lineer interpolasyon kullanir — egitim pipeline (preprocess.py) ile
        ve web (pytorch_predictor.py) ile birebir ayni.
        """
        n = len(self.sign_frames)
        if n == 0:
            return []

        # Kalite kontrolu: gercek (sifir olmayan) frame sayisi
        nonzero = sum(1 for f in self.sign_frames if np.any(f != 0))
        if nonzero < 3:
            return []  # cok az gercel veri
        hand_frames = sum(1 for frame in self.sign_frames if self._has_hand_landmarks(frame))
        min_required_hand_frames = max(3, n // self.MIN_HAND_FRAMES_DIVISOR)
        if hand_frames < min_required_hand_frames:
            self.prediction_history.clear()
            return []

        # Lineer interpolasyon — preprocess.py ile ayni
        indices = np.linspace(0, n - 1, SEQUENCE_LENGTH)
        frames = []
        for idx in indices:
            lower  = int(np.floor(idx))
            upper  = min(int(np.ceil(idx)), n - 1)
            weight = idx - lower
            if lower == upper:
                frames.append(self.sign_frames[lower].copy())
            else:
                interp = ((1 - weight) * self.sign_frames[lower]
                          + weight * self.sign_frames[upper])
                frames.append(interp.astype(np.float32))

        sequence = np.array(frames, dtype=np.float32)
        predictions, probs = self.predict(sequence, return_probs=True)
        predictions = [p for p in predictions if p['confidence'] >= self.confidence_threshold * 100]
        if not predictions:
            return []

        top_class_id = predictions[0]['class_id']
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

        ambiguous = any(p.get('ambiguous', False) for p in predictions)
        ordered_class_ids = [voted_class_id]
        for idx in np.argsort(probs)[::-1]:
            idx = int(idx)
            if idx != voted_class_id and float(probs[idx]) * 100 >= self.confidence_threshold * 100:
                ordered_class_ids.append(idx)
            if len(ordered_class_ids) == TOP_K_PREDICTIONS:
                break

        return [
            self._build_prediction(
                idx,
                float(probs[idx]),
                ambiguous=ambiguous,
                low_confidence=float(probs[idx]) < self.confidence_threshold,
            )
            for idx in ordered_class_ids
        ]

    def process_frame(self, frame) -> Tuple[List[Dict], tuple, str]:
        """
        Process frame with motion-based sign segmentation.
        Returns predictions only when a complete sign is detected.
        """
        landmarks, results = self.extract_landmarks(frame)
        self.pre_buffer.append(landmarks.copy())

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.prev_landmarks = landmarks.copy()
            return [], results, self.state

        motion = self._compute_motion(landmarks)

        predictions = []

        if self.state == "idle":
            if motion > self.MOTION_THRESHOLD:
                self.signing_frames += 1
                if self.signing_frames >= self.START_FRAMES:
                    self.state = "signing"
                    self.sign_frames = [buffered.copy() for buffered in self.pre_buffer]
                    self.prediction_history.clear()
                    self.idle_frames = 0
                    self.signing_frames = 0
            else:
                self.signing_frames = 0

        elif self.state == "signing":
            self.sign_frames.append(landmarks)

            if motion < self.IDLE_THRESHOLD:
                self.idle_frames += 1
                if self.idle_frames >= self.IDLE_FRAMES_TO_STOP:
                    if len(self.sign_frames) >= self.MIN_SIGN_FRAMES:
                        predictions = self._predict_sign()
                        self.last_predictions = predictions
                        if predictions:
                            self.cooldown_counter = self.COOLDOWN_FRAMES
                    self.state = "idle"
                    self.sign_frames = []
                    self.idle_frames = 0
            else:
                # Hareket devam ediyor, idle counter'ı sıfırla
                self.idle_frames = 0

            if len(self.sign_frames) >= self.MAX_SIGN_FRAMES:
                predictions = self._predict_sign()
                self.last_predictions = predictions
                if predictions:
                    self.cooldown_counter = self.COOLDOWN_FRAMES
                self.state = "idle"
                self.sign_frames = []
                self.idle_frames = 0

        return predictions, results, self.state

    def draw_landmarks(self, frame, results):
        """Draw landmarks on frame"""
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

    def _draw_predictions_panel(self, frame, predictions: List[Dict]):
        """Draw predictions panel on frame"""
        h, w = frame.shape[:2]

        panel_width = 350
        panel_height = 180
        panel_x = w - panel_width - 20
        panel_y = 20

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay,
                     (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # Title
        cv2.putText(frame, "Tespit Edilen Kelimeler",
                   (panel_x + 20, panel_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Predictions
        colors = [(46, 204, 113), (52, 152, 219), (155, 89, 182)]

        y_offset = panel_y + 60
        display_preds = predictions if predictions else self.last_predictions
        for i, pred in enumerate(display_preds[:3]):
            color = colors[i] if i < len(colors) else (200, 200, 200)
            conf = pred['confidence']

            text = f"{pred['label_tr'].upper()}"
            cv2.putText(frame, text,
                       (panel_x + 20, y_offset + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            bar_width = int((panel_width - 40) * conf / 100)
            cv2.rectangle(frame,
                         (panel_x + 20, y_offset + 15),
                         (panel_x + 20 + bar_width, y_offset + 30),
                         color, -1)

            cv2.putText(frame, f"%{conf:.1f}",
                       (panel_x + panel_width - 60, y_offset + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            y_offset += 40

    def run_webcam(self, camera_id: int = 0):
        """Run real-time prediction from webcam"""
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("\n" + "=" * 50)
        print("TID Real-time Recognition")
        print("=" * 50)
        print("Press 'Q' to quit")
        print("=" * 50 + "\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Process frame
            predictions, results, sign_state = self.process_frame(frame)

            # Draw landmarks
            frame = self.draw_landmarks(frame, results)

            # Draw state indicator
            color = (0, 255, 0) if sign_state == "signing" else (100, 100, 100)
            label = "KAYIT" if sign_state == "signing" else "Bekliyor..."
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Draw predictions panel
            self._draw_predictions_panel(frame, predictions)

            # Show frame
            cv2.imshow("TID Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def close(self):
        """Release resources"""
        self.pose_landmarker.close()
        self.hand_landmarker.close()


def main():
    """Run real-time prediction"""
    predictor = RealTimePredictor(
        confidence_threshold=0.4,
        motion_threshold=0.0065,
        idle_threshold=0.0070,
        min_sign_frames=12,
        idle_frames_to_stop=8,
        start_frames=1,
    )

    try:
        predictor.run_webcam()
    finally:
        predictor.close()


if __name__ == "__main__":
    main()
