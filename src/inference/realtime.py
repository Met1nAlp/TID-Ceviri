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
from PIL import Image, ImageDraw, ImageFont

sys.path.append(str(Path(__file__).parent.parent.parent))

from app.digit_selection_predictor import DigitSelectionPredictor
from app.pytorch_predictor import PyTorchPredictor
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

        # Thresholds come from config.py and stay consistent across platforms
        self.MOTION_THRESHOLD    = motion_threshold
        self.IDLE_THRESHOLD      = idle_threshold
        self.MIN_SIGN_FRAMES     = min_sign_frames
        self.MIN_DECISION_FRAMES = self.MIN_SIGN_FRAMES
        self.MAX_SIGN_FRAMES     = max(SEQUENCE_LENGTH + 24, 72)
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
            print(f"Model loaded from {model_path} (val acc: {acc})")
        else:
            print(f"Warning: model not found at {model_path}, using untrained model")

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

        print("MediaPipe Tasks API initialized (VIDEO mode, matches runtime pipeline)")

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
        effective_frames = self.sign_frames
        trailing_idle_trim = max(0, self.idle_frames - 2)
        if trailing_idle_trim > 0 and len(self.sign_frames) - trailing_idle_trim >= 1:
            effective_frames = self.sign_frames[:-trailing_idle_trim]

        n = len(effective_frames)
        if n == 0:
            return []

        # Kalite kontrolu: gercek (sifir olmayan) frame sayisi
        nonzero = sum(1 for f in effective_frames if np.any(f != 0))
        if nonzero < 3:
            return []  # cok az gercel veri
        hand_frames = sum(1 for frame in effective_frames if self._has_hand_landmarks(frame))
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
                frames.append(effective_frames[lower].copy())
            else:
                interp = ((1 - weight) * effective_frames[lower]
                          + weight * effective_frames[upper])
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
                    if len(self.sign_frames) >= self.MIN_DECISION_FRAMES:
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


class DesktopWebStyleApp:
    """Desktop dashboard that mirrors the web experience."""

    WINDOW_NAME = "DeepSign-TID Desktop"

    def __init__(self):
        self.predictor = PyTorchPredictor(
            model_path="models/best_model.pth",
            enable_temporal_smoothing=True,
            use_video_landmarkers=True,
            swap_handedness=False,
            motion_threshold=0.0080,
            idle_threshold=0.0060,
            min_sign_frames=15,
            idle_frames_to_stop=7,
            start_frames=2,
            confidence_threshold=0.4,
        )
        self.selection_predictor = self._init_selection_predictor()
        self.current_predictions: List[Dict] = []
        self.current_sentence: List[str] = []
        self.prediction_regions: Dict[int, Tuple[int, int, int, int]] = {}
        self.button_regions: Dict[str, Tuple[int, int, int, int]] = {}
        self._closed = False
        self.ui_colors = {
            "bg_primary": (26, 29, 35),
            "bg_secondary": (37, 41, 50),
            "bg_tertiary": (45, 50, 60),
            "text_primary": (255, 255, 255),
            "text_secondary": (160, 165, 177),
            "accent_green": (46, 204, 113),
            "accent_blue": (52, 152, 219),
            "accent_purple": (155, 89, 182),
            "accent_amber": (243, 156, 18),
            "accent_red": (231, 76, 60),
            "border": (61, 66, 80),
        }
        self._font_cache: Dict[Tuple[int, bool], ImageFont.FreeTypeFont] = {}

    def _init_selection_predictor(self):
        try:
            return DigitSelectionPredictor(device=str(self.predictor.device))
        except Exception as exc:
            print(f"Warning: digit selection model disabled ({exc})")
            return None

    def _font(self, size: int, bold: bool = False):
        key = (size, bold)
        if key in self._font_cache:
            return self._font_cache[key]

        font_candidates = []
        if bold:
            font_candidates.extend(
                [
                    "C:/Windows/Fonts/segoeuib.ttf",
                    "C:/Windows/Fonts/arialbd.ttf",
                    "C:/Windows/Fonts/calibrib.ttf",
                ]
            )
        else:
            font_candidates.extend(
                [
                    "C:/Windows/Fonts/segoeui.ttf",
                    "C:/Windows/Fonts/arial.ttf",
                    "C:/Windows/Fonts/calibri.ttf",
                ]
            )

        for font_path in font_candidates:
            path_obj = Path(font_path)
            if path_obj.exists():
                try:
                    font = ImageFont.truetype(str(path_obj), size=size)
                    self._font_cache[key] = font
                    return font
                except OSError:
                    continue

        font = ImageFont.load_default()
        self._font_cache[key] = font
        return font

    def _measure_text(self, draw, text: str, font) -> Tuple[int, int]:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    def _wrap_text(self, draw, text: str, font, max_width: int, max_lines: int = None) -> List[str]:
        if not text:
            return [""]

        wrapped_lines: List[str] = []
        for paragraph in str(text).splitlines():
            words = paragraph.split()
            if not words:
                wrapped_lines.append("")
                continue

            current = words[0]
            for word in words[1:]:
                trial = f"{current} {word}"
                width, _ = self._measure_text(draw, trial, font)
                if width <= max_width:
                    current = trial
                else:
                    wrapped_lines.append(current)
                    current = word
            wrapped_lines.append(current)

        if max_lines is not None and len(wrapped_lines) > max_lines:
            clipped = wrapped_lines[:max_lines]
            if len(clipped[-1]) > 3:
                clipped[-1] = clipped[-1][:-3].rstrip() + "..."
            else:
                clipped[-1] = clipped[-1] + "..."
            return clipped

        return wrapped_lines

    def _draw_wrapped_text(
        self,
        draw,
        text: str,
        x: int,
        y: int,
        font,
        fill,
        max_width: int,
        line_gap: int = 6,
        max_lines: int = None,
    ) -> int:
        lines = self._wrap_text(draw, text, font, max_width=max_width, max_lines=max_lines)
        _, line_height = self._measure_text(draw, "Ag", font)
        for line in lines:
            draw.text((x, y), line, font=font, fill=fill)
            y += line_height + line_gap
        return y

    def _fit_frame(self, frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = min(target_w / max(w, 1), target_h / max(h, 1))
        resized_w = max(1, int(w * scale))
        resized_h = max(1, int(h * scale))
        return cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    def _selection_snapshot(self) -> Dict:
        if self.selection_predictor is None:
            return {
                "status_text": "Secim modeli kullanilamiyor.",
                "countdown": "-",
                "detected": "-",
                "chosen": "-",
                "highlight_digit": None,
            }

        selection = self.selection_predictor.get_status()
        active = bool(selection.get("active"))
        selected = selection.get("last_selected") or None

        status_text = "Tahmin geldikten sonra 1, 2 veya 3 goster."
        if active:
            status_text = "Top-3 adayi secmek icin 1, 2 veya 3 goster."
        elif selection.get("last_event") == "selected" and selected:
            status_text = f"{selected['digit_value']} ile secildi: {selected['candidate']['label_tr']}"
        elif selection.get("last_event") == "timeout":
            status_text = "Secim zamani doldu. Yeni tahmini bekliyor."
        elif selection.get("last_event") == "cancelled":
            status_text = "Secim iptal edildi."

        detected_parts = []
        if selection.get("last_digit_value"):
            detected_parts.append(str(selection["last_digit_value"]))
        elif selection.get("last_digit_label") == "other_digit":
            detected_parts.append("DIGER")
        else:
            detected_parts.append("-")

        last_confidence = selection.get("last_confidence", 0)
        if isinstance(last_confidence, (int, float)) and last_confidence > 0:
            detected_parts.append(f"%{last_confidence:.1f}")

        chosen_text = "-"
        if selected:
            chosen_text = (
                f"{selected['digit_value']} -> {selected['candidate']['label_tr']} "
                f"(%{selected['confidence']})"
            )

        highlight_digit = None
        if active:
            highlight_digit = selection.get("stable_digit") or selection.get("last_digit_value")

        countdown = f"{selection.get('remaining_ms', 0) / 1000.0:.1f} sn" if active else "-"
        return {
            "status_text": status_text,
            "countdown": countdown,
            "detected": " | ".join(detected_parts),
            "chosen": chosen_text,
            "highlight_digit": highlight_digit,
        }

    def _camera_backends(self):
        backends = []
        if sys.platform.startswith("win") and hasattr(cv2, "CAP_DSHOW"):
            backends.append(("DirectShow", cv2.CAP_DSHOW))
        if hasattr(cv2, "CAP_MSMF"):
            backends.append(("MSMF", cv2.CAP_MSMF))
        backends.append(("Default", None))
        return backends

    def _open_camera(self, camera_id: int):
        for backend_name, backend in self._camera_backends():
            cap = cv2.VideoCapture(camera_id, backend) if backend is not None else cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            for _ in range(10):
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"Camera backend: {backend_name}")
                    return cap
                time.sleep(0.05)

            cap.release()

        return None

    def _handle_click(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONUP:
            return

        for action, rect in self.button_regions.items():
            if self._point_in_rect(x, y, rect):
                if action == "clear_predictions":
                    self.clear_predictions()
                elif action == "clear_sentence":
                    self.clear_sentence()
                return

        for index, rect in self.prediction_regions.items():
            if self._point_in_rect(x, y, rect):
                self.select_prediction(index)
                return

    @staticmethod
    def _point_in_rect(x: int, y: int, rect: Tuple[int, int, int, int]) -> bool:
        left, top, right, bottom = rect
        return left <= x <= right and top <= y <= bottom

    def clear_predictions(self):
        self.current_predictions = []
        if self.selection_predictor:
            self.selection_predictor.cancel("predictions_cleared")

    def clear_sentence(self):
        self.current_sentence = []

    def select_prediction(self, index: int):
        if index < 0 or index >= len(self.current_predictions):
            return

        candidate = dict(self.current_predictions[index])
        self.current_sentence.append(candidate["label_tr"])

        if self.selection_predictor:
            self.selection_predictor.active = False
            self.selection_predictor.vote_history.clear()
            self.selection_predictor.last_event = "selected"
            self.selection_predictor.last_reason = "manual_selection"
            self.selection_predictor.last_selected = {
                "digit_value": index + 1,
                "candidate_index": index,
                "candidate": candidate,
                "confidence": candidate.get("confidence", 0.0),
            }
            self.selection_predictor.selection_serial += 1

        self.current_predictions = []

    def _render_dashboard(self, frame: np.ndarray, sign_state: str) -> np.ndarray:
        width, height = 1540, 920
        canvas = Image.new("RGB", (width, height), self.ui_colors["bg_primary"])
        draw = ImageDraw.Draw(canvas)

        self.prediction_regions = {}
        self.button_regions = {}

        header_rect = (20, 20, width - 20, 90)
        draw.rounded_rectangle(
            header_rect,
            radius=18,
            fill=self.ui_colors["bg_secondary"],
            outline=self.ui_colors["border"],
            width=2,
        )
        draw.text(
            (header_rect[0] + 30, header_rect[1] + 18),
            "TURK ISARET DILI CEVIRI SISTEMI",
            font=self._font(30, bold=True),
            fill=self.ui_colors["text_primary"],
        )
        draw.text(
            (header_rect[0] + 32, header_rect[1] + 52),
            "Desktop canli tanima arayuzu - web panel duzeni",
            font=self._font(15),
            fill=self.ui_colors["text_secondary"],
        )

        left_x = 20
        left_w = 1060
        right_x = 1105
        right_w = width - right_x - 20

        video_panel = (left_x, 115, left_x + left_w, 690)
        draw.rounded_rectangle(
            video_panel,
            radius=18,
            fill=self.ui_colors["bg_secondary"],
            outline=self.ui_colors["border"],
            width=2,
        )

        video_padding = 18
        video_area = (
            video_panel[0] + video_padding,
            video_panel[1] + video_padding,
            video_panel[2] - video_padding,
            video_panel[3] - video_padding,
        )
        fitted = self._fit_frame(frame, video_area[2] - video_area[0], video_area[3] - video_area[1])
        fitted_rgb = cv2.cvtColor(fitted, cv2.COLOR_BGR2RGB)
        fitted_pil = Image.fromarray(fitted_rgb)
        paste_x = video_area[0] + ((video_area[2] - video_area[0]) - fitted.shape[1]) // 2
        paste_y = video_area[1] + ((video_area[3] - video_area[1]) - fitted.shape[0]) // 2
        canvas.paste(fitted_pil, (paste_x, paste_y))

        if sign_state == "signing":
            status_fill = self.ui_colors["accent_green"]
            status_text = "KAYIT"
        elif sign_state == "selection":
            status_fill = self.ui_colors["accent_blue"]
            status_text = "SECIM 1/2/3"
        else:
            status_fill = (100, 100, 100)
            status_text = "BEKLIYOR"

        badge_rect = (video_panel[0] + 22, video_panel[1] + 20, video_panel[0] + 230, video_panel[1] + 62)
        draw.rounded_rectangle(badge_rect, radius=14, fill=status_fill)
        draw.text(
            (badge_rect[0] + 18, badge_rect[1] + 9),
            status_text,
            font=self._font(20, bold=True),
            fill=(255, 255, 255),
        )

        helper_text = "Q: Cikis   C: Kelimeleri temizle   X: Cumleyi temizle   1/2/3: Manuel secim"
        draw.text(
            (video_panel[0] + 26, video_panel[3] - 32),
            helper_text,
            font=self._font(15),
            fill=self.ui_colors["text_secondary"],
        )

        sentence_panel = (left_x, 715, left_x + left_w, 900)
        draw.rounded_rectangle(
            sentence_panel,
            radius=18,
            fill=self.ui_colors["bg_secondary"],
            outline=self.ui_colors["border"],
            width=2,
        )
        draw.text(
            (sentence_panel[0] + 26, sentence_panel[1] + 18),
            "Cumle",
            font=self._font(24, bold=True),
            fill=self.ui_colors["text_primary"],
        )
        draw.text(
            (sentence_panel[0] + 26, sentence_panel[1] + 52),
            "Tahmine tikla veya 1/2/3 gostererek secim yap.",
            font=self._font(15),
            fill=self.ui_colors["text_secondary"],
        )

        chip_x = sentence_panel[0] + 26
        chip_y = sentence_panel[1] + 92
        chip_max_x = sentence_panel[2] - 26
        if not self.current_sentence:
            draw.text(
                (chip_x, chip_y + 10),
                "Kelime eklemek icin tahmine tiklayin veya secim hareketi yapin.",
                font=self._font(18),
                fill=self.ui_colors["text_secondary"],
            )
        else:
            for word in self.current_sentence:
                chip_font = self._font(18, bold=True)
                text_w, text_h = self._measure_text(draw, word, chip_font)
                chip_w = text_w + 30
                chip_h = text_h + 16
                if chip_x + chip_w > chip_max_x:
                    chip_x = sentence_panel[0] + 26
                    chip_y += chip_h + 10
                if chip_y + chip_h > sentence_panel[3] - 18:
                    draw.text(
                        (chip_x, chip_y),
                        "...",
                        font=self._font(18, bold=True),
                        fill=self.ui_colors["text_primary"],
                    )
                    break
                chip_rect = (chip_x, chip_y, chip_x + chip_w, chip_y + chip_h)
                draw.rounded_rectangle(
                    chip_rect,
                    radius=16,
                    fill=(52, 94, 165),
                    outline=(126, 177, 255),
                    width=1,
                )
                draw.text(
                    (chip_x + 15, chip_y + 7),
                    word,
                    font=chip_font,
                    fill=(255, 255, 255),
                )
                chip_x += chip_w + 10

        right_panel = (right_x, 115, right_x + right_w, 900)
        draw.rounded_rectangle(
            right_panel,
            radius=18,
            fill=self.ui_colors["bg_secondary"],
            outline=self.ui_colors["border"],
            width=2,
        )
        draw.text(
            (right_panel[0] + 26, right_panel[1] + 20),
            "Tespit Edilen Kelimeler",
            font=self._font(24, bold=True),
            fill=self.ui_colors["text_primary"],
        )

        prediction_colors = [
            self.ui_colors["accent_green"],
            self.ui_colors["accent_blue"],
            self.ui_colors["accent_purple"],
        ]
        selection_snapshot = self._selection_snapshot()
        highlight_digit = selection_snapshot["highlight_digit"]
        card_y = right_panel[1] + 68

        for index in range(3):
            card_rect = (right_panel[0] + 20, card_y, right_panel[2] - 20, card_y + 108)
            card_fill = self.ui_colors["bg_tertiary"]
            outline = self.ui_colors["border"]
            if highlight_digit == index + 1:
                outline = self.ui_colors["accent_amber"]
            draw.rounded_rectangle(card_rect, radius=14, fill=card_fill, outline=outline, width=2)

            accent_rect = (card_rect[0], card_rect[1], card_rect[0] + 8, card_rect[3])
            draw.rounded_rectangle(accent_rect, radius=14, fill=prediction_colors[index])

            prediction = self.current_predictions[index] if index < len(self.current_predictions) else None
            label = f"{index + 1}. -"
            confidence = 0.0
            if prediction:
                label = f"{index + 1}. {prediction['label_tr']}"
                confidence = float(prediction["confidence"])
                self.prediction_regions[index] = card_rect

            draw.text(
                (card_rect[0] + 24, card_rect[1] + 16),
                label,
                font=self._font(20, bold=True),
                fill=self.ui_colors["text_primary"],
            )

            bar_rect = (card_rect[0] + 24, card_rect[1] + 56, card_rect[2] - 24, card_rect[1] + 70)
            draw.rounded_rectangle(bar_rect, radius=7, fill=(66, 71, 85))
            if confidence > 0:
                fill_w = int((bar_rect[2] - bar_rect[0]) * min(confidence, 100.0) / 100.0)
                draw.rounded_rectangle(
                    (bar_rect[0], bar_rect[1], bar_rect[0] + max(fill_w, 14), bar_rect[3]),
                    radius=7,
                    fill=prediction_colors[index],
                )

            draw.text(
                (card_rect[0] + 24, card_rect[1] + 78),
                f"%{confidence:.1f}",
                font=self._font(16),
                fill=self.ui_colors["text_secondary"],
            )
            card_y += 126

        selection_rect = (right_panel[0] + 20, card_y + 6, right_panel[2] - 20, card_y + 210)
        draw.rounded_rectangle(
            selection_rect,
            radius=14,
            fill=(31, 56, 43),
            outline=(67, 133, 96),
            width=2,
        )
        draw.text(
            (selection_rect[0] + 20, selection_rect[1] + 16),
            "1-2-3 ILE SEC",
            font=self._font(18, bold=True),
            fill=self.ui_colors["accent_green"],
        )
        next_y = self._draw_wrapped_text(
            draw,
            selection_snapshot["status_text"],
            selection_rect[0] + 20,
            selection_rect[1] + 52,
            self._font(16),
            self.ui_colors["text_primary"],
            max_width=selection_rect[2] - selection_rect[0] - 40,
            max_lines=3,
        )

        info_items = [
            ("Kalan Sure", selection_snapshot["countdown"]),
            ("Algilanan Sayi", selection_snapshot["detected"]),
            ("Son Secim", selection_snapshot["chosen"]),
        ]
        info_y = max(next_y + 10, selection_rect[1] + 116)
        box_width = (selection_rect[2] - selection_rect[0] - 50) // 2
        for item_index, (label, value) in enumerate(info_items):
            is_wide = item_index == 2
            box_x = selection_rect[0] + 20
            box_y = info_y + (item_index // 2) * 76
            if item_index == 1:
                box_x = selection_rect[0] + 30 + box_width
                box_y = info_y
            if is_wide:
                box_x = selection_rect[0] + 20
                box_y = info_y + 76
                item_width = selection_rect[2] - selection_rect[0] - 40
            else:
                item_width = box_width

            item_rect = (box_x, box_y, box_x + item_width, box_y + 62)
            draw.rounded_rectangle(
                item_rect,
                radius=10,
                fill=self.ui_colors["bg_tertiary"],
            )
            draw.text(
                (item_rect[0] + 12, item_rect[1] + 10),
                label,
                font=self._font(12),
                fill=self.ui_colors["text_secondary"],
            )
            self._draw_wrapped_text(
                draw,
                value,
                item_rect[0] + 12,
                item_rect[1] + 28,
                self._font(15, bold=True),
                self.ui_colors["text_primary"],
                max_width=item_rect[2] - item_rect[0] - 24,
                max_lines=2,
                line_gap=2,
            )

        buttons = [
            ("clear_predictions", "Kelimeleri Temizle", self.ui_colors["bg_tertiary"], self.ui_colors["text_primary"]),
            ("clear_sentence", "Cumleyi Temizle", (192, 57, 43), (255, 255, 255)),
        ]
        button_y = right_panel[3] - 118
        for index, (action, label, fill, text_fill) in enumerate(buttons):
            btn_rect = (right_panel[0] + 20, button_y + index * 54, right_panel[2] - 20, button_y + index * 54 + 42)
            draw.rounded_rectangle(btn_rect, radius=12, fill=fill, outline=self.ui_colors["border"], width=1)
            text_w, text_h = self._measure_text(draw, label, self._font(16, bold=True))
            draw.text(
                (btn_rect[0] + ((btn_rect[2] - btn_rect[0]) - text_w) // 2, btn_rect[1] + ((btn_rect[3] - btn_rect[1]) - text_h) // 2 - 1),
                label,
                font=self._font(16, bold=True),
                fill=text_fill,
            )
            self.button_regions[action] = btn_rect

        footer_text = "Fare ile tahmin kartina tiklayabilir, 1/2/3 ile manuel secim yapabilirsin."
        draw.text(
            (right_panel[0] + 20, right_panel[3] - 28),
            footer_text,
            font=self._font(13),
            fill=self.ui_colors["text_secondary"],
        )

        return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)

    def run(self, camera_id: int = 0):
        cap = self._open_camera(camera_id)
        if cap is None:
            print("Error: Could not open webcam or read frames from any camera backend")
            return

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, 1540, 920)
        cv2.setMouseCallback(self.WINDOW_NAME, self._handle_click)

        frame_failures = 0
        selection_interrupt_frames = 0
        selection_interrupt_required_frames = 4
        selection_interrupt_motion_threshold = 0.0115

        print("\n" + "=" * 60)
        print("DeepSign-TID Desktop Dashboard")
        print("Q: quit | C: clear predictions | X: clear sentence | 1/2/3: select candidate")
        print("=" * 60 + "\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    frame_failures += 1
                    if frame_failures >= 15:
                        print("Error: webcam stopped returning frames")
                        break
                    time.sleep(0.03)
                    continue

                frame_failures = 0

                frame = cv2.flip(frame, 1)
                sign_state = "idle"

                selection_active = bool(
                    self.selection_predictor and self.selection_predictor.get_status().get("active", False)
                )

                if selection_active:
                    landmarks, results = self.predictor.extract_landmarks(frame)
                    selection_event = self.selection_predictor.process_hand_result(results[1])

                    if selection_event and selection_event.get("event") == "selected":
                        selected_word = selection_event["candidate"].get("label_tr", "")
                        if selected_word:
                            self.current_sentence.append(selected_word)
                        self.current_predictions = []
                        selection_interrupt_frames = 0
                        self.predictor.prev_landmarks = landmarks.copy()
                        sign_state = "selection"
                    elif selection_event and selection_event.get("event") == "timeout":
                        selection_interrupt_frames = 0
                        self.predictor.prev_landmarks = landmarks.copy()
                        sign_state = "idle"
                    elif self.selection_predictor.is_interrupt_guard_active():
                        selection_interrupt_frames = 0
                        self.predictor.prev_landmarks = landmarks.copy()
                        sign_state = "selection"
                    elif self.selection_predictor.has_digit_evidence():
                        selection_interrupt_frames = 0
                        self.predictor.prev_landmarks = landmarks.copy()
                        sign_state = "selection"
                    else:
                        selection_motion = self.predictor.preview_motion(landmarks)
                        if selection_motion > selection_interrupt_motion_threshold:
                            selection_interrupt_frames += 1
                        else:
                            selection_interrupt_frames = 0

                        if selection_interrupt_frames >= selection_interrupt_required_frames:
                            self.selection_predictor.cancel("new_sign_started")
                            self.current_predictions = []
                            selection_interrupt_frames = 0
                            predictions, _, sign_state = self.predictor.process_landmarks(landmarks, results)
                            if predictions:
                                self.current_predictions = predictions
                                self.predictor.pre_buffer.clear()
                                self.selection_predictor.start_selection(self.current_predictions)
                        else:
                            self.predictor.prev_landmarks = landmarks.copy()
                            sign_state = "selection"

                    frame = self.predictor.draw_landmarks(frame, results)
                else:
                    selection_interrupt_frames = 0
                    predictions, results, sign_state = self.predictor.process_frame(frame)
                    if predictions:
                        self.current_predictions = predictions
                        self.predictor.pre_buffer.clear()
                        if self.selection_predictor:
                            self.selection_predictor.start_selection(self.current_predictions)
                    frame = self.predictor.draw_landmarks(frame, results)

                dashboard = self._render_dashboard(frame, sign_state)
                cv2.imshow(self.WINDOW_NAME, dashboard)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    self.clear_predictions()
                if key == ord("x"):
                    self.clear_sentence()
                if key in (ord("1"), ord("2"), ord("3")):
                    self.select_prediction(int(chr(key)) - 1)
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.close()

    def close(self):
        if self._closed:
            return
        self.predictor.pose_landmarker.close()
        self.predictor.hand_landmarker.close()
        self._closed = True


def main():
    """Run real-time prediction"""
    app = DesktopWebStyleApp()

    try:
        app.run()
    finally:
        app.close()


if __name__ == "__main__":
    main()
