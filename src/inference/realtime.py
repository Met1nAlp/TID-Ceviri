"""
Real-time Sign Language Recognition with Sliding Window
Provides Top-3 predictions for webcam input
"""

import cv2
import numpy as np
import torch
import mediapipe as mp
from collections import deque
from pathlib import Path
import pandas as pd
from typing import List, Tuple, Dict
import time
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.training.config import (
    DEVICE, SEQUENCE_LENGTH, LANDMARK_FEATURES, NUM_CLASSES,
    SLIDING_WINDOW_SIZE, SLIDING_WINDOW_STRIDE,
    CONFIDENCE_THRESHOLD, TOP_K_PREDICTIONS,
    DATA_DIR, MODEL_DIR
)
from src.models.hybrid_model import get_model


class RealTimePredictor:
    """
    Real-time sign language recognition using sliding window approach
    """
    
    def __init__(
        self,
        model_path: str = None,
        model_type: str = "landmark_only",
        device: str = DEVICE,
        window_size: int = SLIDING_WINDOW_SIZE,
        stride: int = SLIDING_WINDOW_STRIDE,
        confidence_threshold: float = CONFIDENCE_THRESHOLD
    ):
        self.device = device
        self.window_size = window_size
        self.stride = stride
        self.confidence_threshold = confidence_threshold
        self.model_type = model_type
        
        # Load model
        self.model = self._load_model(model_path, model_type)
        self.model.eval()
        
        # Load class labels
        self.class_labels = self._load_class_labels()
        
        # Initialize MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,  # Medium complexity for real-time
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Frame buffer for sliding window
        self.frame_buffer = deque(maxlen=window_size)
        
        # Frame counter for stride control
        self.frame_count = 0
        
        # Prediction cache
        self.last_predictions = []
        self.last_prediction_time = 0
        
        # Drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
    
    def _load_model(self, model_path: str, model_type: str) -> torch.nn.Module:
        """Load trained model"""
        model = get_model(model_type, use_visual=False)
        
        if model_path is None:
            model_path = MODEL_DIR / "best_model.pth"
        
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded model from {model_path}")
        else:
            print(f"⚠ Model not found at {model_path}, using untrained model")
        
        return model.to(self.device)
    
    def _load_class_labels(self) -> Dict[int, Tuple[str, str]]:
        """Load class ID to label mapping"""
        class_map_path = DATA_DIR / "SignList_ClassId_TR_EN.csv"
        
        if class_map_path.exists():
            df = pd.read_csv(class_map_path)
            return {row['ClassId']: (row['TR'], row['EN']) 
                    for _, row in df.iterrows()}
        else:
            return {i: (f"Class_{i}", f"Class_{i}") for i in range(NUM_CLASSES)}
    
    def extract_landmarks(self, frame) -> np.ndarray:
        """Extract landmarks from a single frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb_frame)
        
        landmarks = []
        
        # Pose landmarks
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            landmarks.extend([0.0] * 132)
        
        # Left hand
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 63)
        
        # Right hand
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 63)
        
        return np.array(landmarks, dtype=np.float32), results
    
    @torch.no_grad()
    def predict(self, landmarks_sequence: np.ndarray) -> List[Tuple[str, str, float]]:
        """
        Make prediction from landmark sequence
        
        Returns:
            List of (turkish_label, english_label, confidence) tuples
        """
        # Ensure correct shape
        if landmarks_sequence.shape[0] < self.window_size:
            # Pad if needed
            pad_size = self.window_size - landmarks_sequence.shape[0]
            padding = np.zeros((pad_size, LANDMARK_FEATURES), dtype=np.float32)
            landmarks_sequence = np.vstack([landmarks_sequence, padding])
        
        # Convert to tensor
        x = torch.from_numpy(landmarks_sequence).unsqueeze(0).to(self.device)
        
        # Forward pass
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = probs.topk(TOP_K_PREDICTIONS, dim=1)
        
        predictions = []
        for i in range(TOP_K_PREDICTIONS):
            idx = top_indices[0, i].item()
            prob = top_probs[0, i].item()
            
            if idx in self.class_labels:
                tr_label, en_label = self.class_labels[idx]
            else:
                tr_label, en_label = f"Class_{idx}", f"Class_{idx}"
            
            predictions.append((tr_label, en_label, prob))
        
        return predictions
    
    def process_frame(self, frame) -> Tuple[List[Tuple[str, str, float]], any]:
        """
        Process a single frame and return predictions if available
        
        Returns:
            predictions: Top-3 predictions or empty list
            results: MediaPipe results for visualization
        """
        # Extract landmarks
        landmarks, results = self.extract_landmarks(frame)
        
        # Add to buffer
        self.frame_buffer.append(landmarks)
        self.frame_count += 1
        
        predictions = self.last_predictions
        
        # Make prediction every stride frames if buffer is full
        if len(self.frame_buffer) >= self.window_size and self.frame_count >= self.stride:
            self.frame_count = 0
            
            # Get sequence from buffer
            sequence = np.array(list(self.frame_buffer))
            
            # Predict
            predictions = self.predict(sequence)
            
            # Filter by confidence
            predictions = [p for p in predictions if p[2] >= self.confidence_threshold]
            
            self.last_predictions = predictions
        
        return predictions, results
    
    def draw_landmarks(self, frame, results):
        """Draw landmarks on frame for visualization"""
        # Draw pose landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
        
        # Draw left hand
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
            )
        
        # Draw right hand
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
        
        return frame
    
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
            predictions, results = self.process_frame(frame)
            
            # Draw landmarks
            frame = self.draw_landmarks(frame, results)
            
            # Draw predictions panel
            self._draw_predictions_panel(frame, predictions)
            
            # Show frame
            cv2.imshow("TID Recognition", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.holistic.close()
    
    def _draw_predictions_panel(self, frame, predictions: List[Tuple[str, str, float]]):
        """Draw predictions panel on frame"""
        h, w = frame.shape[:2]
        
        # Panel dimensions
        panel_width = 350
        panel_height = 180
        panel_x = w - panel_width - 20
        panel_y = 20
        
        # Draw semi-transparent background
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
        
        # Draw predictions
        colors = [(46, 204, 113), (52, 152, 219), (155, 89, 182)]  # Green, Blue, Purple
        
        y_offset = panel_y + 60
        for i, (tr_label, en_label, conf) in enumerate(predictions[:3]):
            color = colors[i] if i < len(colors) else (200, 200, 200)
            
            # Label text
            text = f"{tr_label.upper()}"
            cv2.putText(frame, text,
                       (panel_x + 20, y_offset + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Confidence bar
            bar_width = int((panel_width - 40) * conf)
            cv2.rectangle(frame,
                         (panel_x + 20, y_offset + 15),
                         (panel_x + 20 + bar_width, y_offset + 30),
                         color, -1)
            
            # Confidence percentage
            cv2.putText(frame, f"%{conf*100:.1f}",
                       (panel_x + panel_width - 60, y_offset + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            y_offset += 40
        
        # Instructions
        cv2.putText(frame, "Q: Cikis",
                   (20, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    def close(self):
        """Release resources"""
        self.holistic.close()


def main():
    """Run real-time prediction"""
    predictor = RealTimePredictor(
        model_type="landmark_only",
        confidence_threshold=0.5
    )
    
    try:
        predictor.run_webcam()
    finally:
        predictor.close()


if __name__ == "__main__":
    main()
