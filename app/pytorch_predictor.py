"""
PyTorch Predictor for Web Application - using MediaPipe Tasks API
"""

import sys
from pathlib import Path
import torch
import numpy as np
import cv2
from collections import deque
import json
import urllib.request

sys.path.append(str(Path(__file__).parent.parent))

from src.models.ultra_simple import SimpleLSTM
from src.training.config import SEQUENCE_LENGTH, LANDMARK_FEATURES, NUM_CLASSES

# MediaPipe Tasks API
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image as mp_Image, ImageFormat


class PyTorchPredictor:
    """Predictor using PyTorch model with MediaPipe Tasks API"""
    
    def __init__(self, model_path="models/best_model.pth", device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load PyTorch model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model
        self.model = SimpleLSTM()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"✓ Model loaded! Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Load class labels
        self.class_labels = self._load_class_labels()
        
        # Initialize MediaPipe Tasks
        print("Initializing MediaPipe...")
        self._download_models()
        self._init_mediapipe()
        print("✓ MediaPipe initialized!")
        
        # Frame buffer for sign detection
        self.frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.prev_landmarks = None
        
        # Motion-based segmentation state
        self.state = "idle"          # idle → signing → cooldown
        self.sign_frames = []        # frames collected during signing
        self.idle_frames = 0         # consecutive still frames
        self.signing_frames = 0      # consecutive moving frames
        
        # Thresholds
        self.MOTION_THRESHOLD = 0.015    # min movement to detect sign start
        self.IDLE_THRESHOLD = 0.005      # max movement to detect sign end
        self.MIN_SIGN_FRAMES = 15        # min frames for valid sign (~0.5s)
        self.MAX_SIGN_FRAMES = SEQUENCE_LENGTH  # max frames
        self.IDLE_FRAMES_TO_STOP = 8    # still frames before sign ends
        self.START_FRAMES = 3            # moving frames before sign starts
    
    def _download_models(self):
        """Download MediaPipe model files if not present"""
        self.model_dir = Path(__file__).parent.parent / "src" / "data" / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        models = {
            "pose_landmarker_heavy.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
            "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        }
        
        for filename, url in models.items():
            filepath = self.model_dir / filename
            if not filepath.exists():
                print(f"  Downloading {filename}...")
                urllib.request.urlretrieve(url, filepath)
    
    def _init_mediapipe(self):
        """Initialize MediaPipe landmarkers in VIDEO mode"""
        # Pose landmarker - VIDEO mode
        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(self.model_dir / "pose_landmarker_heavy.task")
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
        
        # Hand landmarker - VIDEO mode
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(self.model_dir / "hand_landmarker.task")
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
        
        self.timestamp_ms = 0
    
    def _load_class_labels(self):
        # Try to load from class_mapping.json first
        class_map_path = Path("class_mapping.json")
        if class_map_path.exists():
            with open(class_map_path, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
                return {int(k): (v, v) for k, v in mapping.items()}
        
        # Fall back to CSV
        csv_path = Path("AUTSL/SignList_ClassId_TR_EN.csv")
        if csv_path.exists():
            import pandas as pd
            df = pd.read_csv(csv_path)
            return {row['ClassId']: (row['TR'], row['EN']) for _, row in df.iterrows()}
        
        return {i: (f"class_{i}", f"class_{i}") for i in range(NUM_CLASSES)}
    
    def extract_landmarks(self, frame):
        """Extract landmarks using MediaPipe Tasks - returns 258 features"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp_Image(image_format=ImageFormat.SRGB, data=rgb_frame)
        
        # Process with pose landmarker
        self.timestamp_ms += 33  # Approximate 30 FPS
        pose_result = self.pose_landmarker.detect_for_video(mp_image, self.timestamp_ms)
        hand_result = self.hand_landmarker.detect_for_video(mp_image, self.timestamp_ms)
        
        landmarks = []
        
        # Pose landmarks (33 * 4 = 132 features: x, y, z, visibility)
        if pose_result.pose_landmarks:
            for lm in pose_result.pose_landmarks[0]:
                landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            landmarks.extend([0.0] * 132)
        
        # Left hand (21 * 3 = 63 features: x, y, z)
        left_hand_found = False
        right_hand_found = False
        
        if hand_result.hand_landmarks and hand_result.handedness:
            for hand_landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
                hand_label = handedness[0].category_name
                
                hand_coords = [lm.x for lm in hand_landmarks] + [lm.y for lm in hand_landmarks] + [lm.z for lm in hand_landmarks]
                
                if hand_label == "Left" and not left_hand_found:
                    left_hand_found = True
                    left_coords = []
                    for lm in hand_landmarks:
                        left_coords.extend([lm.x, lm.y, lm.z])
                elif hand_label == "Right" and not right_hand_found:
                    right_hand_found = True
                    right_coords = []
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
    
    @torch.no_grad()
    def predict(self, sequence):
        """Make prediction with PyTorch model"""
        # Normalize sequence
        mean = sequence.mean()
        std = sequence.std() + 1e-8
        sequence = (sequence - mean) / std
        
        # Convert to tensor
        x = torch.from_numpy(sequence).unsqueeze(0).to(self.device)
        
        # Predict
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        
        # Get top 3
        top_indices = np.argsort(probs)[-3:][::-1]
        
        predictions = []
        for idx in top_indices:
            prob = probs[idx]
            tr_label, en_label = self.class_labels.get(int(idx), (f"class_{idx}", f"class_{idx}"))
            predictions.append({
                'label_tr': tr_label,
                'label_en': en_label,
                'confidence': round(float(prob) * 100, 1)
            })
        
        return predictions
    
    def _compute_motion(self, landmarks):
        """Compute motion magnitude between current and previous landmarks"""
        if self.prev_landmarks is None:
            self.prev_landmarks = landmarks
            return 0.0
        
        # Focus on hand landmarks (indices 132-258) - most informative
        curr_hands = landmarks[132:]
        prev_hands = self.prev_landmarks[132:]
        
        # Mean absolute difference
        motion = np.mean(np.abs(curr_hands - prev_hands))
        self.prev_landmarks = landmarks.copy()
        return motion
    
    def process_frame(self, frame):
        """
        Process frame with motion-based sign segmentation.
        Returns predictions only when a complete sign is detected.
        """
        landmarks, results = self.extract_landmarks(frame)
        motion = self._compute_motion(landmarks)
        
        predictions = []
        status = self.state  # for UI feedback
        
        if self.state == "idle":
            # Waiting for sign to start
            if motion > self.MOTION_THRESHOLD:
                self.signing_frames += 1
                if self.signing_frames >= self.START_FRAMES:
                    # Sign started!
                    self.state = "signing"
                    self.sign_frames = []
                    self.idle_frames = 0
                    self.signing_frames = 0
            else:
                self.signing_frames = 0
        
        elif self.state == "signing":
            # Collecting sign frames
            self.sign_frames.append(landmarks)
            
            if motion < self.IDLE_THRESHOLD:
                self.idle_frames += 1
                if self.idle_frames >= self.IDLE_FRAMES_TO_STOP:
                    # Sign ended - predict!
                    if len(self.sign_frames) >= self.MIN_SIGN_FRAMES:
                        predictions = self._predict_sign()
                    self.state = "idle"
                    self.sign_frames = []
                    self.idle_frames = 0
            else:
                self.idle_frames = 0
            
            # Force predict if too many frames
            if len(self.sign_frames) >= self.MAX_SIGN_FRAMES:
                predictions = self._predict_sign()
                self.state = "idle"
                self.sign_frames = []
                self.idle_frames = 0
        
        return predictions, results, self.state
    
    def _predict_sign(self):
        """Predict sign from collected frames"""
        frames = self.sign_frames
        
        # Pad or trim to SEQUENCE_LENGTH
        if len(frames) < SEQUENCE_LENGTH:
            # Pad by repeating last frame
            while len(frames) < SEQUENCE_LENGTH:
                frames.append(frames[-1])
        else:
            # Take evenly spaced frames
            indices = np.linspace(0, len(frames)-1, SEQUENCE_LENGTH, dtype=int)
            frames = [frames[i] for i in indices]
        
        sequence = np.array(frames, dtype=np.float32)
        return self.predict(sequence)
    
    def draw_landmarks(self, frame, results):
        """Draw landmarks on frame (simple drawing)"""
        pose_result, hand_result = results
        
        # Draw pose landmarks
        if pose_result.pose_landmarks:
            for lm in pose_result.pose_landmarks[0]:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 3, (245, 117, 66), -1)
        
        # Draw hand landmarks
        if hand_result.hand_landmarks:
            for hand_landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
                color = (121, 22, 76) if handedness[0].category_name == "Left" else (80, 110, 10)
                for lm in hand_landmarks:
                    x = int(lm.x * frame.shape[1])
                    y = int(lm.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 3, color, -1)
        
        return frame
