"""
MediaPipe Landmark Extraction
Extracts pose and hand landmarks from sign language videos
Uses IMAGE mode for each frame - more reliable detection
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.training.config import (
    DATA_DIR, PROCESSED_DIR, 
    SEQUENCE_LENGTH, TRAIN_CSV, VAL_CSV, TEST_CSV
)

# MediaPipe imports
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class LandmarkExtractor:
    """Extract pose and hand landmarks using MediaPipe Tasks API - IMAGE mode"""
    
    def __init__(self):
        # Download model files if needed
        self._download_models()
        
        # Create pose landmarker - IMAGE mode for each frame
        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(self.model_dir / "pose_landmarker_heavy.task")
            ),
            running_mode=vision.RunningMode.IMAGE,  # Use IMAGE mode
            num_poses=1,
            min_pose_detection_confidence=0.3,  # Lower threshold
            min_pose_presence_confidence=0.3
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
        
        # Create hand landmarker - IMAGE mode
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(self.model_dir / "hand_landmarker.task")
            ),
            running_mode=vision.RunningMode.IMAGE,  # Use IMAGE mode
            num_hands=2,
            min_hand_detection_confidence=0.3,  # Lower threshold
            min_hand_presence_confidence=0.3
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
    
    def _download_models(self):
        """Download MediaPipe model files if not present"""
        import urllib.request
        
        self.model_dir = Path(__file__).parent / "models"
        self.model_dir.mkdir(exist_ok=True)
        
        models = {
            "pose_landmarker_heavy.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
            "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        }
        
        for filename, url in models.items():
            filepath = self.model_dir / filename
            if not filepath.exists():
                print(f"Downloading {filename}...")
                urllib.request.urlretrieve(url, filepath)
                print(f"Downloaded {filename}")
    
    def extract_landmarks(self, frame):
        """
        Extract landmarks from a single frame using IMAGE mode
        Returns array of shape (258,)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        landmarks = []
        
        # Pose landmarks (33 * 4 = 132 features)
        try:
            pose_result = self.pose_landmarker.detect(mp_image)
            if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
                for lm in pose_result.pose_landmarks[0]:
                    visibility = lm.visibility if hasattr(lm, 'visibility') and lm.visibility is not None else 1.0
                    landmarks.extend([lm.x, lm.y, lm.z, visibility])
            else:
                landmarks.extend([0.0] * 132)
        except Exception as e:
            landmarks.extend([0.0] * 132)
        
        # Hand landmarks (21 * 3 = 63 features per hand)
        try:
            hand_result = self.hand_landmarker.detect(mp_image)
            
            left_hand = [0.0] * 63
            right_hand = [0.0] * 63
            
            if hand_result.hand_landmarks:
                for i, hand_landmarks in enumerate(hand_result.hand_landmarks):
                    if i < len(hand_result.handedness):
                        handedness = hand_result.handedness[i][0].category_name
                        hand_data = []
                        for lm in hand_landmarks:
                            hand_data.extend([lm.x, lm.y, lm.z])
                        
                        if handedness == "Left":
                            left_hand = hand_data[:63]
                        else:
                            right_hand = hand_data[:63]
            
            landmarks.extend(left_hand)
            landmarks.extend(right_hand)
        except Exception as e:
            landmarks.extend([0.0] * 126)
        
        return np.array(landmarks, dtype=np.float32)
    
    def process_video(self, video_path, target_frames=SEQUENCE_LENGTH):
        """
        Process a video file and extract landmarks for all frames
        Returns array of shape (target_frames, 258)
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
        
        frames_landmarks = []
        frame_count = 0
        valid_frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            landmarks = self.extract_landmarks(frame)
            frames_landmarks.append(landmarks)
            frame_count += 1
            
            # Check if we got valid landmarks (not all zeros)
            if np.any(landmarks != 0):
                valid_frame_count += 1
        
        cap.release()
        
        if len(frames_landmarks) == 0:
            return None
        
        # Convert to numpy array
        landmarks_array = np.stack(frames_landmarks)
        
        # Pad or sample to target_frames
        landmarks_array = self._normalize_sequence_length(
            landmarks_array, target_frames
        )
        
        return landmarks_array
    
    def _normalize_sequence_length(self, landmarks, target_frames):
        """
        Normalize sequence length to target_frames using interpolation
        """
        current_frames = landmarks.shape[0]
        
        if current_frames == target_frames:
            return landmarks
        
        # Use linear interpolation for temporal normalization
        indices = np.linspace(0, current_frames - 1, target_frames)
        normalized = np.zeros((target_frames, landmarks.shape[1]), dtype=np.float32)
        
        for i, idx in enumerate(indices):
            lower = int(np.floor(idx))
            upper = min(int(np.ceil(idx)), current_frames - 1)
            weight = idx - lower
            
            if lower == upper:
                normalized[i] = landmarks[lower]
            else:
                normalized[i] = (1 - weight) * landmarks[lower] + weight * landmarks[upper]
        
        return normalized
    
    def close(self):
        """Release MediaPipe resources"""
        self.pose_landmarker.close()
        self.hand_landmarker.close()


def process_dataset(csv_path, video_dir, output_dir, split_name):
    """
    Process all videos in a dataset split
    """
    import sys
    
    # Read CSV
    df = pd.read_csv(csv_path, header=None, names=['video', 'label'])
    
    # Create output directory
    output_path = Path(output_dir) / split_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize extractor
    extractor = LandmarkExtractor()
    
    # Process videos
    successful = 0
    failed = 0
    empty_landmarks = 0
    skipped = 0
    
    print(f"\nProcessing {split_name} split ({len(df)} videos)...")
    sys.stdout.flush()
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=split_name, 
                         file=sys.stdout, ncols=100, mininterval=1.0):
        video_name = row['video']
        label = row['label']
        
        # Skip if already processed (for resume functionality)
        output_file = output_path / f"{video_name.replace('.mp4', '')}.npy"
        if output_file.exists():
            skipped += 1
            continue
        
        video_path = Path(video_dir) / split_name / video_name
        
        if not video_path.exists():
            print(f"\nVideo not found: {video_name}")
            sys.stdout.flush()
            failed += 1
            continue
        
        try:
            landmarks = extractor.process_video(video_path)
            
            if landmarks is not None:
                # Check if landmarks have meaningful data
                if np.mean(np.abs(landmarks)) > 0.01:
                    # Save landmarks
                    np.save(output_file, landmarks)
                    successful += 1
                else:
                    empty_landmarks += 1
                    failed += 1
            else:
                # Video couldn't be opened or processed
                failed += 1
                
        except Exception as e:
            # Print error but continue processing
            print(f"\nError processing {video_name}: {str(e)[:100]}")
            sys.stdout.flush()
            failed += 1
    
    extractor.close()
    
    print(f"\n{split_name}: Skipped: {skipped}, Successful: {successful}, Failed: {failed}, Empty: {empty_landmarks}")
    sys.stdout.flush()
    
    # Save metadata - only for successful files
    # Re-read to get only successful ones
    successful_files = [f.stem for f in (output_path).glob("*.npy")]
    metadata_rows = []
    for _, row in df.iterrows():
        video_stem = row['video'].replace('.mp4', '')
        if video_stem in successful_files:
            metadata_rows.append({'video': video_stem, 'label': row['label']})
    
    metadata = pd.DataFrame(metadata_rows)
    metadata.to_csv(output_path / "metadata.csv", index=False)
    
    return successful, failed


def main():
    """Main function to process all dataset splits"""
    print("=" * 60)
    print("MediaPipe Landmark Extraction for AUTSL Dataset")
    print("Using IMAGE mode for reliable detection")
    print("Supports RESUME - existing files will be skipped")
    print("=" * 60)
    
    # Process each split
    splits = [
        (TRAIN_CSV, DATA_DIR, PROCESSED_DIR, "train"),
        (VAL_CSV, DATA_DIR, PROCESSED_DIR, "val"),
        (TEST_CSV, DATA_DIR, PROCESSED_DIR, "test"),
    ]
    
    total_successful = 0
    total_failed = 0
    
    for csv_path, video_dir, output_dir, split_name in splits:
        successful, failed = process_dataset(csv_path, video_dir, output_dir, split_name)
        total_successful += successful
        total_failed += failed
    
    print("\n" + "=" * 60)
    print(f"Total: Successful: {total_successful}, Failed: {total_failed}")
    print("=" * 60)


if __name__ == "__main__":
    main()
