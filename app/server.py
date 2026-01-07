"""
Flask Web Application for TID Recognition
Uses pre-trained H5 model with OLD MediaPipe Solutions API
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from collections import deque
import json
from threading import Lock

sys.path.append(str(Path(__file__).parent.parent))

# OLD MediaPipe Solutions API
import mediapipe as mp


# Custom AttentionLayer for loading H5 model
class AttentionLayer(tf.keras.layers.Layer):
    """Özel Attention katmanı"""
    
    def __init__(self, units, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        self.W = Dense(units, activation='tanh')
        self.U = Dense(1, activation='softmax')
        
    def call(self, inputs):
        attention_scores = self.W(inputs)
        attention_weights = self.U(attention_scores)
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)
        return context_vector, attention_weights
    
    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config

# Flask app
app = Flask(__name__, 
           template_folder='templates',
           static_folder='static')
CORS(app)

# Global variables
predictor = None
camera = None
camera_lock = Lock()
current_predictions = []
current_sentence = []

# Model config - from the H5 model (30 frames, 258 features)
SEQUENCE_LENGTH = 30  # Model expects 30 frames
LANDMARK_FEATURES = 258  # Model expects 258 features (pose 33*4 + hands 21*3*2)
SLIDING_WINDOW_STRIDE = 10


class KerasPredictor:
    """Predictor using pre-trained Keras model with OLD MediaPipe API"""
    
    def __init__(self, model_path="models/best_model.h5"):
        # Load Keras model with custom objects
        print(f"Loading model from {model_path}...")
        self.model = tf.keras.models.load_model(
            model_path, 
            custom_objects={'AttentionLayer': AttentionLayer}
        )
        print(f"✓ Model loaded! Input shape: {self.model.input_shape}")
        
        # Load class labels
        self.class_labels = self._load_class_labels()
        
        # Initialize OLD MediaPipe Holistic
        print("Initializing MediaPipe Holistic...")
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        print("✓ MediaPipe initialized!")
        
        # Buffer
        self.frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.frame_count = 0
    
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
        
        return {i: (f"class_{i}", f"class_{i}") for i in range(226)}
    
    def extract_landmarks(self, frame):
        """Extract landmarks using OLD MediaPipe Holistic - returns 258 features"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb)
        
        landmarks = []
        
        # Pose landmarks (33 * 4 = 132 features: x, y, z, visibility)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            landmarks.extend([0.0] * 132)
        
        # Left hand (21 * 3 = 63 features: x, y, z)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 63)
        
        # Right hand (21 * 3 = 63 features: x, y, z)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 63)
        
        # Total: 132 + 63 + 63 = 258 features
        return np.array(landmarks, dtype=np.float32), results
    
    def predict(self, sequence):
        """Make prediction with Keras model"""
        x = np.expand_dims(sequence, axis=0)
        probs = self.model.predict(x, verbose=0)[0]
        
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
    
    def process_frame(self, frame):
        landmarks, results = self.extract_landmarks(frame)
        self.frame_buffer.append(landmarks)
        self.frame_count += 1
        
        predictions = []
        
        if len(self.frame_buffer) >= SEQUENCE_LENGTH and self.frame_count >= SLIDING_WINDOW_STRIDE:
            self.frame_count = 0
            sequence = np.array(list(self.frame_buffer))
            predictions = self.predict(sequence)
        
        return predictions, results
    
    def draw_landmarks(self, frame, results):
        """Draw landmarks using OLD MediaPipe drawing utils"""
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2)
            )
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2)
            )
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=2)
            )
        return frame


def generate_frames():
    global predictor, camera, current_predictions
    
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    while True:
        with camera_lock:
            success, frame = camera.read()
        
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        
        if predictor:
            predictions, results = predictor.process_frame(frame)
            if predictions:
                current_predictions = predictions
            frame = predictor.draw_landmarks(frame, results)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/predictions')
def get_predictions():
    global current_predictions
    return jsonify(current_predictions)


@app.route('/add_word', methods=['POST'])
def add_word():
    global current_sentence
    data = request.json
    word = data.get('word', '')
    if word:
        current_sentence.append(word)
    return jsonify({'sentence': current_sentence})


@app.route('/clear_sentence', methods=['POST'])
def clear_sentence():
    global current_sentence
    current_sentence = []
    return jsonify({'sentence': current_sentence})


@app.route('/get_sentence')
def get_sentence():
    global current_sentence
    return jsonify({'sentence': current_sentence})


@app.route('/remove_word', methods=['POST'])
def remove_word():
    global current_sentence
    if current_sentence:
        current_sentence.pop()
    return jsonify({'sentence': current_sentence})


def init_predictor():
    global predictor
    try:
        predictor = KerasPredictor()
    except Exception as e:
        print(f"Error initializing predictor: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("TID Recognition Web Server (Keras H5 Model)")
    print("Using OLD MediaPipe Solutions API")
    print("=" * 50)
    
    init_predictor()
    
    print("\n✓ Server running at http://localhost:5000")
    print("=" * 50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
