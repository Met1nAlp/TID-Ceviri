"""
Flask Web Application for TID Recognition
Uses PyTorch MLP Model
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
from threading import Lock

sys.path.append(str(Path(__file__).parent.parent))

# Import PyTorch predictor
from app.pytorch_predictor import PyTorchPredictor



# Flask app
app = Flask(__name__, 
           template_folder='templates',
           static_folder='static')
CORS(app)

# Global variables
predictor = None          # Web predictor
mobile_predictor = None   # Mobil predictor (ayrı state)
camera = None
camera_lock = Lock()
current_predictions = []
current_sentence = []


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
            predictions, results, sign_state = predictor.process_frame(frame)
            if predictions:
                current_predictions = predictions
            frame = predictor.draw_landmarks(frame, results)
            
            # Show state indicator on frame
            color = (0, 255, 0) if sign_state == "signing" else (100, 100, 100)
            label = "🔴 KAYIT" if sign_state == "signing" else "Bekliyor..."
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
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


@app.route('/debug_status')
def get_debug_status():
    global predictor
    if predictor is None:
        return jsonify({})
    return jsonify(predictor.get_debug_status())


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
    data = request.json
    index = data.get('index', -1)
    
    # Tıklanan kelimeyi sil
    if 0 <= index < len(current_sentence):
        current_sentence.pop(index)
    
    return jsonify({'sentence': current_sentence})


@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    """Mobile: tek frame gönder, motion state machine ile çalışır"""
    global mobile_predictor
    if mobile_predictor is None:
        return jsonify({'error': 'Predictor not initialized'}), 503
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame provided'}), 400
    try:
        import numpy as np
        file = request.files['frame']
        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': 'Invalid image'}), 400
        frame = cv2.flip(frame, 1)
        predictions, _, sign_state = mobile_predictor.process_frame(frame)
        return jsonify({
            'predictions': [{'label_tr': p['label_tr'], 'label_en': p['label_en'], 'confidence': p['confidence']} for p in predictions],
            'sign_state': sign_state
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_sign', methods=['POST'])
def predict_sign():
    """
    Mobil batch endpoint: birden fazla JPEG frame alır, 
    Python MediaPipe ile landmark çıkarır, tahmin yapar.
    Motion detection MOBILE tarafında yapılır.
    """
    global mobile_predictor
    if mobile_predictor is None:
        return jsonify({'error': 'Predictor not initialized'}), 503

    try:
        import numpy as np

        # Multipart'tan tüm frame'leri al
        frames = []
        for key in sorted(request.files.keys()):
            file = request.files[key]
            npimg = np.frombuffer(file.read(), np.uint8)
            frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            if frame is not None:
                frame = cv2.flip(frame, 1)
                frames.append(frame)

        if len(frames) < 5:
            return jsonify({'error': f'Too few frames: {len(frames)}', 'predictions': [], 'sign_state': 'idle'})

        # Her frame'den landmark çıkar (Python MediaPipe — eğitimle aynı)
        landmark_sequence = []
        for f in frames:
            landmarks, _ = mobile_predictor.extract_landmarks(f)
            landmark_sequence.append(landmarks)

        # SEQUENCE_LENGTH'e pad/trim
        from src.training.config import SEQUENCE_LENGTH
        if len(landmark_sequence) < SEQUENCE_LENGTH:
            while len(landmark_sequence) < SEQUENCE_LENGTH:
                landmark_sequence.append(landmark_sequence[-1])
        else:
            indices = np.linspace(0, len(landmark_sequence) - 1, SEQUENCE_LENGTH, dtype=int)
            landmark_sequence = [landmark_sequence[i] for i in indices]

        sequence = np.array(landmark_sequence, dtype=np.float32)
        predictions = mobile_predictor.predict(sequence)

        return jsonify({
            'predictions': [
                {'label_tr': p['label_tr'], 'label_en': p['label_en'], 'confidence': p['confidence']}
                for p in predictions
            ],
            'sign_state': 'idle'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'predictions': [], 'sign_state': 'idle'}), 500


@app.route('/ping')
def ping():
    """Bağlantı kontrolü"""
    return jsonify({'status': 'ok'})


def init_predictor():
    global predictor, mobile_predictor
    try:
        print("Web predictor yükleniyor...")
        predictor = PyTorchPredictor(
            enable_temporal_smoothing=True,
            use_video_landmarkers=True,
            swap_handedness=False,
            motion_threshold=0.0065,
            idle_threshold=0.0070,
            min_sign_frames=12,
            idle_frames_to_stop=8,
            start_frames=1
        )
        print("Mobil predictor yükleniyor...")
        mobile_predictor = PyTorchPredictor(enable_temporal_smoothing=False)
        print("✓ Her iki predictor hazır")
    except Exception as e:
        print(f"Error initializing predictor: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("TID Recognition Web Server (PyTorch MLP Model)")
    print("GPU-Accelerated Inference")
    print("Web + Mobil desteği")
    print("=" * 50)
    
    init_predictor()
    
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print(f"\n✓ Server running at http://localhost:5000")
    print(f"✓ Mobil bağlantı: http://{local_ip}:5000")
    print("=" * 50 + "\n")
    
    # Flask log'larını kapat
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
