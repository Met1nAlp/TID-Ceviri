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
predictor = None
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
        predictor = PyTorchPredictor()
    except Exception as e:
        print(f"Error initializing predictor: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("TID Recognition Web Server (PyTorch MLP Model)")
    print("GPU-Accelerated Inference")
    print("=" * 50)
    
    init_predictor()
    
    print("\n✓ Server running at http://localhost:5000")
    print("=" * 50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
