"""
Flask Web Application for TID Recognition
Uses PyTorch models for both word prediction and digit-based candidate selection.
"""

import sys
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS
from threading import Lock

sys.path.append(str(Path(__file__).parent.parent))

from app.digit_selection_predictor import DigitSelectionPredictor
from app.pytorch_predictor import PyTorchPredictor


app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Global state
predictor = None
mobile_predictor = None
selection_predictor = None
camera = None
camera_lock = Lock()
current_predictions = []
current_sentence = []


def generate_frames():
    global predictor, camera, current_predictions, current_sentence, selection_predictor

    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    selection_interrupt_frames = 0
    selection_interrupt_required_frames = 4
    selection_interrupt_motion_threshold = 0.0115

    while True:
        with camera_lock:
            success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        if predictor:
            selection_active = bool(
                selection_predictor and selection_predictor.get_status().get("active", False)
            )

            if selection_active:
                landmarks, results = predictor.extract_landmarks(frame)
                selection_event = selection_predictor.process_hand_result(results[1])

                if selection_event and selection_event.get("event") == "selected":
                    selected_word = selection_event["candidate"].get("label_tr", "")
                    if selected_word:
                        current_sentence.append(selected_word)
                    current_predictions = []
                    selection_interrupt_frames = 0
                    predictor.prev_landmarks = landmarks.copy()
                    sign_state = "selection"
                elif selection_event and selection_event.get("event") == "timeout":
                    selection_interrupt_frames = 0
                    predictor.prev_landmarks = landmarks.copy()
                    sign_state = "idle"
                elif selection_predictor.is_interrupt_guard_active():
                    selection_interrupt_frames = 0
                    predictor.prev_landmarks = landmarks.copy()
                    sign_state = "selection"
                elif selection_predictor.has_digit_evidence():
                    selection_interrupt_frames = 0
                    predictor.prev_landmarks = landmarks.copy()
                    sign_state = "selection"
                else:
                    selection_motion = predictor.preview_motion(landmarks)
                    if selection_motion > selection_interrupt_motion_threshold:
                        selection_interrupt_frames += 1
                    else:
                        selection_interrupt_frames = 0

                    if selection_interrupt_frames >= selection_interrupt_required_frames:
                        selection_predictor.cancel("new_sign_started")
                        current_predictions = []
                        selection_interrupt_frames = 0
                        predictions, _, sign_state = predictor.process_landmarks(landmarks, results)
                        if predictions:
                            current_predictions = predictions
                            predictor.pre_buffer.clear()
                            selection_predictor.start_selection(current_predictions)
                    else:
                        predictor.prev_landmarks = landmarks.copy()
                        sign_state = "selection"

                frame = predictor.draw_landmarks(frame, results)
            else:
                selection_interrupt_frames = 0
                predictions, results, sign_state = predictor.process_frame(frame)
                if predictions:
                    current_predictions = predictions
                    predictor.pre_buffer.clear()
                    if selection_predictor:
                        selection_predictor.start_selection(current_predictions)
                frame = predictor.draw_landmarks(frame, results)

            if sign_state == "signing":
                color = (0, 255, 0)
                label = "KAYIT"
            elif sign_state == "selection":
                color = (0, 191, 255)
                selection_status = selection_predictor.get_status() if selection_predictor else {}
                remaining_seconds = selection_status.get("remaining_ms", 0) / 1000.0
                label = f"SECIM 1/2/3 ({remaining_seconds:.1f}s)"
            else:
                color = (100, 100, 100)
                label = "Bekliyor..."

            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/predictions")
def get_predictions():
    global current_predictions
    return jsonify(current_predictions)


@app.route("/selection_status")
def get_selection_status():
    global selection_predictor, current_sentence, current_predictions

    if selection_predictor is None:
        return jsonify(
            {
                "active": False,
                "sentence": current_sentence,
                "predictions": current_predictions,
            }
        )

    status = selection_predictor.get_status()
    status["sentence"] = current_sentence
    status["predictions"] = current_predictions
    return jsonify(status)


@app.route("/debug_status")
def get_debug_status():
    global predictor
    if predictor is None:
        return jsonify({})
    return jsonify(predictor.get_debug_status())


@app.route("/add_word", methods=["POST"])
def add_word():
    global current_sentence, selection_predictor
    data = request.json
    word = data.get("word", "")
    if word:
        current_sentence.append(word)
        if selection_predictor:
            selection_predictor.cancel("manual_word_added")
    return jsonify({"sentence": current_sentence})


@app.route("/clear_sentence", methods=["POST"])
def clear_sentence():
    global current_sentence
    current_sentence = []
    return jsonify({"sentence": current_sentence})


@app.route("/clear_predictions", methods=["POST"])
def clear_predictions():
    global current_predictions, selection_predictor
    current_predictions = []
    if selection_predictor:
        selection_predictor.cancel("predictions_cleared")
    return jsonify({"predictions": current_predictions})


@app.route("/get_sentence")
def get_sentence():
    global current_sentence
    return jsonify({"sentence": current_sentence})


@app.route("/remove_word", methods=["POST"])
def remove_word():
    global current_sentence
    data = request.json
    index = data.get("index", -1)

    if 0 <= index < len(current_sentence):
        current_sentence.pop(index)

    return jsonify({"sentence": current_sentence})


@app.route("/predict_frame", methods=["POST"])
def predict_frame():
    """Mobile single-frame endpoint. Left untouched."""
    global mobile_predictor
    if mobile_predictor is None:
        return jsonify({"error": "Predictor not initialized"}), 503
    if "frame" not in request.files:
        return jsonify({"error": "No frame provided"}), 400

    try:
        import numpy as np

        file = request.files["frame"]
        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Invalid image"}), 400
        frame = cv2.flip(frame, 1)
        predictions, _, sign_state = mobile_predictor.process_frame(frame)
        return jsonify(
            {
                "predictions": [
                    {
                        "label_tr": p["label_tr"],
                        "label_en": p["label_en"],
                        "confidence": p["confidence"],
                    }
                    for p in predictions
                ],
                "sign_state": sign_state,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/predict_sign", methods=["POST"])
def predict_sign():
    """
    Mobile batch endpoint. Left untouched.
    """
    global mobile_predictor
    if mobile_predictor is None:
        return jsonify({"error": "Predictor not initialized"}), 503

    try:
        import numpy as np

        frames = []
        for key in sorted(request.files.keys()):
            file = request.files[key]
            npimg = np.frombuffer(file.read(), np.uint8)
            frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            if frame is not None:
                frame = cv2.flip(frame, 1)
                frames.append(frame)

        if len(frames) < 5:
            return jsonify(
                {
                    "error": f"Too few frames: {len(frames)}",
                    "predictions": [],
                    "sign_state": "idle",
                }
            )

        landmark_sequence = []
        for frame in frames:
            landmarks, _ = mobile_predictor.extract_landmarks(frame)
            landmark_sequence.append(landmarks)

        from src.training.config import SEQUENCE_LENGTH

        if len(landmark_sequence) < SEQUENCE_LENGTH:
            while len(landmark_sequence) < SEQUENCE_LENGTH:
                landmark_sequence.append(landmark_sequence[-1])
        else:
            indices = np.linspace(0, len(landmark_sequence) - 1, SEQUENCE_LENGTH, dtype=int)
            landmark_sequence = [landmark_sequence[i] for i in indices]

        sequence = np.array(landmark_sequence, dtype=np.float32)
        predictions = mobile_predictor.predict(sequence)

        return jsonify(
            {
                "predictions": [
                    {
                        "label_tr": p["label_tr"],
                        "label_en": p["label_en"],
                        "confidence": p["confidence"],
                    }
                    for p in predictions
                ],
                "sign_state": "idle",
            }
        )

    except Exception as exc:
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(exc), "predictions": [], "sign_state": "idle"}), 500


@app.route("/ping")
def ping():
    return jsonify({"status": "ok"})


def init_predictor():
    global predictor, mobile_predictor, selection_predictor
    try:
        print("Web predictor yukleniyor...")
        predictor = PyTorchPredictor(
            enable_temporal_smoothing=True,
            use_video_landmarkers=True,
            swap_handedness=False,
            motion_threshold=0.0080,
            idle_threshold=0.0060,
            min_sign_frames=15,
            idle_frames_to_stop=7,
            start_frames=2,
        )
        print("Secim predictor yukleniyor...")
        selection_predictor = DigitSelectionPredictor()
        print("Mobil predictor yukleniyor...")
        mobile_predictor = PyTorchPredictor(enable_temporal_smoothing=False)
        print("Her iki predictor hazir")
    except Exception as exc:
        print(f"Error initializing predictor: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("TID Recognition Web Server (PyTorch Model)")
    print("GPU-Accelerated Inference")
    print("Web + Mobil destegi")
    print("=" * 50)

    init_predictor()

    import socket

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    print(f"\nServer running at http://localhost:5000")
    print(f"Mobil baglanti: http://{local_ip}:5000")
    print("=" * 50 + "\n")

    import logging

    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
