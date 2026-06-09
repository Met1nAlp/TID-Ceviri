"""
Flask Web Application for TID Recognition

İki-thread mimarisi:
  Thread 1 (capture):  Kamerayı ~60fps okur → raw_frame paylaşımlı değişkene yazar
  Thread 2 (process):  Son raw_frame'i alır → MediaPipe + predictor → results paylaşımlı yazar
  generate_frames:     Son raw_frame + son results'ı birleştirir → ~60fps MJPEG serve eder

Sonuç: Video görüntüsü kamera hızında (60fps) akar,
       landmark overlay MediaPipe hızında (~25-30fps) güncellenir.
"""

import sys
import time
from pathlib import Path
from threading import Lock, Thread

import cv2
from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS

sys.path.append(str(Path(__file__).parent.parent))

from app.digit_selection_predictor import DigitSelectionPredictor
from app.pytorch_predictor import PyTorchPredictor


app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# ── Global model state ────────────────────────────────────────────────────────
predictor = None
mobile_predictor = None
selection_predictor = None

# ── Uygulama state (thread-safe) ──────────────────────────────────────────────
_state_lock = Lock()
current_predictions = []
current_sentence = []
sign_state_global = "idle"

# ── Thread 1: Ham kamera frame'i ─────────────────────────────────────────────
_raw_lock  = Lock()
_raw_frame = None          # OpenCV BGR, en son okunan ham kare

# ── Thread 2: Son MediaPipe sonuçları ────────────────────────────────────────
_results_lock    = Lock()
_last_results    = None    # (pose_result, hand_result) tüplü
_last_sign_state = "idle"

# ── MJPEG stream için encode edilmiş frame ────────────────────────────────────
_frame_lock        = Lock()
_latest_frame_bytes = None

# ── Thread lifecycle ─────────────────────────────────────────────────────────
_cap_thread  = None
_proc_thread = None
_threads_running = False

_client_count = 0
_client_lock  = Lock()


# ── Thread 1: Hızlı kamera yakalama ──────────────────────────────────────────
def _capture_loop():
    """
    Sadece kamera okuması yapar. MediaPipe yok, işlem yok.
    raw_frame'e yazarak Thread 2'yi besler.
    """
    global _raw_frame, _threads_running

    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FPS, 60)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # Her zaman en taze frame

    print("[capture] Kamera thread baslatildi")
    while _threads_running:
        success, frame = camera.read()
        if not success:
            time.sleep(0.002)
            continue
        frame = cv2.flip(frame, 1)
        with _raw_lock:
            _raw_frame = frame   # Thread 2 ve generate_frames buradan okur

    camera.release()
    print("[capture] Kamera thread durduruldu")


# ── Thread 2: Yavaş MediaPipe + Predictor ────────────────────────────────────
def _process_loop():
    """
    Son raw_frame'i alır, MediaPipe + predictor çalıştırır.
    Sonuçları _last_results ve sign_state_global'e yazar.
    """
    global _last_results, _last_sign_state, _threads_running
    global current_predictions, current_sentence, sign_state_global

    selection_interrupt_frames = 0
    SIR = 4        # selection_interrupt_required_frames
    SIM = 0.0115   # selection_interrupt_motion_threshold

    print("[process] MediaPipe thread baslatildi")
    while _threads_running:
        # En güncel ham frame'i al
        with _raw_lock:
            frame = _raw_frame
        if frame is None or predictor is None:
            time.sleep(0.005)
            continue

        sign_state = "idle"

        sel_active = bool(
            selection_predictor and selection_predictor.get_status().get("active", False)
        )

        if sel_active:
            landmarks, results = predictor.extract_landmarks(frame)
            selection_event = selection_predictor.process_hand_result(results[1])

            if selection_event and selection_event.get("event") == "selected":
                selected_word = selection_event["candidate"].get("label_tr", "")
                with _state_lock:
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
                sel_motion = predictor.preview_motion(landmarks)
                if sel_motion > SIM:
                    selection_interrupt_frames += 1
                else:
                    selection_interrupt_frames = 0

                if selection_interrupt_frames >= SIR:
                    selection_predictor.cancel("new_sign_started")
                    with _state_lock:
                        current_predictions = []
                    selection_interrupt_frames = 0
                    predictions, _, sign_state = predictor.process_landmarks(landmarks, results)
                    if predictions:
                        with _state_lock:
                            current_predictions = predictions
                        predictor.pre_buffer.clear()
                        selection_predictor.start_selection(predictions)
                else:
                    predictor.prev_landmarks = landmarks.copy()
                    sign_state = "selection"

        else:
            selection_interrupt_frames = 0
            predictions, results, sign_state = predictor.process_frame(frame)
            if predictions:
                with _state_lock:
                    current_predictions = predictions
                predictor.pre_buffer.clear()
                if selection_predictor:
                    selection_predictor.start_selection(predictions)

        # Sonuçları paylaş
        with _results_lock:
            _last_results    = results
            _last_sign_state = sign_state

        with _state_lock:
            sign_state_global = sign_state

    print("[process] MediaPipe thread durduruldu")


def _start_threads():
    global _cap_thread, _proc_thread, _threads_running, _raw_frame, _latest_frame_bytes
    _threads_running = True
    _raw_frame = None
    with _frame_lock:
        pass  # reset

    _cap_thread  = Thread(target=_capture_loop,  daemon=True, name="Capture")
    _proc_thread = Thread(target=_process_loop,  daemon=True, name="Process")
    _cap_thread.start()
    _proc_thread.start()
    print("[server] Capture + Process thread'leri baslatildi")


def _stop_threads():
    global _threads_running
    _threads_running = False
    print("[server] Thread'ler durduruluyor...")


# ── MJPEG stream ──────────────────────────────────────────────────────────────
def generate_frames():
    """
    Thread 1'den gelen son ham frame'i alır,
    Thread 2'den gelen son MediaPipe sonuçlarını üstüne çizer,
    60fps hedefinde JPEG encode edip serve eder.
    """
    global _client_count, _threads_running, _cap_thread, _proc_thread

    with _client_lock:
        _client_count += 1
        if not _threads_running or \
           (_cap_thread and not _cap_thread.is_alive()):
            _start_threads()

    target_interval = 1.0 / 60   # 60fps hedef

    try:
        while _threads_running:
            t0 = time.perf_counter()

            # Son ham frame
            with _raw_lock:
                frame = _raw_frame
            if frame is None:
                time.sleep(0.005)
                continue

            frame = frame.copy()  # Thread güvenliği için kopya

            # Son MediaPipe sonuçları + durum etiketi
            with _results_lock:
                results    = _last_results
                sign_state = _last_sign_state

            if predictor and results is not None:
                frame = predictor.draw_landmarks(frame, results)

            # Durum etiketi
            if sign_state == "signing":
                color, label = (0, 220, 120), "KAYIT"
            elif sign_state == "selection":
                if selection_predictor:
                    rem = selection_predictor.get_status().get("remaining_ms", 0) / 1000.0
                    label = f"SECIM 1/2/3 ({rem:.1f}s)"
                else:
                    label = "SECIM"
                color = (0, 191, 255)
            else:
                color, label = (80, 80, 80), "Bekliyor..."

            # Sol üst köşe metni kaldırıldı — sağ üstteki video badge zaten gösteriyor

            # JPEG encode
            ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
            if ret:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
                )

            elapsed = time.perf_counter() - t0
            wait = target_interval - elapsed
            if wait > 0:
                time.sleep(wait)

    except GeneratorExit:
        pass
    finally:
        with _client_lock:
            _client_count -= 1
            if _client_count <= 0:
                _client_count = 0
                _stop_threads()
                print("[stream] Son istemci ayrildi — thread'ler durduruluyor")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/predictions")
def get_predictions():
    with _state_lock:
        return jsonify(list(current_predictions))


@app.route("/selection_status")
def get_selection_status():
    with _state_lock:
        preds = list(current_predictions)
        sent  = list(current_sentence)

    if selection_predictor is None:
        return jsonify({"active": False, "sentence": sent, "predictions": preds})

    status = selection_predictor.get_status()
    status["sentence"]    = sent
    status["predictions"] = preds
    return jsonify(status)


@app.route("/debug_status")
def get_debug_status():
    if predictor is None:
        return jsonify({})
    debug = predictor.get_debug_status()
    with _state_lock:
        debug["sign_state"] = sign_state_global
    return jsonify(debug)


@app.route("/add_word", methods=["POST"])
def add_word():
    data = request.json
    word = data.get("word", "")
    with _state_lock:
        if word:
            current_sentence.append(word)
    if selection_predictor:
        selection_predictor.cancel("manual_word_added")
    with _state_lock:
        return jsonify({"sentence": list(current_sentence)})


@app.route("/clear_sentence", methods=["POST"])
def clear_sentence():
    with _state_lock:
        current_sentence.clear()
        return jsonify({"sentence": []})


@app.route("/clear_predictions", methods=["POST"])
def clear_predictions():
    with _state_lock:
        current_predictions.clear()
    if selection_predictor:
        selection_predictor.cancel("predictions_cleared")
    return jsonify({"predictions": []})


@app.route("/get_sentence")
def get_sentence():
    with _state_lock:
        return jsonify({"sentence": list(current_sentence)})


@app.route("/remove_word", methods=["POST"])
def remove_word():
    data  = request.json
    index = data.get("index", -1)
    with _state_lock:
        if 0 <= index < len(current_sentence):
            current_sentence.pop(index)
        return jsonify({"sentence": list(current_sentence)})


@app.route("/predict_frame", methods=["POST"])
def predict_frame():
    """Mobile single-frame endpoint."""
    if mobile_predictor is None:
        return jsonify({"error": "Predictor not initialized"}), 503
    if "frame" not in request.files:
        return jsonify({"error": "No frame provided"}), 400
    try:
        import numpy as np
        file  = request.files["frame"]
        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Invalid image"}), 400
        frame = cv2.flip(frame, 1)
        predictions, _, sign_state = mobile_predictor.process_frame(frame)
        return jsonify({
            "predictions": [
                {"label_tr": p["label_tr"], "label_en": p["label_en"], "confidence": p["confidence"]}
                for p in predictions
            ],
            "sign_state": sign_state,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/predict_sign", methods=["POST"])
def predict_sign():
    """Mobile batch endpoint."""
    if mobile_predictor is None:
        return jsonify({"error": "Predictor not initialized"}), 503
    try:
        import numpy as np
        frames = []
        for key in sorted(request.files.keys()):
            file  = request.files[key]
            npimg = np.frombuffer(file.read(), np.uint8)
            frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            if frame is not None:
                frames.append(cv2.flip(frame, 1))

        if len(frames) < 5:
            return jsonify({"error": f"Too few frames: {len(frames)}", "predictions": [], "sign_state": "idle"})

        landmark_sequence = [mobile_predictor.extract_landmarks(f)[0] for f in frames]

        from src.training.config import SEQUENCE_LENGTH
        if len(landmark_sequence) < SEQUENCE_LENGTH:
            while len(landmark_sequence) < SEQUENCE_LENGTH:
                landmark_sequence.append(landmark_sequence[-1])
        else:
            indices = np.linspace(0, len(landmark_sequence) - 1, SEQUENCE_LENGTH, dtype=int)
            landmark_sequence = [landmark_sequence[i] for i in indices]

        predictions = mobile_predictor.predict(np.array(landmark_sequence, dtype=np.float32))
        return jsonify({
            "predictions": [
                {"label_tr": p["label_tr"], "label_en": p["label_en"], "confidence": p["confidence"]}
                for p in predictions
            ],
            "sign_state": "idle",
        })
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc), "predictions": [], "sign_state": "idle"}), 500


@app.route("/ping")
def ping():
    return jsonify({"status": "ok"})


# ── Init ──────────────────────────────────────────────────────────────────────
def init_predictor():
    global predictor, mobile_predictor, selection_predictor
    try:
        print("Web predictor yukleniyor...")
        predictor = PyTorchPredictor(
            enable_temporal_smoothing=True,
            use_video_landmarkers=True,
            swap_handedness=False,
            motion_threshold=0.0095,
            idle_threshold=0.0075,   # 0.0090'dan düşürüldü — orta hızlı hareketlerde yanlış idle olmasın
            min_sign_frames=15,
            idle_frames_to_stop=8,   # 3'ten artırıldı — 8 kare (~265ms) makul bir bekleme
            start_frames=2,
        )
        print("Secim predictor yukleniyor...")
        selection_predictor = DigitSelectionPredictor()
        print("Mobil predictor yukleniyor...")
        mobile_predictor = PyTorchPredictor(enable_temporal_smoothing=False)
        print("Tum predictorlar hazir")
    except Exception as exc:
        print(f"Predictor yuklenirken hata: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  DeepSign TID — Recognition Web Server")
    print("  2-Thread: Capture(60fps) + Process(MediaPipe) ayrı")
    print("  Kamera: arayüz açıldığında başlar, kapanınca durur")
    print("=" * 60)

    init_predictor()
    print("  Modeller hazır. Kamera arayüz açılınca başlayacak.")

    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"\n  Web    : http://localhost:5000")
    print(f"  Mobil  : http://{local_ip}:5000")
    print("=" * 60 + "\n")

    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
