# -*- coding: utf-8 -*-
"""
Faxe - Face Recognition Flask App
- UI:            GET  /
- Health:        GET  /health
- Predict:       POST /predict           (multipart/form-data 'image')
                 POST /api/predict       (alias)
- PredictFrame:  POST /predict_frame     (JSON {"image": "data:image/jpeg;base64,..."} )
                 POST /api/predict_frame (alias)
"""

# ====== Reduce TF logs & force CPU (Windows w/o CUDA) ======
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import base64
import io
from typing import List, Dict, Any, Tuple

import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template

try:
    from flask_cors import CORS
    _HAS_CORS = True
except Exception:
    _HAS_CORS = False

import tensorflow as tf
from tensorflow import keras

# ====== ABSOLUTE PATHS TO .h5 & .txt (theo yêu cầu) ======
MODEL_PATH  = r"C:\New folder\lonai\faxe\converted_keras\keras_model.h5"
LABELS_PATH = r"C:\New folder\lonai\faxe\converted_keras\labels.txt"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Không tìm thấy model: {MODEL_PATH}")
if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(f"Không tìm thấy labels: {LABELS_PATH}")

# ====== Load model & infer input size ======
model = keras.models.load_model(MODEL_PATH)

def _infer_img_size(m) -> int:
    try:
        shp = m.input_shape
        if isinstance(shp, list):
            shp = shp[0]
        h, w = int(shp[1]), int(shp[2])
        if h == w and h > 0:
            return h
    except Exception:
        pass
    return 160  # fallback

IMG_SIZE = _infer_img_size(model)

# ====== Load labels (.txt supports "0 label" or "label") ======
def load_labels_txt(path: str) -> List[str]:
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split(" ", 1)
            if len(parts) == 2 and parts[0].isdigit():
                labels.append(parts[1])
            else:
                labels.append(s)
    return labels

LABELS = load_labels_txt(LABELS_PATH)

# ====== Haar face detector ======
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ====== Flask app ======
app = Flask(__name__, static_folder="static", template_folder="templates")
if _HAS_CORS:
    CORS(app)
else:
    @app.after_request
    def add_cors_headers(resp):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        resp.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
        return resp

# ====== Helpers ======
def _decode_file_to_bgr(file_storage) -> np.ndarray:
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return bgr

def _decode_base64_to_bgr(s: str) -> np.ndarray:
    if "," in s:
        s = s.split(",", 1)[1]
    img_bytes = base64.b64decode(s)
    arr = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return bgr

def _smart_resize_for_detection(bgr: np.ndarray, max_w: int = 384) -> Tuple[np.ndarray, float]:
    """Resize ảnh về max_w (nếu lớn hơn) để detect nhanh hơn. Trả về ảnh đã resize và scale so với ảnh input."""
    h, w = bgr.shape[:2]
    if w <= max_w:
        return bgr, 1.0
    scale = max_w / float(w)
    resized = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale

def _preprocess_face_rgb(face_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    x = rgb.astype("float32") / 255.0
    return np.expand_dims(x, 0)

def _predict_crop(face_bgr: np.ndarray) -> Tuple[int, float]:
    """Return (class_idx, prob) for a face crop."""
    x = _preprocess_face_rgb(face_bgr)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return idx, float(probs[idx])

def _detect_and_classify(bgr: np.ndarray) -> Dict[str, Any]:
    """Detect faces; for each face, run classifier; return faces + overall top-1."""
    # 1) Resize nhẹ để detect nhanh
    small, scale = _smart_resize_for_detection(bgr, max_w=384)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # 2) Haar detect (nhanh + ổn định)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.22,   # lớn hơn -> ít khung & nhanh hơn
        minNeighbors=5,
        minSize=(60, 60),
    )

    faces_out = []
    overall_best = (-1, 0.0, "")  # (idx, prob, label)

    if len(faces) == 0:
        # Không phát hiện mặt: phân loại toàn ảnh
        idx, prob = _predict_crop(bgr)
        label = LABELS[idx] if idx < len(LABELS) else str(idx)
        return {
            "faces": [],  # không có bbox
            "top_class_idx": idx,
            "top_class_name": label,
            "top_class_prob": prob,
        }

    # Map bbox về toạ độ ảnh gốc
    inv = 1.0 / scale
    for (x, y, w, h) in faces:
        gx, gy = int(round(x * inv)), int(round(y * inv))
        gw, gh = int(round(w * inv)), int(round(h * inv))

        # Thêm padding 12% để crop đủ mặt
        pad = int(0.12 * max(gw, gh))
        x0 = max(0, gx - pad); y0 = max(0, gy - pad)
        x1 = min(bgr.shape[1], gx + gw + pad); y1 = min(bgr.shape[0], gy + gh + pad)
        crop = bgr[y0:y1, x0:x1]
        if crop.size == 0:
            continue

        idx, prob = _predict_crop(crop)
        label = LABELS[idx] if idx < len(LABELS) else str(idx)

        faces_out.append({
            "x": gx, "y": gy, "w": gw, "h": gh,
            "label": label, "score": prob,
        })

        if prob > overall_best[1]:
            overall_best = (idx, prob, label)

    # Nếu vì lý do gì không push được mặt nào, fallback toàn ảnh
    if not faces_out:
        idx, prob = _predict_crop(bgr)
        label = LABELS[idx] if idx < len(LABELS) else str(idx)
        top_idx, top_prob, top_name = idx, prob, label
    else:
        top_idx, top_prob, top_name = overall_best

    return {
        "faces": faces_out,
        "top_class_idx": int(top_idx),
        "top_class_name": top_name,
        "top_class_prob": float(top_prob),
    }

# ====== Routes ======
@app.route("/", methods=["GET"])
def ui():
    """Render UI."""
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    """Health check JSON."""
    return jsonify({
        "status": "ok",
        "img_size": IMG_SIZE,
        "classes": LABELS,
        "model_path": MODEL_PATH,
        "labels_path": LABELS_PATH
    })

# --- Predict from uploaded file ---
@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])  # alias
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Thiếu file 'image'"}), 400
    bgr = _decode_file_to_bgr(request.files["image"])
    if bgr is None:
        return jsonify({"error": "Không đọc được ảnh"}), 400
    try:
        result = _detect_and_classify(bgr)
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Predict from base64 frame (webcam) ---
@app.route("/predict_frame", methods=["POST"])
@app.route("/api/predict_frame", methods=["POST"])  # alias
def predict_frame():
    data = request.get_json(silent=True) or {}
    if "image" not in data:
        return jsonify({"error": "Thiếu khóa 'image' trong JSON"}), 400
    bgr = _decode_base64_to_bgr(data["image"])
    if bgr is None:
        return jsonify({"error": "Base64 không hợp lệ"}), 400
    try:
        result = _detect_and_classify(bgr)
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ====== Main ======
if __name__ == "__main__":
    # Chạy: python app.py  -> http://127.0.0.1:5000/
    app.run(host="0.0.0.0", port=5000, debug=False)
