# face_detect.py
"""
YOLOv8-face detector + Buffalo_L ONNX embedding extractor.

Functions:
- detect_faces(image, conf_threshold=0.60) -> list of [x,y,w,h,conf]
- get_embedding(face_image) -> 1D numpy float32 L2-normalized vector

Model locations (edit paths if needed):
- YOLOv8: models/yolov8_face/best.pt
- Buffalo ONNX: models/buffalo_l/*.onnx  (first .onnx will be used)

Requires: ultralytics, onnxruntime, opencv-python, numpy
"""

import os
import glob
import numpy as np
import cv2

# Try to import ultralytics YOLO and onnxruntime
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except Exception:
    YOLO = None
    _YOLO_AVAILABLE = False

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except Exception:
    ort = None
    _ORT_AVAILABLE = False

# Default model paths - adjust to your project layout if needed
YOLO_MODEL_PATH = os.path.join(os.getcwd(), "models", "yolov8_face", "best.pt")
BUFFALO_FOLDER = os.path.join(os.getcwd(), "models", "buffalo_l")

# Globals to hold loaded models
_yolo_model = None
_ort_session = None
_ort_input_shape = None

def _load_yolo():
    global _yolo_model
    if _yolo_model is not None:
        return True
    if not _YOLO_AVAILABLE:
        print("Detect error: YOLO (ultralytics) not available. Install: pip install ultralytics")
        return False
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"Detect error: YOLO weights not found at {YOLO_MODEL_PATH}")
        return False
    try:
        _yolo_model = YOLO(YOLO_MODEL_PATH)
        print("YOLO loaded:", YOLO_MODEL_PATH)
        return True
    except Exception as e:
        print("YOLO load error:", e)
        _yolo_model = None
        return False

def _load_buffalo_onnx():
    global _ort_session, _ort_input_shape
    if _ort_session is not None:
        return True
    if not _ORT_AVAILABLE:
        print("Detect error: onnxruntime not available. Install: pip install onnxruntime")
        return False
    if not os.path.isdir(BUFFALO_FOLDER):
        print(f"Buffalo folder not found: {BUFFALO_FOLDER}")
        return False
    onnx_files = glob.glob(os.path.join(BUFFALO_FOLDER, "*.onnx"))
    if len(onnx_files) == 0:
        print("No ONNX file found in buffalo_l folder:", BUFFALO_FOLDER)
        return False
    onnx_path = onnx_files[0]
    try:
        providers = ['CPUExecutionProvider']
        _ort_session = ort.InferenceSession(onnx_path, providers=providers)
        inp = _ort_session.get_inputs()[0].shape  # e.g. (1,3,112,112) or (1,3,224,224)
        _ort_input_shape = tuple(inp)
        print("Buffalo ONNX loaded:", onnx_path, "input shape:", _ort_input_shape)
        return True
    except Exception as e:
        print("Buffalo ONNX load error:", e)
        _ort_session = None
        _ort_input_shape = None
        return False

def detect_faces(image, conf_threshold=0.60):
    """
    Detect faces and return list of boxes [x,y,w,h,conf].
    Uses ultralytics YOLO if available and model path exists.
    """
    if not _load_yolo():
        # fallback: simple OpenCV Haar cascade (very weak) — but we prefer CI to require ultralytics
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.exists(cascade_path):
            face_cascade = cv2.CascadeClassifier(cascade_path)
            rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            return [[int(x),int(y),int(w),int(h), 0.0] for (x,y,w,h) in rects]
        return []

    # Run inference
    results = _yolo_model(image)  # ultralytics returns Results object(s)
    boxes = []
    # take first (batch size 1)
    try:
        res = results[0]
        for det in res.boxes.data.cpu().numpy():
            # det format: x1,y1,x2,y2,conf,cls
            x1, y1, x2, y2, conf, cls = det
            if conf < conf_threshold:
                continue
            x = int(max(0, round(x1)))
            y = int(max(0, round(y1)))
            w = int(max(1, round(x2 - x1)))
            h = int(max(1, round(y2 - y1)))
            boxes.append([x, y, w, h, float(conf)])
    except Exception:
        # structured access fallback
        try:
            for r in results:
                for box in r.boxes:
                    b = box.xyxy.cpu().numpy()[0]
                    conf = float(box.conf.cpu().numpy()[0])
                    if conf < conf_threshold:
                        continue
                    x1, y1, x2, y2 = b
                    x = int(max(0, round(x1)))
                    y = int(max(0, round(y1)))
                    w = int(max(1, round(x2 - x1)))
                    h = int(max(1, round(y2 - y1)))
                    boxes.append([x, y, w, h, conf])
        except Exception:
            pass
    return boxes

def _preprocess_for_onnx(face_img, target_shape):
    """
    face_img: BGR HxW(x3) uint8
    target_shape: something like (1,3,112,112) or (1,3,224,224)
    returns NCHW float32
    """
    _, _, th, tw = target_shape if len(target_shape) == 4 else (None, None, target_shape[2], target_shape[3])
    if th is None or tw is None:
        th, tw = 112, 112
    # convert to RGB
    img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (tw, th))
    arr = img.astype(np.float32) / 255.0
    # Normalize to [-1,1] (common). If your buffalo model expects other normalization adjust here.
    arr = (arr - 0.5) / 0.5
    # HWC -> CHW
    arr = np.transpose(arr, (2, 0, 1)).astype(np.float32)
    arr = np.expand_dims(arr, 0)
    return arr

def get_embedding(face_img):
    """
    Compute Buffalo embedding for a single face crop (BGR uint8).
    Returns 1D L2-normalized numpy float32 embedding or None if failure.
    """
    if not _load_buffalo_onnx():
        return None
    try:
        inp_shape = _ort_input_shape
        arr = _preprocess_for_onnx(face_img, inp_shape)
        out_name = _ort_session.get_outputs()[0].name
        res = _ort_session.run([out_name], { _ort_session.get_inputs()[0].name: arr })[0]
        # res shape (1, dim)
        emb = np.array(res).reshape(-1).astype(np.float32)
        # L2 normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb
    except Exception as e:
        print("Buffalo embedding error:", e)
        return None

# quick check at import
if __name__ == "__main__":
    print("face_detect.py loaded. YOLO available:", _YOLO_AVAILABLE, "onnxruntime available:", _ORT_AVAILABLE)
    print("YOLO model path:", YOLO_MODEL_PATH)
    print("Buffalo ONNX path:", BUFFALO_FOLDER)
