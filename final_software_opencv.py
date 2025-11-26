# final_software_opencv.py
"""
Recognition pipeline using:
- YOLOv8-face (detection): face_detect.detect_faces(...)
- Buffalo_L ONNX (embeddings): face_detect.get_embedding(...)
Recognition uses cosine similarity against stored per-person mean embeddings.
Manual marking: press 'm' to mark currently recognized persons (no auto-mark).
Keys in recognition:
 - q: quit
 - m: mark currently recognized
 - c: clear today's attendance
 - s: save snapshot of current frame
Dataset creation:
 - Press 's' to start a burst capture (100 face saves) to create dataset quickly.
Training:
 - Calculate embedding for each image, average per person -> save classifier_buffalo.pkl
Classifier format:
 {'embeddings': {name: mean_embedding}, 'threshold': 0.6 }
"""

import os
import cv2
import pickle
import numpy as np
from datetime import datetime
import face_detect
import attendance

def _parse_box(box):
    """
    Normalize detection box into (x,y,w,h,conf).
    Accepts formats like [x,y,w,h,conf] or [x1,y1,x2,y2,conf,...].
    Returns ints for x,y,w,h and float for conf.
    """
    try:
        b = list(box)
        if len(b) < 4:
            raise ValueError("box must have at least four values")
        a0, a1, a2, a3 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        # If values look like x1,y1,x2,y2 (x2 > x1 and y2 > y1) convert to x,y,w,h
        if a2 > a0 and a3 > a1 and (a2 - a0) >= 2 and (a3 - a1) >= 2:
            x = int(max(0, round(a0)))
            y = int(max(0, round(a1)))
            w = int(max(1, round(a2 - a0)))
            h = int(max(1, round(a3 - a1)))
        else:
            x = int(max(0, round(a0)))
            y = int(max(0, round(a1)))
            w = int(max(1, round(a2)))
            h = int(max(1, round(a3)))
        conf = float(b[4]) if len(b) > 4 else 0.0
        return x, y, w, h, conf
    except Exception as e:
        raise ValueError(f"Invalid box format: {box} -> {e}")

DEFAULT_SAVE_BURST = 100
CLASSIFIER_FILENAME = "classifier_buffalo.pkl"
DEFAULT_THRESHOLD = 0.60  # cosine similarity threshold (higher = stricter)

def abs_path_or_default(path):
    if path and path.strip() != "":
        return os.path.abspath(path)
    return os.path.abspath('output')

# ---------- Dataset creation (burst mode save) ----------
def dataset_creation(parameters):
    """
    parameters: (path1, webcam, face_dim, gpu_unused, username, vid_path)
    Press 's' in camera window to capture a burst of DEFAULT_SAVE_BURST face crops and save into the user folder.
    """
    path1, webcam, face_dim, gpu, username, vid_path = parameters

    # If the caller provided an explicit path1 use it directly as the output root.
    # Previously this code joined path1 + 'output' which caused nested directories
    # when path1 already pointed at the 'output' folder (=> output/output). Use
    # path1 itself and ensure the folder exists.
    if path1 and path1.strip() != "":
        output_root = os.path.abspath(path1)
    else:
        output_root = abs_path_or_default(path1)
    os.makedirs(output_root, exist_ok=True)

    res = webcam
    if not res:
        res = (640, 480)
    else:
        try:
            res = tuple(map(int, res.split('x')))
        except:
            res = (640, 480)

    face_size = (160,160)
    personNo = 1

    while True:
        ask = username.replace(" ", "_") if username else ""
        folder_name = ask if ask != "" else 'person' + str(personNo)
        personNo += 1

        users_folder = os.path.join(output_root, folder_name)
        os.makedirs(users_folder, exist_ok=True)
        print("Dataset: saving into", users_folder)

        data_type = vid_path if vid_path else 0
        device = cv2.VideoCapture(data_type)
        if not device or not device.isOpened():
            print("Unable to open camera/video:", data_type)
            return 0
        device.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
        device.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])

        cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
        print(f"Camera running. Press 's' to save images loop (will capture {DEFAULT_SAVE_BURST} frames), 'q' to quit this person.")

        while True:
            ret, frame = device.read()
            if not ret or frame is None:
                break
            disp = frame.copy()
            cv2.putText(disp, "Press 's' to burst-save, 'q' to quit", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow("Output", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                # burst capture
                print("Starting capture loop for", DEFAULT_SAVE_BURST, "frames...")
                saved = 0
                frame_idx = 0
                while saved < DEFAULT_SAVE_BURST:
                    ret2, frame2 = device.read()
                    if not ret2 or frame2 is None:
                        break
                    # detect faces, take largest
                    boxes = face_detect.detect_faces(frame2, conf_threshold=0.60)
                    if boxes:
                        boxes_sorted = sorted(boxes, key=lambda b: _parse_box(b)[2]*_parse_box(b)[3], reverse=True)
                        x,y,w,h,_ = _parse_box(boxes_sorted[0])
                        x1,y1 = max(0,x), max(0,y)
                        x2,y2 = min(frame2.shape[1], x+w), min(frame2.shape[0], y+h)
                        if x2 > x1 and y2 > y1:
                            face = frame2[y1:y2, x1:x2]
                            # save full-color face, not resized (resize preserved for training pipeline)
                            fname = os.path.join(users_folder, f"{folder_name}_{str(saved+1).zfill(4)}.png")
                            cv2.imwrite(fname, face)
                            saved += 1
                            if saved % 10 == 0:
                                print("Saved", saved, "images...")
                    frame_idx += 1
                    # show a quick visual
                    cv2.putText(frame2, f"Saving: {saved}/{DEFAULT_SAVE_BURST}", (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    cv2.imshow("Output", frame2)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                print("Burst capture complete: saved", saved, "images.")
        device.release()
        cv2.destroyAllWindows()
        return 1

# ---------- Training: collect Buffalo embeddings and average per class ----------
def _collect_embeddings_labels(root_folder):
    names = []
    embeddings = []
    labels = []
    label_map = {}
    idx = 0
    if not os.path.isdir(root_folder):
        return np.zeros((0,)), np.array([]), {}
    for class_name in sorted(os.listdir(root_folder)):
        class_path = os.path.join(root_folder, class_name)
        if not os.path.isdir(class_path):
            continue
        # collect all embeddings for this person
        emb_list = []
        for fname in os.listdir(class_path):
            if not fname.lower().endswith(('.png','.jpg','.jpeg')):
                continue
            img_path = os.path.join(class_path, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue
            # detect face in the saved image (expect face roughly centered)
            boxes = face_detect.detect_faces(img, conf_threshold=0.5)
            if not boxes:
                continue
            boxes_sorted = sorted(boxes, key=lambda b: _parse_box(b)[2]*_parse_box(b)[3], reverse=True)
            x,y,w,h,_ = _parse_box(boxes_sorted[0])
            x1,y1 = max(0,x), max(0,y)
            x2,y2 = min(img.shape[1], x+w), min(img.shape[0], y+h)
            if x2 <= x1 or y2 <= y1:
                continue
            face = img[y1:y2, x1:x2]
            emb = face_detect.get_embedding(face)
            if emb is not None:
                emb_list.append(emb)
        if len(emb_list) == 0:
            print("No embeddings for class:", class_name, "- skipping")
            continue
        mean_emb = np.mean(np.vstack(emb_list), axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm
        label_map[idx] = class_name
        embeddings.append(mean_emb)
        labels.append(idx)
        idx += 1
    if len(embeddings) == 0:
        return np.zeros((0,)), np.array([]), {}
    return np.vstack(embeddings), np.array(labels), label_map

def train(parameters):
    """
    parameters: (dataset_root, unused, unused, unused, unused, classifier_name, unused, unused)
    Creates classifier_buffalo.pkl with:
     {'embeddings': dict(name->embedding), 'threshold': threshold}
    """
    path1, path2, batch, img_dim, gpu, clf_name, split_percent, split_data = parameters
    dataset_root = path1 if path1 else abs_path_or_default(path1)
    if not os.path.isdir(dataset_root):
        print("Dataset folder not found:", dataset_root)
        return 0
    print("Training (collecting Buffalo embeddings)...")
    emb_array, labels, label_map = _collect_embeddings_labels(dataset_root)
    if emb_array.shape[0] == 0:
        print("No training embeddings found.")
        return 0
    # build name->embedding dict
    embeddings_dict = {}
    for idx, name in label_map.items():
        embeddings_dict[name] = emb_array[idx]
    # Save classifier
    classifier_filename = (clf_name + '.pkl') if clf_name else CLASSIFIER_FILENAME
    with open(classifier_filename, 'wb') as f:
        pickle.dump({'embeddings': embeddings_dict, 'threshold': DEFAULT_THRESHOLD}, f)
    classifier_filename_abs = os.path.abspath(classifier_filename)
    print("Saved classifier to", classifier_filename_abs)
    return 1

# ---------- Recognize (manual marking using cosine similarity) ----------
def _cosine(a, b):
    if a is None or b is None:
        return -1.0
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return -1.0
    return float(np.dot(a, b) / (na * nb))

def recognize(mode, parameters):
    """
    mode: 'w' webcam, 'v' video, 'i' images
    parameters: (classifier_path, ..., resolution, img_path, out_img_path, vid_path, vid_save, vid_see)
    Manual marking only: press 'm' to mark currently seen names.
    Returns comma-separated names marked this run.
    """
    classifier_path, _, _, _, _, _, resolution, img_path, out_img_path, vid_path, vid_save, vid_see = parameters
    # Allow classifier names with or without the .pkl extension
    classifier_path = classifier_path if classifier_path else CLASSIFIER_FILENAME
    if not classifier_path.lower().endswith('.pkl'):
        candidate = classifier_path + '.pkl'
    else:
        candidate = classifier_path
    candidate_abs = os.path.abspath(candidate)
    # Use provided value if it exists or fallback to CLASSIFIER_FILENAME
    if os.path.exists(candidate_abs):
        classifier_path = candidate_abs
    else:
        classifier_path = os.path.abspath(CLASSIFIER_FILENAME)
    if not os.path.exists(classifier_path):
        # Treat missing classifier as an error so the UI can show an error dialog
        raise FileNotFoundError(f"Classifier file not found: {classifier_path}")

    with open(classifier_path, 'rb') as f:
        data = pickle.load(f)
    embeddings_dict = data.get('embeddings', {})
    threshold = data.get('threshold', DEFAULT_THRESHOLD)

    marked_this_run = set()
    # prepare arrays for fast compare
    names = list(embeddings_dict.keys())
    if len(names) == 0:
        print("No embeddings in classifier.")
        return ""
    emb_matrix = np.vstack([embeddings_dict[n] for n in names]).astype(np.float32)

    if mode == "w":
        source = 0
    elif mode == "v":
        source = vid_path
    else:
        # images folder: process and auto-mark (image mode keeps old behavior)
        image_folder = img_path if img_path else ""
        files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png','.jpg','.jpeg'))]
        for f in files:
            img = cv2.imread(os.path.join(image_folder, f))
            if img is None:
                continue
            boxes = face_detect.detect_faces(img, conf_threshold=0.6)
            if not boxes:
                continue
            boxes_sorted = sorted(boxes, key=lambda b: _parse_box(b)[2]*_parse_box(b)[3], reverse=True)
            x,y,w,h,_ = _parse_box(boxes_sorted[0])
            face = img[y:y+h, x:x+w]
            emb = face_detect.get_embedding(face)
            if emb is None:
                continue
            # cosine similarities
            sims = (emb_matrix @ emb).astype(np.float32)  # emb_matrix rows are normalized; emb normalized in loader
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            if best_sim >= threshold:
                name = names[best_idx]
                ok = attendance.mark_present(name)
                if ok:
                    marked_this_run.add(name)
        return ",".join(sorted(list(marked_this_run)))

    # open camera/video
    cap = cv2.VideoCapture(source)
    if not cap or not cap.isOpened():
        print("Cannot open source:", source)
        return ""
    if resolution and mode == "w":
        try:
            wres, hres = tuple(map(int, resolution.split('x')))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, wres)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hres)
        except:
            pass

    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    print("Recognition started (manual marking). Keys: m=mark  q=quit  c=clear today  s=snapshot")
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            disp = frame.copy()
            boxes = face_detect.detect_faces(frame, conf_threshold=0.6)
            current_recognized = set()
            for b in boxes:
                x,y,w,h,conf = _parse_box(b)
                x1,y1 = max(0,x), max(0,y)
                x2,y2 = min(frame.shape[1], x+w), min(frame.shape[0], y+h)
                if x2 <= x1 or y2 <= y1:
                    continue
                face = frame[y1:y2, x1:x2]
                emb = face_detect.get_embedding(face)
                if emb is None:
                    label_text = "NoEmb"
                else:
                    # compare with stored embeddings: cosine similarity
                    sims = emb_matrix @ emb
                    best_idx = int(np.argmax(sims))
                    best_sim = float(sims[best_idx])
                    name = names[best_idx]
                    if best_sim >= threshold:
                        label_text = f"{name} ({best_sim:.2f})"
                        current_recognized.add(name)
                    else:
                        label_text = f"Unknown ({best_sim:.2f})"
                cv2.rectangle(disp, (x1,y1), (x2,y2), (0,255,0), 1)
                cv2.putText(disp, label_text, (x1+2, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            cv2.putText(disp, "q:quit  m:mark current  c:clear today  s:snapshot", (8, disp.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            cv2.imshow("Output", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('m'):
                if not current_recognized:
                    print("No recognized faces to mark.")
                else:
                    for nm in current_recognized:
                        ok = attendance.mark_present(nm)
                        if ok:
                            marked_this_run.add(nm)
                            print("Marked present:", nm)
                        else:
                            print("Already marked or error for:", nm)
            if key == ord('c'):
                attendance.clear_today()
                marked_this_run.clear()
                print("Cleared today's attendance.")
            if key == ord('s'):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = os.path.join(os.getcwd(), f"snapshot_{ts}.png")
                cv2.imwrite(fname, frame)
                print("Saved snapshot:", fname)
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return ",".join(sorted(list(marked_this_run)))
