# final_software_opencv.py
"""
Recognition pipeline:
- YOLOv8-face detector → face_detect.detect_faces()
- Buffalo_L ONNX embeddings → face_detect.get_embedding()
Using cosine similarity + strong Unknown detection (top-2 rule)
"""

import os
import cv2
import pickle
import numpy as np
from datetime import datetime
import face_detect
import attendance


def _parse_box(box):
    x, y, w, h, conf = box
    return int(x), int(y), int(w), int(h), float(conf)


DEFAULT_SAVE_BURST = 100
CLASSIFIER_FILENAME = "classifier_buffalo.pkl"

# FIXED threshold for YOLO + buffalo_l
DEFAULT_THRESHOLD = 0.82     # IMPORTANT


# ==========================================
# TRAINING (unchanged except threshold)
# ==========================================
def _collect_embeddings_labels(root_folder):
    embeddings_dict = {}
    for class_name in sorted(os.listdir(root_folder)):
        class_path = os.path.join(root_folder, class_name)
        if not os.path.isdir(class_path):
            continue

        emb_list = []
        for f in os.listdir(class_path):
            if not f.lower().endswith(('.png','.jpg','.jpeg')):
                continue

            img = cv2.imread(os.path.join(class_path, f))
            if img is None:
                continue

            boxes = face_detect.detect_faces(img, conf_threshold=0.75)
            if not boxes:
                continue

            x,y,w,h,conf = _parse_box(sorted(boxes, key=lambda x:x[2]*x[3], reverse=True)[0])
            face = img[y:y+h, x:x+w]
            emb = face_detect.get_embedding(face)
            if emb is not None:
                emb_list.append(emb)

        if len(emb_list) == 0:
            continue

        mean_emb = np.mean(emb_list, axis=0)
        mean_emb /= (np.linalg.norm(mean_emb) + 1e-6)

        embeddings_dict[class_name] = mean_emb

    return embeddings_dict


def train(parameters):
    dataset_root, _, _, _, _, clf_name, _, _ = parameters
    clf_path = clf_name + ".pkl"

    embeddings_dict = _collect_embeddings_labels(dataset_root)
    data = {"embeddings": embeddings_dict, "threshold": DEFAULT_THRESHOLD}

    with open(clf_path, "wb") as f:
        pickle.dump(data, f)

    print("Saved classifier:", clf_path)
    return 1


# ==========================================
# MAIN RECOGNITION (FIXED!!)
# ==========================================
def recognize(mode, params):
    classifier_path, _, _, _, _, _, resolution, img_folder, out_img, vid_path, vid_save, vid_see = params

    classifier_path = classifier_path if classifier_path.endswith(".pkl") else classifier_path + ".pkl"
    with open(classifier_path, "rb") as f:
        data = pickle.load(f)

    embeddings_dict = data["embeddings"]
    threshold = data.get("threshold", DEFAULT_THRESHOLD)

    names = list(embeddings_dict.keys())
    emb_matrix = np.vstack([embeddings_dict[n] for n in names]).astype(np.float32)

    cap = cv2.VideoCapture(0 if mode == "w" else vid_path)
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)

    print("Recognition started — UNKNOWN fix enabled")

    marked = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        disp = frame.copy()
        boxes = face_detect.detect_faces(frame, conf_threshold=0.85)
        current = set()

        for b in boxes:
            x,y,w,h,conf = _parse_box(b)
            face = frame[y:y+h, x:x+w]

            emb = face_detect.get_embedding(face)
            if emb is None:
                label = "NoEmb"
            else:
                emb = emb / (np.linalg.norm(emb) + 1e-6)

                sims = emb_matrix @ emb

                sorted_idx = np.argsort(sims)[::-1]
                best_idx = sorted_idx[0]
                second_idx = sorted_idx[1] if len(sorted_idx) > 1 else None

                best_sim = sims[best_idx]
                second_sim = sims[second_idx] if second_idx is not None else -1

                best_name = names[best_idx]

                # RULE 1 — threshold
                if best_sim < threshold:
                    label = f"Unknown ({best_sim:.2f})"

                # RULE 2 — top2 separation
                elif (best_sim - second_sim) < 0.05:
                    label = f"Unknown ({best_sim:.2f})"

                else:
                    label = f"{best_name} ({best_sim:.2f})"
                    current.add(best_name)

            cv2.rectangle(disp, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(disp, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.imshow("Output", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('m'):
            for nm in current:
                attendance.mark_present(nm)
                marked.add(nm)

    cap.release()
    cv2.destroyAllWindows()

    return ",".join(sorted(marked))
