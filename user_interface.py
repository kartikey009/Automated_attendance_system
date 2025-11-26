# user_interface.py
"""
PyQt6 modern UI for the Attendance System.
Wires to final_software_opencv.dataset_creation, train, recognize.
"""

import sys
import os
import threading
import pickle
import traceback
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QFileDialog, QMessageBox, QListWidget, QTextEdit,
    QGroupBox, QGridLayout, QProgressBar
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer
import numpy as _np
import cv2

# import backend modules
import final_software_opencv as final
import attendance
import face_detect

# Simple worker thread wrapper using threading.Thread
class _Invoker(QObject):
    """Helper QObject that invokes a Python callable in the QObject's thread
    by emitting a signal from other threads. This ensures UI callbacks are
    executed on the main (Qt) thread."""
    call = pyqtSignal(object, object)

    def __init__(self):
        super().__init__()
        self.call.connect(self._on_call)

    def _on_call(self, cb, args):
        try:
            cb(*args)
        except Exception:
            traceback.print_exc()


# create a single invoker instance tied to the main thread
_INVOKER = _Invoker()


class Worker(threading.Thread):
    def __init__(self, fn, args=(), on_done=None):
        super().__init__()
        self.fn = fn
        self.args = args
        self.on_done = on_done
        self.result = None
        self.exc = None

    def run(self):
        try:
            self.result = self.fn(*self.args)
        except Exception as e:
            self.exc = e
            traceback.print_exc()
        finally:
            if self.on_done:
                try:
                    # If on main thread, call directly; otherwise schedule via Qt invoker
                    if threading.current_thread() is threading.main_thread():
                        self.on_done(self.result, self.exc)
                    else:
                        # If no Qt event loop / QApplication exists (e.g. running headless tests),
                        # call the callback directly to avoid losing the result. When a
                        # QApplication exists, schedule the callback on the Qt/main thread
                        # via our invoker so UI work happens on the correct thread.
                        try:
                            from PyQt6.QtWidgets import QApplication
                            if QApplication.instance() is None:
                                # no Qt app: call directly
                                self.on_done(self.result, self.exc)
                            else:
                                _INVOKER.call.emit(self.on_done, (self.result, self.exc))
                        except Exception:
                            # fallback: direct call
                            try:
                                self.on_done(self.result, self.exc)
                            except Exception:
                                traceback.print_exc()
                except Exception:
                    traceback.print_exc()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Attendance System (Buffalo+YOLOv8)")
        self.resize(950, 650)
        self._recog_timer = None
        self._build_ui()
        self.log("UI ready")

    def _build_ui(self):
        central = QWidget()
        main = QVBoxLayout(central)

        # Header
        header = QLabel("Automated Attendance — Buffalo_L embeddings + YOLOv8 Face Detector")
        header.setStyleSheet("font-size:18px; font-weight:600; padding:8px;")
        main.addWidget(header)

        # Top controls: Dataset / Train / Recognize
        group = QGroupBox("Actions")
        gbox = QGridLayout(group)
        gbox.addWidget(QLabel("Dataset Output Root:"), 0, 0)
        self.inp_ds_out = QLineEdit(os.path.join(os.getcwd(), "output"))
        gbox.addWidget(self.inp_ds_out, 0, 1)
        btn_ds_browse = QPushButton("Browse")
        btn_ds_browse.clicked.connect(self.browse_dataset_root)
        gbox.addWidget(btn_ds_browse, 0, 2)

        gbox.addWidget(QLabel("Username (for dataset):"), 1, 0)
        self.inp_ds_name = QLineEdit("person1")
        gbox.addWidget(self.inp_ds_name, 1, 1)

        btn_create = QPushButton("Create Dataset (Burst)")
        btn_create.clicked.connect(self.create_dataset)
        gbox.addWidget(btn_create, 1, 2)

        gbox.addWidget(QLabel("Dataset Root for Training:"), 2, 0)
        self.inp_train_root = QLineEdit(os.path.join(os.getcwd(), "output"))
        gbox.addWidget(self.inp_train_root, 2, 1)
        btn_train = QPushButton("Train (compute Buffalo embeddings)")
        btn_train.clicked.connect(self.start_train)
        gbox.addWidget(btn_train, 2, 2)

        gbox.addWidget(QLabel("Classifier filename:"), 3, 0)
        self.inp_classifier = QLineEdit("classifier_buffalo")
        gbox.addWidget(self.inp_classifier, 3, 1)
        btn_reload = QPushButton("Load Classifier")
        btn_reload.clicked.connect(self.load_classifier)
        gbox.addWidget(btn_reload, 3, 2)

        gbox.addWidget(QLabel("Recognition resolution (e.g. 640x480):"), 4, 0)
        self.inp_res = QLineEdit("640x480")
        gbox.addWidget(self.inp_res, 4, 1)

        # Recog controls moved into Camera area below (embedded view)

        main.addWidget(group)
        mid = QHBoxLayout()
        # Camera display (embedded inside UI)
        cam_box = QGroupBox("Camera / Recognition")
        cam_v = QVBoxLayout(cam_box)
        # camera display QLabel
        self.lbl_camera = QLabel()
        self.lbl_camera.setMinimumSize(640, 480)
        self.lbl_camera.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_camera.setStyleSheet("background: #222; border: 1px solid #444;")
        cam_v.addWidget(self.lbl_camera)
        # Controls below camera
        cam_controls = QHBoxLayout()
        self.btn_start_rec = QPushButton("Start Recognition (embedded)")
        self.btn_start_rec.clicked.connect(self.start_recognition)
        self.btn_stop_rec = QPushButton("Stop Recognition")
        self.btn_stop_rec.clicked.connect(self.stop_recognition)
        self.btn_stop_rec.setEnabled(False)
        self.btn_mark_current = QPushButton("Mark Current")
        self.btn_mark_current.clicked.connect(self._mark_current)
        cam_controls.addWidget(self.btn_start_rec)
        cam_controls.addWidget(self.btn_stop_rec)
        cam_controls.addWidget(self.btn_mark_current)
        cam_v.addLayout(cam_controls)
        mid.addWidget(cam_box, 3)
        # Logs
        logs_box = QGroupBox("Logs / Status")
        logs_layout = QVBoxLayout(logs_box)
        self.txt_logs = QTextEdit()
        self.txt_logs.setReadOnly(True)
        logs_layout.addWidget(self.txt_logs)
        mid.addWidget(logs_box, 2)

        # Attendance list
        att_box = QGroupBox("Today's Attendance")
        att_layout = QVBoxLayout(att_box)
        self.lst_att = QListWidget()
        att_layout.addWidget(self.lst_att)
        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self.refresh_attendance)
        btn_clear = QPushButton("Clear Today")
        btn_clear.clicked.connect(self.clear_attendance)
        att_layout.addWidget(btn_refresh)
        att_layout.addWidget(btn_clear)
        mid.addWidget(att_box, 1)

        main.addLayout(mid)

        # Footer
        footer = QLabel("Keys during recognition: m = manual mark, q = quit, c = clear today's attendance, s = snapshot")
        footer.setStyleSheet("color: gray;")
        main.addWidget(footer)

        self.setCentralWidget(central)

        # initial refresh
        self.refresh_attendance()

        # recognition state
        self._recog_worker = None
        self._recog_running = False
        self._recog_current_names = set()

    def browse_dataset_root(self):
        d = QFileDialog.getExistingDirectory(self, "Select dataset root", self.inp_ds_out.text())
        if d:
            self.inp_ds_out.setText(d)

    def create_dataset(self):
        out = self.inp_ds_out.text().strip()
        name = self.inp_ds_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "Please enter a user name")
            return
        params = (out, "", "", "", name, "")
        self.log("Starting dataset creation thread...")
        w = Worker(final.dataset_creation, args=(params,), on_done=self._on_dataset_done)
        w.start()

    def _on_dataset_done(self, result, exc):
        if exc:
            self.log("Dataset creation error: " + str(exc))
            QMessageBox.warning(self, "Dataset", f"Error: {exc}")
        else:
            self.log("Dataset creation finished.")
            QMessageBox.information(self, "Dataset", "Dataset creation finished.")

    def start_train(self):
        root = self.inp_train_root.text().strip()
        if not os.path.isdir(root):
            QMessageBox.warning(self, "Error", "Invalid dataset root")
            return
        clf_name = self.inp_classifier.text().strip()
        params = (root, "", "", "", "", clf_name, "", "")
        self.log("Starting training thread...")
        w = Worker(final.train, args=(params,), on_done=self._on_train_done)
        w.start()

    def _on_train_done(self, result, exc):
        if exc:
            self.log("Training error: " + str(exc))
            QMessageBox.warning(self, "Training", f"Error: {exc}")
        else:
            self.log("Training finished.")
            QMessageBox.information(self, "Training", "Training finished. Load classifier to use recognition.")

    def load_classifier(self):
        fname = self.inp_classifier.text().strip()
        if not fname:
            fname = "classifier_buffalo"
        # Accept names with or without .pkl and search some likely locations
        cand_name = fname if fname.endswith(".pkl") else fname + ".pkl"
        candidates = []
        # If user typed an absolute path already, try that first
        if os.path.isabs(fname):
            candidates.append(fname)
        # Common locations to check (in order): as typed, project cwd, train root, dataset output
        candidates.append(os.path.abspath(cand_name))
        train_root = self.inp_train_root.text().strip()
        if train_root:
            candidates.append(os.path.abspath(os.path.join(train_root, cand_name)))
        ds_out = self.inp_ds_out.text().strip()
        if ds_out:
            candidates.append(os.path.abspath(os.path.join(ds_out, cand_name)))
        # remove duplicates preserving order
        seen = set()
        unique = []
        for p in candidates:
            if p and p not in seen:
                unique.append(p);
                seen.add(p)
        found = None
        for p in unique:
            if os.path.exists(p):
                found = p
                break
        if not found:
            QMessageBox.warning(self, "Classifier",
                                f"Classifier not found. Tried:\n\n" + "\n".join(unique) + "\n\nRun training to create a classifier or select the correct path.")
            return
        path = found
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self.classifier_loaded = path
            self.log(f"Classifier loaded: {path}  (contains {len(data.get('embeddings', {}))} classes)")
        except Exception as e:
            self.log("Load classifier error: " + str(e))
            QMessageBox.warning(self, "Classifier", f"Error loading classifier: {e}")

    def start_recognition(self):
        # build params tuple
        clf = self.inp_classifier.text().strip()
        if not clf:
            clf = "classifier_buffalo"
        params = (clf, "", "", "", "", "", self.inp_res.text().strip(), "", "", "", "", "")
        # Pre-check classifier exists (allow search in several probable locations)
        cand_name = clf if clf.endswith('.pkl') else clf + '.pkl'
        candidates = [os.path.abspath(cand_name)]
        train_root = self.inp_train_root.text().strip()
        if train_root:
            candidates.append(os.path.abspath(os.path.join(train_root, cand_name)))
        ds_out = self.inp_ds_out.text().strip()
        if ds_out:
            candidates.append(os.path.abspath(os.path.join(ds_out, cand_name)))
        found = None
        for p in candidates:
            if os.path.exists(p):
                found = p
                break
        if not found:
            self.log(f"Classifier file not found (tried): {', '.join(candidates)}")
            QMessageBox.warning(self, "Recognition", f"Classifier file not found.\nTried:\n\n" + "\n".join(candidates))
            return
        # replace the classifier name with absolute path so backend finds it reliably
        params = (found, "", "", "", "", "", self.inp_res.text().strip(), "", "", "", "", "")
        # Embedded recognition: start a worker that feeds frames into the camera QLabel
        # Find classifier file (support names with/without .pkl in some common locations)
        cand_name = clf if clf.endswith('.pkl') else clf + '.pkl'
        candidates = [os.path.abspath(cand_name)]
        train_root = self.inp_train_root.text().strip()
        if train_root:
            candidates.append(os.path.abspath(os.path.join(train_root, cand_name)))
        ds_out = self.inp_ds_out.text().strip()
        if ds_out:
            candidates.append(os.path.abspath(os.path.join(ds_out, cand_name)))
        found = None
        for p in candidates:
            if os.path.exists(p):
                found = p
                break
        if not found:
            self.log(f"Classifier file not found (tried): {', '.join(candidates)}")
            QMessageBox.warning(self, "Recognition", f"Classifier file not found.\nTried:\n\n" + "\n".join(candidates))
            return

        # load classifier
        try:
            with open(found, 'rb') as fh:
                data = pickle.load(fh)
        except Exception as e:
            QMessageBox.warning(self, "Classifier", f"Error loading classifier: {e}")
            return

        embeddings_dict = data.get('embeddings', {})
        if len(embeddings_dict) == 0:
            QMessageBox.warning(self, "Recognition", "Classifier contains no embeddings (train first)")
            return

        threshold = data.get('threshold', 0.6)
        names = list(embeddings_dict.keys())
        emb_matrix = _np.vstack([embeddings_dict[n] for n in names]).astype(_np.float32)

        # start worker
        self._recog_running = True
        self.btn_stop_rec.setEnabled(True)
        self.btn_start_rec.setEnabled(False)
        self._recog_worker = Worker(self._recognition_loop, args=(emb_matrix, names, threshold, self.inp_res.text().strip()), on_done=self._on_recog_done)
        # keep showing attendance updates while recognition is running
        try:
            if self._recog_timer is None:
                self._recog_timer = QTimer(self)
                self._recog_timer.timeout.connect(self.refresh_attendance)
                self._recog_timer.start(1000)
        except Exception:
            traceback.print_exc()
        self._recog_worker.start()

    def _on_recog_done(self, result, exc):
        # stop live refresh timer
        try:
            if getattr(self, '_recog_timer', None):
                self._recog_timer.stop()
                self._recog_timer = None
        except Exception:
            traceback.print_exc()
        if exc:
            self.log("Error in recognize: " + str(exc))
            QMessageBox.warning(self, "Recognition", f"Error: {exc}")
        else:
            self.log("Recognition finished. Marked: " + (result if result else "(none)"))
            QMessageBox.information(self, "Recognition finished", "Marked: " + (result if result else "(none)"))
            self.refresh_attendance()
        # reset recognition UI state
        try:
            self.btn_stop_rec.setEnabled(False)
            self.btn_start_rec.setEnabled(True)
            self._recog_running = False
            self._recog_worker = None
        except Exception:
            pass

    def refresh_attendance(self):
        self.lst_att.clear()
        names = attendance.get_marked()
        for n in names:
            self.lst_att.addItem(n)

    def clear_attendance(self):
        confirm = QMessageBox.question(self, "Confirm", "Clear today's attendance?")
        if confirm == QMessageBox.StandardButton.Yes:
            attendance.clear_today()
            self.refresh_attendance()
            self.log("Today's attendance cleared.")

    def log(self, message):
        ts = datetime.now().strftime("%H:%M:%S")
        self.txt_logs.append(f"[{ts}] {message}")
        self.txt_logs.ensureCursorVisible()

    # Embedded recognition helpers
    def stop_recognition(self):
        """Signal the background recognition worker to stop."""
        if self._recog_worker and self._recog_running:
            self.log("Stopping recognition...")
            try:
                self._recog_worker._stop_request = True
            except Exception:
                pass

    def _mark_current(self):
        """User clicks 'Mark Current' to record currently recognized names in attendance."""
        if not getattr(self, '_recog_current_names', None):
            QMessageBox.information(self, "Mark Current", "No recognized people to mark right now.")
            return
        for nm in list(self._recog_current_names):
            ok = attendance.mark_present(nm)
            if ok:
                self.log(f"Marked present: {nm}")
        self.refresh_attendance()

    def _display_frame(self, frame, recognized):
        """Update QLabel with a frame (main thread)."""
        try:
            self._recog_current_names = set(recognized)
            if frame is None:
                return
            img = frame
            if img.ndim == 3 and img.shape[2] == 3:
                h, w, ch = img.shape
                bytes_per_line = ch * w
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            else:
                h, w = img.shape[:2]
                qimg = QImage(img.data, w, h, QImage.Format.Format_Grayscale8)
            pix = QPixmap.fromImage(qimg).scaled(self.lbl_camera.size(), Qt.AspectRatioMode.KeepAspectRatio)
            self.lbl_camera.setPixmap(pix)
        except Exception:
            traceback.print_exc()

    def _recognition_loop(self, emb_matrix, names, threshold, resolution):
        """Worker thread: capture frames, run detection & embedding, draw overlays and send frames to UI."""
        try:
            cap = cv2.VideoCapture(0)
            if not cap or not cap.isOpened():
                self.log("Cannot open camera for recognition")
                return ""
            if resolution:
                try:
                    wres, hres = tuple(map(int, resolution.split('x')))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, wres)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hres)
                except Exception:
                    pass

            marked_set = set()
            while True:
                if getattr(threading.current_thread(), '_stop_request', False):
                    break
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                boxes = face_detect.detect_faces(frame, conf_threshold=0.6)
                current_rec = []
                for b in boxes:
                    try:
                        x, y, w, h, conf = int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(b[4] if len(b)>4 else 0.0)
                    except Exception:
                        try:
                            x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
                            x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
                            conf = float(b[4]) if len(b)>4 else 0.0
                        except Exception:
                            continue
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    face = frame[y1:y2, x1:x2]
                    emb = face_detect.get_embedding(face)
                    label_text = "NoEmb"
                    if emb is not None and emb_matrix.shape[0] > 0:
                        sims = emb_matrix @ emb
                        best_idx = int(_np.argmax(sims))
                        best_sim = float(sims[best_idx])
                        name = names[best_idx]
                        if best_sim >= threshold:
                            label_text = f"{name} ({best_sim:.2f})"
                            current_rec.append(name)
                        else:
                            label_text = f"Unknown ({best_sim:.2f})"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.putText(frame, label_text, (x1 + 2, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                try:
                    _INVOKER.call.emit(self._display_frame, (frame.copy(), current_rec))
                except Exception:
                    pass

                # brief wait so loop yields
                cv2.waitKey(1)

            cap.release()
            return ",".join(sorted(set(marked_set)))
        except Exception:
            traceback.print_exc()
            return ""

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
