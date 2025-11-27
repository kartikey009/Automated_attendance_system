# user_interface.py
"""
PySide6 modern UI for the Attendance System.
Restored "Professional" Look: Dark Theme, Sidebar Navigation, Modular Pages.
Switched from PyQt6 to PySide6 to resolve DLL load errors.
"""

import sys
import os
import threading
import pickle
import traceback
from datetime import datetime

# Switch to PySide6
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QFileDialog, QMessageBox, QListWidget, QTextEdit,
    QGroupBox, QGridLayout, QStackedWidget, QFrame, QSizePolicy
)
from PySide6.QtGui import QPixmap, QImage, QIcon, QFont, QColor, QPalette
from PySide6.QtCore import Qt, Signal, QObject, QTimer, QSize
import numpy as _np
import cv2

# import backend modules
import final_software_opencv as final
import attendance
import face_detect

# -------------------------------------------------------------------------
# STYLESHEET (Dark Theme)
# -------------------------------------------------------------------------
DARK_STYLESHEET = """
QMainWindow {
    background-color: #1e1e1e;
}
QWidget {
    background-color: #1e1e1e;
    color: #f0f0f0;
    font-family: "Segoe UI", sans-serif;
    font-size: 14px;
}
QGroupBox {
    border: 1px solid #3d3d3d;
    border-radius: 6px;
    margin-top: 24px;
    font-weight: bold;
    color: #e0e0e0;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 5px;
    left: 10px;
    color: #007acc;
}
QLineEdit {
    background-color: #2d2d2d;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    padding: 6px;
    color: #ffffff;
    selection-background-color: #007acc;
}
QLineEdit:focus {
    border: 1px solid #007acc;
}
QPushButton {
    background-color: #007acc;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #0098ff;
}
QPushButton:pressed {
    background-color: #005c99;
}
QPushButton:disabled {
    background-color: #3d3d3d;
    color: #808080;
}
QListWidget, QTextEdit {
    background-color: #252526;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    color: #e0e0e0;
}
QLabel {
    color: #cccccc;
}
/* Sidebar Styles */
QFrame#Sidebar {
    background-color: #252526;
    border-right: 1px solid #3d3d3d;
}
QPushButton[class="SidebarBtn"] {
    background-color: transparent;
    text-align: left;
    padding: 12px 20px;
    border-radius: 0px;
    border-left: 3px solid transparent;
    color: #cccccc;
    font-size: 15px;
}
QPushButton[class="SidebarBtn"]:hover {
    background-color: #2d2d2d;
    color: white;
}
QPushButton[class="SidebarBtn"]:checked {
    background-color: #1e1e1e;
    border-left: 3px solid #007acc;
    color: #007acc;
    font-weight: bold;
}
"""

# -------------------------------------------------------------------------
# THREADING HELPERS
# -------------------------------------------------------------------------

class _Invoker(QObject):
    """Helper to invoke callbacks on the main thread."""
    # In PySide6, Signal is used instead of pyqtSignal
    call = Signal(object, object)

    def __init__(self):
        super().__init__()
        self.call.connect(self._on_call)

    def _on_call(self, cb, args):
        try:
            cb(*args)
        except Exception:
            traceback.print_exc()

_INVOKER = _Invoker()

class Worker(threading.Thread):
    def __init__(self, fn, args=(), on_done=None):
        super().__init__()
        self.fn = fn
        self.args = args
        self.on_done = on_done
        self.result = None
        self.exc = None
        self._stop_request = False

    def run(self):
        try:
            self.result = self.fn(*self.args)
        except Exception as e:
            self.exc = e
            traceback.print_exc()
        finally:
            if self.on_done:
                # Schedule callback on main thread
                try:
                    _INVOKER.call.emit(self.on_done, (self.result, self.exc))
                except Exception:
                    pass


class VideoCaptureThread(threading.Thread):
    """
    Background thread that opens a cv2.VideoCapture and reads frames continuously.
    It calls on_open(ok, device_index) and on_frame(frame) back on the main/UI thread
    via the global _INVOKER.
    """
    def __init__(self, device, on_frame=None, on_open=None, read_fps=30):
        super().__init__(daemon=True)
        self.device = device
        self.on_frame = on_frame
        self.on_open = on_open
        self.read_fps = read_fps
        self._stop = threading.Event()
        self.cap = None

    def run(self):
        # Try to coerce device into int index or accept string filename
        device_index = self.device
        try:
            if device_index is None:
                device_index = 0
            # handle numpy bools
            try:
                import numpy as _np
                is_np_bool = isinstance(device_index, _np.bool_)
            except Exception:
                is_np_bool = False
            if isinstance(device_index, bool) or is_np_bool:
                device_index = 0
            if isinstance(device_index, str) and device_index.strip().isdigit():
                device_index = int(device_index.strip())
            else:
                if not isinstance(device_index, (int, str, os.PathLike)):
                    try:
                        device_index = int(device_index)
                    except Exception:
                        device_index = 0
        except Exception:
            device_index = 0

        # Select an API preference on Windows to avoid slow default backends
        api_pref = None
        try:
            if os.name == 'nt':
                api_pref = cv2.CAP_DSHOW
        except Exception:
            api_pref = None

        try:
            if api_pref is None:
                self.cap = cv2.VideoCapture(device_index)
            else:
                # pass api preference for faster open on Windows
                try:
                    self.cap = cv2.VideoCapture(device_index, api_pref)
                except TypeError:
                    # older OpenCV may not accept api pref; fallback
                    self.cap = cv2.VideoCapture(device_index)

            opened = self.cap is not None and self.cap.isOpened()
            # Notify main thread about open result
            if self.on_open:
                _INVOKER.call.emit(self.on_open, (opened, device_index))

            if not opened:
                return

            # Read frames until stopped
            delay = 1.0 / max(1, int(self.read_fps))
            while not self._stop.is_set():
                try:
                    ret, frame = self.cap.read()
                    if not ret or frame is None:
                        # small sleep if read fails
                        time.sleep(0.01)
                        continue
                    if self.on_frame:
                        _INVOKER.call.emit(self.on_frame, (frame,))
                    time.sleep(delay)
                except Exception:
                    time.sleep(0.01)
                    continue
        finally:
            try:
                if self.cap is not None:
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = None
            except Exception:
                pass

    def stop(self):
        self._stop.set()

# -------------------------------------------------------------------------
# PAGES
# -------------------------------------------------------------------------

class DatasetPage(QWidget):
    def __init__(self, log_callback):
        super().__init__()
        self.log = log_callback
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        title = QLabel("Create Dataset")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffffff;")
        layout.addWidget(title)

        group = QGroupBox("Dataset Configuration")
        gbox = QGridLayout(group)
        gbox.setVerticalSpacing(15)

        gbox.addWidget(QLabel("Output Directory:"), 0, 0)
        self.inp_ds_out = QLineEdit(os.path.join(os.getcwd(), "output"))
        gbox.addWidget(self.inp_ds_out, 0, 1)
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self.browse_root)
        gbox.addWidget(btn_browse, 0, 2)

        gbox.addWidget(QLabel("User Name:"), 1, 0)
        self.inp_ds_name = QLineEdit("person1")
        self.inp_ds_name.setPlaceholderText("Enter name of the person")
        gbox.addWidget(self.inp_ds_name, 1, 1)

        layout.addWidget(group)

        # Camera preview (embedded in UI)
        self.lbl_preview = QLabel("Camera Preview")
        self.lbl_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview.setMinimumSize(640, 480)
        self.lbl_preview.setStyleSheet("background-color: #000; border: 2px solid #3d3d3d; border-radius: 4px;")
        layout.addWidget(self.lbl_preview)

        # Camera controls
        ctrl_layout = QHBoxLayout()

        self.btn_start_cam = QPushButton("Open Camera")
        self.btn_start_cam.setMinimumHeight(40)
        self.btn_start_cam.setStyleSheet("font-size: 14px; background-color: #007acc;")
        self.btn_start_cam.clicked.connect(self.start_camera)
        ctrl_layout.addWidget(self.btn_start_cam)

        self.btn_burst = QPushButton("Start Burst Capture")
        self.btn_burst.setMinimumHeight(40)
        self.btn_burst.setStyleSheet("font-size: 14px; background-color: #f39c12; color: #000;")
        self.btn_burst.clicked.connect(self.start_burst_capture)
        self.btn_burst.setEnabled(False)
        ctrl_layout.addWidget(self.btn_burst)

        self.btn_stop_cam = QPushButton("Stop Camera")
        self.btn_stop_cam.setMinimumHeight(40)
        self.btn_stop_cam.setStyleSheet("font-size: 14px; background-color: #d9534f;")
        self.btn_stop_cam.clicked.connect(self.stop_camera)
        self.btn_stop_cam.setEnabled(False)
        ctrl_layout.addWidget(self.btn_stop_cam)

        layout.addLayout(ctrl_layout)
        
        layout.addStretch()

    def browse_root(self):
        d = QFileDialog.getExistingDirectory(self, "Select dataset root", self.inp_ds_out.text())
        if d:
            self.inp_ds_out.setText(d)

    def create_dataset(self):
        # Legacy create_dataset kept as compatibility route: start the camera instead
        self.start_camera()

    def _on_done(self, result, exc):
        if exc:
            self.log(f"Dataset creation failed: {exc}")
            QMessageBox.warning(self, "Error", f"Dataset creation failed: {exc}")
        else:
            self.log("Dataset creation completed successfully.")
            QMessageBox.information(self, "Success", "Dataset creation finished.")

    # ---------------------- Embedded camera logic ----------------------
    def start_camera(self, device_index=0, *args):
        # Open camera and start timer
        if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
            return
        try:
            # When connected to a QPushButton.clicked signal the slot may receive
            # a boolean checked argument (False) or other unexpected types. Normalize
            # common values into an int camera index or a filename-like string.
            if device_index is None:
                device_index = 0
            # treat numpy.bool_ and bool both as default camera (0)
            try:
                import numpy as _np
                is_np_bool = isinstance(device_index, _np.bool_)
            except Exception:
                is_np_bool = False

            if isinstance(device_index, bool) or is_np_bool:
                device_index = 0

            # numeric-like: '0', '1' or floats (0.0) -> int
            if isinstance(device_index, str) and device_index.strip().isdigit():
                device_index = int(device_index.strip())
            else:
                # try to coerce floats or numeric objects to int safely
                if not isinstance(device_index, (int, str, os.PathLike)):
                    try:
                        device_index = int(device_index)
                    except Exception:
                        # fallback to default camera
                        device_index = 0
            self.cap = cv2.VideoCapture(device_index)
            if not self.cap.isOpened():
                QMessageBox.warning(self, "Error", f"Unable to open camera: {device_index}")
                self.cap = None
                return
            # set reasonable resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            self.timer = QTimer(self)
            self.timer.timeout.connect(self._update_preview)
            self.timer.start(30)

            self.last_frame = None
            self.btn_start_cam.setEnabled(False)
            self.btn_burst.setEnabled(True)
            self.btn_stop_cam.setEnabled(True)
            self.log("Camera opened for dataset creation.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to open camera: {e}")

    def stop_camera(self, *args):
        # Stop timer and release camera
        try:
            if hasattr(self, 'timer') and self.timer is not None:
                self.timer.stop()
        except Exception:
            pass
        try:
            if hasattr(self, 'cap') and self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
        except Exception:
            pass

        self.lbl_preview.setPixmap(QPixmap())
        self.btn_start_cam.setEnabled(True)
        self.btn_burst.setEnabled(False)
        self.btn_stop_cam.setEnabled(False)
        self.log("Camera stopped.")

    def _update_preview(self):
        # Read a frame and display
        if not hasattr(self, 'cap') or self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return
        # save last frame for burst worker
        self.last_frame = frame.copy()
        # draw any helpful text
        disp = frame.copy()
        cv2.putText(disp, "Dataset Capture - press 'Start Burst Capture' to save frames.", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        # Convert to QImage
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(self.lbl_preview.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.lbl_preview.setPixmap(pix)

    def start_burst_capture(self, *args):
        # Verify path & username
        out = self.inp_ds_out.text().strip()
        name = self.inp_ds_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "Please enter a user name")
            return
        out_root = out if out else os.path.join(os.getcwd(), 'output')
        os.makedirs(out_root, exist_ok=True)
        folder_name = name.replace(' ', '_') if name.strip() != '' else 'person1'
        users_folder = os.path.join(out_root, folder_name)
        os.makedirs(users_folder, exist_ok=True)

        # disable controls
        self.btn_burst.setEnabled(False)
        self.btn_start_cam.setEnabled(False)
        self.btn_stop_cam.setEnabled(False)
        self.log(f"Starting burst capture into: {users_folder}")

        # run worker to save DEFAULT_SAVE_BURST frames
        w = Worker(self._burst_worker, args=(users_folder, final.DEFAULT_SAVE_BURST), on_done=self._on_burst_done)
        w.start()

    def _burst_worker(self, users_folder, count):
        saved = 0
        tries = 0
        # read from last_frame repeatedly until we saved `count` faces or camera closes
        while saved < count:
            if not hasattr(self, 'last_frame') or self.last_frame is None:
                tries += 1
                if tries > 100:
                    break
                import time; time.sleep(0.05)
                continue
            frame = self.last_frame.copy()
            # perform detection on current frame
            boxes = face_detect.detect_faces(frame, conf_threshold=0.60)
            if boxes:
                # pick largest
                boxes_sorted = sorted(boxes, key=lambda b: (b[2]*b[3]) if len(b) >= 4 else 0, reverse=True)
                # parse
                try:
                    x,y,w,h,_ = final._parse_box(boxes_sorted[0])
                except Exception:
                    # fallback: use list structure from detect_faces
                    b = boxes_sorted[0]
                    x,y,w,h = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                x1,y1 = max(0,x), max(0,y)
                x2,y2 = min(frame.shape[1], x+w), min(frame.shape[0], y+h)
                if x2 > x1 and y2 > y1:
                    face = frame[y1:y2, x1:x2]
                    fname = os.path.join(users_folder, f"{os.path.basename(users_folder)}_{str(saved+1).zfill(4)}.png")
                    try:
                        cv2.imwrite(fname, face)
                        saved += 1
                    except Exception:
                        pass
            import time; time.sleep(0.03)
        return saved

    def _on_burst_done(self, result, exc):
        if exc:
            self.log(f"Burst capture failed: {exc}")
            QMessageBox.warning(self, "Error", f"Burst capture failed: {exc}")
        else:
            self.log(f"Burst capture completed: saved {result} images")
            QMessageBox.information(self, "Done", f"Saved {result} images.")

        # re-enable controls if camera still active
        self.btn_start_cam.setEnabled(True)
        self.btn_stop_cam.setEnabled(True if hasattr(self,'cap') and self.cap is not None else False)
        self.btn_burst.setEnabled(True)


class TrainPage(QWidget):
    def __init__(self, log_callback):
        super().__init__()
        self.log = log_callback
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        title = QLabel("Train Model")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffffff;")
        layout.addWidget(title)

        group = QGroupBox("Training Configuration")
        gbox = QGridLayout(group)
        gbox.setVerticalSpacing(15)

        gbox.addWidget(QLabel("Dataset Root:"), 0, 0)
        self.inp_train_root = QLineEdit(os.path.join(os.getcwd(), "output"))
        gbox.addWidget(self.inp_train_root, 0, 1)
        
        gbox.addWidget(QLabel("Classifier Filename:"), 1, 0)
        self.inp_classifier = QLineEdit("classifier_buffalo")
        self.inp_classifier.setPlaceholderText("e.g. classifier_buffalo")
        gbox.addWidget(self.inp_classifier, 1, 1)

        layout.addWidget(group)

        btn_train = QPushButton("Start Training")
        btn_train.setMinimumHeight(50)
        btn_train.setStyleSheet("font-size: 16px; background-color: #28a745;") # Green for train
        btn_train.clicked.connect(self.start_train)
        layout.addWidget(btn_train)

        layout.addStretch()

    def start_train(self):
        root = self.inp_train_root.text().strip()
        clf_name = self.inp_classifier.text().strip()
        
        if not os.path.isdir(root):
            QMessageBox.warning(self, "Error", "Invalid dataset root directory")
            return
            
        params = (root, "", "", "", "", clf_name, "", "")
        self.log(f"Starting training with root='{root}'...")
        w = Worker(final.train, args=(params,), on_done=self._on_done)
        w.start()

    def _on_done(self, result, exc):
        if exc:
            self.log(f"Training failed: {exc}")
            QMessageBox.warning(self, "Error", f"Training failed: {exc}")
        else:
            self.log("Training finished successfully.")
            QMessageBox.information(self, "Success", "Training finished. You can now use the classifier for recognition.")


class RecognitionPage(QWidget):
    def __init__(self, log_callback, attendance_callback):
        super().__init__()
        self.log = log_callback
        self.refresh_attendance = attendance_callback
        self._recog_worker = None
        self._recog_running = False
        self._recog_current_names = set()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        # Top bar: Settings
        top_layout = QHBoxLayout()
        self.inp_clf_path = QLineEdit("classifier_buffalo")
        self.inp_clf_path.setPlaceholderText("Classifier Path")
        btn_load = QPushButton("Load Classifier")
        btn_load.clicked.connect(self.load_classifier)
        
        top_layout.addWidget(QLabel("Classifier:"))
        top_layout.addWidget(self.inp_clf_path)
        top_layout.addWidget(btn_load)
        layout.addLayout(top_layout)

        # Camera View
        self.lbl_camera = QLabel("Camera Feed")
        self.lbl_camera.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.lbl_camera.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_camera.setStyleSheet("background-color: #000; border: 2px solid #3d3d3d; border-radius: 4px;")
        self.lbl_camera.setMinimumSize(640, 480)
        layout.addWidget(self.lbl_camera)

        # Controls
        controls = QHBoxLayout()
        self.btn_start = QPushButton("Start Recognition")
        self.btn_start.clicked.connect(self.start_recognition)
        self.btn_start.setStyleSheet("background-color: #007acc; font-size: 16px; padding: 10px;")
        
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_recognition)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("background-color: #d9534f; font-size: 16px; padding: 10px;")

        self.btn_mark = QPushButton("Mark Present")
        self.btn_mark.clicked.connect(self._mark_current)
        self.btn_mark.setStyleSheet("background-color: #f0ad4e; color: #000; font-size: 16px; padding: 10px;")

        controls.addWidget(self.btn_start)
        controls.addWidget(self.btn_stop)
        controls.addWidget(self.btn_mark)
        layout.addLayout(controls)

    def load_classifier(self):
        fname = self.inp_clf_path.text().strip()
        if not fname: fname = "classifier_buffalo"
        
        # Search logic
        cand_name = fname if fname.endswith(".pkl") else fname + ".pkl"
        candidates = [os.path.abspath(cand_name)]
        # Add common paths
        candidates.append(os.path.join(os.getcwd(), cand_name))
        candidates.append(os.path.join(os.getcwd(), "output", cand_name))
        
        found = None
        for p in candidates:
            if os.path.exists(p):
                found = p
                break
        
        if found:
            self.inp_clf_path.setText(found)
            self.log(f"Classifier found: {found}")
            QMessageBox.information(self, "Classifier", f"Found: {found}")
            return found
        else:
            self.log(f"Classifier not found. Tried: {candidates}")
            QMessageBox.warning(self, "Error", "Classifier not found.")
            return None

    def start_recognition(self):
        path = self.load_classifier()
        if not path: return

        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            self.log(f"Error loading pickle: {e}")
            return

        embeddings_dict = data.get('embeddings', {})
        if not embeddings_dict:
            QMessageBox.warning(self, "Error", "No embeddings in classifier.")
            return

        threshold = data.get('threshold', 0.6)
        names = list(embeddings_dict.keys())
        emb_matrix = _np.vstack([embeddings_dict[n] for n in names]).astype(_np.float32)

        self._recog_running = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        
        self._recog_worker = Worker(self._recognition_loop, args=(emb_matrix, names, threshold), on_done=self._on_recog_done)
        self._recog_worker.start()
        self.log("Recognition started.")

    def stop_recognition(self):
        if self._recog_worker:
            self._recog_worker._stop_request = True
            self.log("Stopping recognition...")

    def _on_recog_done(self, result, exc):
        self._recog_running = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_camera.clear()
        self.lbl_camera.setText("Camera Stopped")
        if exc:
            self.log(f"Recognition error: {exc}")
        else:
            self.log(f"Recognition finished. Result: {result}")

    def _mark_current(self):
        if not self._recog_current_names:
            self.log("No faces recognized to mark.")
            return
        for nm in list(self._recog_current_names):
            if attendance.mark_present(nm):
                self.log(f"Marked present: {nm}")
        self.refresh_attendance()

    def _display_frame(self, frame, recognized):
        self._recog_current_names = set(recognized)
        if frame is None: return
        
        try:
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
            pass

    def _recognition_loop(self, emb_matrix, names, threshold):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Could not open camera")
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        marked_set = set()
        
        while True:
            if getattr(threading.current_thread(), '_stop_request', False):
                break
            
            ret, frame = cap.read()
            if not ret: break

            boxes = face_detect.detect_faces(frame, conf_threshold=0.6)
            current_rec = []

            for b in boxes:
                x, y, w, h = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
                
                face = frame[y1:y2, x1:x2]
                if face.size == 0: continue

                emb = face_detect.get_embedding(face)
                label_text = "Unknown"
                color = (0, 0, 255) # Red for unknown

                if emb is not None:
                    sims = emb_matrix @ emb
                    best_idx = int(_np.argmax(sims))
                    best_sim = float(sims[best_idx])
                    
                    if best_sim >= threshold:
                        name = names[best_idx]
                        label_text = f"{name} ({best_sim:.2f})"
                        current_rec.append(name)
                        color = (0, 255, 0) # Green for known

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Send to UI
            _INVOKER.call.emit(self._display_frame, (frame.copy(), current_rec))
            cv2.waitKey(1)

        cap.release()
        return list(marked_set)


class AttendancePage(QWidget):
    def __init__(self, log_callback):
        super().__init__()
        self.log_callback = log_callback
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        title = QLabel("Attendance Records")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffffff;")
        layout.addWidget(title)

        # List
        self.lst_att = QListWidget()
        layout.addWidget(self.lst_att, 2)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_refresh = QPushButton("Refresh List")
        btn_refresh.clicked.connect(self.refresh_list)
        btn_clear = QPushButton("Clear Today's Data")
        btn_clear.clicked.connect(self.clear_data)
        btn_clear.setStyleSheet("background-color: #d9534f;") # Red warning color
        
        btn_layout.addWidget(btn_refresh)
        btn_layout.addWidget(btn_clear)
        layout.addLayout(btn_layout)

        # Logs
        layout.addWidget(QLabel("System Logs:"))
        self.txt_logs = QTextEdit()
        self.txt_logs.setReadOnly(True)
        layout.addWidget(self.txt_logs, 1)

        self.refresh_list()

    def refresh_list(self):
        self.lst_att.clear()
        names = attendance.get_marked()
        for n in names:
            self.lst_att.addItem(n)
    
    def clear_data(self):
        if QMessageBox.question(self, "Confirm", "Clear all attendance for today?") == QMessageBox.StandardButton.Yes:
            attendance.clear_today()
            self.refresh_list()
            self.log("Attendance cleared.")

    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.txt_logs.append(f"[{ts}] {msg}")
        self.txt_logs.ensureCursorVisible()


# -------------------------------------------------------------------------
# MAIN WINDOW
# -------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Automated Attendance System")
        self.resize(1100, 750)
        
        # Central Widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar
        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(220)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(0, 20, 0, 20)
        sidebar_layout.setSpacing(5)

        # Sidebar Title
        lbl_title = QLabel("Attendance\nSystem")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title.setStyleSheet("font-size: 20px; font-weight: bold; color: #ffffff; padding-bottom: 20px;")
        sidebar_layout.addWidget(lbl_title)

        # Sidebar Buttons
        self.btn_ds = self._create_sidebar_btn("Dataset Creation")
        self.btn_train = self._create_sidebar_btn("Train Model")
        self.btn_recog = self._create_sidebar_btn("Recognition")
        self.btn_att = self._create_sidebar_btn("Attendance & Logs")
        
        sidebar_layout.addWidget(self.btn_ds)
        sidebar_layout.addWidget(self.btn_train)
        sidebar_layout.addWidget(self.btn_recog)
        sidebar_layout.addWidget(self.btn_att)
        sidebar_layout.addStretch()
        
        # Footer in sidebar
        lbl_ver = QLabel("v2.0 Pro")
        lbl_ver.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_ver.setStyleSheet("color: #666; font-size: 12px;")
        sidebar_layout.addWidget(lbl_ver)

        main_layout.addWidget(self.sidebar)

        # Content Area
        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack)

        # Initialize Pages
        self.page_att = AttendancePage(self.log_message) # Init first to receive logs
        self.page_ds = DatasetPage(self.log_message)
        self.page_train = TrainPage(self.log_message)
        self.page_recog = RecognitionPage(self.log_message, self.page_att.refresh_list)

        self.stack.addWidget(self.page_ds)    # Index 0
        self.stack.addWidget(self.page_train) # Index 1
        self.stack.addWidget(self.page_recog) # Index 2
        self.stack.addWidget(self.page_att)   # Index 3

        # Connect Buttons
        self.btn_ds.clicked.connect(lambda: self.switch_page(0, self.btn_ds))
        self.btn_train.clicked.connect(lambda: self.switch_page(1, self.btn_train))
        self.btn_recog.clicked.connect(lambda: self.switch_page(2, self.btn_recog))
        self.btn_att.clicked.connect(lambda: self.switch_page(3, self.btn_att))

        # Default Page
        self.btn_recog.click()

    def _create_sidebar_btn(self, text):
        btn = QPushButton(text)
        btn.setCheckable(True)
        btn.setAutoExclusive(True)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        # We use setProperty to style specific buttons if needed, 
        # but here we rely on the class selector in QSS: QPushButton[class="SidebarBtn"]
        btn.setProperty("class", "SidebarBtn") 
        return btn

    def switch_page(self, index, btn_sender):
        self.stack.setCurrentIndex(index)
        # Ensure button is checked (handled by AutoExclusive, but good to be explicit if needed)
        btn_sender.setChecked(True)

    def log_message(self, msg):
        # Forward logs to the attendance page log viewer
        self.page_att.log(msg)


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)
    
    # Fix for high DPI displays
    if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt.ApplicationAttribute, 'AA_UseHighDpiPixmaps'):
        app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
