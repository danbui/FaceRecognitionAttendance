import sys
import time
import queue
import cv2
import numpy as np
import argparse
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QStackedWidget, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QLineEdit, QFormLayout, QMessageBox
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont

from .camera_service import CameraService
from .face_detector import FaceDetector, is_face_inside_guide
from .face_embedder import FaceEmbedder
from .attendance_service import (
    enroll_employee, recognize_and_attend,
    confirm_enroll_update, enroll_new_with_embedding,
)
from .database import init_db
from .config import STABLE_FACE_SECONDS

# ── Colors for OpenCV drawing ───────────────────────────
COLOR_YELLOW = (0, 255, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY = (180, 180, 180)
COLOR_DARK_BG = (30, 30, 30)
COLOR_ORANGE = (0, 140, 255)


# ═══════════════════════════════════════════════════════════
#  Helper Functions for OpenCV Drawing
# ═══════════════════════════════════════════════════════════

def guide_box(frame):
    """Calculate the guide frame rectangle (center of screen)."""
    h, w, _ = frame.shape
    gw, gh = int(w * 0.35), int(h * 0.50)
    gx, gy = (w - gw) // 2, (h - gh) // 2
    return gx, gy, gw, gh

def draw_text_centered(frame, text, y, font_scale=0.8, color=COLOR_WHITE, thickness=2):
    """Draw text centered horizontally on the frame."""
    h, w = frame.shape[:2]
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    x = (w - text_size[0]) // 2
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def draw_status_bar(frame, text, color, bg_alpha=0.7):
    """Draw a status bar at the top of the screen."""
    h, w = frame.shape[:2]
    bar_h = 60
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

def draw_result_overlay(frame, text, color, sub_text=""):
    """Draw a large centered result overlay (for success/error/duplicate)."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h // 3), (w, 2 * h // 3), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    draw_text_centered(frame, text, h // 2 - 10, 0.9, color, 2)
    if sub_text:
        draw_text_centered(frame, sub_text, h // 2 + 30, 0.6, COLOR_GRAY, 1)


# ═══════════════════════════════════════════════════════════
#  QThreads
# ═══════════════════════════════════════════════════════════

class CameraWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, camera_service: CameraService):
        super().__init__()
        self.cam = camera_service
        self.running = False

    def run(self):
        self.running = True
        for frame in self.cam.frames():
            if not self.running:
                break
            self.frame_ready.emit(frame)

    def stop(self):
        self.running = False
        self.wait()


class AIWorker(QThread):
    result_ready = pyqtSignal(dict)

    def __init__(self, embedder: FaceEmbedder):
        super().__init__()
        self.embedder = embedder
        self.request_q = queue.Queue()
        self.running = False

    def submit(self, task: dict):
        self.request_q.put(task)

    def run(self):
        self.running = True
        while self.running:
            try:
                task = self.request_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if task is None:
                continue

            try:
                if task["action"] == "enroll":
                    res = enroll_employee(
                        task["employee_code"],
                        task["full_name"],
                        task["department"],
                        task["frame"],
                        task["face_raw"],
                        self.embedder,
                    )
                elif task["action"] == "enroll_add":
                    confirm_enroll_update(
                        task["employee_id"],
                        task["embedding"],
                        task["image_path"],
                    )
                    res = {
                        "status": "enrolled",
                        "employee_id": task["employee_id"],
                        "employee_code": task["employee_code"],
                        "full_name": task["full_name"],
                        "image_path": task["image_path"],
                    }
                elif task["action"] == "enroll_force":
                    res = enroll_new_with_embedding(
                        task["employee_code"],
                        task["full_name"],
                        task["department"],
                        task["embedding"],
                        task["image_path"],
                    )
                elif task["action"] == "attend":
                    res = recognize_and_attend(
                        task["frame"],
                        task["face_raw"],
                        self.embedder,
                        check_type=task["check_type"],
                    )
            except Exception as e:
                res = {"status": "error", "message": str(e)}

            self.result_ready.emit(res)

    def stop(self):
        self.running = False
        self.wait()


# ═══════════════════════════════════════════════════════════
#  Main Window
# ═══════════════════════════════════════════════════════════

class KioskWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setWindowTitle("Edge Attendance Kiosk")
        self.resize(800, 600)
        self.setStyleSheet("background-color: #1e1e1e; color: white;")

        # State
        self.current_mode = None
        self.stable_start = None
        self.last_result = None
        self.result_display_until = 0
        self.ai_busy = False
        self.pending_confirm = None

        # Init components
        init_db()
        self.cam_service = CameraService(args.camera, demo=args.demo).start()
        self.detector = FaceDetector()
        self.embedder = FaceEmbedder()

        # Threads
        self.cam_worker = CameraWorker(self.cam_service)
        self.cam_worker.frame_ready.connect(self.process_frame)
        self.cam_worker.start()

        self.ai_worker = AIWorker(self.embedder)
        self.ai_worker.result_ready.connect(self.handle_ai_result)
        self.ai_worker.start()

        # UI Setup
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.setup_mode_selection_ui()
        self.setup_enroll_input_ui()
        self.setup_camera_ui()

    def closeEvent(self, event):
        self.cam_worker.stop()
        self.ai_worker.stop()
        self.cam_service.release()
        super().closeEvent(event)

    # ── UI Construction ──

    def setup_mode_selection_ui(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignCenter)

        title = QLabel("EDGE ATTENDANCE SYSTEM")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        layout.addSpacing(30)

        btn_enroll = QPushButton("1. GHI DANH (Enroll)")
        btn_enroll.setMinimumHeight(60)
        btn_enroll.setStyleSheet("background-color: #32a0ff; color: #1e1e1e; font-weight: bold; font-size: 16px; border-radius: 5px;")
        btn_enroll.clicked.connect(lambda: self.switch_to_enroll_input())
        layout.addWidget(btn_enroll)
        layout.addSpacing(10)

        btn_checkin = QPushButton("2. ĐIỂM DANH VÀO (Check-in)")
        btn_checkin.setMinimumHeight(60)
        btn_checkin.setStyleSheet("background-color: #00ff00; color: #1e1e1e; font-weight: bold; font-size: 16px; border-radius: 5px;")
        btn_checkin.clicked.connect(lambda: self.switch_to_camera("check_in"))
        layout.addWidget(btn_checkin)
        layout.addSpacing(10)

        btn_checkout = QPushButton("3. ĐIỂM DANH RA (Check-out)")
        btn_checkout.setMinimumHeight(60)
        btn_checkout.setStyleSheet("background-color: #ff8c00; color: #1e1e1e; font-weight: bold; font-size: 16px; border-radius: 5px;")
        btn_checkout.clicked.connect(lambda: self.switch_to_camera("check_out"))
        layout.addWidget(btn_checkout)

        self.stack.addWidget(page)

    def setup_enroll_input_ui(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignCenter)

        title = QLabel("NHẬP THÔNG TIN NHÂN VIEN")
        title.setFont(QFont("Arial", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        layout.addSpacing(30)

        form_layout = QFormLayout()
        form_layout.setLabelAlignment(Qt.AlignRight)
        
        self.input_code = QLineEdit()
        self.input_code.setMinimumHeight(40)
        self.input_code.setStyleSheet("background-color: #323232; border: 1px solid #555; border-radius: 3px; font-size: 16px; padding: 5px;")
        form_layout.addRow(QLabel("Mã nhân viên:"), self.input_code)

        self.input_name = QLineEdit()
        self.input_name.setMinimumHeight(40)
        self.input_name.setStyleSheet("background-color: #323232; border: 1px solid #555; border-radius: 3px; font-size: 16px; padding: 5px;")
        form_layout.addRow(QLabel("Họ và tên:"), self.input_name)

        self.input_dept = QLineEdit()
        self.input_dept.setMinimumHeight(40)
        self.input_dept.setStyleSheet("background-color: #323232; border: 1px solid #555; border-radius: 3px; font-size: 16px; padding: 5px;")
        form_layout.addRow(QLabel("Phòng ban:"), self.input_dept)

        layout.addLayout(form_layout)
        layout.addSpacing(30)

        btn_submit = QPushButton("Tiếp tục (Enter)")
        btn_submit.setMinimumHeight(50)
        btn_submit.setStyleSheet("background-color: #00ff00; color: #1e1e1e; font-weight: bold; font-size: 16px; border-radius: 5px;")
        btn_submit.clicked.connect(self.submit_enroll_form)
        layout.addWidget(btn_submit)

        btn_cancel = QPushButton("Hủy (ESC)")
        btn_cancel.setMinimumHeight(40)
        btn_cancel.setStyleSheet("background-color: #555; color: white; font-weight: bold; font-size: 14px; border-radius: 5px;")
        btn_cancel.clicked.connect(self.switch_to_mode_selection)
        layout.addWidget(btn_cancel)

        self.stack.addWidget(page)

    def setup_camera_ui(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.camera_label)

        # Footer overlay (pseudo) using a fixed layout or just let OpenCV draw it
        # Here we let OpenCV draw the overlays directly on the frame.
        self.stack.addWidget(page)

    # ── Transitions ──

    def switch_to_mode_selection(self):
        self.current_mode = None
        self.stack.setCurrentIndex(0)

    def switch_to_enroll_input(self):
        self.input_code.clear()
        self.input_name.clear()
        self.input_dept.clear()
        self.input_code.setFocus()
        self.stack.setCurrentIndex(1)

    def submit_enroll_form(self):
        if not self.input_code.text().strip() or not self.input_name.text().strip():
            QMessageBox.warning(self, "Lỗi", "Vui lòng nhập đủ Mã nhân viên và Họ tên!")
            return
        self.switch_to_camera("enroll")

    def switch_to_camera(self, mode):
        self.current_mode = mode
        self.stable_start = None
        self.last_result = None
        self.ai_busy = False
        self.stack.setCurrentIndex(2)

    def keyPressEvent(self, event):
        if self.stack.currentIndex() == 0:  # Mode selection
            if event.key() == Qt.Key_1:
                self.switch_to_enroll_input()
            elif event.key() == Qt.Key_2:
                self.switch_to_camera("check_in")
            elif event.key() == Qt.Key_3:
                self.switch_to_camera("check_out")
            elif event.key() == Qt.Key_Q:
                self.close()

        elif self.stack.currentIndex() == 1:  # Enroll input
            if event.key() == Qt.Key_Escape:
                self.switch_to_mode_selection()
            elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
                self.submit_enroll_form()

        elif self.stack.currentIndex() == 2:  # Camera
            if event.key() == Qt.Key_Escape:
                self.switch_to_mode_selection()

    # ── Logic ──

    def handle_ai_result(self, result):
        self.ai_busy = False
        if result.get("status") == "confirm_duplicate":
            # Handle duplicate confirmation via QMessageBox
            reply = QMessageBox.question(
                self, "Cảnh báo trùng lặp",
                f"Khuôn mặt này giống với nhân viên đã có:\n"
                f"- Tên: {result['existing_full_name']}\n"
                f"- Mã NV: {result['existing_employee_code']}\n"
                f"- Độ tin cậy: {result['confidence']:.3f}\n\n"
                f"Bạn có muốn cập nhật thêm khuôn mặt này cho nhân viên cũ không?\n"
                f"(Yes = Cập nhật NV cũ | No = Vẫn tạo NV mới)",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )

            if reply == QMessageBox.Yes:
                task = {
                    "action": "enroll_add",
                    "employee_id": result["existing_employee_id"],
                    "employee_code": result["existing_employee_code"],
                    "full_name": result["existing_full_name"],
                    "embedding": result["embedding"],
                    "image_path": result["image_path"],
                }
                self.ai_worker.submit(task)
                self.ai_busy = True
            elif reply == QMessageBox.No:
                task = {
                    "action": "enroll_force",
                    "employee_code": result["new_employee_code"],
                    "full_name": result["new_full_name"],
                    "department": result["new_department"],
                    "embedding": result["embedding"],
                    "image_path": result["image_path"],
                }
                self.ai_worker.submit(task)
                self.ai_busy = True
            else:
                self.switch_to_enroll_input()
        else:
            self.last_result = result
            self.result_display_until = time.time() + 4.0

    def process_frame(self, frame):
        if self.stack.currentIndex() != 2:
            return  # Not on camera page

        now = time.time()
        
        # ── OpenCV Drawing Logic ──
        guide = guide_box(frame)
        gx, gy, gw, gh = guide
        guide_color = COLOR_YELLOW
        status_text = "Dua mat vao khung vang"

        showing_result = self.last_result and now < self.result_display_until

        if showing_result or self.ai_busy:
            face_box, face_raw = None, None
            self.stable_start = None
        else:
            face_box, face_raw = self.detector.detect_largest_with_raw(frame)
            if self.last_result and now >= self.result_display_until:
                # Auto return after enrollment
                if self.current_mode == "enroll" and self.last_result.get("status") == "enrolled":
                    self.switch_to_enroll_input()
                    return
                self.last_result = None

        if face_box:
            fx, fy, fw, fh = face_box
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 255, 0), 2)

            if is_face_inside_guide(face_box, guide):
                guide_color = COLOR_GREEN

                if self.stable_start is None:
                    self.stable_start = now

                elapsed = now - self.stable_start
                if elapsed >= STABLE_FACE_SECONDS:
                    if self.current_mode == "enroll":
                        task = {
                            "action": "enroll",
                            "employee_code": self.input_code.text(),
                            "full_name": self.input_name.text(),
                            "department": self.input_dept.text(),
                            "frame": frame.copy(),
                            "face_raw": face_raw.copy(),
                        }
                    else:
                        task = {
                            "action": "attend",
                            "frame": frame.copy(),
                            "face_raw": face_raw.copy(),
                            "check_type": "CHECK_IN" if self.current_mode == "check_in" else "CHECK_OUT",
                        }
                    
                    self.ai_worker.submit(task)
                    self.ai_busy = True
                    self.stable_start = None
                else:
                    progress = elapsed / STABLE_FACE_SECONDS
                    status_text = f"Giu yen... {progress * 100:.0f}%"
                    # Progress bar
                    bar_x, bar_y = gx, gy + gh + 10
                    bar_w = int(gw * progress)
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + gw, bar_y + 8), (50, 50, 50), -1)
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8), COLOR_GREEN, -1)
            else:
                self.stable_start = None
                status_text = "Can chinh mat vao giua khung"
        else:
            self.stable_start = None

        # Draw guide
        cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), guide_color, 3)
        corner_len = 20
        corners = [
            ((gx, gy), (gx + corner_len, gy), (gx, gy + corner_len)),
            ((gx + gw, gy), (gx + gw - corner_len, gy), (gx + gw, gy + corner_len)),
            ((gx, gy + gh), (gx + corner_len, gy + gh), (gx, gy + gh - corner_len)),
            ((gx + gw, gy + gh), (gx + gw - corner_len, gy + gh), (gx + gw, gy + gh - corner_len)),
        ]
        for pt, pt_h, pt_v in corners:
            cv2.line(frame, pt, pt_h, guide_color, 4)
            cv2.line(frame, pt, pt_v, guide_color, 4)

        # Draw result overlay
        if self.last_result and now < self.result_display_until:
            r = self.last_result
            if r["status"] == "enrolled":
                draw_result_overlay(frame, f"Da ghi danh: {r['full_name']}", COLOR_GREEN, f"Ma NV: {r['employee_code']}")
            elif r["status"] == "success":
                draw_result_overlay(frame, f"{r['full_name']}", COLOR_GREEN, f"{r['check_type']} - Do tin cay: {r['confidence']:.3f}")
            elif r["status"] == "duplicate":
                draw_result_overlay(frame, f"Da ghi nhan: {r['full_name']}", COLOR_ORANGE, r["message"])
            elif r["status"] == "not_found":
                draw_result_overlay(frame, "Khong tim thay nhan vien", COLOR_RED, "Vui long ghi danh truoc")
            elif r["status"] == "error":
                draw_result_overlay(frame, "LOI HE THONG", COLOR_RED, r.get("message", ""))
            status_text = ""

        if self.ai_busy:
            status_text = "Dang xu ly..."

        mode_labels = {
            "enroll": "GHI DANH",
            "check_in": "DIEM DANH VAO",
            "check_out": "DIEM DANH RA",
        }
        draw_status_bar(frame, f"[{mode_labels.get(self.current_mode, '')}] {status_text}", guide_color)

        cv2.putText(frame, "ESC: Quay lai", (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GRAY, 1)

        # ── Convert to QImage and display ──
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(qimg))


def main():
    parser = argparse.ArgumentParser(description="Edge Attendance Kiosk (Qt Version)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--demo", action="store_true", help="Run without camera")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    # Set default font for the whole app
    app.setFont(QFont("Arial", 12))
    
    window = KioskWindow(args)
    # window.showFullScreen() # Uncomment for production deployment on Pi
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
